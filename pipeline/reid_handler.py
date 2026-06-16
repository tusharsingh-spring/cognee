"""Person Re-Identification using CLIP for cross-session visual matching."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    DATA_DIR,
    REID_ENABLED,
    REID_MATCH_THRESHOLD,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class ReIDHandler:
    def __init__(self) -> None:
        self._loaded = False
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._global_db: Dict[str, Dict] = {}
        self._db_path = DATA_DIR / "reid" / "identity_db.json"
        self._session_matches: Dict[int, str] = {}
        self._id_counter = 0
        self._use_clip = False

    def load(self) -> None:
        if self._loaded:
            return
        if not REID_ENABLED:
            logger.info("[REID] Re-identification disabled")
            self._loaded = True
            return

        logger.info("[REID] Loading person re-identification system...")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_db()

        try:
            import torch
            import open_clip
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            self._tokenizer = open_clip.get_tokenizer("ViT-B-32")
            self._model.eval()
            self._use_clip = True
            logger.info("[REID] CLIP ViT-B-32 loaded for visual embeddings")
        except Exception as e:
            logger.warning(f"[REID] CLIP unavailable ({e}), using fallback")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self._use_clip = False
                logger.info("[REID] Using sentence-transformers fallback")
            except Exception as e2:
                logger.warning(f"[REID] Fallback unavailable: {e2}")

        self._loaded = True
        logger.info(f"[REID] Loaded {len(self._global_db)} identities from DB")

    def extract_features(self, crop: np.ndarray) -> Optional[np.ndarray]:
        if crop is None or crop.size == 0:
            return None

        try:
            if self._use_clip:
                import torch
                from PIL import Image
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                img_tensor = self._preprocess(pil).unsqueeze(0)
                with torch.no_grad():
                    emb = self._model.encode_image(img_tensor)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                return emb.numpy().flatten()
            elif self._model is not None:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                small = cv2.resize(rgb, (128, 256))
                from PIL import Image
                pil = Image.fromarray(small)
                text_desc = self._describe_appearance(small)
                emb = self._model.encode(text_desc, convert_to_numpy=True)
                return emb
        except Exception as e:
            logger.debug(f"[REID] Feature extraction error: {e}")

        return self._simple_embedding(crop)

    def _simple_embedding(self, crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(crop, (64, 128))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) > 2 else resized
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        color_hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        color_hist = cv2.normalize(color_hist, color_hist).flatten()
        hog = self._compute_hog(gray)
        texture = cv2.Laplacian(gray, cv2.CV_64F).var()
        emb = np.concatenate([color_hist, hog[:64], [texture / 1000.0]])
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb

    def _compute_hog(self, gray: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bins = np.linspace(0, np.pi, 9)
        hist = np.histogram(ang.flatten(), bins=bins, weights=mag.flatten())[0]
        hist = hist / (hist.sum() + 1e-8)
        return hist

    def _describe_appearance(self, img: np.ndarray) -> str:
        h, w = img.shape[:2]
        upper = img[:h//2, :]
        lower = img[h//2:, :]
        upper_mean = upper.mean(axis=(0, 1))
        lower_mean = lower.mean(axis=(0, 1))
        parts = []
        r, g, b_val = upper_mean[2], upper_mean[1], upper_mean[0]
        if r > max(g, b_val) + 20:
            parts.append("red top")
        elif g > max(r, b_val) + 20:
            parts.append("green top")
        elif b_val > max(r, g) + 20:
            parts.append("blue top")
        else:
            gray_val = int(upper.mean())
            parts.append("dark top" if gray_val < 80 else "light top" if gray_val > 150 else "neutral top")
        r2, g2, b2 = lower_mean[2], lower_mean[1], lower_mean[0]
        if r2 > max(g2, b2) + 20:
            parts.append("red bottom")
        elif g2 > max(r2, b2) + 20:
            parts.append("green bottom")
        elif b2 > max(r2, g2) + 20:
            parts.append("blue bottom")
        else:
            gray2 = int(lower.mean())
            parts.append("dark bottom" if gray2 < 80 else "light bottom" if gray2 > 150 else "neutral bottom")
        return "person with " + " and ".join(parts)

    def match_identity(self, track_id: int, crop: np.ndarray) -> Optional[Dict]:
        emb = self.extract_features(crop)
        if emb is None:
            return None

        best_id = None
        best_sim = -1.0
        best_info = {}

        for gid, info in self._global_db.items():
            stored_emb = np.array(info.get("embedding", []))
            if stored_emb.size == 0 or stored_emb.size != emb.size:
                continue
            sim = float(np.dot(emb, stored_emb) / (np.linalg.norm(emb) * np.linalg.norm(stored_emb) + 1e-8))
            if sim > best_sim and sim >= REID_MATCH_THRESHOLD:
                best_sim = sim
                best_id = gid
                best_info = info

        if best_id:
            prev = self._session_matches.get(track_id)
            self._session_matches[track_id] = best_id
            self._update_identity(best_id, emb, track_id)
            if prev != best_id:
                logger.info(f"[REID] Person_{track_id} matched {best_id} (sim={best_sim:.2f})")
            return {"global_id": best_id, "similarity": best_sim, "info": best_info}

        new_id = f"person_{int(time.time())}_{self._id_counter}"
        self._id_counter += 1
        self._store_identity(new_id, emb, track_id)
        self._session_matches[track_id] = new_id
        logger.info(f"[REID] New identity: {new_id} (Person_{track_id})")
        return {"global_id": new_id, "similarity": 1.0, "info": {"first_seen": time.time()}}

    def _store_identity(self, gid: str, emb: np.ndarray, track_id: int) -> None:
        self._global_db[gid] = {
            "embedding": emb.tolist(),
            "first_seen": time.time(),
            "last_seen": time.time(),
            "sightings": 1,
            "track_ids": [track_id],
        }
        self._save_db()

    def _update_identity(self, gid: str, emb: np.ndarray, track_id: int) -> None:
        if gid in self._global_db:
            info = self._global_db[gid]
            stored = np.array(info["embedding"])
            if stored.shape == emb.shape:
                alpha = 0.85
                info["embedding"] = (alpha * stored + (1 - alpha) * emb).tolist()
            info["last_seen"] = time.time()
            info["sightings"] = info.get("sightings", 0) + 1
            if track_id not in info.get("track_ids", []):
                info.setdefault("track_ids", []).append(track_id)

    def get_global_id(self, track_id: int) -> Optional[str]:
        return self._session_matches.get(track_id)

    def get_identity_info(self, gid: str) -> Optional[Dict]:
        return self._global_db.get(gid)

    def get_session_history(self) -> List[Dict]:
        return [
            {"global_id": gid, "first_seen": info.get("first_seen"),
             "last_seen": info.get("last_seen"), "sightings": info.get("sightings", 0)}
            for gid, info in self._global_db.items()
        ]

    def _save_db(self) -> None:
        try:
            data = {gid: {k: v for k, v in info.items()} for gid, info in self._global_db.items()}
            with open(self._db_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"[REID] Save error: {e}")

    def _load_db(self) -> None:
        if not self._db_path.exists():
            return
        try:
            with open(self._db_path) as f:
                data = json.load(f)
            self._global_db = data
            self._id_counter = len(data)
            logger.info(f"[REID] Loaded {len(data)} identities")
        except Exception as e:
            logger.warning(f"[REID] Load error: {e}")

    @property
    def identity_count(self) -> int:
        return len(self._global_db)

    @property
    def is_ready(self) -> bool:
        return self._loaded
