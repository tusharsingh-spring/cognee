"""Face detection + recognition + identity tracking using DeepFace + MediaPipe."""

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    FACE_DETECTION_ENABLED,
    FACE_MIN_CONFIDENCE,
    REID_MATCH_THRESHOLD,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class FaceRecognizer:
    def __init__(self) -> None:
        self._loaded = False
        self._detector = None
        self._face_db: Dict[str, Tuple[np.ndarray, str, float]] = {}
        self._person_face_map: Dict[int, str] = {}
        self._face_cache: Dict[int, List[Dict]] = {}
        self._use_deepface = False
        self._deepface_model = "Facenet"

    def load(self) -> None:
        if self._loaded:
            return
        if not FACE_DETECTION_ENABLED:
            logger.info("[FACE] Face recognition disabled")
            self._loaded = True
            return

        logger.info("[FACE] Loading face recognition models...")

        try:
            face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._detector = cv2.CascadeClassifier(face_cascade_path)
            self._mp_type = "opencv"
            logger.info("[FACE] Using OpenCV Haar cascade")
        except Exception:
            logger.warning("[FACE] Face detection unavailable")
            self._loaded = True
            return

        try:
            from deepface import DeepFace
            self._deepface_available = True
            logger.info(f"[FACE] DeepFace available for recognition")
        except Exception as e:
            logger.warning(f"[FACE] DeepFace unavailable: {e}")
            self._deepface_available = False

        self._loaded = True
        logger.info("[FACE] Face recognition ready")

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if not self._loaded or not self._detector:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []

        try:
            detected = self._detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            for (x, y, w, h) in detected:
                faces.append({
                    "bbox": (x, y, x + w, y + h),
                    "confidence": 1.0,
                    "landmarks": [],
                })
        except Exception:
            mp_type = getattr(self, "_mp_type", "")
            if mp_type == "mediapipe_v1" and hasattr(self, "_mp_face_detection"):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._detector.process(rgb)
                if results.detections:
                    fh, fw = frame.shape[:2]
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x1 = int(bbox.xmin * fw)
                        y1 = int(bbox.ymin * fh)
                        x2 = int((bbox.xmin + bbox.width) * fw)
                        y2 = int((bbox.ymin + bbox.height) * fh)
                        score = detection.score[0] if detection.score else 0.5
                        if score >= FACE_MIN_CONFIDENCE:
                            faces.append({
                                "bbox": (x1, y1, x2, y2),
                                "confidence": float(score),
                                "landmarks": [],
                            })

        return faces

    def extract_embedding(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = face_bbox
        if x2 <= x1 or y2 <= y1:
            return None
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        if hasattr(self, "_deepface_available") and self._deepface_available:
            try:
                from deepface import DeepFace
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                embedding_objs = DeepFace.represent(
                    img_path=face_rgb,
                    model_name=self._deepface_model,
                    enforce_detection=False,
                )
                if embedding_objs and len(embedding_objs) > 0:
                    emb = np.array(embedding_objs[0]["embedding"])
                    return emb / (np.linalg.norm(emb) + 1e-8)
            except Exception as e:
                logger.debug(f"[FACE] DeepFace embedding failed: {e}")

        face_crop = cv2.resize(face_crop, (112, 112))
        mean = face_crop.mean(axis=(0, 1))
        std = face_crop.std(axis=(0, 1)) + 1e-6
        face_crop = (face_crop - mean) / std
        return face_crop.flatten() / np.linalg.norm(face_crop.flatten())

    def match_face(self, embedding: np.ndarray) -> Optional[str]:
        best_id = None
        best_sim = -1.0
        for fid, (emb, _, _) in self._face_db.items():
            if emb.shape != embedding.shape:
                continue
            sim = float(np.dot(embedding, emb))
            if sim > best_sim and sim >= REID_MATCH_THRESHOLD:
                best_sim = sim
                best_id = fid
        return best_id

    def register_face(self, face_id: str, embedding: np.ndarray, label: str = "") -> None:
        self._face_db[face_id] = (embedding, label, time.time())
        logger.debug(f"[FACE] Registered face_id: {face_id}")

    def process_person(self, track_id: int, crop: np.ndarray) -> List[Dict]:
        faces = self.detect_faces(crop)
        results = []
        for face in faces:
            emb = self.extract_embedding(crop, face["bbox"])
            if emb is None:
                continue
            matched_id = self.match_face(emb)
            if matched_id is None:
                matched_id = f"face_{int(time.time() * 1000)}"
                self.register_face(matched_id, emb)
                logger.info(f"[FACE] New face detected: {matched_id} on Person_{track_id}")
            else:
                logger.debug(f"[FACE] Person_{track_id} matched face: {matched_id}")
            self._person_face_map[track_id] = matched_id
            results.append({
                "track_id": track_id,
                "face_id": matched_id,
                "bbox": face["bbox"],
                "confidence": face["confidence"],
            })
        return results

    def get_face_id(self, track_id: int) -> Optional[str]:
        return self._person_face_map.get(track_id)

    def get_face_count(self) -> int:
        return len(self._face_db)

    @property
    def is_available(self) -> bool:
        return FACE_DETECTION_ENABLED and self._loaded and self._detector is not None
