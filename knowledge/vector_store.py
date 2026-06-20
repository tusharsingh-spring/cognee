"""ChromaDB vector store for person embeddings with multi-modal metadata support."""

import time
from typing import Dict, List, Optional, Tuple

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from config.settings import CHROMA_COLLECTION, CHROMA_DIR, VSS_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    def __init__(self) -> None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._ensure_frame_collection()
        self._count = self.collection.count()
        self._encoder: Optional[SentenceTransformer] = None
        logger.info(f"[VECTOR] ChromaDB ready, {self._count} existing embeddings")

    def _ensure_frame_collection(self) -> None:
        try:
            self._frame_collection = self.client.get_or_create_collection(
                name=f"{CHROMA_COLLECTION}_frames",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            self._frame_collection = None

    def _get_encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            self._encoder = SentenceTransformer(VSS_MODEL)
        return self._encoder

    def search_by_text(self, query: str, n_results: int = 5) -> List[Dict]:
        if self.collection.count() == 0:
            return []
        encoder = self._get_encoder()
        embedding = encoder.encode(query, convert_to_numpy=True).tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, self.collection.count()),
        )
        matches = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                matches.append({
                    "id": results["ids"][0][i],
                    "distance": float(results["distances"][0][i]) if results.get("distances") else 0,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                })
        return matches

    def store(
        self,
        track_id: int,
        embedding: list,
        caption: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        doc_id = f"person_{track_id}"
        meta = metadata or {}
        meta.update({"caption": caption, "timestamp": time.time()})

        existing = self.collection.get(ids=[doc_id])
        if existing and existing["ids"]:
            self.collection.update(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[meta],
            )
        else:
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[meta],
            )
        self._count += 1
        logger.debug(f"[VECTOR] Stored embedding for Person_{track_id}")

    def store_frame(
        self,
        frame_idx: int,
        timestamp: float,
        persons_meta: Dict[int, Dict],
        caption_text: str = "",
        scene_text: str = "",
    ) -> bool:
        if self._frame_collection is None:
            return False
        try:
            meta_parts = []
            for tid, pm in persons_meta.items():
                action = pm.get("action", "")
                contact = pm.get("contact", "")
                gaze = pm.get("gaze", "")
                depth = pm.get("depth", "")
                hand = pm.get("hand", "")
                meta_parts.append(f"P{tid}:{action}:{contact}:{gaze}:{depth}:{hand}")
            text_to_embed = f"{caption_text} {scene_text} {' '.join(meta_parts)}"[:2000]
            if not text_to_embed.strip():
                text_to_embed = f"frame {frame_idx} at {timestamp:.1f}s"
            encoder = self._get_encoder()
            embedding = encoder.encode(text_to_embed, convert_to_numpy=True).tolist()
            doc_id = f"frame_{frame_idx}"
            meta = {
                "frame": frame_idx,
                "timestamp": timestamp,
                "persons": str(list(persons_meta.keys())),
                "text_snippet": text_to_embed[:500],
            }
            self._frame_collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[meta],
            )
            return True
        except Exception as e:
            logger.debug(f"[VECTOR] Frame store failed: {e}")
            return False

    def search_frames(self, query: str, n_results: int = 5) -> List[Dict]:
        if self._frame_collection is None or self._frame_collection.count() == 0:
            return []
        try:
            encoder = self._get_encoder()
            embedding = encoder.encode(query, convert_to_numpy=True).tolist()
            results = self._frame_collection.query(
                query_embeddings=[embedding],
                n_results=min(n_results, self._frame_collection.count()),
            )
            matches = []
            if results and results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    matches.append({
                        "id": results["ids"][0][i],
                        "distance": float(results["distances"][0][i]) if results.get("distances") else 0,
                        "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    })
            return matches
        except Exception as e:
            logger.debug(f"[VECTOR] Frame search failed: {e}")
            return []

    def search(
        self, embedding: list, n_results: int = 5
    ) -> List[Tuple[str, float, Dict]]:
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, self.collection.count()),
        )

        matches = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                distance = results["distances"][0][i] if results.get("distances") else 0
                metadata = (
                    results["metadatas"][0][i] if results.get("metadatas") else {}
                )
                matches.append((doc_id, float(distance), metadata))
        return matches

    def get(self, track_id: int) -> Optional[Dict]:
        doc_id = f"person_{track_id}"
        result = self.collection.get(ids=[doc_id])
        if result and result["ids"]:
            return {
                "embedding": result["embeddings"][0] if result.get("embeddings") else None,
                "metadata": result["metadatas"][0] if result.get("metadatas") else {},
            }
        return None

    def delete(self, track_id: int) -> None:
        doc_id = f"person_{track_id}"
        self.collection.delete(ids=[doc_id])
        self._count = max(0, self._count - 1)

    @property
    def count(self) -> int:
        return self.collection.count()
