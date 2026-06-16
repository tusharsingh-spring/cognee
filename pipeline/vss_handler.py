"""Visual Similarity Search handler using sentence-transformers."""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import VSS_MODEL, VSS_SIMILARITY_THRESHOLD, VSS_MAX_EMBEDDINGS
from utils.logger import get_logger

logger = get_logger(__name__)


class VSSHandler:
    def __init__(self) -> None:
        self.model = None
        self._loaded = False
        self._embeddings: Dict[int, Tuple[np.ndarray, str, float]] = {}

    def load(self) -> None:
        if self._loaded:
            return
        logger.info(f"[VSS] Loading embedding model: {VSS_MODEL}")
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(VSS_MODEL)
        self._loaded = True
        logger.info("[VSS] Embedding model loaded")

    def compute_embedding(self, text: str) -> np.ndarray:
        if self.model is None:
            self.load()
        return self.model.encode(text, convert_to_numpy=True)

    def store(self, track_id: int, caption: str) -> Optional[np.ndarray]:
        if len(self._embeddings) >= VSS_MAX_EMBEDDINGS:
            oldest = min(self._embeddings.items(), key=lambda x: x[1][2])
            del self._embeddings[oldest[0]]
            logger.debug(f"[VSS] Evicted oldest embedding (ID:{oldest[0]})")

        embedding = self.compute_embedding(caption)
        self._embeddings[track_id] = (embedding, caption, time.time())
        logger.debug(f"[VSS] Person {track_id} embedding stored")
        return embedding

    def search_similar(
        self, track_id: int, caption: str
    ) -> List[Tuple[int, float]]:
        if not self._embeddings:
            return []

        query_emb = self.compute_embedding(caption)
        matches = []

        for tid, (emb, _, ts) in self._embeddings.items():
            if tid == track_id:
                continue
            similarity = float(
                np.dot(query_emb, emb)
                / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
            )
            if similarity >= VSS_SIMILARITY_THRESHOLD:
                matches.append((tid, similarity))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def get_embedding(self, track_id: int) -> Optional[np.ndarray]:
        entry = self._embeddings.get(track_id)
        return entry[0] if entry else None
