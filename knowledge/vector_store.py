"""ChromaDB vector store for person embeddings."""

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
        self._count = self.collection.count()
        self._encoder: Optional[SentenceTransformer] = None
        logger.info(f"[VECTOR] ChromaDB ready, {self._count} existing embeddings")

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
