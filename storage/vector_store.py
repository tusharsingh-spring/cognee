"""ChromaDB vector store with BGE-M3 embeddings for event retrieval.

Stores: person descriptions, perception packets, VLM outputs, LLM narratives.
3-tier storage: recent events, daily summaries, behavioral baselines.
"""

import time
from typing import Dict, List, Optional, Tuple

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from config.settings import CHROMA_COLLECTION, CHROMA_DIR, VSS_MODEL, VSS_SIMILARITY_THRESHOLD
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
        self._frame_collection = self.client.get_or_create_collection(
            name=f"{CHROMA_COLLECTION}_frames",
            metadata={"hnsw:space": "cosine"},
        )
        self._daily_collection = self.client.get_or_create_collection(
            name=f"{CHROMA_COLLECTION}_daily",
            metadata={"hnsw:space": "cosine"},
        )
        self._encoder = None
        logger.info(f"[VECTOR] ChromaDB ready ({self.collection.count()} embeddings)")

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(VSS_MODEL)
        return self._encoder

    def search_by_text(self, query: str, n_results: int = 10) -> List[Dict]:
        encoder = self._get_encoder()
        embedding = encoder.encode(query, convert_to_numpy=True).tolist()
        matches = []

        for collection in [self._frame_collection, self.collection]:
            if collection.count() == 0:
                continue
            results = collection.query(query_embeddings=[embedding], n_results=min(n_results, collection.count()))
            if results and results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    dist = float(results["distances"][0][i]) if results.get("distances") else 0
                    matches.append({
                        "id": results["ids"][0][i],
                        "distance": round(dist, 4),
                        "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                        "document": results["documents"][0][i] if results.get("documents") else "",
                    })
        matches.sort(key=lambda m: m.get("distance", 1.0))
        return matches[:n_results]

    def store(self, track_id, embedding: list, caption: str, metadata: Optional[Dict] = None) -> None:
        doc_id = f"person_{track_id}_{int(time.time())}"
        meta = metadata or {}
        meta.update({"caption": caption[:500], "timestamp": time.time()})
        self.collection.add(ids=[doc_id], embeddings=[embedding], metadatas=[meta], documents=[caption[:2000]])
        logger.debug(f"[VECTOR] Stored: {doc_id}")

    def search(self, embedding: list, n_results: int = 10) -> List[Tuple[str, float, Dict]]:
        if self.collection.count() == 0:
            return []
        results = self.collection.query(query_embeddings=[embedding], n_results=min(n_results, self.collection.count()))
        matches = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                matches.append((results["ids"][0][i], float(results["distances"][0][i]) if results.get("distances") else 0,
                                results["metadatas"][0][i] if results.get("metadatas") else {}))
        return matches

    def get_by_id(self, doc_id: str) -> Optional[Dict]:
        result = self.collection.get(ids=[doc_id])
        if result and result["ids"]:
            return {"id": result["ids"][0], "metadata": result["metadatas"][0] if result.get("metadatas") else {},
                    "document": result["documents"][0] if result.get("documents") else ""}
        return None

    def store_event(self, event_type: str, data: dict, text: str) -> bool:
        try:
            encoder = self._get_encoder()
            embedding = encoder.encode(text, convert_to_numpy=True).tolist()
            doc_id = f"evt_{int(time.time() * 1000)}"
            self._frame_collection.add(
                ids=[doc_id], embeddings=[embedding],
                metadatas=[{"event_type": event_type, "timestamp": time.time(), **data}],
                documents=[text[:2000]],
            )
            return True
        except Exception as e:
            logger.debug(f"[VECTOR] Event store failed: {e}")
            return False

    def search_events(self, query: str, n_results: int = 10) -> List[Dict]:
        if self._frame_collection.count() == 0:
            return []
        encoder = self._get_encoder()
        embedding = encoder.encode(query, convert_to_numpy=True).tolist()
        results = self._frame_collection.query(query_embeddings=[embedding], n_results=min(n_results, self._frame_collection.count()))
        matches = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                matches.append({
                    "id": results["ids"][0][i],
                    "distance": round(float(results["distances"][0][i]), 4) if results.get("distances") else 0,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                })
        return matches

    @property
    def count(self) -> int:
        return self.collection.count()
