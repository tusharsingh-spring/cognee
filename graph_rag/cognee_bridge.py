"""Cognee Bridge — file-based event store for timelined perception + VLM output.

No LLM calls. Events are saved to JSONL and served via simple text search.
Groq is reserved exclusively for the chatbot UI.
"""

import json
import os
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

from config.settings import (
    DATA_DIR,
    COGNEE_ENABLED,
)
from utils.logger import get_logger

logger = get_logger(__name__)

COGNEE_FLUSH_INTERVAL = 30.0
COGNEE_MAX_BUFFER = 500
COGNEE_DATASET = "argus_events"
COGNEE_PERCEPTION_EVERY_N = 10


class CogneeBridge:
    """File-based event store for perception + VLM + LLM outputs with timestamped JSONL."""

    def __init__(self, readonly: bool = False) -> None:
        self.enabled = COGNEE_ENABLED
        self._buffer: List[str] = []
        self._lock = threading.Lock()
        self._event_count = 0
        self._last_flush = time.time()
        self._running = False
        self._ready = False
        self._stats: Dict = {"nodes": 0, "datasets": 0, "events_indexed": 0}
        self._frame_count = 0
        self._readonly = readonly

        self._save_path = DATA_DIR / "cognee_events.jsonl"
        self._load_events()

        if self.enabled and not self._readonly:
            self._running = True
            self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
            self._flush_thread.start()
            logger.info("[COGNEE] File-based event store started (flush every %ds, no LLM)",
                        int(COGNEE_FLUSH_INTERVAL))
        elif self.enabled:
            logger.info("[COGNEE] Bridge started in readonly mode")

    # ── Public API (sync, called from main loop) ──

    def ingest_perception(self, packet, frame_time: float = 0.0) -> None:
        if not self.enabled:
            return
        self._frame_count += 1
        if self._frame_count % COGNEE_PERCEPTION_EVERY_N != 0:
            return
        entry = self._build_perception_entry(packet, frame_time)
        self._append(entry)

    def ingest_vlm_output(self, track_id: int, vlm_text: str, task: str = "dense", frame_time: float = 0.0) -> None:
        if not self.enabled or not vlm_text:
            return
        ts = frame_time if frame_time > 0 else time.time()
        entry = {
            "type": "vlm_output",
            "track_id": track_id,
            "task": task,
            "text": vlm_text[:500],
            "timestamp": ts,
        }
        self._append(json.dumps(entry))

    def ingest_llm_output(self, track_id: int, reasoning: dict, frame_time: float = 0.0) -> None:
        if not self.enabled:
            return
        ts = frame_time if frame_time > 0 else time.time()
        entry = {
            "type": "llm_output",
            "track_id": track_id,
            "narrative": reasoning.get("narrative", ""),
            "intent": reasoning.get("intent", ""),
            "anomaly_score": reasoning.get("anomaly_score", 0),
            "urgency": reasoning.get("urgency", "none"),
            "tags": reasoning.get("store_tags", []),
            "notify": reasoning.get("notify", False),
            "timestamp": ts,
        }
        self._append(json.dumps(entry))

    def log_event(self, event_type: str, track_id: int, data: dict, frame_time: float = 0.0) -> None:
        if not self.enabled:
            return
        ts = frame_time if frame_time > 0 else time.time()
        entry = {
            "type": "event",
            "event_type": event_type,
            "track_id": track_id,
            "data": data,
            "timestamp": ts,
        }
        self._append(json.dumps(entry))

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        if not self.enabled:
            return []
        return self._fallback_search(query, top_k)

    def retrieve_context(self, query: str, top_k: int = 10) -> List[Dict]:
        if not self.enabled:
            return []
        return self.search(query, top_k=top_k)

    def get_recent_events(self, n: int = 10) -> List[Dict]:
        events = []
        for line in reversed(self._buffer[-n:]):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def get_person_history(self, track_id: int) -> Dict:
        results = self.search(f"Person_{track_id} actions interactions", top_k=20)
        return {
            "person": {"track_id": track_id},
            "events": results,
            "actions": [r for r in results if "action" in str(r).lower()],
            "interactions": [r for r in results if "interact" in str(r).lower()],
        }

    def get_daily_patterns(self) -> List[Dict]:
        return []

    def detect_patterns(self) -> None:
        pass

    def purge_old(self, max_age: float = 3600.0) -> None:
        pass

    def purge_dataset(self) -> None:
        if not self.enabled:
            return
        try:
            if self._save_path.is_file():
                self._save_path.unlink()
            self._event_count = 0
            self._stats["nodes"] = 0
            with self._lock:
                self._buffer = []
            logger.info("[COGNEE] Event store purged")
        except Exception as e:
            logger.error("[COGNEE] purge_dataset failed: %s", e)

    def improve_graph(self) -> None:
        pass

    def get_stats(self) -> Dict:
        return {
            "total_nodes": self._stats.get("nodes", 0),
            "total_edges": 0,
            "node_types": {"Event": self._event_count},
            "events_indexed": self._event_count,
        }

    def save(self) -> None:
        self._flush_to_cognee()

    def close(self) -> None:
        self._running = False
        if hasattr(self, '_flush_thread') and self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=10.0)
        self._flush_to_cognee()
        logger.info("[COGNEE] File store closed")

    # ── Internals ──

    def _build_perception_entry(self, packet, frame_time: float = 0.0) -> str:
        ts = frame_time if frame_time > 0 else packet.timestamp
        lines = [f"[PERCEPTION] Frame {packet.frame_number} | {datetime.fromtimestamp(ts).strftime('%H:%M:%S')}"]

        for person in packet.persons:
            tid = person.track_id
            parts = [f"Person_{tid} detected (conf={person.confidence:.2f})"]

            if hasattr(packet, 'actions') and tid in packet.actions:
                a = packet.actions[tid]
                parts.append(f"action: {a.action} (conf={a.confidence:.2f})")
            if hasattr(packet, 'poses') and tid in packet.poses:
                parts.append(f"pose: {packet.poses[tid].visible_count}/17 joints visible")
            if hasattr(packet, 'gaze') and tid in packet.gaze:
                g = packet.gaze[tid]
                parts.append(f"gaze: {g.direction}")
                if g.target_person_id is not None:
                    parts.append(f"looking at Person_{g.target_person_id}")
            if hasattr(packet, 'flow') and tid in packet.flow:
                parts.append(f"flow: {packet.flow[tid].mean_magnitude:.1f}px")

            lines.append(" | ".join(parts))

        if packet.objects:
            obj_names = set(o.name for o in packet.objects)
            lines.append(f"Objects seen: {', '.join(obj_names)}")

        if hasattr(packet, 'contacts'):
            for c in packet.contacts:
                if c.contact:
                    lines.append(f"Contact: Person_{c.person_a} <-> Person_{c.person_b} (score={c.score:.2f})")

        # Store as structured JSON for cognee
        entry = {
            "type": "perception",
            "frame": packet.frame_number,
            "timestamp": ts,
            "text": "\n".join(lines),
            "persons": [{"track_id": p.track_id, "confidence": p.confidence, "bbox": list(p.bbox)} for p in packet.persons],
            "person_count": len(packet.persons),
            "object_count": len(packet.objects),
        }
        return json.dumps(entry)

    def _append(self, entry: str) -> None:
        with self._lock:
            self._buffer.append(entry)
            self._event_count += 1

    def _flush_loop(self) -> None:
        while self._running:
            time.sleep(COGNEE_FLUSH_INTERVAL)
            if not self._running:
                break
            try:
                self._flush_to_cognee()
            except Exception as e:
                logger.error("[COGNEE] Flush error: %s", e)

    def _flush_to_cognee(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            batch = list(self._buffer)
            self._buffer = []

        n = len(batch)
        logger.info("[COGNEE] Saving %d events to JSONL...", n)
        try:
            self._save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._save_path, "a", encoding="utf-8") as f:
                for entry in batch:
                    f.write(entry + "\n")
            self._last_flush = time.time()
            self._ready = True
            self._stats["nodes"] = self._stats.get("nodes", 0) + n
            logger.info("[COGNEE] Saved %d events", n)
        except Exception as e:
            logger.error("[COGNEE] Flush error: %s", e)
            with self._lock:
                self._buffer = batch + self._buffer

    def _fallback_search(self, query: str, top_k: int = 10) -> List[Dict]:
        results = []
        words = query.lower().split()

        try:
            if self._save_path.is_file():
                with open(self._save_path, "r", encoding="utf-8") as fh:
                    content = fh.read()
                content_lower = content.lower()
                if any(w in content_lower for w in words):
                    lines = content.strip().split("\n")
                    for line in lines[-500:]:
                        if any(w in line.lower() for w in words):
                            try:
                                obj = json.loads(line)
                                txt = obj.get("text", obj.get("type", str(obj)))
                            except Exception:
                                txt = line
                            results.append({
                                "node": txt[:80],
                                "type": "Entity",
                                "label": txt[:500],
                                "neighbors": [],
                                "score": 0.8,
                            })
                            if len(results) >= top_k * 3:
                                break

            deduped = []
            seen = set()
            for r in results:
                key = r["label"][:100]
                if key not in seen:
                    seen.add(key)
                    deduped.append(r)
            if deduped:
                logger.info("[COGNEE] Search found %d results", len(deduped))
            return deduped[:top_k]
        except Exception as e:
            logger.debug("[COGNEE] Search error: %s", e)
            return []

    def _load_events(self) -> None:
        if not self._save_path.is_file():
            return
        try:
            with open(self._save_path, encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            self._buffer = lines[-COGNEE_MAX_BUFFER:]
            self._event_count = len(lines)
            logger.info("[COGNEE] Loaded %d persisted events", len(lines))
        except Exception:
            pass
