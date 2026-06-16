"""Temporal Action Engine - tracks what happens, when, and in what sequence.

Stores action timelines with timestamps, builds causal sequences, 
enables temporal querying via Graph RAG and Vector DB.
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Action:
    track_id: int
    action_type: str  # appeared, left, walking, holding, interacting, etc.
    description: str
    timestamp: float
    confidence: float = 1.0
    bbox: Optional[Tuple] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TemporalActionEngine:
    def __init__(self) -> None:
        self.actions: List[Action] = []
        self.person_timelines: Dict[int, List[Action]] = defaultdict(list)
        self.sequences: List[List[Action]] = []
        self._active_sequences: Dict[str, List[Action]] = {}
        self._person_last_action: Dict[int, str] = {}
        self._object_interactions: Dict[str, List[Dict]] = defaultdict(list)
        self._scene_context: Dict[str, Any] = {}
        self._sequence_counter = 0

    def log_action(
        self,
        track_id: int,
        action_type: str,
        description: str,
        bbox: Optional[Tuple] = None,
        metadata: Optional[Dict] = None,
    ) -> Action:
        action = Action(
            track_id=track_id,
            action_type=action_type,
            description=description,
            timestamp=time.time(),
            bbox=bbox,
            metadata=metadata or {},
        )
        self.actions.append(action)
        self.person_timelines[track_id].append(action)

        if action_type != self._person_last_action.get(track_id):
            self._person_last_action[track_id] = action_type
            self._check_sequence(action)

        for obj_name in self._extract_objects(description):
            self._object_interactions[obj_name].append({
                "track_id": track_id,
                "action_type": action_type,
                "timestamp": action.timestamp,
                "description": description,
            })

        if len(self.actions) > 10000:
            self.actions = self.actions[-5000:]
            for tid in self.person_timelines:
                self.person_timelines[tid] = self.person_timelines[tid][-200:]

        logger.debug(
            f"[ACTION] Person_{track_id} | {action_type}: {description[:80]}"
        )
        return action

    def _check_sequence(self, action: Action) -> None:
        key = f"person_{action.track_id}"
        if key not in self._active_sequences:
            self._active_sequences[key] = [action]
        else:
            self._active_sequences[key].append(action)
            if len(self._active_sequences[key]) >= 3:
                seq = self._active_sequences[key][-3:]
                combined = " → ".join(a.action_type for a in seq)
                self.sequences.append(seq)
                self._sequence_counter += 1

    def _extract_objects(self, text: str) -> List[str]:
        objects = []
        obj_keywords = ["cup", "bottle", "phone", "book", "laptop", "chair", "bag",
                        "knife", "spoon", "fork", "bowl", "plate", "glass", "remote",
                        "keyboard", "mouse", "pen", "paper", "document", "wallet"]
        text_lower = text.lower()
        for kw in obj_keywords:
            if kw in text_lower:
                objects.append(kw)
        return objects

    def get_person_timeline(self, track_id: int) -> List[Dict]:
        return [
            {
                "action_type": a.action_type,
                "description": a.description,
                "timestamp": a.timestamp,
                "time_str": time.strftime("%H:%M:%S", time.localtime(a.timestamp)),
            }
            for a in self.person_timelines.get(track_id, [])
        ]

    def get_all_timelines(self) -> Dict[int, List[Dict]]:
        return {
            tid: self.get_person_timeline(tid)
            for tid in self.person_timelines
        }

    def get_sequences(self) -> List[Dict]:
        return [
            {
                "actions": [
                    {
                        "track_id": a.track_id,
                        "action_type": a.action_type,
                        "description": a.description,
                        "timestamp": a.timestamp,
                    }
                    for a in seq
                ],
                "summary": " → ".join(a.action_type for a in seq),
                "start_time": seq[0].timestamp,
                "end_time": seq[-1].timestamp,
            }
            for seq in self.sequences
        ]

    def query_by_time(self, start: float, end: float) -> List[Dict]:
        return [
            {
                "track_id": a.track_id,
                "action_type": a.action_type,
                "description": a.description,
                "timestamp": a.timestamp,
                "time_str": time.strftime("%H:%M:%S", time.localtime(a.timestamp)),
            }
            for a in self.actions
            if start <= a.timestamp <= end
        ]

    def query_by_type(self, action_type: str) -> List[Dict]:
        return [
            {
                "track_id": a.track_id,
                "description": a.description,
                "timestamp": a.timestamp,
            }
            for a in self.actions
            if a.action_type == action_type
        ]

    def query_object_interactions(self, object_name: str) -> List[Dict]:
        return self._object_interactions.get(object_name, [])

    def set_scene_context(self, context: Dict) -> None:
        self._scene_context = context

    def get_scene_summary(
        self, duration_seconds: Optional[float] = None
    ) -> Dict:
        now = time.time()
        cutoff = now - (duration_seconds or 300)
        recent = [a for a in self.actions if a.timestamp >= cutoff]

        person_counts = defaultdict(int)
        action_counts = defaultdict(int)
        for a in recent:
            person_counts[a.track_id] += 1
            action_counts[a.action_type] += 1

        return {
            "total_actions": len(recent),
            "unique_persons": len(person_counts),
            "action_breakdown": dict(action_counts),
            "sequences_detected": len(self.sequences),
            "object_interactions": {
                obj: len(events) for obj, events in self._object_interactions.items()
            },
            "time_range": {
                "start": time.strftime(
                    "%H:%M:%S", time.localtime(recent[0].timestamp)
                ) if recent else "N/A",
                "end": time.strftime("%H:%M:%S", time.localtime(now)),
            },
        }

    def natural_query(self, query: str) -> Dict:
        query_lower = query.lower()
        now = time.time()
        results = {"matches": [], "type": "unknown"}

        if "timeline" in query_lower or "history" in query_lower:
            for tid_str in query.split():
                try:
                    tid = int(tid_str)
                    results["matches"] = self.get_person_timeline(tid)
                    results["type"] = "timeline"
                    break
                except ValueError:
                    pass
            if not results["matches"]:
                results["matches"] = self.get_all_timelines()
                results["type"] = "all_timelines"

        elif "sequence" in query_lower:
            results["matches"] = self.get_sequences()
            results["type"] = "sequences"

        elif "last" in query_lower or "recent" in query_lower:
            mins = 5
            for word in query.split():
                try:
                    mins = int(word)
                except ValueError:
                    pass
            results["matches"] = self.query_by_time(now - mins * 60, now)
            results["type"] = "recent"

        elif "object" in query_lower or "holding" in query_lower or "carrying" in query_lower:
            for obj in ["cup", "phone", "book", "laptop", "bottle", "bag"]:
                if obj in query_lower:
                    results["matches"] = self.query_object_interactions(obj)
                    results["type"] = f"object_{obj}"
                    break

        elif "appeared" in query_lower:
            results["matches"] = self.query_by_type("appeared")
            results["type"] = "appearances"

        elif "left" in query_lower or "departed" in query_lower:
            results["matches"] = self.query_by_type("left")
            results["type"] = "departures"

        elif "interacting" in query_lower:
            results["matches"] = self.query_by_type("interacting")
            results["type"] = "interactions"

        else:
            recent = self.query_by_time(now - 300, now)
            results["matches"] = recent
            results["type"] = "recent_default"

        results["summary"] = self.get_scene_summary()
        return results

    @property
    def action_count(self) -> int:
        return len(self.actions)
