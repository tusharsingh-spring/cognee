"""Comprehensive Scene Analyzer - gathers dense information about everything in frame.

Combines VLM, YOLO, Re-ID, Face Recognition, and Graph RAG to build
a complete picture of the scene at any moment.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class SceneAnalyzer:
    def __init__(self) -> None:
        self._scene_history: List[Dict] = []
        self._last_scene_query = 0.0
        self._query_interval = 30.0

    def analyze(
        self,
        frame: np.ndarray,
        persons: List[Dict],
        objects: List[Dict],
        vlm_engine,
        detector,
        face_recognizer,
        reid_handler,
        graph_store,
        action_engine,
    ) -> Dict[str, Any]:
        analysis = {
            "timestamp": time.time(),
            "time_str": time.strftime("%H:%M:%S", time.localtime()),
            "person_count": len(persons),
            "object_count": len(objects),
            "persons": [],
            "objects": [o["name"] for o in objects],
            "interactions": [],
            "scene_summary": "",
        }

        for person in persons:
            tid = person["track_id"]
            bbox = person["bbox"]
            crop = person.get("crop")

            person_info = {
                "track_id": tid,
                "bbox": bbox,
                "caption": "",
                "appearance": "",
                "action": "",
                "context": "",
                "face_id": face_recognizer.get_face_id(tid) if face_recognizer.is_available else None,
                "global_id": reid_handler.get_global_id(tid) if reid_handler.is_ready else None,
                "nearby_objects": [],
                "timeline": action_engine.get_person_timeline(tid)[-3:] if action_engine else [],
            }

            if vlm_engine and vlm_engine._loaded:
                details = vlm_engine.get_person_details(tid)
                if details:
                    dense = details.get("dense", "")
                    person_info["caption"] = dense
                    person_info["appearance"] = dense
                    person_info["action"] = dense
                    person_info["context"] = ""

            if objects:
                nearby = detector.get_person_object_proximity(bbox, objects)
                person_info["nearby_objects"] = nearby[:5]

            analysis["persons"].append(person_info)

        for i, p1 in enumerate(persons):
            for p2 in persons[i + 1:]:
                dist = self._distance(p1["bbox"], p2["bbox"])
                if dist < 300:
                    analysis["interactions"].append({
                        "person_a": p1["track_id"],
                        "person_b": p2["track_id"],
                        "distance": round(dist, 1),
                        "caption_a": analysis["persons"][i].get("caption", ""),
                        "caption_b": analysis["persons"][i + 1].get("caption", ""),
                    })

        analysis["scene_summary"] = self._generate_summary(analysis)

        if time.time() - self._last_scene_query > self._query_interval:
            self._last_scene_query = time.time()
            self._scene_history.append(analysis)
            if len(self._scene_history) > 100:
                self._scene_history = self._scene_history[-50:]
            logger.info(f"[SCENE] {analysis['scene_summary']}")

        return analysis

    def _distance(self, bbox1, bbox2) -> float:
        c1 = ((bbox1[0] + bbox1[2]) // 2, (bbox1[1] + bbox1[3]) // 2)
        c2 = ((bbox2[0] + bbox2[2]) // 2, (bbox2[1] + bbox2[3]) // 2)
        return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    def _generate_summary(self, analysis: Dict) -> str:
        parts = []
        n = analysis["person_count"]

        if n == 0:
            parts.append("Empty scene")
        elif n == 1:
            p = analysis["persons"][0]
            cap = p.get("caption", "") or p.get("appearance", "")
            parts.append(f"Person_{p['track_id']}: {cap[:100]}")
        else:
            parts.append(f"{n} persons present")
            for p in analysis["persons"]:
                parts.append(f"  Person_{p['track_id']}: {p.get('caption','')[:80]}")

        if analysis["objects"]:
            parts.append(f"Objects: {', '.join(analysis['objects'][:8])}")

        if analysis["interactions"]:
            ints = [f"P{i['person_a']}-P{i['person_b']}" for i in analysis["interactions"]]
            parts.append(f"Interactions: {', '.join(ints)}")

        return " | ".join(parts)

    def get_history(self, limit: int = 10) -> List[Dict]:
        return self._scene_history[-limit:]

    def gather_context_for_llm(
        self,
        action_engine,
        graph_store,
        reid_handler,
        session_manager,
    ) -> Dict[str, Any]:
        return {
            "actions": action_engine.get_scene_summary(300),
            "sequences": action_engine.get_sequences()[-10:],
            "graph_stats": graph_store.get_stats(),
            "identities": reid_handler.get_session_history()[-20:],
            "session": session_manager.get_current_summary(),
            "scene_snapshots": self._scene_history[-5:],
        }
