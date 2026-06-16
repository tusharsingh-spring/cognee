"""End LLM Aggregator - collects all pipeline data and generates comprehensive understanding.

Takes all gathered information (VLM captions, YOLO objects, action sequences, 
Re-ID identities, graph state, face data) and synthesizes a complete narrative
of what's happening, what happened, and the state of everything.
"""

import time
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class LLMAggregator:
    def __init__(self) -> None:
        self._last_aggregation = 0.0
        self._aggregation_interval = 60.0
        self._history: List[Dict] = []

    def aggregate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        now = time.time()

        result = {
            "timestamp": now,
            "time_str": time.strftime("%H:%M:%S", time.localtime(now)),
            "narrative": "",
            "summary": "",
            "stats": {},
            "events": [],
            "recommendations": [],
        }

        graph = context.get("graph_stats", {})
        actions = context.get("actions", {})
        sequences = context.get("sequences", [])
        identities = context.get("identities", [])
        session = context.get("session", {})
        snapshots = context.get("scene_snapshots", [])

        result["stats"] = {
            "persons_tracked": graph.get("person_nodes", 0),
            "objects_detected": graph.get("object_nodes", 0),
            "actions_recorded": actions.get("total_actions", 0),
            "sequences_detected": actions.get("sequences_detected", 0),
            "unique_identities": len(identities),
            "session_duration": session.get("duration", 0),
            "graph_edges": graph.get("total_edges", 0),
        }

        events = []
        for seq in sequences[-5:]:
            events.append({
                "type": "sequence",
                "description": seq.get("summary", ""),
                "time": seq.get("start_time", 0),
            })

        for snap in snapshots[-3:]:
            for p in snap.get("persons", []):
                caption = p.get("caption", "") or p.get("appearance", "")
                if caption:
                    events.append({
                        "type": "person_observation",
                        "track_id": p["track_id"],
                        "description": caption[:200],
                        "time": snap.get("timestamp", 0),
                    })

            if snap.get("interactions"):
                for inter in snap["interactions"]:
                    events.append({
                        "type": "interaction",
                        "description": f"Person_{inter['person_a']} near Person_{inter['person_b']} ({inter['distance']}px)",
                        "time": snap.get("timestamp", 0),
                    })

        result["events"] = events[-20:]

        result["narrative"] = self._build_narrative(result, context)
        result["summary"] = self._build_summary(result, context)

        if now - self._last_aggregation > self._aggregation_interval:
            self._last_aggregation = now
            self._history.append(result)
            if len(self._history) > 50:
                self._history = self._history[-25:]
            logger.info(f"[AGGREGATE] {result['summary']}")

        return result

    def _build_narrative(self, result: Dict, context: Dict) -> str:
        stats = result["stats"]
        parts = []

        parts.append(f"Session has been running for {int(stats['session_duration'])} seconds.")
        parts.append(
            f"Tracked {stats['persons_tracked']} persons across {stats['unique_identities']} "
            f"unique identities. Recorded {stats['actions_recorded']} actions and "
            f"{stats['sequences_detected']} action sequences."
        )
        parts.append(f"Detected {stats['objects_detected']} objects with {stats['graph_edges']} graph connections.")

        if result["events"]:
            parts.append("\nRecent events:")
            for evt in result["events"][-8:]:
                ts = time.strftime("%H:%M:%S", time.localtime(evt["time"])) if evt.get("time") else ""
                parts.append(f"  [{ts}] {evt['type']}: {evt['description'][:120]}")

        actions = context.get("actions", {})
        if actions.get("action_breakdown"):
            parts.append(f"\nAction breakdown: {actions['action_breakdown']}")

        return "\n".join(parts)

    def _build_summary(self, result: Dict, context: Dict) -> str:
        stats = result["stats"]
        return (
            f"{stats['persons_tracked']} persons | "
            f"{stats['objects_detected']} objects | "
            f"{stats['actions_recorded']} actions | "
            f"{stats['sequences_detected']} sequences | "
            f"{stats['unique_identities']} identities | "
            f"{stats['session_duration']:.0f}s"
        )

    def get_history(self, limit: int = 10) -> List[Dict]:
        return self._history[-limit:]

    def answer_question(self, question: str, context: Dict) -> Dict:
        q = question.lower()
        stats = self._stats_from_context(context)

        answer = {
            "question": question,
            "answer": "",
            "confidence": 0.5,
            "source": "aggregator",
        }

        if "how many" in q and "person" in q:
            answer["answer"] = f"There are {stats['persons_tracked']} persons tracked. {stats['unique_identities']} unique identities."
            answer["confidence"] = 0.9

        elif "how many" in q and "object" in q:
            answer["answer"] = f"Detected {stats['objects_detected']} objects."
            answer["confidence"] = 0.9

        elif "what happened" in q or "what is happening" in q:
            events = result.get("events", context.get("events", []))
            if events:
                lines = [f"[{time.strftime('%H:%M:%S', time.localtime(e['time']))}] {e['description'][:100]}" for e in events[-5:]]
                answer["answer"] = "Recent events:\n" + "\n".join(lines)
            else:
                answer["answer"] = "No significant events recorded recently."

        elif "how long" in q or "duration" in q:
            answer["answer"] = f"Session has been running for {int(stats['session_duration'])} seconds."

        elif "summary" in q:
            narratives = self.get_history(1)
            if narratives:
                answer["answer"] = narratives[0].get("narrative", "")
                answer["confidence"] = 0.8
            else:
                answer["answer"] = self._build_narrative({"stats": stats, "events": []}, context)

        else:
            narratives = self.get_history(1)
            if narratives:
                answer["answer"] = narratives[0].get("narrative", f"System active. {stats}")
            else:
                answer["answer"] = f"System running: {stats['persons_tracked']} persons, {stats['actions_recorded']} actions."

        return answer

    def _stats_from_context(self, context: Dict) -> Dict:
        graph = context.get("graph_stats", {})
        actions = context.get("actions", {})
        identities = context.get("identities", [])
        session = context.get("session", {})
        return {
            "persons_tracked": graph.get("person_nodes", 0),
            "objects_detected": graph.get("object_nodes", 0),
            "actions_recorded": actions.get("total_actions", 0),
            "sequences_detected": actions.get("sequences_detected", 0),
            "unique_identities": len(identities),
            "session_duration": session.get("duration", 0),
        }
