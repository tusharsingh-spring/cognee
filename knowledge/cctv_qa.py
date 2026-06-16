"""CCTV Q&A Engine - natural language queries over the recorded video intelligence.

Answers questions like:
- "Who was here between 10:00 and 10:15?"
- "What was Person_3 doing?"
- "Did anyone pick up the phone?"
- "Show me all interactions with laptops"
- "What happened after Person_1 left?"

Uses Graph RAG + Vector DB + Temporal Action Engine + SQLite for comprehensive answers.
"""

import json
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


class CCTVQA:
    def __init__(self) -> None:
        self._qa_history: List[Dict] = []

    def answer(
        self,
        question: str,
        graph_store,
        vector_store,
        sqlite_store,
        action_engine,
        session_manager,
        reid_handler,
    ) -> Dict[str, Any]:
        q = question.lower().strip()
        result = {
            "question": question,
            "answer": "",
            "evidence": [],
            "confidence": 0.0,
            "type": "unknown",
        }

        if self._handle_who(q, result, graph_store, reid_handler, action_engine):
            pass
        elif self._handle_what(q, result, action_engine, graph_store, sqlite_store):
            pass
        elif self._handle_when(q, result, action_engine, sqlite_store):
            pass
        elif self._handle_where(q, result, graph_store, action_engine):
            pass
        elif self._handle_did(q, result, action_engine, graph_store):
            pass
        elif self._handle_graph_query(q, result, graph_store):
            pass
        else:
            self._handle_general(q, result, graph_store, action_engine, session_manager)

        result["timestamp"] = time.time()
        self._qa_history.append(result)
        if len(self._qa_history) > 100:
            self._qa_history = self._qa_history[-100:]

        return result

    def _handle_who(self, q, result, graph_store, reid_handler, action_engine) -> bool:
        if "who" not in q:
            return False

        result["type"] = "who"

        time_range = self._extract_time_range(q)
        persons_found = []

        for node, data in graph_store.graph.nodes(data=True):
            if data.get("type") != "Person":
                continue
            label = data.get("label", "")
            ts = data.get("timestamp", 0)
            if time_range and not (time_range[0] <= ts <= time_range[1]):
                continue
            if any(kw in label.lower() for kw in q.split() if len(kw) > 2):
                persons_found.append(node)

        if persons_found:
            descriptions = []
            for p in persons_found[:5]:
                neighbors = graph_store.query_neighbors(int(p.split("_")[1]) if "_" in p else 0)
                actions = [f"{n[1]}" for n in neighbors[:3]]
                descriptions.append(f"{p}: {actions}")
            result["answer"] = f"Found {len(persons_found)} persons: {'; '.join(descriptions)}"
            result["evidence"] = persons_found
            result["confidence"] = 0.8
        else:
            identities = reid_handler.get_session_history()
            result["answer"] = f"No specific person found with those criteria. {len(identities)} identities tracked this session."
            result["evidence"] = identities
            result["confidence"] = 0.4

        return True

    def _handle_what(self, q, result, action_engine, graph_store, sqlite_store) -> bool:
        if "what" not in q:
            return False

        result["type"] = "what"

        for tid_str in re.findall(r"\d+", q):
            try:
                tid = int(tid_str)
                timeline = action_engine.get_person_timeline(tid)
                if timeline:
                    acts = [f"{a['action_type']}({a['time_str']})" for a in timeline[-5:]]
                    result["answer"] = f"Person_{tid}: {', '.join(acts)}"
                    result["evidence"] = timeline
                    result["confidence"] = 0.9
                    return True
            except ValueError:
                pass

        summary = action_engine.get_scene_summary()
        result["answer"] = (
            f"In the last 5 minutes: {summary['unique_persons']} persons, "
            f"{summary['total_actions']} actions. "
            f"Breakdown: {summary['action_breakdown']}"
        )
        result["evidence"] = summary
        result["confidence"] = 0.7
        return True

    def _handle_when(self, q, result, action_engine, sqlite_store) -> bool:
        if "when" not in q:
            return False

        result["type"] = "when"

        for tid_str in re.findall(r"\d+", q):
            try:
                tid = int(tid_str)
                timeline = action_engine.get_person_timeline(tid)
                if timeline:
                    first = timeline[0]
                    last = timeline[-1]
                    result["answer"] = (
                        f"Person_{tid} first seen at {first['time_str']}, "
                        f"last action: {last['action_type']} at {last['time_str']}"
                    )
                    result["evidence"] = timeline[:3]
                    result["confidence"] = 0.8
                    return True
            except ValueError:
                pass

        events = sqlite_store.get_recent_events(limit=5)
        if events:
            lines = [f"{e['event_type']} at {time.strftime('%H:%M:%S', time.localtime(e['timestamp']))}" for e in events]
            result["answer"] = f"Recent events: {'; '.join(lines)}"
            result["evidence"] = events
            result["confidence"] = 0.6
        return True

    def _handle_where(self, q, result, graph_store, action_engine) -> bool:
        if "where" not in q:
            return False

        result["type"] = "where"

        locations = []
        for node, data in graph_store.graph.nodes(data=True):
            if data.get("type") == "Location":
                neighbors = []
                for _, target, edata in graph_store.graph.out_edges(node, data=True):
                    neighbors.append(target)
                locations.append(f"{node}[{data.get('label','?')}] connected to {neighbors}")

        if locations:
            result["answer"] = f"Locations: {'; '.join(locations[:5])}"
            result["evidence"] = locations
            result["confidence"] = 0.7
        else:
            result["answer"] = "No location data recorded yet."
            result["confidence"] = 0.3
        return True

    def _handle_did(self, q, result, action_engine, graph_store) -> bool:
        if "did" not in q:
            return False

        result["type"] = "did"

        action_keywords = ["pick", "grab", "hold", "take", "leave", "enter", "exit", "open", "close", "use"]
        for kw in action_keywords:
            if kw in q:
                matches = action_engine.query_by_type(kw)
                if matches:
                    result["answer"] = f"Yes, {len(matches)} instances of '{kw}' detected. First: {matches[0]['description'][:80]}"
                    result["evidence"] = matches[:3]
                    result["confidence"] = 0.75
                    return True
                for a in action_engine.actions:
                    if kw in a.description.lower():
                        result["answer"] = f"Yes: Person_{a.track_id} - {a.description[:100]}"
                        result["evidence"] = [a.description]
                        result["confidence"] = 0.7
                        return True

        result["answer"] = f"No evidence found for that action in the last {len(action_engine.actions)} recorded events."
        result["confidence"] = 0.3
        return True

    def _handle_graph_query(self, q, result, graph_store) -> bool:
        rag_results = graph_store.rag_search(q, top_k=5)
        if rag_results:
            items = [f"{r['node']}[{r['type']}]: {r['label'][:60]}" for r in rag_results]
            result["answer"] = f"Graph matches: {' | '.join(items)}"
            result["evidence"] = rag_results
            result["type"] = "graph_rag"
            result["confidence"] = 0.6
            return True
        return False

    def _handle_general(self, q, result, graph_store, action_engine, session_manager) -> None:
        result["type"] = "general"
        stats = graph_store.get_stats()
        scene = action_engine.get_scene_summary()
        session = session_manager.get_current_summary()

        result["answer"] = (
            f"System status: {stats.get('person_nodes',0)} persons tracked, "
            f"{scene['unique_persons']} active now, "
            f"{scene['total_actions']} actions recorded. "
            f"Session: {session.get('session_id','?')} "
            f"({session.get('duration',0):.0f}s elapsed)"
        )
        result["evidence"] = {"graph": stats, "scene": scene, "session": session}
        result["confidence"] = 0.5

    def _extract_time_range(self, text: str) -> Optional[Tuple[float, float]]:
        times = re.findall(r"(\d{1,2}):(\d{2})", text)
        if len(times) >= 2:
            now = time.time()
            today_start = now - (now % 86400)
            t1 = today_start + int(times[0][0]) * 3600 + int(times[0][1]) * 60
            t2 = today_start + int(times[1][0]) * 3600 + int(times[1][1]) * 60
            return (min(t1, t2), max(t1, t2))
        return None

    def get_history(self) -> List[Dict]:
        return self._qa_history
