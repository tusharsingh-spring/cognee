"""Periodic runtime summary generator."""

import time
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class SummaryEngine:
    def __init__(self) -> None:
        self._last_summary = 0.0
        self._stats: Dict[str, int] = {
            "total_persons": 0,
            "total_interactions": 0,
            "total_objects": 0,
            "total_alerts": 0,
            "total_vlm_calls": 0,
            "peak_persons": 0,
        }
        self._frame_count = 0

    def update(self, stats: Dict[str, int]) -> None:
        self._frame_count += 1
        if stats.get("current_persons", 0) > self._stats["peak_persons"]:
            self._stats["peak_persons"] = stats["current_persons"]
        self._latest = stats

    def should_summarize(self, interval: float) -> bool:
        now = time.time()
        if now - self._last_summary > interval:
            self._last_summary = now
            return True
        return False

    def generate(self, graph_stats: Optional[Dict] = None) -> Dict:
        gs = graph_stats or {}
        summary = {
            "total_persons": gs.get("person_nodes", 0),
            "total_objects": gs.get("object_nodes", 0),
            "total_actions": gs.get("action_nodes", 0),
            "total_interactions": self._stats["total_interactions"],
            "total_alerts": self._stats["total_alerts"],
            "peak_persons": self._stats["peak_persons"],
            "graph": gs,
        }
        logger.info(
            f"[SUMMARY] {summary['total_persons']} persons, "
            f"{summary['total_objects']} objects, "
            f"{summary['total_actions']} actions | "
            f"Peak: {summary['peak_persons']} simultaneous"
        )
        return summary

    def reset(self) -> None:
        self._stats = {k: 0 for k in self._stats}
        self._frame_count = 0
        self._stats["peak_persons"] = 0
