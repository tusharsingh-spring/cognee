"""Project Progress Tracker - monitors which ARGUS components are operational.

Uses the knowledge graph + deep learning model statuses to compute
real-time project completion percentage and component health.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ComponentStatus:
    name: str
    category: str
    loaded: bool = False
    active: bool = False
    last_check: float = 0.0
    health_pct: float = 0.0
    message: str = ""
    error_count: int = 0
    success_count: int = 0


# Progress weights per component (must sum to ~1.0)
_PROJECT_BLUEPRINT: Dict[str, List[Dict[str, Any]]] = {
    "Layer 1: Motion Detection": [
        {"name": "Camera Capture", "weight": 0.04, "check": "camera"},
        {"name": "MOG2 Background Subtraction", "weight": 0.04, "check": "mog2"},
    ],
    "Layer 2: Detection + Tracking": [
        {"name": "YOLOv8n Model", "weight": 0.05, "check": "yolo"},
        {"name": "ByteTrack Tracker", "weight": 0.04, "check": "bytetrack"},
        {"name": "Person Crop Extraction", "weight": 0.03, "check": "crop"},
        {"name": "Object Detection", "weight": 0.04, "check": "object"},
    ],
    "Layer 3: Face + Re-Identification": [
        {"name": "Face Detection", "weight": 0.04, "check": "face"},
        {"name": "Person Re-ID", "weight": 0.05, "check": "reid"},
    ],
    "Layer 4: VLM Engine": [
        {"name": "Florence-2 Model", "weight": 0.08, "check": "florence"},
        {"name": "Caption Generation", "weight": 0.06, "check": "caption"},
        {"name": "VQA Handler", "weight": 0.05, "check": "vqa"},
        {"name": "VLM Async Queue", "weight": 0.03, "check": "queue"},
    ],
    "Layer 5: Knowledge + Memory": [
        {"name": "NetworkX Knowledge Graph", "weight": 0.06, "check": "graph"},
        {"name": "ChromaDB Vector Store", "weight": 0.05, "check": "chromadb"},
        {"name": "SQLite Event Store", "weight": 0.03, "check": "sqlite"},
        {"name": "Visual Similarity Search", "weight": 0.05, "check": "vss"},
        {"name": "Session Manager", "weight": 0.04, "check": "session"},
    ],
    "Layer 6: Alerts + Dashboard": [
        {"name": "Alert Engine", "weight": 0.05, "check": "alerts"},
        {"name": "Webhook Delivery", "weight": 0.03, "check": "webhook"},
        {"name": "Streamlit Dashboard", "weight": 0.04, "check": "dashboard"},
        {"name": "Summary Engine", "weight": 0.04, "check": "summary"},
        {"name": "Graph RAG Queries", "weight": 0.06, "check": "rag"},
    ],
}


class ProjectTracker:
    def __init__(self) -> None:
        self.components: Dict[str, ComponentStatus] = {}
        self._start_time = time.time()
        self._overall_progress: float = 0.0
        self._phase: str = "Initializing"

        for category, items in _PROJECT_BLUEPRINT.items():
            for item in items:
                self.components[item["check"]] = ComponentStatus(
                    name=item["name"],
                    category=category,
                )

    def mark_loaded(self, component: str) -> None:
        comp = self.components.get(component)
        if comp:
            comp.loaded = True
            comp.last_check = time.time()
            comp.health_pct = 50.0
            comp.message = "Model loaded"

    def mark_active(self, component: str) -> None:
        comp = self.components.get(component)
        if comp:
            comp.active = True
            comp.last_check = time.time()
            comp.health_pct = 100.0
            comp.message = "Active"
            comp.success_count += 1

    def mark_error(self, component: str, error_msg: str = "") -> None:
        comp = self.components.get(component)
        if comp:
            comp.error_count += 1
            comp.last_check = time.time()
            comp.health_pct = max(0.0, comp.health_pct - 20.0)
            comp.message = error_msg or "Error"

    def mark_inactive(self, component: str) -> None:
        comp = self.components.get(component)
        if comp:
            comp.active = False
            comp.health_pct = max(30.0, comp.health_pct - 5.0)
            comp.message = "Idle"

    def compute_progress(self) -> Tuple[float, str]:
        total_weight = 0.0
        earned_weight = 0.0

        for category, items in _PROJECT_BLUEPRINT.items():
            for item in items:
                weight = item["weight"]
                total_weight += weight
                comp = self.components.get(item["check"])
                if comp:
                    if comp.active:
                        earned_weight += weight * 1.0
                    elif comp.loaded:
                        earned_weight += weight * 0.5
                    elif comp.error_count > 0:
                        earned_weight += weight * 0.1

        self._overall_progress = (earned_weight / total_weight * 100) if total_weight > 0 else 0.0

        if self._overall_progress >= 95:
            self._phase = "Production Ready"
        elif self._overall_progress >= 70:
            self._phase = "Operational"
        elif self._overall_progress >= 40:
            self._phase = "Partial"
        elif self._overall_progress >= 10:
            self._phase = "Warming Up"
        else:
            self._phase = "Initializing"

        return self._overall_progress, self._phase

    def get_category_progress(self) -> List[Dict[str, Any]]:
        result = []
        for category, items in _PROJECT_BLUEPRINT.items():
            cat_total = sum(item["weight"] for item in items)
            cat_earned = 0.0
            active_count = 0
            loaded_count = 0
            total_count = len(items)

            for item in items:
                comp = self.components.get(item["check"])
                if comp:
                    if comp.active:
                        cat_earned += item["weight"]
                        active_count += 1
                    elif comp.loaded:
                        cat_earned += item["weight"] * 0.5
                        loaded_count += 1

            pct = (cat_earned / cat_total * 100) if cat_total > 0 else 0.0
            result.append({
                "category": category,
                "progress": round(pct, 1),
                "active": active_count,
                "loaded": loaded_count,
                "total": total_count,
            })

        return result

    def get_status_display(self) -> Dict[str, Any]:
        overall, phase = self.compute_progress()
        uptime = time.time() - self._start_time
        cats = self.get_category_progress()

        return {
            "overall": round(overall, 1),
            "phase": phase,
            "uptime_sec": int(uptime),
            "uptime_str": f"{int(uptime // 3600):02d}:{int((uptime % 3600) // 60):02d}:{int(uptime % 60):02d}",
            "categories": cats,
            "component_count": len(self.components),
            "active_count": sum(1 for c in self.components.values() if c.active),
            "loaded_count": sum(1 for c in self.components.values() if c.loaded),
            "error_count": sum(c.error_count for c in self.components.values()),
        }

    def get_bar_string(self, width: int = 40) -> str:
        overall, phase = self.compute_progress()
        filled = int(width * overall / 100)
        bar = "#" * filled + "-" * (width - filled)
        return f"[{bar}] {overall:.0f}% | {phase}"

    def get_compact_status(self) -> str:
        overall, phase = self.compute_progress()
        a = sum(1 for c in self.components.values() if c.active)
        l = sum(1 for c in self.components.values() if c.loaded) - a
        return f"{overall:.0f}% | {phase} | {a} active, {l} loaded"

    def log_progress(self) -> None:
        status = self.get_status_display()
        logger.info(
            f"[PROGRESS] {status['overall']:.1f}% | Phase: {status['phase']} | "
            f"Active: {status['active_count']}/{status['component_count']} | "
            f"Uptime: {status['uptime_str']}"
        )
        for cat in status["categories"]:
            logger.debug(
                f"  {cat['category']}: {cat['progress']:.0f}% "
                f"({cat['active']}/{cat['total']} active)"
            )


tracker = ProjectTracker()
