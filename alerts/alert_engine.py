"""Alert condition evaluation engine."""

import time
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class AlertEngine:
    def __init__(self) -> None:
        self._last_alert_time: Dict[str, float] = {}
        self._alert_history: List[Dict] = []
        self._alert_count = 0

    def evaluate_new_person(
        self, track_id: int, caption: Optional[str] = None
    ) -> Optional[Dict]:
        alert = {
            "type": "new_person",
            "track_id": track_id,
            "message": f"New person detected: ID:{track_id}" + (f" - {caption}" if caption else ""),
            "severity": "info",
            "timestamp": time.time(),
        }
        return self._maybe_fire(alert)

    def evaluate_interaction(
        self, person_a: int, person_b: int, caption: Optional[str] = None
    ) -> Optional[Dict]:
        alert = {
            "type": "interaction",
            "track_id": person_a,
            "message": f"Person_{person_a} interacting with Person_{person_b}" + (f" - {caption}" if caption else ""),
            "severity": "info",
            "timestamp": time.time(),
            "data": {"person_a": person_a, "person_b": person_b},
        }
        return self._maybe_fire(alert)

    def evaluate_unusual_behavior(
        self, track_id: int, behavior: str
    ) -> Optional[Dict]:
        unusual_keywords = ["running", "lying down", "fallen", "fighting", "screaming"]
        if any(kw in behavior.lower() for kw in unusual_keywords):
            alert = {
                "type": "unusual_behavior",
                "track_id": track_id,
                "message": f"Unusual behavior: Person_{track_id} - {behavior}",
                "severity": "warning",
                "timestamp": time.time(),
                "data": {"behavior": behavior},
            }
            return self._maybe_fire(alert)
        return None

    def _maybe_fire(self, alert: Dict) -> Optional[Dict]:
        from config.settings import ALERT_DEDUP_SECONDS, ALERT_THROTTLE_SECONDS

        now = time.time()

        alert_key = alert["type"]
        last_fire = self._last_alert_time.get(alert_key, 0)
        if now - last_fire < ALERT_THROTTLE_SECONDS:
            return None

        for past in self._alert_history:
            if (
                past["type"] == alert["type"]
                and past.get("track_id") == alert.get("track_id")
                and now - past["timestamp"] < ALERT_DEDUP_SECONDS
            ):
                return None

        self._last_alert_time[alert_key] = now
        self._alert_history.append(alert)
        self._alert_count += 1

        if len(self._alert_history) > 100:
            self._alert_history = self._alert_history[-100:]

        logger.info(f"[ALERT] {alert['severity'].upper()}: {alert['message']}")
        return alert

    @property
    def alert_count(self) -> int:
        return self._alert_count

    def get_recent(self, limit: int = 10) -> List[Dict]:
        return self._alert_history[-limit:]
