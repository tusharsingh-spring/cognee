"""Notification rules — decide when, what, and how to notify."""

from typing import Dict, Optional

from config.settings import ALERT_DEDUP_SECONDS, ALERT_THROTTLE_SECONDS
from notifications.webhook import WebhookNotifier
from utils.logger import get_logger

logger = get_logger(__name__)


class AlertEngine:
    def __init__(self) -> None:
        self._last_alert_time: Dict[str, float] = {}
        self._alert_history: list = []
        self._alert_count = 0
        self._webhook = WebhookNotifier()

    def evaluate(
        self,
        reasoning: dict,
        perception_context: str,
        track_id: Optional[int] = None,
    ) -> Optional[Dict]:
        """Evaluate LLM reasoning and decide whether to fire an alert."""
        import time

        if not reasoning.get("notify", False):
            return None

        urgency = reasoning.get("urgency", "none")
        if urgency == "none":
            return None

        now = time.time()

        alert_type = f"llm_urgency_{urgency}"
        last_fire = self._last_alert_time.get(alert_type, 0)
        if now - last_fire < ALERT_THROTTLE_SECONDS:
            return None

        dedup_key = reasoning.get("narrative", "")[:100]
        for past in self._alert_history[-20:]:
            if past.get("dedup_key") == dedup_key:
                if now - past.get("timestamp", 0) < ALERT_DEDUP_SECONDS:
                    return None

        alert = {
            "type": "llm_reasoning",
            "track_id": track_id,
            "urgency": urgency,
            "anomaly_score": reasoning.get("anomaly_score", 0),
            "message": reasoning.get("notification_text", reasoning.get("narrative", "")),
            "severity": "warning" if urgency == "high" else "info",
            "timestamp": now,
            "dedup_key": dedup_key,
            "context": perception_context[:500],
        }

        self._last_alert_time[alert_type] = now
        self._alert_history.append(alert)
        if len(self._alert_history) > 100:
            self._alert_history = self._alert_history[-100:]
        self._alert_count += 1

        title = f"ARGUS V2 - {urgency.upper()} Alert"
        if reasoning.get("anomaly_score", 0) > 0.5:
            title += f" [Anomaly: {reasoning['anomaly_score']:.2f}]"

        self._webhook.send_alert(title, alert["message"], severity=alert["severity"])

        logger.info(f"[ALERT] {urgency.upper()}: {alert['message'][:150]}")
        return alert

    def evaluate_threat_keywords(self, text: str, track_id: int) -> Optional[Dict]:
        import time

        threat_words = [
            "knife", "weapon", "gun", "steal", "theft", "robbery", "fight",
            "break-in", "force entry", "suspicious", "threat", "violen", "attack",
            "running away", "hiding", "crouching behind",
        ]
        now = time.time()
        text_lower = text.lower()
        hits = [w for w in threat_words if w in text_lower]

        if not hits:
            return None

        alert_key = f"{track_id}:{','.join(sorted(hits))}"
        for past in self._alert_history[-10:]:
            if past.get("dedup_key") == alert_key:
                if now - past.get("timestamp", 0) < ALERT_DEDUP_SECONDS:
                    return None

        alert = {
            "type": "keyword_threat",
            "track_id": track_id,
            "message": f"THREAT KEYWORDS ({', '.join(hits)}): {text[:300]}",
            "severity": "warning",
            "timestamp": now,
            "dedup_key": alert_key,
            "hits": hits,
        }
        self._alert_history.append(alert)
        self._alert_count += 1

        self._webhook.send_alert(
            f"THREAT: {', '.join(hits)}", f"Person_{track_id}: {text[:300]}", "warning"
        )
        logger.warning(f"[ALERT] {alert['message'][:200]}")
        return alert

    def evaluate_interaction(self, person_a: int, person_b: int) -> None:
        """Evaluate contact/interaction between two persons for alerting."""
        import time
        alert_key = f"contact_{min(person_a, person_b)}_{max(person_a, person_b)}"
        last = self._last_alert_time.get(alert_key, 0)
        if time.time() - last < ALERT_THROTTLE_SECONDS:
            return
        self._last_alert_time[alert_key] = time.time()
        self._alert_count += 1
        logger.info(f"[ALERT] Contact between Person_{person_a} and Person_{person_b}")

    @property
    def alert_count(self) -> int:
        return self._alert_count

    def get_recent(self, limit: int = 10) -> list:
        return self._alert_history[-limit:]
