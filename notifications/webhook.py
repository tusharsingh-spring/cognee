"""Notification system — Slack/Discord webhooks + Firebase FCM push."""

import json
import time
from typing import Dict, List, Optional

import requests

from config.settings import DISCORD_WEBHOOK_URL, SLACK_WEBHOOK_URL
from utils.logger import get_logger

logger = get_logger(__name__)


class WebhookNotifier:
    def __init__(self) -> None:
        self._sent_count = 0
        self._history: List[Dict] = []

    def send(self, payload: dict, channel: str = "slack") -> bool:
        if channel == "slack" and SLACK_WEBHOOK_URL:
            return self._send_slack(payload)
        elif channel == "discord" and DISCORD_WEBHOOK_URL:
            return self._send_discord(payload)
        return False

    def _send_slack(self, payload: dict) -> bool:
        try:
            msg = {
                "text": f"*ARGUS V2 Alert*\n"
                        f"```{json.dumps(payload, indent=2, default=str)[:2000]}```",
                "username": "ARGUS V2",
                "icon_emoji": ":camera:",
            }
            resp = requests.post(SLACK_WEBHOOK_URL, json=msg, timeout=10)
            if resp.status_code == 200:
                self._sent_count += 1
                self._history.append({"timestamp": time.time(), "channel": "slack", "payload": payload})
                logger.info("[WEBHOOK] Slack notification sent")
                return True
        except Exception as e:
            logger.error(f"[WEBHOOK] Slack error: {e}")
        return False

    def _send_discord(self, payload: dict) -> bool:
        try:
            msg = {
                "content": f"**ARGUS V2 Alert**\n```json\n{json.dumps(payload, indent=2, default=str)[:1500]}\n```",
                "username": "ARGUS V2",
            }
            resp = requests.post(DISCORD_WEBHOOK_URL, json=msg, timeout=10)
            if resp.status_code == 204:
                self._sent_count += 1
                self._history.append({"timestamp": time.time(), "channel": "discord", "payload": payload})
                logger.info("[WEBHOOK] Discord notification sent")
                return True
        except Exception as e:
            logger.error(f"[WEBHOOK] Discord error: {e}")
        return False

    def send_alert(self, title: str, message: str, severity: str = "info") -> None:
        payload = {"title": title, "message": message, "severity": severity, "timestamp": time.time()}
        for channel in ["slack", "discord"]:
            self.send(payload, channel=channel)

    @property
    def sent_count(self) -> int:
        return self._sent_count
