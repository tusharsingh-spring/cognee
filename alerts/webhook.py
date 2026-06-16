"""Webhook delivery for Slack and Discord."""

import json
from typing import Any, Dict, Optional

import requests

from config.settings import DISCORD_WEBHOOK_URL, SLACK_WEBHOOK_URL
from utils.logger import get_logger

logger = get_logger(__name__)


class WebhookNotifier:
    def __init__(self) -> None:
        self.slack_url = SLACK_WEBHOOK_URL
        self.discord_url = DISCORD_WEBHOOK_URL

    def send(self, alert: Dict) -> None:
        if self.slack_url:
            self._send_slack(alert)
        if self.discord_url:
            self._send_discord(alert)

    def _send_slack(self, alert: Dict) -> None:
        severity = alert.get("severity", "info")
        color_map = {"info": "#36a64f", "warning": "#ffcc00", "critical": "#ff0000"}
        color = color_map.get(severity, "#cccccc")

        payload = {
            "text": f"ARGUS Alert: {alert['type']}",
            "attachments": [
                {
                    "title": alert.get("message", ""),
                    "text": json.dumps(alert.get("data", {}), indent=2),
                    "color": color,
                }
            ],
        }

        try:
            resp = requests.post(self.slack_url, json=payload, timeout=5)
            if resp.status_code == 200:
                logger.debug(f"[WEBHOOK] Sent to Slack: {alert['type']}")
            else:
                logger.warning(f"[WEBHOOK] Slack returned {resp.status_code}")
        except Exception as e:
            logger.error(f"[WEBHOOK] Slack error: {e}")

    def _send_discord(self, alert: Dict) -> None:
        severity = alert.get("severity", "info")
        color_map = {"info": 65280, "warning": 16776960, "critical": 16711680}
        color = color_map.get(severity, 8421504)

        payload = {
            "content": f"**ARGUS Alert: {alert['type']}**",
            "embeds": [
                {
                    "title": alert.get("message", ""),
                    "description": json.dumps(alert.get("data", {}), indent=2),
                    "color": color,
                }
            ],
        }

        try:
            resp = requests.post(self.discord_url, json=payload, timeout=5)
            if resp.status_code in (200, 204):
                logger.debug(f"[WEBHOOK] Sent to Discord: {alert['type']}")
            else:
                logger.warning(f"[WEBHOOK] Discord returned {resp.status_code}")
        except Exception as e:
            logger.error(f"[WEBHOOK] Discord error: {e}")
