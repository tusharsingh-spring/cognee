"""VLM trigger manager — decides when to call the VLM for each person.

Smart gating: only trigger VLM when it would add value.
Rules: new person, action change, periodic refresh, contact event.
"""

import time
from typing import Dict, Optional, Set, Tuple

from config.settings import VLM_REFRESH_INTERVAL
from utils.logger import get_logger

logger = get_logger(__name__)


class VLMTriggerManager:
    def __init__(self) -> None:
        self._active_ids: Dict[int, float] = {}
        self._last_vlm_time: Dict[int, float] = {}
        self._person_actions: Dict[int, str] = {}
        self._person_contacts: Dict[int, float] = {}
        self._trigger_count: int = 0

    def register_person(self, track_id: int) -> None:
        self._active_ids[track_id] = time.time()

    def unregister_person(self, track_id: int) -> None:
        self._active_ids.pop(track_id, None)

    def get_active_ids(self) -> Set[int]:
        now = time.time()
        stale = [tid for tid, t in self._active_ids.items() if now - t > 30]
        for tid in stale:
            del self._active_ids[tid]
        return set(self._active_ids.keys())

    def should_trigger(
        self,
        track_id: int,
        current_action: str,
        is_new_person: bool = False,
        has_contact: bool = False,
        movement_px: float = 0.0,
    ) -> Tuple[bool, str]:
        now = time.time()
        last_time = self._last_vlm_time.get(track_id, 0)

        if is_new_person:
            self._trigger_count += 1
            self._last_vlm_time[track_id] = now
            self._person_actions[track_id] = current_action
            return True, "new_person"

        prev_action = self._person_actions.get(track_id, "")
        if prev_action and prev_action != current_action:
            if current_action in ("walking", "running", "falling", "reaching", "grabbing"):
                self._trigger_count += 1
                self._last_vlm_time[track_id] = now
                self._person_actions[track_id] = current_action
                return True, f"action_change:{prev_action}->{current_action}"

        if has_contact:
            self._person_contacts[track_id] = now
            if now - last_time > VLM_REFRESH_INTERVAL:
                self._trigger_count += 1
                self._last_vlm_time[track_id] = now
                return True, "contact_event"

        if now - last_time > VLM_REFRESH_INTERVAL:
            self._trigger_count += 1
            self._last_vlm_time[track_id] = now
            self._person_actions[track_id] = current_action
            return True, "periodic_refresh"

        return False, "not_needed"

    @property
    def trigger_count(self) -> int:
        return self._trigger_count
