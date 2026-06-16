"""Decides WHEN to call the VLM based on trigger rules."""

import time
from typing import Callable, Dict, List, Optional

import numpy as np

from config.settings import (
    VLM_CAPTION_DELAY_NEW,
    VLM_REFRESH_INTERVAL,
    VLM_STATE_CHANGE_THRESHOLD,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class VLMTriggerManager:
    def __init__(self) -> None:
        self._person_state: Dict[int, Dict] = {}
        self._pending_new: Dict[int, float] = {}

    def register_person(self, track_id: int, embedding: Optional[np.ndarray] = None) -> None:
        now = time.time()
        if track_id not in self._person_state:
            self._pending_new[track_id] = now
            self._person_state[track_id] = {
                "first_seen": now,
                "last_caption": 0.0,
                "last_vqa": 0.0,
                "last_embedding": embedding,
                "caption_count": 0,
            }

    def unregister_person(self, track_id: int) -> None:
        self._person_state.pop(track_id, None)
        self._pending_new.pop(track_id, None)

    def should_caption(self, track_id: int, embedding: Optional[np.ndarray] = None) -> str:
        state = self._person_state.get(track_id)
        if state is None:
            return ""

        now = time.time()
        pending_time = self._pending_new.get(track_id)

        if pending_time and now - pending_time > VLM_CAPTION_DELAY_NEW:
            del self._pending_new[track_id]
            logger.debug(f"[TRIGGER] New person {track_id} → requesting caption")
            state["last_caption"] = now
            state["caption_count"] += 1
            state["last_embedding"] = embedding
            return "new_person"

        if now - state["last_caption"] > VLM_REFRESH_INTERVAL:
            logger.debug(f"[TRIGGER] Periodic refresh for Person {track_id}")
            state["last_caption"] = now
            state["caption_count"] += 1
            state["last_embedding"] = embedding
            return "periodic"

        if embedding is not None and state.get("last_embedding") is not None:
            try:
                similarity = float(
                    np.dot(embedding, state["last_embedding"])
                    / (np.linalg.norm(embedding) * np.linalg.norm(state["last_embedding"]))
                )
                if similarity < (1.0 - VLM_STATE_CHANGE_THRESHOLD):
                    logger.debug(
                        f"[TRIGGER] Person {track_id} state changed (sim={similarity:.2f})"
                    )
                    state["last_caption"] = now
                    state["caption_count"] += 1
                    state["last_embedding"] = embedding
                    return "state_change"
            except Exception:
                pass

        return ""

    def should_vqa(self, track_id: int) -> bool:
        from config.settings import VQA_REFRESH_INTERVAL

        state = self._person_state.get(track_id)
        if state is None:
            return False

        now = time.time()
        if state["last_vqa"] == 0.0 or now - state["last_vqa"] > VQA_REFRESH_INTERVAL:
            state["last_vqa"] = now
            return True
        return False

    def get_active_ids(self) -> List[int]:
        return list(self._person_state.keys())
