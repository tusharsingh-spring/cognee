"""Hand tracking using MediaPipe Hands.

Tracks 21 landmarks per hand, classifies hand state (open/grip/neutral).
"""

from typing import Dict, List, Optional

import cv2
import numpy as np

from config.settings import HAND_CONFIDENCE, HAND_ENABLED, HAND_EVERY_N, HAND_MAX_HANDS
from utils.logger import get_logger
from utils.profiler import profiler

logger = get_logger(__name__)


class HandTracker:
    def __init__(self) -> None:
        self.enabled = HAND_ENABLED
        self.is_ready = False
        self._model = None
        self._frame_count = 0

        if self.enabled:
            self._load()

    def _load(self) -> None:
        try:
            import mediapipe as mp
            mp_hands = mp.solutions.hands
            self._model = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=HAND_MAX_HANDS,
                min_detection_confidence=HAND_CONFIDENCE,
                min_tracking_confidence=HAND_CONFIDENCE,
            )
            self.is_ready = True
            logger.info("[HAND] MediaPipe Hands ready")
        except Exception as e:
            logger.warning(f"[HAND] Init failed: {e}")

    def track(
        self, frame: np.ndarray, persons: List[Dict], should_process: bool = True
    ) -> Dict[int, List[Dict]]:
        if not self.is_ready or not persons:
            return {}

        self._frame_count += 1
        if not should_process and self._frame_count % HAND_EVERY_N != 0:
            return {}

        prober = profiler.get("hand")
        prober.start()

        results = {}
        for person in persons:
            tid = person["track_id"]
            crop = person.get("crop")
            if crop is None or crop.size == 0:
                continue
            hands = self._track_single(crop)
            if hands:
                results[tid] = hands

        prober.stop()
        return results

    def _track_single(self, crop: np.ndarray) -> Optional[List[Dict]]:
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result = self._model.process(rgb)
            if not result.multi_hand_landmarks:
                return None

            hands = []
            for idx, landmarks in enumerate(result.multi_hand_landmarks):
                h, w = crop.shape[:2]
                lm_list = [[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks.landmark]

                handedness = result.multi_handedness[idx].classification[0].label \
                    if result.multi_handedness and idx < len(result.multi_handedness) \
                    else "unknown"

                state = self._classify_hand_state(lm_list)

                hands.append({
                    "handedness": handedness.lower(),
                    "state": state,
                    "landmarks": [[round(p[0], 1), round(p[1], 1), round(p[2], 1)] for p in lm_list],
                    "is_open": state == "open",
                    "is_grip": state == "grip",
                })
            return hands
        except Exception:
            return None

    def _classify_hand_state(self, landmarks: List[List[float]]) -> str:
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]

        open_count = 0
        for tip_idx, pip_idx in zip(tips, pips):
            tip_y = landmarks[tip_idx][1]
            pip_y = landmarks[pip_idx][1]
            if tip_y < pip_y:
                open_count += 1

        if open_count >= 4:
            return "open"
        elif open_count <= 1:
            return "grip"
        return "neutral"
