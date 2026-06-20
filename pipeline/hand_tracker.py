"""Hand tracking using MediaPipe Hands (tasks API v0.10.x)."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import HAND_CONFIDENCE, HAND_ENABLED, HAND_EVERY_N, HAND_MAX_HANDS, MODEL_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


class HandTracker:
    def __init__(self) -> None:
        self._enabled = HAND_ENABLED
        self._confidence = HAND_CONFIDENCE
        self._max_hands = HAND_MAX_HANDS
        self._hands = None
        self._loaded = False
        self._frame_counter = 0
        self._last_landmarks: Dict[int, List[Dict]] = {}

        if self._enabled:
            self._try_load()

    def _try_load(self) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_path = str(MODEL_DIR / "hand_landmarker.task")
            if not Path(model_path).exists():
                logger.warning(f"[Hand] Model not found: {model_path}")
                self._try_cv_fallback()
                return

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=self._max_hands,
                min_hand_detection_confidence=self._confidence,
                min_tracking_confidence=self._confidence * 0.8,
            )
            self._hands = vision.HandLandmarker.create_from_options(options)
            self._loaded = True
            logger.info("[Hand] MediaPipe Hands loaded (tasks API)")
        except ImportError:
            self._loaded = False
            logger.warning("[Hand] mediapipe not installed.")
        except Exception as e:
            self._loaded = False
            logger.warning(f"[Hand] Load failed: {e}")
            self._try_cv_fallback()

    def _try_cv_fallback(self) -> None:
        try:
            import mediapipe as mp
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
                mp_hands = mp.solutions.hands
                self._hands = mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self._max_hands,
                    min_detection_confidence=self._confidence,
                    min_tracking_confidence=self._confidence * 0.8,
                )
                self._mp_draw = mp.solutions.drawing_utils
                self._mp_hands_mod = mp_hands
                self._use_legacy = True
                self._loaded = True
                logger.info("[Hand] MediaPipe Hands legacy API loaded")
        except Exception as e:
            logger.warning(f"[Hand] All load methods failed: {e}")

    @property
    def is_ready(self) -> bool:
        return self._enabled and self._loaded

    @property
    def last_landmarks(self) -> Dict[int, List[Dict]]:
        return self._last_landmarks

    def track(
        self, frame: np.ndarray, persons: List[Dict], force: bool = False
    ) -> Dict[int, List[Dict]]:
        if not self.is_ready:
            return {}

        self._frame_counter += 1
        if not force and self._frame_counter % HAND_EVERY_N != 0:
            return self._last_landmarks

        results: Dict[int, List[Dict]] = {}
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if getattr(self, "_use_legacy", False):
            self._last_landmarks = self._track_legacy(rgb, persons, h, w)
            return self._last_landmarks

        for person in persons:
            tid = person["track_id"]
            bbox = person["bbox"]
            x1, y1, x2, y2 = bbox

            pad_w = int((x2 - x1) * 0.3)
            pad_h = int((y2 - y1) * 0.25)
            cx1 = max(0, int(x1) - pad_w)
            cy1 = max(0, int(y1) - pad_h)
            cx2 = min(w, int(x2) + pad_w)
            cy2 = min(h, int(y2) + pad_h)
            if cx2 <= cx1 or cy2 <= cy1:
                continue

            crop = rgb[cy1:cy2, cx1:cx2]
            mp_image = __import__("mediapipe").Image(
                image_format=__import__("mediapipe").ImageFormat.SRGB, data=crop
            )
            try:
                hand_result = self._hands.detect(mp_image)
            except Exception:
                continue

            if hand_result.hand_landmarks:
                person_hands = []
                for idx, hand_lms in enumerate(hand_result.hand_landmarks):
                    handedness = "unknown"
                    if hand_result.handedness and idx < len(hand_result.handedness):
                        handedness = hand_result.handedness[idx][0].category_name

                    kpts = []
                    for lm in hand_lms:
                        x = lm.x * (cx2 - cx1) + cx1
                        y = lm.y * (cy2 - cy1) + cy1
                        kpts.append((x, y, 1.0))

                    is_open = self._is_open_palm(kpts)
                    is_grip = self._is_grip(kpts)
                    person_hands.append({
                        "handedness": handedness,
                        "keypoints": kpts,
                        "is_open": is_open,
                        "is_grip": is_grip,
                    })
                results[tid] = person_hands

        self._last_landmarks = results
        return results

    def _track_legacy(self, rgb, persons, h, w):
        results = {}
        for person in persons:
            tid = person["track_id"]
            bbox = person["bbox"]
            x1, y1, x2, y2 = bbox
            pad_w = int((x2 - x1) * 0.3)
            pad_h = int((y2 - y1) * 0.25)
            cx1 = max(0, int(x1) - pad_w)
            cy1 = max(0, int(y1) - pad_h)
            cx2 = min(w, int(x2) + pad_w)
            cy2 = min(h, int(y2) + pad_h)
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            crop = rgb[cy1:cy2, cx1:cx2]
            try:
                hr = self._hands.process(crop)
            except Exception:
                continue
            if hr.multi_hand_landmarks:
                person_hands = []
                for idx, hlm in enumerate(hr.multi_hand_landmarks):
                    handedness = "unknown"
                    if hr.multi_handedness and idx < len(hr.multi_handedness):
                        handedness = hr.multi_handedness[idx].classification[0].label
                    kpts = [(lm.x * (cx2 - cx1) + cx1, lm.y * (cy2 - cy1) + cy1, 1.0) for lm in hlm.landmark]
                    is_open = self._is_open_palm(kpts)
                    is_grip = self._is_grip(kpts)
                    person_hands.append({"handedness": handedness, "keypoints": kpts, "is_open": is_open, "is_grip": is_grip})
                results[tid] = person_hands
        return results

    def _is_open_palm(self, keypoints: List[Tuple]) -> bool:
        if len(keypoints) < 21:
            return False
        palm_center = np.mean([keypoints[0][:2], keypoints[5][:2], keypoints[17][:2]], axis=0)
        extended = 0
        for tip, base in [(4, 2), (8, 6), (12, 10), (16, 14), (20, 18)]:
            if tip < len(keypoints) and base < len(keypoints):
                d_tip = np.linalg.norm(np.array(keypoints[tip][:2]) - palm_center)
                d_base = np.linalg.norm(np.array(keypoints[base][:2]) - palm_center)
                if d_tip > d_base * 1.3:
                    extended += 1
        return extended >= 4

    def _is_grip(self, keypoints: List[Tuple]) -> bool:
        if len(keypoints) < 21:
            return False
        folded = 0
        for tip, pip, base in [(8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)]:
            if tip < len(keypoints) and pip < len(keypoints) and base < len(keypoints):
                d_tip = np.linalg.norm(np.array(keypoints[tip][:2]) - np.array(keypoints[base][:2]))
                d_pip = np.linalg.norm(np.array(keypoints[pip][:2]) - np.array(keypoints[base][:2]))
                if d_tip < d_pip * 1.2:
                    folded += 1
        return folded >= 3

    def check_hand_to_face(self, hands: List[Dict], face_bbox: Optional[Tuple[int, int, int, int]]) -> bool:
        if not hands or face_bbox is None:
            return False
        fx1, fy1, fx2, fy2 = face_bbox
        for hand in hands:
            wrist = hand["keypoints"][0]
            if fx1 <= wrist[0] <= fx2 and fy1 <= wrist[1] <= fy2:
                return True
        return False

    def close(self) -> None:
        if self._hands is not None and hasattr(self._hands, "close"):
            self._hands.close()
