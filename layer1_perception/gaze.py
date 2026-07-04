"""Gaze estimation using MediaPipe FaceMesh iris landmarks.

Estimates yaw/pitch from 468 face mesh landmarks with iris tracking.
Computes gaze direction and determines if person is looking at another person.
"""

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    GAZE_ANGLE_THRESHOLD,
    GAZE_CONFIDENCE,
    GAZE_ENABLED,
    GAZE_EVERY_N,
)
from utils.logger import get_logger
from utils.profiler import profiler

logger = get_logger(__name__)


class GazeEstimator:
    def __init__(self) -> None:
        self.enabled = GAZE_ENABLED
        self.is_ready = False
        self._model = None
        self._frame_count = 0

        if self.enabled:
            self._load()

    def _load(self) -> None:
        try:
            import mediapipe as mp
            mp_face_mesh = mp.solutions.face_mesh
            self._model = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=2,
                refine_landmarks=True,
                min_detection_confidence=GAZE_CONFIDENCE,
                min_tracking_confidence=GAZE_CONFIDENCE,
            )
            self.is_ready = True
            logger.info("[GAZE] MediaPipe FaceMesh ready")
        except Exception as e:
            logger.warning(f"[GAZE] Init failed: {e}")

    def estimate(
        self, frame: np.ndarray, persons: List[Dict], should_process: bool = True
    ) -> Dict[int, Dict]:
        if not self.is_ready or not persons:
            return {}

        self._frame_count += 1
        if not should_process and self._frame_count % GAZE_EVERY_N != 0:
            return {}

        prober = profiler.get("gaze")
        prober.start()

        results = {}
        for person in persons:
            tid = person["track_id"]
            bbox = person["bbox"]
            crop = person.get("crop")
            if crop is None or crop.size == 0:
                continue
            gd = self._estimate_single(crop)
            if gd:
                results[tid] = gd

        prober.stop()
        return results

    def _estimate_single(self, crop: np.ndarray) -> Optional[Dict]:
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result = self._model.process(rgb)
            if not result.multi_face_landmarks:
                return None

            face = result.multi_face_landmarks[0]
            h, w = crop.shape[:2]

            left_iris = face.landmark[468]  # left iris center
            right_iris = face.landmark[473]  # right iris center
            left_eye_center = face.landmark[33]
            right_eye_center = face.landmark[263]
            nose_tip = face.landmark[1]
            face_center = face.landmark[168]

            iris_cx = (left_iris.x + right_iris.x) / 2
            iris_cy = (left_iris.y + right_iris.y) / 2
            eye_cx = (left_eye_center.x + right_eye_center.x) / 2
            eye_cy = (left_eye_center.y + right_eye_center.y) / 2

            yaw = math.degrees(math.atan2(
                2 * (nose_tip.x - face_center.x), nose_tip.z + 0.01
            ))
            pitch = math.degrees(math.atan2(
                2 * (nose_tip.y - face_center.y), nose_tip.z + 0.01
            ))

            iris_offset_x = iris_cx - eye_cx
            iris_offset_y = iris_cy - eye_cy

            if abs(yaw) < 15 and abs(pitch) < 15:
                direction = "center"
            elif iris_offset_x < -0.05:
                direction = "left"
            elif iris_offset_x > 0.05:
                direction = "right"
            elif iris_offset_y < -0.05:
                direction = "up"
            elif iris_offset_y > 0.05:
                direction = "down"
            else:
                direction = "center"

            return {
                "yaw": round(yaw, 1),
                "pitch": round(pitch, 1),
                "direction": direction,
                "iris_offset": (round(iris_offset_x, 3), round(iris_offset_y, 3)),
                "has_face": True,
            }
        except Exception:
            return None

    def compute_gaze_target(
        self, gaze_data: Dict, person_bboxes: Dict[int, Tuple[int, int, int, int]]
    ) -> Optional[int]:
        direction = gaze_data.get("direction", "center")
        yaw = gaze_data.get("yaw", 0)

        if direction == "center":
            return None

        best_id = None
        best_score = 0

        for tid, bbox in person_bboxes.items():
            cx = (bbox[0] + bbox[2]) / 2
            score = 1.0 / (abs(cx) + 1)
            if abs(yaw) > 20 and abs(yaw) < 90:
                if (yaw > 0 and cx > 0) or (yaw < 0 and cx < 0):
                    score *= 2
            if score > best_score:
                best_score = score
                best_id = tid

        return best_id if best_score > 0.01 else None
