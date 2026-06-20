"""Gaze estimation using MediaPipe FaceMesh iris (tasks API v0.10.x)."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import GAZE_ANGLE_THRESHOLD, GAZE_CONFIDENCE, GAZE_ENABLED, GAZE_EVERY_N, MODEL_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]


class GazeEstimator:
    def __init__(self) -> None:
        self._enabled = GAZE_ENABLED
        self._confidence = GAZE_CONFIDENCE
        self._angle_threshold = GAZE_ANGLE_THRESHOLD
        self._face_mesh = None
        self._loaded = False
        self._frame_counter = 0
        self._last_gaze: Dict[int, Dict] = {}

        if self._enabled:
            self._try_load()

    def _try_load(self) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_path = str(MODEL_DIR / "face_landmarker.task")
            if not Path(model_path).exists():
                logger.warning(f"[Gaze] Model not found: {model_path}")
                self._try_legacy()
                return

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=10,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                min_face_detection_confidence=self._confidence,
                min_tracking_confidence=self._confidence * 0.8,
            )
            self._face_mesh = vision.FaceLandmarker.create_from_options(options)
            self._loaded = True
            logger.info("[Gaze] MediaPipe FaceMesh loaded (tasks API)")
        except ImportError:
            self._loaded = False
            logger.warning("[Gaze] mediapipe not installed.")
        except Exception as e:
            self._loaded = False
            logger.warning(f"[Gaze] Load failed: {e}")
            self._try_legacy()

    def _try_legacy(self) -> None:
        try:
            import mediapipe as mp
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=10,
                    refine_landmarks=True,
                    min_detection_confidence=self._confidence,
                    min_tracking_confidence=self._confidence * 0.8,
                )
                self._use_legacy = True
                self._loaded = True
                logger.info("[Gaze] MediaPipe FaceMesh legacy API loaded")
        except Exception as e:
            logger.warning(f"[Gaze] Legacy load also failed: {e}")

    @property
    def is_ready(self) -> bool:
        return self._enabled and self._loaded

    @property
    def last_gaze(self) -> Dict[int, Dict]:
        return self._last_gaze

    def estimate(
        self, frame: np.ndarray, persons: List[Dict], force: bool = False
    ) -> Dict[int, Dict]:
        if not self.is_ready:
            return {}

        self._frame_counter += 1
        if not force and self._frame_counter % GAZE_EVERY_N != 0:
            return self._last_gaze

        results: Dict[int, Dict] = {}
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if getattr(self, "_use_legacy", False):
            self._last_gaze = self._estimate_legacy(rgb, persons, h, w)
            return self._last_gaze

        for person in persons:
            tid = person["track_id"]
            bbox = person["bbox"]
            x1, y1, x2, y2 = bbox

            pad = int((x2 - x1) * 0.2)
            cx1 = max(0, int(x1) - pad)
            cy1 = max(0, int(y1) - pad)
            cx2 = min(w, int(x2) + pad)
            cy2 = min(h, int(y2) + pad)
            if cx2 <= cx1 or cy2 <= cy1:
                continue

            crop = rgb[cy1:cy2, cx1:cx2]
            mp_image = __import__("mediapipe").Image(
                image_format=__import__("mediapipe").ImageFormat.SRGB, data=crop
            )
            try:
                fm_result = self._face_mesh.detect(mp_image)
            except Exception:
                continue

            if not fm_result.face_landmarks:
                continue

            gaze_data = self._extract_gaze_tasks(
                fm_result.face_landmarks[0], cx1, cy1, cx2 - cx1, cy2 - cy1
            )
            if gaze_data:
                face_bbox = self._face_bbox_from_landmarks(
                    fm_result.face_landmarks[0], cx1, cy1, cx2 - cx1, cy2 - cy1
                )
                gaze_data["face_bbox"] = face_bbox
                gaze_data["person_bbox"] = bbox
                results[tid] = gaze_data

        self._last_gaze = results
        return results

    def _extract_gaze_tasks(self, landmarks, ox, oy, rw, rh):
        try:
            left_iris_center = np.mean([[landmarks[i].x * rw + ox, landmarks[i].y * rh + oy] for i in LEFT_IRIS], axis=0)
            left_eye_l = np.array([landmarks[LEFT_EYE_CORNERS[0]].x * rw + ox, landmarks[LEFT_EYE_CORNERS[0]].y * rh + oy])
            left_eye_r = np.array([landmarks[LEFT_EYE_CORNERS[1]].x * rw + ox, landmarks[LEFT_EYE_CORNERS[1]].y * rh + oy])
            left_eye_width = float(np.linalg.norm(left_eye_r - left_eye_l))
            if left_eye_width < 1:
                return None
            left_iris_rel = float((left_iris_center[0] - left_eye_l[0]) / left_eye_width)
            left_yaw = (left_iris_rel - 0.5) * 60.0

            right_iris_center = np.mean([[landmarks[i].x * rw + ox, landmarks[i].y * rh + oy] for i in RIGHT_IRIS], axis=0)
            right_eye_l = np.array([landmarks[RIGHT_EYE_CORNERS[0]].x * rw + ox, landmarks[RIGHT_EYE_CORNERS[0]].y * rh + oy])
            right_eye_r = np.array([landmarks[RIGHT_EYE_CORNERS[1]].x * rw + ox, landmarks[RIGHT_EYE_CORNERS[1]].y * rh + oy])
            right_eye_width = float(np.linalg.norm(right_eye_r - right_eye_l))
            if right_eye_width < 1:
                right_yaw = left_yaw
            else:
                right_iris_rel = float((right_iris_center[0] - right_eye_l[0]) / right_eye_width)
                right_yaw = (right_iris_rel - 0.5) * 60.0

            yaw = float((left_yaw + right_yaw) / 2)
            eyes_center_x = (left_iris_center[0] + right_iris_center[0]) / 2
            eyes_center_y = (left_iris_center[1] + right_iris_center[1]) / 2

            direction = "center"
            if yaw < -5:
                direction = "left"
            elif yaw > 5:
                direction = "right"

            return {
                "yaw": round(yaw, 1),
                "direction": direction,
                "eyes_center": (int(eyes_center_x), int(eyes_center_y)),
                "left_eye_width": round(left_eye_width, 1),
                "right_eye_width": round(right_eye_width, 1),
            }
        except Exception as e:
            logger.debug(f"[Gaze] Extraction failed: {e}")
            return None

    def _face_bbox_from_landmarks(self, landmarks, ox, oy, rw, rh):
        xs = [lm.x * rw + ox for lm in landmarks]
        ys = [lm.y * rh + oy for lm in landmarks]
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

    def _estimate_legacy(self, rgb, persons, h, w):
        results = {}
        for person in persons:
            tid = person["track_id"]
            bbox = person["bbox"]
            x1, y1, x2, y2 = bbox
            pad = int((x2 - x1) * 0.2)
            cx1 = max(0, int(x1) - pad)
            cy1 = max(0, int(y1) - pad)
            cx2 = min(w, int(x2) + pad)
            cy2 = min(h, int(y2) + pad)
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            crop = rgb[cy1:cy2, cx1:cx2]
            try:
                fm = self._face_mesh.process(crop)
            except Exception:
                continue
            if not fm.multi_face_landmarks:
                continue
            gd = self._extract_gaze_legacy(fm.multi_face_landmarks[0], cx1, cy1, cx2 - cx1, cy2 - cy1)
            if gd:
                face_bbox = self._face_bbox_legacy(fm.multi_face_landmarks[0], cx1, cy1, cx2 - cx1, cy2 - cy1)
                gd["face_bbox"] = face_bbox
                gd["person_bbox"] = bbox
                results[tid] = gd
        return results

    def _extract_gaze_legacy(self, face_lms, ox, oy, rw, rh):
        try:
            left_iris_center = np.mean([[face_lms.landmark[i].x * rw + ox, face_lms.landmark[i].y * rh + oy] for i in LEFT_IRIS], axis=0)
            left_eye_l = np.array([face_lms.landmark[LEFT_EYE_CORNERS[0]].x * rw + ox, face_lms.landmark[LEFT_EYE_CORNERS[0]].y * rh + oy])
            left_eye_r = np.array([face_lms.landmark[LEFT_EYE_CORNERS[1]].x * rw + ox, face_lms.landmark[LEFT_EYE_CORNERS[1]].y * rh + oy])
            left_eye_width = float(np.linalg.norm(left_eye_r - left_eye_l))
            if left_eye_width < 1:
                return None
            left_yaw = (float((left_iris_center[0] - left_eye_l[0]) / left_eye_width) - 0.5) * 60.0

            right_iris_center = np.mean([[face_lms.landmark[i].x * rw + ox, face_lms.landmark[i].y * rh + oy] for i in RIGHT_IRIS], axis=0)
            right_eye_l = np.array([face_lms.landmark[RIGHT_EYE_CORNERS[0]].x * rw + ox, face_lms.landmark[RIGHT_EYE_CORNERS[0]].y * rh + oy])
            right_eye_r = np.array([face_lms.landmark[RIGHT_EYE_CORNERS[1]].x * rw + ox, face_lms.landmark[RIGHT_EYE_CORNERS[1]].y * rh + oy])
            right_eye_width = float(np.linalg.norm(right_eye_r - right_eye_l))
            right_yaw = (float((right_iris_center[0] - right_eye_l[0]) / right_eye_width) - 0.5) * 60.0 if right_eye_width >= 1 else left_yaw

            yaw = float((left_yaw + right_yaw) / 2)
            direction = "center"
            if yaw < -5:
                direction = "left"
            elif yaw > 5:
                direction = "right"

            return {
                "yaw": round(yaw, 1),
                "direction": direction,
                "eyes_center": (int((left_iris_center[0] + right_iris_center[0]) / 2), int((left_iris_center[1] + right_iris_center[1]) / 2)),
                "left_eye_width": round(left_eye_width, 1),
                "right_eye_width": round(right_eye_width, 1),
            }
        except Exception:
            return None

    def _face_bbox_legacy(self, face_lms, ox, oy, rw, rh):
        xs = [lm.x * rw + ox for lm in face_lms.landmark]
        ys = [lm.y * rh + oy for lm in face_lms.landmark]
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

    def compute_gaze_target(
        self, gaze_data: Dict, other_persons: Dict[int, Tuple[int, int, int, int]], angle_threshold: Optional[float] = None
    ) -> Optional[int]:
        if not gaze_data:
            return None
        thresh = angle_threshold if angle_threshold is not None else self._angle_threshold
        yaw = gaze_data.get("yaw", 0)
        ec = gaze_data.get("eyes_center")
        if ec is None:
            return None
        gc_x, gc_y = ec
        best_tid, best_offset = None, float("inf")
        for tid, (bx1, by1, bx2, by2) in other_persons.items():
            pcx = (bx1 + bx2) / 2
            pcy = (by1 + by2) / 2
            angle = float(np.degrees(np.arctan2(pcx - gc_x, abs(pcy - gc_y) + 1e-6)))
            offset = abs(angle - yaw)
            if offset < thresh and offset < best_offset:
                best_offset, best_tid = offset, tid
        return best_tid

    def close(self) -> None:
        if self._face_mesh is not None and hasattr(self._face_mesh, "close"):
            self._face_mesh.close()
