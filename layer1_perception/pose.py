"""Pose estimation using MediaPipe Tasks API → ONNX RTMPose → CNN fallback.

Multi-backend with graceful degradation. Outputs 17 COCO keypoints per person.
"""

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    MODEL_DIR,
    POSE_COCO_KEYPOINTS,
    POSE_CONFIDENCE,
    POSE_ENABLED,
    POSE_IMG_SIZE,
    POSE_ONNX_PATH,
)
from utils.logger import get_logger
from utils.profiler import profiler

logger = get_logger(__name__)

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

COCO_JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

TORSO_INDICES = [5, 6, 11, 12]
HEAD_INDICES = [0, 1, 2, 3, 4]
ARM_INDICES = [5, 7, 9, 6, 8, 10]
LEG_INDICES = [11, 13, 15, 12, 14, 16]


class PoseEstimator:
    def __init__(self) -> None:
        self.enabled = POSE_ENABLED
        self.is_ready = False
        self._model = None
        self._backend = "none"
        self._prev_poses: Dict[int, np.ndarray] = {}

        if not self.enabled:
            return

        if self._try_mediapipe():
            pass
        elif self._try_onnx():
            pass
        elif self._try_torch_fallback():
            pass

        if self.is_ready:
            logger.info(f"[POSE] Ready (backend={self._backend})")
        else:
            logger.warning("[POSE] All backends failed — pose disabled")

    def _try_mediapipe(self) -> bool:
        try:
            import mediapipe as mp

            mp_pose = mp.solutions.pose
            self._model = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=POSE_CONFIDENCE,
                min_tracking_confidence=POSE_CONFIDENCE,
            )
            self._backend = "mediapipe"
            self.is_ready = True
            return True
        except Exception:
            for mod in list(sys.modules.keys()):
                if mod == "mediapipe" or mod.startswith("mediapipe."):
                    del sys.modules[mod]
            return False

    def _try_onnx(self) -> bool:
        try:
            import onnxruntime as ort

            onnx_path = POSE_ONNX_PATH
            if not onnx_path.is_file():
                onnx_path = MODEL_DIR / "rtmpose_s.onnx"
            if not onnx_path.is_file():
                return False

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._onnx_session = ort.InferenceSession(
                str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self._backend = "onnx_rtmpose"
            self.is_ready = True
            return True
        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"[POSE] ONNX failed: {e}")
            return False

    def _try_torch_fallback(self) -> bool:
        try:
            import torch
            import torch.nn as nn

            self._torch_model = SimplePoseCNN()
            self._backend = "torch_cnn"
            self.is_ready = True
            return True
        except ImportError:
            return False

    def estimate(
        self, frame: np.ndarray, persons: List[Dict]
    ) -> Dict[int, np.ndarray]:
        if not self.is_ready or not persons:
            return {}

        prober = profiler.get("pose")
        prober.start()

        results = {}
        for person in persons:
            tid = person["track_id"]
            bbox = person["bbox"]
            kpts = self._estimate_single(frame, bbox)
            if kpts is not None:
                results[tid] = kpts.astype(np.float32)

        self._prev_poses = {k: v.copy() for k, v in results.items()}
        prober.stop()
        return results

    def _estimate_single(self, frame, bbox) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        if self._backend == "mediapipe":
            return self._mediapipe_infer(crop)
        elif self._backend == "onnx_rtmpose":
            return self._onnx_infer(crop)
        elif self._backend == "torch_cnn":
            return self._torch_infer(crop)
        return None

    def _mediapipe_infer(self, crop) -> Optional[np.ndarray]:
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result = self._model.process(rgb)
            if not result.pose_landmarks:
                return None

            h, w = crop.shape[:2]
            keypoints = np.zeros((POSE_COCO_KEYPOINTS, 3), dtype=np.float32)
            mapping = list(range(11)) + [12] + [14] + [16] + [18] + [20] + [
                lm for lm in [23, 24, 25, 26, 27, 28] if lm <= 32
            ]
            for i in range(min(POSE_COCO_KEYPOINTS, len(result.pose_landmarks.landmark))):
                lm = result.pose_landmarks.landmark[i] if i < len(result.pose_landmarks.landmark) else None
                if lm and lm.visibility > 0.3:
                    keypoints[i] = [lm.x * w, lm.y * h, lm.visibility]
            return keypoints if keypoints[:, 2].max() > 0 else None
        except Exception:
            return None

    def _onnx_infer(self, crop) -> Optional[np.ndarray]:
        try:
            h, w = crop.shape[:2]
            input_h, input_w = POSE_IMG_SIZE[1], POSE_IMG_SIZE[0]
            resized = cv2.resize(crop, (input_w, input_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
            tensor = (tensor - 0.485) / 0.229

            outputs = self._onnx_session.run(None, {"input": tensor})
            heatmap = outputs[0][0]  # (17, H/4, W/4)

            keypoints = np.zeros((POSE_COCO_KEYPOINTS, 3), dtype=np.float32)
            hm_h, hm_w = heatmap.shape[1], heatmap.shape[2]
            scale_x = w / hm_w
            scale_y = h / hm_h

            for j in range(POSE_COCO_KEYPOINTS):
                hm = heatmap[j]
                max_idx = np.argmax(hm)
                py, px = divmod(max_idx, hm_w)
                conf = hm[py, px]
                keypoints[j] = [px * scale_x, py * scale_y, min(float(conf), 1.0)]

            return keypoints if keypoints[:, 2].max() > 0.3 else None
        except Exception:
            return None

    def _torch_infer(self, crop) -> Optional[np.ndarray]:
        try:
            import torch

            resized = cv2.resize(crop, (64, 64))
            tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0

            with torch.no_grad():
                output = self._torch_model(tensor)
                kp = output.squeeze().cpu().numpy()
                kp = kp.reshape(POSE_COCO_KEYPOINTS, 3)
                kp[:, :2] *= max(crop.shape[:2])
                return kp.astype(np.float32)
        except Exception:
            return None

    def get_torso_center(self, kpts: np.ndarray) -> Optional[Tuple[float, float]]:
        if kpts is None:
            return None
        valid = kpts[TORSO_INDICES, 2] > 0.3
        if valid.sum() < 1:
            return None
        pts = kpts[TORSO_INDICES, :2][valid]
        return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))

    def compute_velocity(
        self, tid: int, kpts: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        if tid not in self._prev_poses:
            return None
        prev = self._prev_poses[tid]
        curr_center = self.get_torso_center(kpts)
        prev_center = self.get_torso_center(prev)
        if curr_center is None or prev_center is None:
            return None
        return (curr_center[0] - prev_center[0], curr_center[1] - prev_center[1], 0.0)

    def get_joint_pair_distance(
        self, kpts1: np.ndarray, kpts2: np.ndarray
    ) -> Optional[float]:
        if kpts1 is None or kpts2 is None:
            return None
        valid = (kpts1[:, 2] > 0.3) & (kpts2[:, 2] > 0.3)
        if valid.sum() < 2:
            return None
        diffs = np.linalg.norm(kpts1[valid, :2] - kpts2[valid, :2], axis=1)
        return float(diffs.min())


class SimplePoseCNN:
    """Minimal CNN for pose heatmap regression — CPU fallback."""

    def __init__(self):
        import torch
        import torch.nn as nn

        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, POSE_COCO_KEYPOINTS * 3, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)

    def __call__(self, x):
        return self.forward(x)
