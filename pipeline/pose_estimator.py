"""Pose estimation using MediaPipe Pose (tasks API) with ONNX fallback."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    CV_DEVICE,
    CV_USE_FP16,
    MODEL_DIR,
    POSE_COCO_KEYPOINTS,
    POSE_CONFIDENCE,
    POSE_DET_ONNX_PATH,
    POSE_ENABLED,
    POSE_IMG_SIZE,
    POSE_ONNX_PATH,
)
from utils.logger import get_logger

logger = get_logger(__name__)

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

SKELETON_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
]

COCO_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 255, 0), (255, 128, 0), (0, 128, 255), (128, 0, 255),
    (255, 255, 128), (128, 255, 255), (255, 128, 255), (0, 255, 128), (128, 255, 128),
    (0, 128, 128), (128, 128, 255),
]


class PoseEstimator:
    def __init__(self) -> None:
        self._enabled = POSE_ENABLED
        self._confidence = POSE_CONFIDENCE
        self._img_size = POSE_IMG_SIZE
        self._device = CV_DEVICE
        self._session = None
        self._det_session = None
        self._loaded = False
        self._load_error = None

        if self._enabled:
            self._try_load()

    def _try_load(self) -> None:
        if self._try_mediapipe():
            return
        if self._try_onnx():
            return
        self._load_torch_fallback()

    def _try_mediapipe(self) -> bool:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_path = str(MODEL_DIR / "pose_landmarker_lite.task")
            if not Path(model_path).exists():
                logger.warning(f"[Pose] Model not found: {model_path}")
                return self._try_mediapipe_legacy()

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                num_poses=10,
                min_pose_detection_confidence=self._confidence,
                min_tracking_confidence=self._confidence * 0.8,
            )
            self._mp_pose = vision.PoseLandmarker.create_from_options(options)
            self._loaded = True
            self._mp_enabled = True
            logger.info("[Pose] MediaPipe Pose landmarker loaded (tasks API)")
            return True
        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"[Pose] MediaPipe tasks failed: {e}")
            return self._try_mediapipe_legacy()

    def _try_mediapipe_legacy(self) -> bool:
        try:
            import mediapipe as mp
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
                self._mp_pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=self._confidence,
                    min_tracking_confidence=self._confidence * 0.8,
                )
                self._loaded = True
                self._mp_enabled = True
                self._mp_legacy = True
                logger.info("[Pose] MediaPipe Pose loaded (legacy API)")
                return True
        except Exception as e:
            logger.debug(f"[Pose] MediaPipe legacy failed: {e}")
        return False

    def _try_onnx(self) -> bool:
        try:
            import onnxruntime as ort
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if CV_DEVICE == "cuda"
                else ["CPUExecutionProvider"]
            )
            if POSE_ONNX_PATH.exists():
                self._session = ort.InferenceSession(str(POSE_ONNX_PATH), providers=providers)
                self._loaded = True
                self._onnx_enabled = True
                logger.info(f"[Pose] RTMPose ONNX loaded from {POSE_ONNX_PATH}")
                if POSE_DET_ONNX_PATH.exists():
                    self._det_session = ort.InferenceSession(str(POSE_DET_ONNX_PATH), providers=providers)
                return True
        except Exception as e:
            logger.debug(f"[Pose] ONNX failed: {e}")
        return False

    def _load_torch_fallback(self) -> None:
        try:
            import torch
            import torch.nn as nn

            self._torch_enabled = True
            self._torch_pose_model = None
            logger.info("[Pose] Torch fallback ready, will use lightweight CNN on first frame")
            self._loaded = True
        except ImportError:
            self._loaded = False
            logger.error("[Pose] Torch not available either. Pose estimation disabled.")

    @property
    def is_ready(self) -> bool:
        return self._enabled and self._loaded

    def estimate(
        self, frame: np.ndarray, persons: List[Dict]
    ) -> Dict[int, np.ndarray]:
        results: Dict[int, np.ndarray] = {}
        if not self.is_ready:
            return results

        if getattr(self, "_mp_enabled", False):
            return self._estimate_mediapipe(frame, persons)

        if getattr(self, "_onnx_enabled", False):
            return self._estimate_onnx(frame, persons)

        return results

    def _estimate_mediapipe(
        self, frame: np.ndarray, persons: List[Dict]
    ) -> Dict[int, np.ndarray]:
        results: Dict[int, np.ndarray] = {}
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if getattr(self, "_mp_legacy", False):
            try:
                mp_result = self._mp_pose.process(rgb)
            except Exception:
                return results
            if not mp_result.pose_landmarks:
                return results
            pose_lms = mp_result.pose_landmarks.landmark
        else:
            mp_image = __import__("mediapipe").Image(
                image_format=__import__("mediapipe").ImageFormat.SRGB, data=rgb
            )
            try:
                mp_result = self._mp_pose.detect(mp_image)
            except Exception:
                return results
            if not mp_result.pose_landmarks:
                return results
            pose_lms = mp_result.pose_landmarks[0]

        kpts_all = np.zeros((17, 3), dtype=np.float32)
        mp_to_coco = [0, 0, 0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        for ci, mi in enumerate(mp_to_coco):
            if mi < len(pose_lms):
                lm = pose_lms[mi]
                kpts_all[ci] = [lm.x * w, lm.y * h, lm.visibility if hasattr(lm, "visibility") else 1.0]

        for person in persons:
            tid = person["track_id"]
            bbox = person["bbox"]
            x1, y1, x2, y2 = bbox
            person_kpts = kpts_all.copy()
            in_box = 0
            for ji in range(17):
                if x1 <= person_kpts[ji, 0] <= x2 and y1 <= person_kpts[ji, 1] <= y2:
                    in_box += 1
                else:
                    person_kpts[ji, 2] = 0.0
            if in_box >= 3:
                results[tid] = person_kpts

        return results

    def _estimate_onnx(
        self, frame: np.ndarray, persons: List[Dict]
    ) -> Dict[int, np.ndarray]:
        results: Dict[int, np.ndarray] = {}
        h, w = frame.shape[:2]
        for person in persons:
            tid = person["track_id"]
            bbox = person["bbox"]
            x1, y1, x2, y2 = bbox
            pad = int((x2 - x1) * 0.15)
            x1c = max(0, x1 - pad)
            y1c = max(0, y1 - pad)
            x2c = min(w, x2 + pad)
            y2c = min(h, y2 + pad)
            crop = frame[y1c:y2c, x1c:x2c]
            if crop.size == 0:
                continue
            keypoints = self._infer_onnx_person(crop, (x1c, y1c, x2c - x1c, y2c - y1c))
            if keypoints is not None and len(keypoints) == POSE_COCO_KEYPOINTS:
                results[tid] = keypoints
        return results

    def _infer_onnx_person(
        self, crop: np.ndarray, roi: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        rx, ry, rw, rh = roi

        if self._session is not None:
            try:
                import onnxruntime as ort
                img = cv2.resize(crop, self._img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
                img = np.expand_dims(img, axis=0)
                input_name = self._session.get_inputs()[0].name
                outputs = self._session.run(None, {input_name: img})
                heatmaps = outputs[0]
                kpts = self._decode_simcc(heatmaps[0], rw, rh)
                kpts[:, 0] += rx
                kpts[:, 1] += ry
                return kpts
            except Exception as e:
                logger.debug(f"[Pose] ONNX inference failed: {e}")

        return None

    def _infer_torch_simple(
        self, crop: np.ndarray, rx: int, ry: int, rw: int, rh: int
    ) -> Optional[np.ndarray]:
        try:
            import torch

            if not hasattr(self, "_simple_pose_model") or self._simple_pose_model is None:
                self._simple_pose_model = self._build_simple_pose_cnn()

            img = cv2.resize(crop, self._img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

            if CV_DEVICE == "cuda" and torch.cuda.is_available():
                tensor = tensor.cuda()

            with torch.no_grad():
                heatmaps = self._simple_pose_model(tensor)

            heatmaps = heatmaps.cpu().numpy()
            kpts = self._decode_simcc(heatmaps[0], rw, rh)
            kpts[:, 0] += rx
            kpts[:, 1] += ry
            return kpts
        except Exception as e:
            logger.debug(f"[Pose] Torch inference failed: {e}")
            return None

    def _build_simple_pose_cnn(self):
        import torch
        import torch.nn as nn

        class SimplePoseCNN(nn.Module):
            def __init__(self, num_keypoints=17):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
                )
                self.decoder = nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(32, num_keypoints, 1),
                )

            def forward(self, x):
                return self.decoder(self.encoder(x))

        model = SimplePoseCNN(num_keypoints=POSE_COCO_KEYPOINTS)
        if CV_DEVICE == "cuda" and torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        return model

    def _decode_simcc(
        self, heatmaps: np.ndarray, w: int, h: int
    ) -> np.ndarray:
        n_kpts, hm_h, hm_w = heatmaps.shape
        kpts = np.zeros((n_kpts, 3), dtype=np.float32)

        for i in range(n_kpts):
            hm = heatmaps[i]
            _, conf, _, max_loc = cv2.minMaxLoc(hm)
            if conf < self._confidence:
                kpts[i, 2] = 0.0
                continue
            kpts[i, 0] = float(max_loc[0]) / hm_w * w
            kpts[i, 1] = float(max_loc[1]) / hm_h * h
            kpts[i, 2] = float(conf)

        return kpts

    def compute_velocity(
        self, prev_kpts: np.ndarray, curr_kpts: np.ndarray, dt: float
    ) -> np.ndarray:
        if dt <= 0:
            return np.zeros((prev_kpts.shape[0], 3), dtype=np.float32)
        vel = np.zeros_like(curr_kpts)
        valid = (prev_kpts[:, 2] > self._confidence) & (curr_kpts[:, 2] > self._confidence)
        vel[valid, :2] = (curr_kpts[valid, :2] - prev_kpts[valid, :2]) / dt
        vel[valid, 2] = np.minimum(prev_kpts[valid, 2], curr_kpts[valid, 2])
        return vel

    def get_torso_center(self, kpts: np.ndarray) -> Optional[Tuple[float, float]]:
        l_shoulder = kpts[5]
        r_shoulder = kpts[6]
        l_hip = kpts[11]
        r_hip = kpts[12]
        torso_points = [l_shoulder, r_shoulder, l_hip, r_hip]
        valid = [pt for pt in torso_points if pt[2] > self._confidence]
        if len(valid) < 2:
            return None
        return (float(np.mean([v[0] for v in valid])), float(np.mean([v[1] for v in valid])))

    def get_joint_pair_distance(
        self, kpts_a: np.ndarray, kpts_b: np.ndarray, joint_idx: int
    ) -> Optional[float]:
        if kpts_a[joint_idx, 2] < self._confidence or kpts_b[joint_idx, 2] < self._confidence:
            return None
        return float(np.linalg.norm(kpts_a[joint_idx, :2] - kpts_b[joint_idx, :2]))
