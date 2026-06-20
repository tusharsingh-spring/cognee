"""Optical flow estimation using RAFT-small via torchvision."""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    CV_DEVICE,
    FLOW_ENABLED,
    FLOW_EVERY_N,
    FLOW_RESIZE,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class OpticalFlowEstimator:
    def __init__(self) -> None:
        self._enabled = FLOW_ENABLED
        self._device = CV_DEVICE
        self._resize = FLOW_RESIZE
        self._model = None
        self._loaded = False
        self._load_error = None
        self._frame_counter = 0
        self._prev_frame: Optional[np.ndarray] = None
        self._last_flow: Optional[np.ndarray] = None

        if self._enabled:
            self._try_load()

    def _try_load(self) -> None:
        try:
            import torch
            import torchvision

            logger.info("[Flow] Loading RAFT-small from torchvision...")
            self._model = torchvision.models.optical_flow.raft_small(pretrained=True)
            if CV_DEVICE == "cuda" and torch.cuda.is_available():
                self._model = self._model.cuda()
            self._model.eval()
            self._loaded = True
            logger.info("[Flow] RAFT-small loaded")
        except Exception as e:
            self._load_error = str(e)
            logger.warning(f"[Flow] RAFT load failed: {e}")
            self._try_cv_fallback()

    def _try_cv_fallback(self) -> None:
        self._use_cv_fallback = True
        self._loaded = True
        logger.info("[Flow] Using OpenCV Farneback optical flow as fallback")

    @property
    def is_ready(self) -> bool:
        return self._enabled and self._loaded

    @property
    def last_flow(self) -> Optional[np.ndarray]:
        return self._last_flow

    def compute(self, prev_frame: np.ndarray, curr_frame: np.ndarray, force: bool = False) -> Optional[np.ndarray]:
        if not self.is_ready:
            return None

        self._frame_counter += 1
        if not force and self._frame_counter % FLOW_EVERY_N != 0:
            return self._last_flow

        try:
            if getattr(self, "_use_cv_fallback", False):
                flow = self._compute_farneback(prev_frame, curr_frame)
            else:
                flow = self._compute_raft(prev_frame, curr_frame)

            if flow is not None:
                self._last_flow = flow
            return self._last_flow
        except Exception as e:
            logger.debug(f"[Flow] Computation failed: {e}")
            return self._last_flow

    def _compute_raft(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            import torch

            rh, rw = self._resize
            p = cv2.resize(prev_frame, (rw, rh))
            c = cv2.resize(curr_frame, (rw, rh))

            p_tensor = torch.from_numpy(p).permute(2, 0, 1).unsqueeze(0).float()
            c_tensor = torch.from_numpy(c).permute(2, 0, 1).unsqueeze(0).float()

            if CV_DEVICE == "cuda" and torch.cuda.is_available():
                p_tensor = p_tensor.cuda()
                c_tensor = c_tensor.cuda()

            with torch.no_grad():
                flows = self._model(p_tensor, c_tensor)

            flow = flows[-1] if isinstance(flows, list) else flows
            flow = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()

            h, w = curr_frame.shape[:2]
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
            flow[:, :, 0] *= (w / rw)
            flow[:, :, 1] *= (h / rh)

            return flow
        except Exception as e:
            logger.debug(f"[Flow] RAFT failed: {e}")
            return self._compute_farneback(prev_frame, curr_frame)

    def _compute_farneback(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            p_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            c_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                p_gray, c_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            return flow
        except Exception as e:
            logger.debug(f"[Flow] Farneback failed: {e}")
            return None

    def get_flow_magnitude(self, flow: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        f = flow if flow is not None else self._last_flow
        if f is None:
            return None
        mag, _ = cv2.cartToPolar(f[..., 0], f[..., 1])
        return mag

    def get_person_flow_stats(
        self, flow: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[Dict[str, float]]:
        if flow is None:
            return None

        x1, y1, x2, y2 = bbox
        h, w = flow.shape[:2]
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(w, int(x2)); y2 = min(h, int(y2))

        if x2 <= x1 or y2 <= y1:
            return None

        region = flow[y1:y2, x1:x2]
        if region.size == 0:
            return None

        mag = np.sqrt(region[..., 0]**2 + region[..., 1]**2)

        return {
            "mean_magnitude": float(np.mean(mag)),
            "max_magnitude": float(np.max(mag)),
            "std_magnitude": float(np.std(mag)),
            "median_magnitude": float(np.median(mag)),
        }

    def get_flow_visualization(self, flow: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        f = flow if flow is not None else self._last_flow
        if f is None:
            return None

        mag, ang = cv2.cartToPolar(f[..., 0], f[..., 1])
        hsv = np.zeros((f.shape[0], f.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    def detect_motion_boundaries(
        self, flow: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        if flow is None:
            return np.zeros((1, 1), dtype=np.uint8)

        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        mag_norm = mag / (mag.max() + 1e-8)

        grad_x = cv2.Sobel(mag_norm, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(mag_norm, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        boundaries = (grad_mag > threshold).astype(np.uint8) * 255
        return boundaries
