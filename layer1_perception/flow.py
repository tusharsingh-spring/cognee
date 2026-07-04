"""Optical flow using RAFT-small from torchvision with Farneback fallback."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import FLOW_ENABLED, FLOW_EVERY_N, FLOW_RESIZE
from utils.logger import get_logger
from utils.profiler import profiler

logger = get_logger(__name__)


class OpticalFlowEstimator:
    def __init__(self) -> None:
        self.enabled = FLOW_ENABLED
        self.is_ready = False
        self._model = None
        self._frame_count = 0
        self._prev_gray: Optional[np.ndarray] = None

        if self.enabled:
            self._load()

    def _load(self) -> None:
        try:
            import torch
            from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

            self._model = raft_small(weights=Raft_Small_Weights.DEFAULT)
            self._model.eval()
            self._is_raft = True
            self.is_ready = True
            logger.info("[FLOW] RAFT-small ready")
        except Exception as e:
            logger.warning(f"[FLOW] RAFT failed: {e}, using Farneback fallback")
            self._is_raft = False
            self.is_ready = True

    def compute(
        self, prev_frame: np.ndarray, curr_frame: np.ndarray, should_process: bool = True
    ) -> Optional[np.ndarray]:
        if not self.is_ready:
            return None

        self._frame_count += 1
        if not should_process and self._frame_count % FLOW_EVERY_N != 0:
            return None

        prober = profiler.get("flow")
        prober.start()

        try:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            rw, rh = FLOW_RESIZE
            prev_small = cv2.resize(prev_gray, (rw, rh))
            curr_small = cv2.resize(curr_gray, (rw, rh))

            if getattr(self, "_is_raft", False) and self._model is not None:
                flow = self._raft_compute(prev_small, curr_small)
            else:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_small, curr_small, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )

            h, w = prev_frame.shape[:2]
            if flow.shape[:2] != (h, w):
                flow = cv2.resize(flow, (w, h))
                flow[:, :, 0] *= w / rw
                flow[:, :, 1] *= h / rh

            prober.stop()
            return flow
        except Exception as e:
            logger.debug(f"[FLOW] Compute failed: {e}")
            prober.stop()
            return None

    def _raft_compute(self, prev_gray, curr_gray) -> np.ndarray:
        import torch

        p = torch.from_numpy(prev_gray).float().unsqueeze(0).unsqueeze(0) / 255.0
        c = torch.from_numpy(curr_gray).float().unsqueeze(0).unsqueeze(0) / 255.0
        with torch.no_grad():
            flow = self._model(p, c)[-1]
        flow_np = flow.squeeze().permute(1, 2, 0).cpu().numpy()
        return flow_np

    def get_person_flow_stats(
        self, flow: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[Dict]:
        if flow is None:
            return None

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(flow.shape[1], x2)
        y2 = min(flow.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return None

        patch = flow[y1:y2, x1:x2]
        mag = np.sqrt(patch[:, :, 0] ** 2 + patch[:, :, 1] ** 2)

        mean_mag = float(np.mean(mag))
        max_mag = float(np.max(mag))
        mean_angle = float(np.mean(np.arctan2(patch[:, :, 1], patch[:, :, 0])))

        return {
            "mean_magnitude": round(mean_mag, 2),
            "max_magnitude": round(max_mag, 2),
            "direction_degrees": round(np.degrees(mean_angle), 1),
        }

    def get_flow_visualization(self, flow: np.ndarray) -> np.ndarray:
        if flow is None:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[:, :, 0] = ang * 180 / np.pi / 2
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
