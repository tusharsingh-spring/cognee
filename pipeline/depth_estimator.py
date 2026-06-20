"""Depth estimation using Depth Anything V2 Small via ONNX or torch hub."""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    CV_DEVICE,
    CV_USE_FP16,
    DEPTH_CONTACT_Z_THRESHOLD,
    DEPTH_DEVICE,
    DEPTH_ENABLED,
    DEPTH_EVERY_N,
    DEPTH_RESIZE,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class DepthEstimator:
    def __init__(self) -> None:
        self._enabled = DEPTH_ENABLED
        self._device = CV_DEVICE
        self._resize = DEPTH_RESIZE
        self._model = None
        self._cv_net = None
        self._model_type = ""
        self._loaded = False
        self._load_error = None
        self._frame_counter = 0
        self._last_depth: Optional[np.ndarray] = None

        if self._enabled:
            self._try_load()

    def _try_load(self) -> None:
        self._try_opencv_dnn()
        if self._loaded:
            return
        self._try_torch_hub()
        if self._loaded:
            return
        self._try_local_pth()

    def _try_opencv_dnn(self) -> None:
        try:
            from config.settings import MODEL_DIR
            model_path = MODEL_DIR / "midas_v2_small.onnx"
            if not model_path.exists():
                logger.info("[Depth] No ONNX model found, trying download...")
                import urllib.request
                url = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
                try:
                    urllib.request.urlretrieve(url, str(model_path))
                    logger.info(f"[Depth] Downloaded MiDaS ONNX to {model_path}")
                except Exception:
                    logger.warning("[Depth] Could not download MiDaS ONNX")
                    return

            self._cv_net = cv2.dnn.readNetFromONNX(str(model_path))
            self._cv_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._cv_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._loaded = True
            self._model_type = "opencv_midas"
            logger.info("[Depth] MiDaS Small ONNX loaded via OpenCV DNN")
        except Exception as e:
            logger.debug(f"[Depth] OpenCV DNN load failed: {e}")

    def _try_torch_hub(self) -> None:
        try:
            import torch
            logger.info("[Depth] Loading Depth Anything V2 Small from torch hub...")
            repo = "DepthAnything/Depth-Anything-V2-Small"
            self._model = torch.hub.load(repo, "DepthAnythingV2Small", pretrained=True, trust_repo=True, skip_validation=True)
            if CV_DEVICE == "cuda" and torch.cuda.is_available():
                self._model = self._model.cuda()
            self._model.eval()
            self._loaded = True
            self._model_type = "torch_hub"
            logger.info("[Depth] Depth Anything V2 Small loaded via torch hub")
        except Exception as e:
            logger.debug(f"[Depth] torch hub load failed: {e}")

    def _try_local_pth(self) -> None:
        try:
            import torch
            from config.settings import MODEL_DIR
            local_path = MODEL_DIR / "depth_anything_v2_vits.pth"
            if local_path.exists():
                from depth_anything_v2.dpt import DepthAnythingV2
                self._model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
                self._model.load_state_dict(torch.load(str(local_path), map_location="cpu"))
                if CV_DEVICE == "cuda" and torch.cuda.is_available():
                    self._model = self._model.cuda()
                self._model.eval()
                self._loaded = True
                self._model_type = "local_pth"
                logger.info("[Depth] Loaded from local weights")
            else:
                logger.warning("[Depth] No depth model available. Disabling.")
        except Exception as e:
            logger.warning(f"[Depth] Local load failed: {e}")

    @property
    def is_ready(self) -> bool:
        return self._enabled and self._loaded

    @property
    def last_depth(self) -> Optional[np.ndarray]:
        return self._last_depth

    def estimate(self, frame: np.ndarray, force: bool = False) -> Optional[np.ndarray]:
        if not self.is_ready:
            return None

        self._frame_counter += 1
        if not force and self._frame_counter % DEPTH_EVERY_N != 0:
            return self._last_depth

        try:
            if getattr(self, "_model_type", "") == "opencv_midas":
                depth = self._estimate_cv(frame)
            else:
                depth = self._estimate_torch(frame)

            if depth is not None:
                self._last_depth = depth
            return self._last_depth

        except Exception as e:
            logger.debug(f"[Depth] Inference failed: {e}")
            return self._last_depth

    def _estimate_cv(self, frame: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        rh, rw = 256, 256
        inp = cv2.resize(frame, (rw, rh))
        blob = cv2.dnn.blobFromImage(inp, 1/255.0, (rw, rh), (0, 0, 0), swapRB=True, crop=False)
        self._cv_net.setInput(blob)
        output = self._cv_net.forward()
        output = output.squeeze()
        output = cv2.resize(output, (w, h), interpolation=cv2.INTER_LINEAR)
        output = (output - output.min()) / (output.max() - output.min() + 1e-8)
        return output

    def _estimate_torch(self, frame: np.ndarray) -> Optional[np.ndarray]:
        import torch

        h, w = frame.shape[:2]
        rh, rw = self._resize
        inp = cv2.resize(frame, (rw, rh))
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        inp_tensor = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        if CV_DEVICE == "cuda" and torch.cuda.is_available():
            inp_tensor = inp_tensor.cuda()

        with torch.no_grad():
            depth = self._model(inp_tensor)

        if isinstance(depth, (list, tuple)):
            depth = depth[-1]

        depth = depth.squeeze().cpu().numpy()
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth

    def get_person_depth(
        self, depth_map: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[Dict[str, float]]:
        if depth_map is None:
            return None

        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(w, int(x2)); y2 = min(h, int(y2))

        if x2 <= x1 or y2 <= y1:
            return None

        region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            return None

        torso_cy = y1 + (y2 - y1) // 3
        torso_ch = (y2 - y1) // 3
        ty1 = max(y1, int(torso_cy - torso_ch // 2))
        ty2 = min(y2, int(torso_cy + torso_ch // 2))
        tx1 = max(x1, int(x1 + (x2 - x1) * 0.3))
        tx2 = min(x2, int(x2 - (x2 - x1) * 0.3))

        if ty2 > ty1 and tx2 > tx1:
            torso_region = depth_map[ty1:ty2, tx1:tx2]
            if torso_region.size > 0:
                return {
                    "mean_depth": float(np.mean(region)),
                    "median_depth": float(np.median(region)),
                    "torso_depth": float(np.mean(torso_region)),
                    "std_depth": float(np.std(region)),
                }

        return {
            "mean_depth": float(np.mean(region)),
            "median_depth": float(np.median(region)),
            "torso_depth": float(np.mean(region)),
            "std_depth": float(np.std(region)),
        }

    def compute_3d_distance(
        self,
        depth_a: Dict[str, float],
        depth_b: Dict[str, float],
        pixel_dist: float,
        focal_length_px: float = 800.0,
    ) -> Optional[float]:
        if depth_a is None or depth_b is None:
            return None

        z_a = depth_a.get("torso_depth", depth_a.get("mean_depth", 0))
        z_b = depth_b.get("torso_depth", depth_b.get("mean_depth", 0))

        if z_a <= 0 or z_b <= 0:
            return None

        x_a = (pixel_dist * 0.5) / focal_length_px * z_a * DEPTH_CONTACT_Z_THRESHOLD
        x_b = (pixel_dist * 0.5) / focal_length_px * z_b * DEPTH_CONTACT_Z_THRESHOLD

        d_xy = (x_a + x_b) / 2
        d_z = abs(z_a - z_b) * DEPTH_CONTACT_Z_THRESHOLD

        return float(np.sqrt(d_xy**2 + d_z**2))

    def get_depth_heatmap(self, depth_map: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        dmap = depth_map if depth_map is not None else self._last_depth
        if dmap is None:
            return None
        heatmap = (dmap * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO)
        return heatmap
