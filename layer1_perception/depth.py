"""Monocular depth estimation using MiDaS ONNX with Depth Anything V2 fallback."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    DEPTH_CONTACT_Z_THRESHOLD,
    DEPTH_ENABLED,
    DEPTH_EVERY_N,
    DEPTH_ONNX_PATH,
    DEPTH_RESIZE,
    MODEL_DIR,
)
from utils.logger import get_logger
from utils.profiler import profiler

logger = get_logger(__name__)


class DepthEstimator:
    def __init__(self) -> None:
        self.enabled = DEPTH_ENABLED
        self.is_ready = False
        self._model = None
        self._backend = "none"
        self._frame_count = 0
        self._transforms = None

        if self.enabled:
            self._load()

    def _load(self) -> None:
        if self._try_onnx():
            return
        if self._try_torch():
            return
        logger.warning("[DEPTH] All backends failed")

    def _try_onnx(self) -> bool:
        try:
            import onnxruntime as ort

            onnx_path = DEPTH_ONNX_PATH
            if not onnx_path.is_file():
                onnx_path = MODEL_DIR / "midas_v2_small.onnx"
            if not onnx_path.is_file():
                return False

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._onnx_session = ort.InferenceSession(
                str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self._backend = "onnx_midas"
            self.is_ready = True
            logger.info("[DEPTH] MiDaS ONNX ready")
            return True
        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"[DEPTH] ONNX failed: {e}")
            return False

    def _try_torch(self) -> bool:
        try:
            import torch

            self._model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self._model.eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self._transforms = midas_transforms.small_transform
            self._backend = "torch_midas"
            self.is_ready = True
            logger.info("[DEPTH] MiDaS torch ready")
            return True
        except Exception:
            try:
                import torch
                self._model = torch.hub.load(
                    "depth-anything/Depth-Anything-V2-Small", "Depth-Anything-V2-Small",
                    source="local", pretrained=False
                )
                self._model.eval()
                self._transforms = None
                self._backend = "torch_depth_anything"
                self.is_ready = True
                logger.info("[DEPTH] Depth-Anything-V2 torch ready")
                return True
            except Exception:
                return False

    def estimate(
        self, frame: np.ndarray, should_process: bool = True
    ) -> Optional[np.ndarray]:
        if not self.is_ready:
            return None

        self._frame_count += 1
        if not should_process and self._frame_count % DEPTH_EVERY_N != 0:
            return None

        prober = profiler.get("depth")
        prober.start()

        try:
            rw, rh = DEPTH_RESIZE
            resized = cv2.resize(frame, (rw, rh))

            if self._backend == "onnx_midas":
                depth = self._onnx_infer(resized)
            elif self._backend == "torch_midas":
                depth = self._torch_infer(resized)
            elif self._backend == "torch_depth_anything":
                depth = self._torch_da_infer(resized)
            else:
                depth = None

            if depth is not None:
                depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))

            prober.stop()
            return depth
        except Exception as e:
            logger.debug(f"[DEPTH] Estimate failed: {e}")
            prober.stop()
            return None

    def _onnx_infer(self, img) -> Optional[np.ndarray]:
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
            tensor = (tensor - 0.485) / 0.229
            output = self._onnx_session.run(None, {"input": tensor.astype(np.float32)})
            return np.squeeze(output[0])
        except Exception:
            return None

    def _torch_infer(self, img) -> Optional[np.ndarray]:
        import torch
        if self._transforms is not None:
            tensor = self._transforms(img)
        else:
            tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            pred = self._model(tensor)
            if isinstance(pred, torch.Tensor):
                return pred.squeeze().cpu().numpy()
        return None

    def _torch_da_infer(self, img) -> Optional[np.ndarray]:
        import torch
        tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            pred = self._model(tensor)
            if isinstance(pred, torch.Tensor):
                return pred.squeeze().cpu().numpy()
        return None

    def get_person_depth(
        self, depth_map: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[Dict]:
        if depth_map is None:
            return None

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(depth_map.shape[1], x2)
        y2 = min(depth_map.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return None

        patch = depth_map[y1:y2, x1:x2]
        if patch.size == 0:
            return None

        center_y1 = y1 + (y2 - y1) // 3
        center_y2 = y1 + 2 * (y2 - y1) // 3
        torso = depth_map[center_y1:center_y2, x1:x2]

        return {
            "mean_depth": round(float(np.mean(patch)), 3),
            "torso_depth": round(float(np.mean(torso)) if torso.size > 0 else 0, 3),
            "min_depth": round(float(np.min(patch)), 3),
            "max_depth": round(float(np.max(patch)), 3),
        }

    def compute_3d_distance(
        self,
        depth_a: Optional[Dict],
        depth_b: Optional[Dict],
        pixel_dist: float,
    ) -> Optional[float]:
        if depth_a is None or depth_b is None:
            return None
        z_a = depth_a.get("torso_depth", 0)
        z_b = depth_b.get("torso_depth", 0)
        z_diff = abs(z_a - z_b) * DEPTH_CONTACT_Z_THRESHOLD / 1000.0
        return float(np.sqrt(pixel_dist ** 2 + z_diff ** 2))

    def get_depth_heatmap(self, depth_map: np.ndarray) -> np.ndarray:
        if depth_map is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        d_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)
