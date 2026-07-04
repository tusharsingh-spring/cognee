"""Segmentation using MobileSAM (CPU-optimized)."""

from typing import Dict, List, Optional

import cv2
import numpy as np

from config.settings import SEG_ALPHA, SEG_ENABLED, SEG_EVERY_N, MODEL_DIR
from utils.logger import get_logger
from utils.profiler import profiler

logger = get_logger(__name__)


class Segmenter:
    def __init__(self) -> None:
        self.enabled = SEG_ENABLED
        self.is_ready = False
        self._model = None
        self._frame_count = 0

        if self.enabled:
            self._load()

    def _load(self) -> None:
        try:
            from ultralytics import SAM
            sam_path = MODEL_DIR / "mobile_sam.pt"
            if not sam_path.is_file():
                sam_path = "mobile_sam.pt"
            self._model = SAM(str(sam_path))
            self.is_ready = True
            logger.info("[SEG] MobileSAM ready")
        except Exception as e:
            logger.warning(f"[SEG] Init failed: {e}")

    def segment(
        self, frame: np.ndarray, persons: List[Dict], should_process: bool = True
    ) -> Dict[int, np.ndarray]:
        if not self.is_ready or not persons:
            return {}

        self._frame_count += 1
        if not should_process and self._frame_count % SEG_EVERY_N != 0:
            return {}

        prober = profiler.get("seg")
        prober.start()

        masks = {}
        for person in persons:
            tid = person["track_id"]
            bbox = person["bbox"]
            mask = self._segment_person(frame, bbox)
            if mask is not None:
                masks[tid] = mask

        prober.stop()
        return masks

    def _segment_person(self, frame, bbox) -> Optional[np.ndarray]:
        try:
            results = self._model(frame, bboxes=[list(bbox)], verbose=False)
            if results and len(results) > 0 and results[0].masks is not None:
                mask = results[0].masks.data[0].cpu().numpy()
                return (mask > 0.5).astype(np.uint8)
        except Exception as e:
            logger.debug(f"[SEG] Segment failed: {e}")
        return None

    def overlay_mask(self, frame, mask, color=(0, 255, 0)) -> np.ndarray:
        if mask is None or mask.shape[:2] != frame.shape[:2]:
            return frame
        overlay = frame.copy()
        overlay[mask > 0] = (
            overlay[mask > 0] * (1 - SEG_ALPHA) + np.array(color) * SEG_ALPHA
        ).astype(np.uint8)
        return overlay
