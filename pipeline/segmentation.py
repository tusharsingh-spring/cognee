"""Segmentation using SAM2 / MobileSAM for per-person pixel-accurate masks."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    CV_DEVICE,
    SEG_ALPHA,
    SEG_DEVICE,
    SEG_ENABLED,
    SEG_EVERY_N,
    SEG_MODEL,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class Segmenter:
    def __init__(self) -> None:
        self._enabled = SEG_ENABLED
        self._device = CV_DEVICE
        self._alpha = SEG_ALPHA
        self._model = None
        self._predictor = None
        self._loaded = False
        self._frame_counter = 0
        self._last_masks: Dict[int, np.ndarray] = {}

        if self._enabled:
            self._try_load()

    def _try_load(self) -> None:
        try:
            from ultralytics import SAM

            logger.info("[Seg] Loading MobileSAM via Ultralytics...")
            self._model = SAM("mobile_sam.pt")
            self._loaded = True
            logger.info("[Seg] MobileSAM loaded via Ultralytics")
            return
        except Exception as e:
            logger.debug(f"[Seg] Ultralytics SAM failed: {e}")

        try:
            try:
                from segment_anything import sam_model_registry, SamPredictor
            except ImportError:
                from segment_anything_hq import sam_model_registry, SamPredictor

            logger.info(f"[Seg] Loading {SEG_MODEL} via segment-anything...")
            checkpoint = f"{SEG_MODEL}.pth"
            self._model = sam_model_registry["vit_h"](checkpoint=checkpoint)
            if CV_DEVICE == "cuda":
                self._model = self._model.cuda()
            self._predictor = SamPredictor(self._model)
            self._loaded = True
            logger.info(f"[Seg] SAM predictor ready")
        except Exception as e:
            self._load_error = str(e)
            self._loaded = False
            logger.warning(f"[Seg] Model load failed: {e}")
            self._try_fallback()

    def _try_fallback(self) -> None:
        self._loaded = True
        logger.info("[Seg] Using grabCut fallback (CPU, per-person bbox)")

    @property
    def is_ready(self) -> bool:
        return self._enabled and self._loaded

    @property
    def last_masks(self) -> Dict[int, np.ndarray]:
        return self._last_masks

    def segment(
        self, frame: np.ndarray, persons: List[Dict], force: bool = False
    ) -> Dict[int, np.ndarray]:
        if not self.is_ready:
            return {}

        self._frame_counter += 1
        if not force and self._frame_counter % SEG_EVERY_N != 0:
            return self._last_masks

        masks: Dict[int, np.ndarray] = {}
        h, w = frame.shape[:2]

        try:
            if self._predictor is not None:
                self._predictor.set_image(frame)
                for person in persons:
                    bbox = person["bbox"]
                    x1, y1, x2, y2 = bbox
                    input_box = np.array([[x1, y1, x2, y2]])
                    mask_out, _, _ = self._predictor.predict(
                        box=input_box, multimask_output=False
                    )
                    masks[person["track_id"]] = mask_out[0].astype(np.uint8)
            elif self._model is not None and hasattr(self._model, "predict"):
                for person in persons:
                    bbox = person["bbox"]
                    x1, y1, x2, y2 = bbox
                    results = self._model(frame, bboxes=[bbox])
                    if results and hasattr(results[0], "masks") and results[0].masks is not None:
                        mask = results[0].masks.data[0].cpu().numpy()
                        mask = (mask > 0.5).astype(np.uint8)
                        masks[person["track_id"]] = mask
            else:
                masks = self._segment_grabcut(frame, persons)
        except Exception as e:
            logger.debug(f"[Seg] Segmentation failed: {e}")
            masks = self._segment_grabcut(frame, persons)

        self._last_masks = masks
        return masks

    def _segment_grabcut(
        self, frame: np.ndarray, persons: List[Dict]
    ) -> Dict[int, np.ndarray]:
        masks: Dict[int, np.ndarray] = {}
        h, w = frame.shape[:2]

        for person in persons:
            tid = person["track_id"]
            x1, y1, x2, y2 = person["bbox"]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))

            if x2 <= x1 + 5 or y2 <= y1 + 5:
                continue

            roi = frame[y1:y2, x1:x2]
            full_mask = np.zeros((h, w), dtype=np.uint8)
            rect = (0, 0, x2 - x1, y2 - y1)

            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            bgd = np.zeros((1, 65), dtype=np.float64)
            fgd = np.zeros((1, 65), dtype=np.float64)

            try:
                cv2.grabCut(roi, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
            except Exception:
                center_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                ch, cw = roi.shape[:2]
                cv2.ellipse(center_mask, (cw // 2, ch // 2), (cw // 3, ch // 3), 0, 0, 360, 1, -1)
                mask[center_mask > 0] = cv2.GC_FGD
                try:
                    cv2.grabCut(roi, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
                except Exception:
                    mask = (center_mask > 0).astype(np.uint8) * 3

            fore = np.where((mask == 1) | (mask == 3), 1, 0).astype(np.uint8)
            full_mask[y1:y2, x1:x2] = fore
            masks[tid] = full_mask

        return masks

    def get_mask_overlay(
        self, frame: np.ndarray, masks: Dict[int, np.ndarray], colors: Optional[Dict[int, Tuple]] = None
    ) -> np.ndarray:
        overlay = frame.copy()
        default_colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
        ]
        for i, (tid, mask) in enumerate(masks.items()):
            if mask.shape[:2] != frame.shape[:2]:
                continue
            color = colors.get(tid, default_colors[i % len(default_colors)]) if colors else default_colors[i % len(default_colors)]
            colored = np.zeros_like(frame)
            colored[mask > 0] = color
            overlay = cv2.addWeighted(overlay, 1.0, colored, self._alpha, 0)
        return overlay

    def compute_mask_overlap(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        if mask_a is None or mask_b is None:
            return 0.0
        inter = np.sum(mask_a & mask_b)
        return float(inter)
