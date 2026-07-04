"""YOLOv8n ONNX gating layer — runs every frame, decides whether to trigger full pipeline.

Lightweight (~6MB model, ~10-15ms on CPU) person detector that gates
the heavier perception pipeline. Only passes frames with real events.
"""

from typing import List, Optional, Set, Tuple

import cv2
import numpy as np

from config.settings import (
    GATE_CONFIDENCE,
    GATE_ENABLED,
    GATE_IMG_SIZE,
    GATE_MODEL,
    GATE_NEW_PERSON_FRAMES,
    GATE_PERSON_CLASS,
)
from utils.logger import get_logger
from utils.profiler import profiler

logger = get_logger(__name__)


class YOLOv8nGate:
    """Fast person detection gate using Ultralytics YOLOv8n.

    Runs inference every frame. Triggers full pipeline on:
    - New person appears
    - Person disappears
    - Significant bbox change
    - Periodic forced trigger
    """

    def __init__(self) -> None:
        self.enabled = GATE_ENABLED
        self._model = None
        self._ready = False
        self._frame_count = 0
        self._forced_interval = 15  # force trigger every N frames

        self._prev_person_ids: Set[int] = set()
        self._prev_bboxes: dict = {}
        self._new_person_counter: dict = {}
        self._lost_person_counter: dict = {}
        self._consecutive_no_persons = 0

        if self.enabled:
            self._load()

    def _load(self) -> None:
        try:
            from ultralytics import YOLO

            self._model = YOLO(GATE_MODEL)
            self._ready = True
            logger.info(f"[GATE] YOLOv8n loaded (imgsz={GATE_IMG_SIZE})")
        except Exception as e:
            logger.warning(f"[GATE] Model load failed: {e}")
            self._load_mog2_fallback()

    def _load_mog2_fallback(self) -> None:
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25, detectShadows=False
        )
        self._ready = True
        logger.info("[GATE] Using MOG2 motion fallback")

    @property
    def is_ready(self) -> bool:
        return self._ready and self.enabled

    def should_process(self, frame: np.ndarray) -> Tuple[bool, str]:
        """Run gating check. Returns (should_process, reason)."""
        if not self.enabled:
            return True, "gate_disabled"

        prober = profiler.get("gate")
        prober.start()

        self._frame_count += 1

        try:
            if hasattr(self, "_mog2"):
                trigger, reason = self._mog2_check(frame)
            else:
                trigger, reason = self._yolo_check(frame)
        except Exception as e:
            logger.debug(f"[GATE] Check failed: {e}")
            trigger, reason = True, "gate_error"

        prober.stop()
        return trigger, reason

    def _yolo_check(self, frame: np.ndarray) -> Tuple[bool, str]:
        if self._model is None:
            return True, "no_model"

        results = self._model(frame, imgsz=GATE_IMG_SIZE, verbose=False, device="cpu")
        if not results or len(results) == 0:
            return self._handle_empty(), "no_detections"

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return self._handle_empty(), "no_detections"

        current_ids: Set[int] = set()
        current_bboxes = {}
        person_detected = False

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == GATE_PERSON_CLASS and conf >= GATE_CONFIDENCE:
                person_detected = True
                xyxy = box.xyxy[0].cpu().numpy()
                tid = box.id[0] if box.id is not None else -1
                tid = int(tid)
                current_ids.add(tid)
                current_bboxes[tid] = xyxy

        if not person_detected:
            return self._handle_empty(), "no_persons"

        trigger = False
        reason = "stable"

        new_ids = current_ids - self._prev_person_ids
        lost_ids = self._prev_person_ids - current_ids

        for tid in new_ids:
            self._new_person_counter[tid] = self._new_person_counter.get(tid, 0) + 1
            if self._new_person_counter[tid] >= 3:
                trigger = True
                reason = "new_person"
                break

        if not trigger:
            for tid in list(self._lost_person_counter.keys()):
                if tid not in current_ids:
                    self._lost_person_counter[tid] += 1
                    if self._lost_person_counter[tid] >= 3:
                        trigger = True
                        reason = "person_left"
                        break

        if not trigger:
            for tid in current_ids:
                if tid in self._prev_bboxes:
                    prev_box = self._prev_bboxes[tid]
                    curr_box = current_bboxes.get(tid)
                    if curr_box is not None:
                        movement = self._bbox_movement(prev_box, curr_box)
                        if movement > 50:
                            trigger = True
                            reason = "significant_movement"
                            break

        if not trigger and self._frame_count % self._forced_interval == 0:
            trigger = True
            reason = "forced_periodic"

        self._prev_person_ids = current_ids
        self._prev_bboxes = current_bboxes
        self._consecutive_no_persons = 0

        return trigger, reason

    def _handle_empty(self) -> bool:
        self._consecutive_no_persons += 1
        self._new_person_counter.clear()
        self._lost_person_counter.clear()
        if self._consecutive_no_persons > 30:
            self._prev_person_ids.clear()
            self._prev_bboxes.clear()
        return False

    def _mog2_check(self, frame: np.ndarray) -> Tuple[bool, str]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        fgmask = self._mog2.apply(gray)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        significant = [c for c in contours if cv2.contourArea(c) > 800]
        if significant:
            return True, "motion_detected"

        if self._frame_count % self._forced_interval == 0:
            return True, "forced_periodic"

        return False, "no_motion"

    def _bbox_movement(self, b1, b2) -> float:
        c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
        c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
        return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    def get_ids(self) -> Set[int]:
        return self._prev_person_ids.copy()

    def reset(self) -> None:
        self._prev_person_ids.clear()
        self._prev_bboxes.clear()
        self._new_person_counter.clear()
        self._lost_person_counter.clear()
        self._frame_count = 0
