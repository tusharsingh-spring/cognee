"""Smart frame gate: skip redundant frames, forward only those with new information."""

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    GATE_CONTACT_EVENT_FORCE,
    GATE_ENABLED,
    GATE_MAX_SKIP,
    GATE_MIN_INTERVAL,
    GATE_MOTION_THRESHOLD,
    GATE_NEW_PERSON_FRAMES,
    GATE_POSE_DELTA,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class FrameGate:
    def __init__(self) -> None:
        self._enabled = GATE_ENABLED
        self._motion_threshold = GATE_MOTION_THRESHOLD
        self._pose_delta = GATE_POSE_DELTA
        self._min_interval = GATE_MIN_INTERVAL
        self._max_skip = GATE_MAX_SKIP
        self._contact_force = GATE_CONTACT_EVENT_FORCE
        self._new_person_frames = GATE_NEW_PERSON_FRAMES

        self._prev_frame_gray: Optional[np.ndarray] = None
        self._prev_poses: Dict[int, np.ndarray] = {}
        self._prev_track_ids: set = set()
        self._prev_contact_states: Dict[Tuple[int, int], bool] = {}
        self._last_forward_time = 0.0
        self._skipped_since_forward = 0
        self._new_person_counters: Dict[int, int] = {}
        self._total_forwarded = 0
        self._total_skipped = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self._enabled = val

    def decide(
        self,
        frame: np.ndarray,
        current_ids: set,
        motion_boxes: list = None,
        poses: Optional[Dict[int, np.ndarray]] = None,
        contact_states: Optional[Dict[Tuple[int, int], bool]] = None,
    ) -> Tuple[bool, str]:
        if not self._enabled:
            self._total_forwarded += 1
            return True, "gate_disabled"

        now = time.time()
        motion_boxes = motion_boxes or []
        has_mog2_motion = len(motion_boxes) > 0

        if has_mog2_motion:
            self._accept_frame("mog2_motion")
            return True, "mog2_motion"

        if now - self._last_forward_time < self._min_interval:
            pass
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_NEAREST)
            if self._prev_frame_gray is not None:
                diff = cv2.absdiff(gray, self._prev_frame_gray)
                motion_fraction = float(np.count_nonzero(diff > 18) / diff.size)
                if motion_fraction > self._motion_threshold:
                    self._prev_frame_gray = gray
                    self._accept_frame("motion")
                    return True, "motion"
            self._prev_frame_gray = gray

        new_ids = current_ids - self._prev_track_ids
        gone_ids = self._prev_track_ids - current_ids

        if new_ids:
            for tid in new_ids:
                self._new_person_counters[tid] = self._new_person_frames
            for tid in gone_ids:
                self._new_person_counters.pop(tid, None)
            self._prev_track_ids = current_ids
            self._accept_frame("new_person")
            return True, "new_person"

        if gone_ids:
            for tid in gone_ids:
                self._new_person_counters.pop(tid, None)
            self._prev_track_ids = current_ids
            self._accept_frame("person_left")
            return True, "person_left"

        for tid, remaining in list(self._new_person_counters.items()):
            if tid in current_ids and remaining > 0:
                self._new_person_counters[tid] = remaining - 1
                self._accept_frame(f"new_dense_{tid}")
                return True, f"new_dense_{tid}"
            elif remaining <= 0:
                self._new_person_counters.pop(tid, None)

        if poses is not None and self._prev_poses:
            max_pose_shift = 0.0
            for tid, kpts in poses.items():
                if tid in self._prev_poses:
                    prev_kpts = self._prev_poses[tid]
                    if kpts.shape == prev_kpts.shape:
                        valid = (kpts[:, 2] > 0.3) & (prev_kpts[:, 2] > 0.3)
                        if valid.any():
                            shift = float(np.mean(np.linalg.norm(
                                kpts[valid, :2] - prev_kpts[valid, :2], axis=1
                            )))
                            max_pose_shift = max(max_pose_shift, shift)
            if max_pose_shift > self._pose_delta:
                self._prev_poses = {k: v.copy() for k, v in poses.items()} if poses else {}
                self._accept_frame("pose_changed")
                return True, "pose_changed"

        self._prev_poses = {k: v.copy() for k, v in poses.items()} if poses else {}

        if contact_states is not None and self._contact_force:
            if contact_states != self._prev_contact_states:
                self._prev_contact_states = dict(contact_states)
                self._accept_frame("contact_change")
                return True, "contact_change"
        self._prev_contact_states = dict(contact_states) if contact_states else {}

        if self._skipped_since_forward >= self._max_skip:
            self._prev_track_ids = current_ids
            self._accept_frame("max_skip_forced")
            return True, "max_skip_forced"

        self._skipped_since_forward += 1
        self._total_skipped += 1
        return False, "skip"

    def _accept_frame(self, reason: str) -> None:
        self._last_forward_time = time.time()
        self._skipped_since_forward = 0
        self._total_forwarded += 1

    def get_stats(self) -> Dict[str, object]:
        total = self._total_forwarded + self._total_skipped
        skip_rate = (self._total_skipped / total * 100) if total > 0 else 0.0
        return {
            "forwarded": self._total_forwarded,
            "skipped": self._total_skipped,
            "total": total,
            "skip_rate": skip_rate,
        }
