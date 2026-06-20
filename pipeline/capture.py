"""Camera capture with MOG2 motion detection to drop idle frames.
Video-file mode in dense captioning bypasses motion gating entirely."""

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_URL,
    CAMERA_WIDTH,
    MOTION_FRAME_SKIP,
    MOTION_HISTORY,
    MOTION_LEARNING_RATE,
    MOTION_MIN_AREA,
    MOTION_THRESHOLD,
    VIDEO_FILE,
    DENSE_FRAME_INTERVAL,
)
from utils.logger import get_logger
from utils.profiler import profiler

logger = get_logger(__name__)


class MotionCapture:
    def __init__(self, source: Optional[str] = None) -> None:
        _source = source or VIDEO_FILE or CAMERA_URL or CAMERA_INDEX
        self._is_video_file = bool(source or VIDEO_FILE)
        self._source_name = _source
        self.cap = cv2.VideoCapture(_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        logger.info(f"[CAPTURE] Source: {_source} (video_file={self._is_video_file})")

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=MOTION_HISTORY,
            varThreshold=MOTION_THRESHOLD,
            detectShadows=False,
        )

        self.frame_count = 0
        self.skip_count = 0
        self.last_frame: Optional[np.ndarray] = None
        self._fps_times: List[float] = []
        self._running = False

    def _get_fps(self) -> float:
        now = time.perf_counter()
        self._fps_times.append(now)
        self._fps_times = [t for t in self._fps_times if now - t < 5.0]
        return len(self._fps_times) / 5.0 if self._fps_times else 0.0

    def read(self) -> Tuple[bool, Optional[np.ndarray], List[Tuple[int, int, int, int]]]:
        ret, frame = self.cap.read()
        if not ret or frame is None:
            if self._is_video_file:
                return False, None, []
            if not ret or frame is None:
                return False, None, []

        self.frame_count += 1
        self.last_frame = frame
        motion_boxes: List[Tuple[int, int, int, int]] = []

        prober = profiler.get("capture")
        prober.start()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        fg_mask = self.bg_subtractor.apply(gray, learningRate=MOTION_LEARNING_RATE)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        has_motion = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > MOTION_MIN_AREA:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                motion_boxes.append((x, y, x + w_box, y + h_box))
                has_motion = True

        if not has_motion:
            self.skip_count += 1
        else:
            self.skip_count = 0

        fps = self._get_fps()
        prober.stop()
        return True, frame, motion_boxes

    def release(self) -> None:
        self._running = False
        if self.cap.isOpened():
            self.cap.release()

    @property
    def is_opened(self) -> bool:
        return self.cap.isOpened()
