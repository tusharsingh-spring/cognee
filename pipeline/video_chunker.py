"""Video chunker: efficient frame ingestion with smart key-frame selection.
Decodes video in chunks, runs MOG2 pre-scan to identify interesting frames,
only forwards frames with significant motion/change to the pipeline."""

import time
from collections import deque
from typing import Callable, Generator, List, Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class VideoChunker:
    def __init__(
        self,
        chunk_size: int = 30,
        motion_threshold: float = 0.01,
        keyframe_interval: float = 0.5,
        resize_dims: Tuple[int, int] = (160, 120),
    ):
        self._chunk_size = chunk_size
        self._motion_threshold = motion_threshold
        self._keyframe_interval = keyframe_interval
        self._resize_dims = resize_dims
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=16, detectShadows=False
        )
        self._prev_gray: Optional[np.ndarray] = None
        self._last_keyframe_time = 0.0
        self._frame_idx = 0
        self._total_read = 0
        self._total_forwarded = 0
        self._total_skipped = 0

    def process_video(
        self,
        video_path: str,
        callback: Callable[[int, np.ndarray, List[Tuple]], None],
        max_frames: int = 0,
    ) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"[Chunker] Cannot open: {video_path}")
            return {"error": "cannot open"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_time = time.time()

        logger.info(f"[Chunker] Processing {video_path}: {total_frames}frames @ {fps:.0f}fps")

        buf: deque = deque()
        buf_motion: deque = deque()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self._total_read += 1
            self._frame_idx += 1

            if max_frames and self._total_read > max_frames:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, self._resize_dims, interpolation=cv2.INTER_NEAREST)

            fg_mask = self._bg_subtractor.apply(gray_small, learningRate=0.005)
            motion_score = float(np.count_nonzero(fg_mask > 200) / fg_mask.size)

            frame_diff = 0.0
            if self._prev_gray is not None:
                diff = cv2.absdiff(gray_small, self._prev_gray)
                frame_diff = float(np.count_nonzero(diff > 20) / diff.size)

            self._prev_gray = gray_small

            motion_regions: List[Tuple] = []
            if motion_score > 0.001:
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) > 20:
                        x, y, wb, hb = cv2.boundingRect(cnt)
                        sx = self._resize_dims[0] / gray.shape[1]
                        sy = self._resize_dims[1] / gray.shape[0]
                        motion_regions.append((
                            int(x / sx), int(y / sy),
                            int((x + wb) / sx), int((y + hb) / sy),
                        ))

            buf.append((self._frame_idx, frame, motion_regions))
            buf_motion.append((self._frame_idx, motion_score, frame_diff, len(motion_regions)))

            if len(buf) >= self._chunk_size:
                self._flush_chunk(buf, buf_motion, callback, fps)
                buf.clear()
                buf_motion.clear()

        if buf:
            self._flush_chunk(buf, buf_motion, callback, fps)

        cap.release()
        elapsed = time.time() - start_time
        skip_pct = (self._total_skipped / max(1, self._total_read)) * 100

        logger.info(
            f"[Chunker] Done: {self._total_read} read, "
            f"{self._total_forwarded} forwarded ({skip_pct:.0f}% skipped) "
            f"in {elapsed:.1f}s"
        )

        return {
            "total_read": self._total_read,
            "total_forwarded": self._total_forwarded,
            "total_skipped": self._total_skipped,
            "skip_rate": round(skip_pct, 1),
            "elapsed": round(elapsed, 1),
            "fps": round(total_frames / elapsed, 1) if elapsed > 0 else 0,
        }

    def _flush_chunk(
        self,
        buf: deque,
        buf_motion: deque,
        callback: Callable,
        fps: float,
    ) -> None:
        chunk_size = len(buf)
        if chunk_size == 0:
            return

        motions = [m[1] for m in buf_motion]
        diffs = [m[2] for m in buf_motion]
        regions = [m[3] for m in buf_motion]

        mean_motion = float(np.mean(motions)) if motions else 0.0
        max_motion = float(np.max(motions)) if motions else 0.0
        max_diff = float(np.max(diffs)) if diffs else 0.0

        key_indices = set()

        if max_motion > self._motion_threshold or max_diff > 0.03:
            for i in range(chunk_size):
                fi, motion_score, frame_diff, n_regions = buf_motion[i]
                score = motion_score * 0.6 + frame_diff * 0.3 + (n_regions > 0) * 0.1
                if score > self._motion_threshold * 0.4:
                    key_indices.add(i)

        if not key_indices:
            key_indices.add(0)
            if chunk_size > 5:
                key_indices.add(chunk_size // 2)
            if chunk_size > 10:
                key_indices.add(chunk_size - 1)

        for i in sorted(key_indices):
            fi, frame, motion_regions = buf[i]
            keyframe_interval_sec = 1.0 / max(fps, 1.0) * self._keyframe_interval * fps
            if time.time() - self._last_keyframe_time < self._keyframe_interval:
                continue

            self._total_forwarded += 1
            self._last_keyframe_time = time.time()
            callback(fi, frame, motion_regions)

        self._total_skipped += chunk_size - len(key_indices)
