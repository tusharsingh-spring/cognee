"""Performance profiling for ARGUS pipeline layers."""

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from config.settings import PROFILE_EVERY_N_FRAMES, PROFILE_ENABLED
from utils.logger import get_logger

logger = get_logger(__name__)


class LayerProfiler:
    def __init__(self, name: str) -> None:
        self.name = name
        self.timings: List[float] = []
        self._start: Optional[float] = None
        self.frame_count = 0
        self.last_report = 0
        self._enabled = PROFILE_ENABLED

    def start(self) -> None:
        if not self._enabled:
            return
        self._start = time.perf_counter()

    def stop(self) -> None:
        if not self._enabled or self._start is None:
            return
        elapsed = time.perf_counter() - self._start
        self.timings.append(elapsed)
        self._start = None
        self.frame_count += 1

        if (
            self.frame_count - self.last_report >= PROFILE_EVERY_N_FRAMES
            and self.timings
        ):
            avg = sum(self.timings[-PROFILE_EVERY_N_FRAMES:]) / min(
                len(self.timings), PROFILE_EVERY_N_FRAMES
            )
            logger.debug(
                f"[PROFILE] {self.name}: avg={avg*1000:.1f}ms "
                f"(last {min(len(self.timings), PROFILE_EVERY_N_FRAMES)} frames)"
            )
            self.last_report = self.frame_count

    @property
    def avg_ms(self) -> float:
        if not self.timings:
            return 0.0
        return (sum(self.timings) / len(self.timings)) * 1000

    def reset(self) -> None:
        self.timings.clear()
        self.frame_count = 0
        self.last_report = 0


class PipelineProfiler:
    def __init__(self) -> None:
        self.layers: Dict[str, LayerProfiler] = {}

    def get(self, name: str) -> LayerProfiler:
        if name not in self.layers:
            self.layers[name] = LayerProfiler(name)
        return self.layers[name]

    def report(self) -> str:
        parts = []
        for name, lp in self.layers.items():
            parts.append(f"{name}: {lp.avg_ms:.1f}ms")
        return " | ".join(parts)

    def reset_all(self) -> None:
        for lp in self.layers.values():
            lp.reset()


profiler = PipelineProfiler()
