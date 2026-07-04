import time
from collections import defaultdict
from typing import Dict, Optional

from utils.logger import get_logger

logger = get_logger("Profiler")


class LayerProfiler:
    __slots__ = ("name", "_start", "_total", "_count", "_min", "_max")

    def __init__(self, name: str) -> None:
        self.name = name
        self._start: Optional[float] = None
        self._total: float = 0.0
        self._count: int = 0
        self._min: float = float("inf")
        self._max: float = 0.0

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        if self._start is None:
            return 0.0
        elapsed = (time.perf_counter() - self._start) * 1000
        self._total += elapsed
        self._count += 1
        if elapsed < self._min:
            self._min = elapsed
        if elapsed > self._max:
            self._max = elapsed
        self._start = None
        return elapsed

    @property
    def avg_ms(self) -> float:
        return self._total / self._count if self._count > 0 else 0.0

    @property
    def fps(self) -> float:
        return 1000.0 / self.avg_ms if self.avg_ms > 0 else 0.0


class PipelineProfiler:
    def __init__(self) -> None:
        self._profilers: Dict[str, LayerProfiler] = {}

    def get(self, name: str) -> LayerProfiler:
        if name not in self._profilers:
            self._profilers[name] = LayerProfiler(name)
        return self._profilers[name]

    def report(self) -> None:
        if not self._profilers:
            return
        lines = ["─" * 50, f"{'Layer':<25} {'Avg ms':>8} {'Count':>8} {'FPS':>8}", "─" * 50]
        for name, prof in sorted(self._profilers.items()):
            lines.append(f"{name:<25} {prof.avg_ms:>8.1f} {prof._count:>8} {prof.fps:>8.1f}")
        lines.append("─" * 50)
        logger.info("\n".join(lines))

    def reset(self) -> None:
        self._profilers.clear()


profiler = PipelineProfiler()
