"""Runtime performance statistics (FPS and per-stage latency)."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Optional

import numpy as np


class PerformanceStats:
    """Rolling FPS plus latency percentiles per named stage."""

    def __init__(self, window: int = 120) -> None:
        self.window = window
        self._frame_times: Deque[float] = deque(maxlen=window)
        self._latency: dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._all_latency: dict[str, list[float]] = defaultdict(list)
        self.frames = 0

    def tick(self, now: Optional[float] = None) -> None:
        """Call once per displayed/processed frame."""
        self._frame_times.append(time.perf_counter() if now is None else now)
        self.frames += 1

    def record(self, stage: str, ms: float) -> None:
        self._latency[stage].append(ms)
        self._all_latency[stage].append(ms)

    @property
    def fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    def latency(self, stage: str) -> float:
        values = self._latency.get(stage)
        return float(np.mean(values)) if values else 0.0

    def summary(self) -> dict[str, float]:
        """Whole-run mean/p50/p95 latency per stage."""
        out: dict[str, float] = {"frames": float(self.frames)}
        for stage, values in self._all_latency.items():
            arr = np.asarray(values)
            out[f"{stage}_mean_ms"] = float(arr.mean())
            out[f"{stage}_p50_ms"] = float(np.percentile(arr, 50))
            out[f"{stage}_p95_ms"] = float(np.percentile(arr, 95))
        if "total" in self._all_latency and out.get("total_mean_ms", 0) > 0:
            out["throughput_fps"] = 1000.0 / out["total_mean_ms"]
        return out


class StageTimer:
    """Context manager: ``with StageTimer(stats, "reid"): ...``."""

    def __init__(self, stats: PerformanceStats, stage: str) -> None:
        self.stats, self.stage = stats, stage

    def __enter__(self) -> "StageTimer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stats.record(self.stage, (time.perf_counter() - self._t0) * 1000.0)
