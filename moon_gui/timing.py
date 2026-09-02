"""UI-only elapsed time and ETA; never imported by the numerical runner.

Use a monotonic clock supplied by the caller. Samples span whole completed
subprocesses (startup, integration, frames and checkpoint output). User pauses
and time before a resumed session are intentionally excluded.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class TimingEstimate:
    elapsed_seconds: float
    remaining_seconds: float | None
    sample_count: int
    segment_overdue: bool = False


class RunTiming:
    minimum_samples = 2

    def __init__(self, start_time_myr: float = 0.0, end_time_myr: float = 0.0) -> None:
        self.completed_myr = float(start_time_myr)
        self.end_time_myr = float(end_time_myr)
        self._elapsed_seconds = 0.0
        self._samples: deque[tuple[float, float]] = deque(maxlen=5)
        self._segment_started: float | None = None
        self._segment_target: float | None = None

    def start_segment(self, target_time_myr: float, now: float) -> None:
        if self._segment_started is not None:
            raise RuntimeError("A timing segment is already active")
        if not self.completed_myr < target_time_myr <= self.end_time_myr:
            raise ValueError("Segment target must advance within the run interval")
        self._segment_started = float(now)
        self._segment_target = float(target_time_myr)

    def _current_elapsed(self, now: float) -> float:
        if self._segment_started is None:
            return 0.0
        return max(0.0, float(now) - self._segment_started)

    def finish_segment(self, now: float) -> None:
        if self._segment_started is None or self._segment_target is None:
            raise RuntimeError("No timing segment is active")
        duration = self._current_elapsed(now)
        span = self._segment_target - self.completed_myr
        self._elapsed_seconds += duration
        self._samples.append((span, duration))
        self.completed_myr = self._segment_target
        self._segment_started = None
        self._segment_target = None

    def stop_segment(self, now: float) -> None:
        """Count time spent on an interrupted segment without claiming progress."""
        self._elapsed_seconds += self._current_elapsed(now)
        self._segment_started = None
        self._segment_target = None

    def estimate(self, now: float) -> TimingEstimate:
        current_elapsed = self._current_elapsed(now)
        elapsed = self._elapsed_seconds + current_elapsed
        count = len(self._samples)
        remaining_myr = max(0.0, self.end_time_myr - self.completed_myr)
        if remaining_myr == 0.0:
            return TimingEstimate(elapsed, 0.0, count)
        if count < self.minimum_samples:
            return TimingEstimate(elapsed, None, count)
        seconds_per_myr = sum(seconds for _, seconds in self._samples) / sum(
            span for span, _ in self._samples
        )
        if seconds_per_myr <= 0.0:
            return TimingEstimate(elapsed, None, count)
        if self._segment_target is not None:
            expected_segment_seconds = (
                self._segment_target - self.completed_myr
            ) * seconds_per_myr
            if current_elapsed > expected_segment_seconds:
                # A topology event or final rendering may outlast the estimate.
                # Do not count down to a false zero while the process is busy.
                return TimingEstimate(elapsed, None, count, segment_overdue=True)
        remaining = max(0.0, remaining_myr * seconds_per_myr - current_elapsed)
        return TimingEstimate(elapsed, remaining, count)


def format_duration(seconds: float) -> str:
    """Compact Russian duration, rounded upward to avoid a premature zero."""
    value = max(0, math.ceil(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, seconds_int = divmod(remainder, 60)
    if hours:
        return f"{hours} ч {minutes} мин"
    if minutes:
        return f"{minutes} мин {seconds_int} с"
    return f"{seconds_int} с"
