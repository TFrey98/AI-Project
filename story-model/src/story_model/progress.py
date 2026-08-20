"""Low-noise progress and ETA reporting for long deterministic stages."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0.0:
        return "unknown"

    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)

    if hours:
        return f"{hours:d}h {minutes:02d}m {remaining_seconds:02d}s"

    if minutes:
        return f"{minutes:d}m {remaining_seconds:02d}s"

    return f"{remaining_seconds:d}s"


@dataclass
class ProgressReporter:
    """Emit throttled progress lines while still guaranteeing a final line."""

    label: str
    total: int
    unit: str
    interval_seconds: float = 10.0
    emit: Callable[[str], None] = print
    clock: Callable[[], float] = time.monotonic
    started_at: float = field(init=False)
    last_emitted_at: float = field(init=False)

    def __post_init__(self) -> None:
        if self.total < 1:
            raise ValueError("progress total must be positive")

        if self.interval_seconds < 0.0:
            raise ValueError("progress interval cannot be negative")

        self.started_at = self.clock()
        self.last_emitted_at = self.started_at

    def update(
        self,
        completed: int,
        detail: str | None = None,
    ) -> None:
        completed = max(0, min(int(completed), self.total))
        now = self.clock()
        finished = completed >= self.total

        if (
            not finished
            and now - self.last_emitted_at < self.interval_seconds
        ):
            return

        elapsed = max(0.0, now - self.started_at)
        rate = completed / elapsed if elapsed > 0.0 else 0.0
        remaining = self.total - completed
        eta = remaining / rate if rate > 0.0 else None
        percent = 100.0 * completed / self.total
        message = (
            f"{self.label}: {completed:,}/{self.total:,} "
            f"{self.unit} ({percent:.1f}%), "
            f"elapsed {format_duration(elapsed)}, "
            f"ETA {format_duration(eta)}"
        )

        if rate > 0.0:
            message += f", {rate:,.0f} {self.unit}/s"

        if detail:
            message += f", {detail}"

        self.emit(message)
        self.last_emitted_at = now
