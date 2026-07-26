"""UTC and experiment-time utilities."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np


def parse_utc_timestamp(value: str) -> datetime:
    """Parse an ISO 8601 UTC timestamp ending in Z."""
    if not value.endswith("Z"):
        raise ValueError("UTC timestamps must end in 'Z'.")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_utc_timestamp(value: datetime) -> str:
    """Format a timezone-aware datetime as ISO 8601 UTC."""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_time_grid(duration_hours: float, output_step_seconds: float) -> np.ndarray:
    """Create a grid starting at zero and including the exact final time."""
    duration_seconds = float(duration_hours) * 3600.0
    step = float(output_step_seconds)
    if not np.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError("duration_hours must be positive and finite.")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("output_step_seconds must be positive and finite.")

    count = int(np.floor(duration_seconds / step))
    grid = np.arange(count + 1, dtype=float) * step
    if grid[-1] < duration_seconds and not np.isclose(grid[-1], duration_seconds):
        grid = np.append(grid, duration_seconds)
    else:
        grid[-1] = duration_seconds
    return grid


def timestamps_from_epoch(epoch_utc: str, elapsed_seconds: np.ndarray) -> tuple[str, ...]:
    """Create UTC timestamps corresponding to elapsed seconds."""
    epoch = parse_utc_timestamp(epoch_utc)
    return tuple(
        format_utc_timestamp(epoch + timedelta(seconds=float(seconds)))
        for seconds in elapsed_seconds
    )
