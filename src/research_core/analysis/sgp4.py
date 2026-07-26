"""SGP4 common-state comparison and TLE-age analysis."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..analysis.comparison import create_error_summary
from ..data_models import StateHistory
from ..time_utils import parse_utc_timestamp


def tle_age_report(
    tle_epoch_utc: str,
    timestamps_utc: tuple[str, ...],
    elapsed_seconds: np.ndarray,
) -> dict[str, Any]:
    """Return signed age of each output epoch relative to the TLE epoch."""
    epoch = parse_utc_timestamp(tle_epoch_utc)
    ages_seconds = np.asarray(
        [(parse_utc_timestamp(timestamp) - epoch).total_seconds() for timestamp in timestamps_utc],
        dtype=float,
    )
    expected = np.asarray(elapsed_seconds, dtype=float)
    if not np.allclose(ages_seconds, expected, rtol=0.0, atol=1e-6):
        raise ValueError("TLE age and elapsed-time grids are inconsistent.")
    return {
        "tle_epoch_utc": tle_epoch_utc,
        "timestamps_utc": timestamps_utc,
        "elapsed_seconds": expected,
        "tle_age_seconds": ages_seconds,
        "tle_age_hours": ages_seconds / 3600.0,
        "start_age_hours": float(ages_seconds[0] / 3600.0),
        "end_age_hours": float(ages_seconds[-1] / 3600.0),
        "maximum_absolute_age_hours": float(np.max(np.abs(ages_seconds)) / 3600.0),
    }


def create_sgp4_model_summary(
    comparisons: Iterable[dict[str, Any]],
    histories: Iterable[StateHistory],
) -> dict[str, Any]:
    """Create comparison summaries and runtime metadata for all common-state models."""
    history_map = {history.model_name: history for history in histories}
    models: dict[str, Any] = {}
    for comparison in comparisons:
        summary = create_error_summary(comparison)
        model = str(summary["comparison_model"])
        history = history_map[model]
        models[model] = {
            **summary,
            "runtime_seconds": float(history.runtime_seconds),
            "function_evaluations": history.function_evaluations,
            "solver_status": history.solver_status,
            "final_position_difference_km": float(summary["position_difference_m"]["final"] / 1000.0),
            "maximum_position_difference_km": float(summary["position_difference_m"]["maximum_absolute"] / 1000.0),
            "rms_position_difference_km": float(summary["position_difference_m"]["rms"] / 1000.0),
        }
    sgp4_history = history_map["sgp4"]
    return {
        "reference_model": "sgp4",
        "frame": sgp4_history.frame,
        "epoch_utc": sgp4_history.epoch_utc,
        "duration_hours": float(sgp4_history.elapsed_seconds[-1] / 3600.0),
        "state_count": int(sgp4_history.elapsed_seconds.size),
        "sgp4_runtime_seconds": float(sgp4_history.runtime_seconds),
        "models": models,
    }


def initial_state_differences_m(
    reference: StateHistory,
    histories: Iterable[StateHistory],
) -> dict[str, dict[str, float]]:
    """Measure initial common-state position and velocity differences."""
    output: dict[str, dict[str, float]] = {}
    for history in histories:
        output[history.model_name] = {
            "position_difference_m": float(
                np.linalg.norm(history.positions_km[0] - reference.positions_km[0]) * 1000.0
            ),
            "velocity_difference_mm_s": float(
                np.linalg.norm(history.velocities_km_s[0] - reference.velocities_km_s[0]) * 1.0e6
            ),
        }
    return output
