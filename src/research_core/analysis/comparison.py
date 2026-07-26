"""Direct model-to-model state comparison."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data_models import StateHistory


def compare_state_histories(
    reference: StateHistory,
    comparison: StateHistory,
) -> dict[str, Any]:
    """Compare two histories at identical epochs and in the same frame."""
    if reference.frame != comparison.frame:
        raise ValueError(
            f"Cannot compare frames {reference.frame!r} and {comparison.frame!r}."
        )
    if reference.epoch_utc != comparison.epoch_utc:
        raise ValueError("Cannot compare histories with different initial epochs.")
    if reference.elapsed_seconds.shape != comparison.elapsed_seconds.shape:
        raise ValueError("State histories have different time-grid shapes.")
    if not np.allclose(
        reference.elapsed_seconds,
        comparison.elapsed_seconds,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("State histories are not evaluated at the same elapsed times.")

    position_delta_km = comparison.positions_km - reference.positions_km
    velocity_delta_km_s = comparison.velocities_km_s - reference.velocities_km_s
    position_difference_m = np.linalg.norm(position_delta_km, axis=1) * 1000.0
    velocity_difference_mm_s = np.linalg.norm(velocity_delta_km_s, axis=1) * 1e6

    return {
        "reference_model": reference.model_name,
        "comparison_model": comparison.model_name,
        "frame": reference.frame,
        "epoch_utc": reference.epoch_utc,
        "elapsed_seconds": reference.elapsed_seconds,
        "timestamps_utc": reference.timestamps_utc,
        "position_delta_x_m": position_delta_km[:, 0] * 1000.0,
        "position_delta_y_m": position_delta_km[:, 1] * 1000.0,
        "position_delta_z_m": position_delta_km[:, 2] * 1000.0,
        "position_difference_m": position_difference_m,
        "velocity_delta_x_mm_s": velocity_delta_km_s[:, 0] * 1e6,
        "velocity_delta_y_mm_s": velocity_delta_km_s[:, 1] * 1e6,
        "velocity_delta_z_mm_s": velocity_delta_km_s[:, 2] * 1e6,
        "velocity_difference_mm_s": velocity_difference_mm_s,
    }


def _statistics(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    absolute = np.abs(array)
    max_index = int(np.argmax(absolute))
    return {
        "initial": float(array[0]),
        "final": float(array[-1]),
        "mean": float(np.mean(array)),
        "mean_absolute": float(np.mean(absolute)),
        "rms": float(np.sqrt(np.mean(array * array))),
        "median": float(np.median(array)),
        "standard_deviation": float(np.std(array)),
        "maximum_absolute": float(absolute[max_index]),
        "percentile_95_absolute": float(np.percentile(absolute, 95.0)),
        "index_of_maximum_absolute": max_index,
    }


def create_error_summary(comparison_data: dict[str, Any]) -> dict[str, Any]:
    """Create summary statistics for position and velocity separation."""
    elapsed = np.asarray(comparison_data["elapsed_seconds"], dtype=float)
    position_stats = _statistics(
        np.asarray(comparison_data["position_difference_m"], dtype=float)
    )
    velocity_stats = _statistics(
        np.asarray(comparison_data["velocity_difference_mm_s"], dtype=float)
    )
    position_index = int(position_stats.pop("index_of_maximum_absolute"))
    velocity_index = int(velocity_stats.pop("index_of_maximum_absolute"))
    position_stats["time_of_maximum_seconds"] = float(elapsed[position_index])
    velocity_stats["time_of_maximum_seconds"] = float(elapsed[velocity_index])

    return {
        "reference_model": comparison_data["reference_model"],
        "comparison_model": comparison_data["comparison_model"],
        "frame": comparison_data["frame"],
        "position_difference_m": position_stats,
        "velocity_difference_mm_s": velocity_stats,
    }
