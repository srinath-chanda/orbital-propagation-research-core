"""Scientific data structures used by Research Core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ClassicalElements:
    """Osculating classical orbital elements for an elliptical orbit."""

    semi_major_axis_km: float
    eccentricity: float
    inclination_rad: float
    raan_rad: float
    argument_of_perigee_rad: float
    true_anomaly_rad: float

    def as_degrees_dict(self) -> dict[str, float]:
        return {
            "semi_major_axis_km": self.semi_major_axis_km,
            "eccentricity": self.eccentricity,
            "inclination_deg": float(np.degrees(self.inclination_rad)),
            "raan_deg": float(np.degrees(self.raan_rad) % 360.0),
            "argument_of_perigee_deg": float(
                np.degrees(self.argument_of_perigee_rad) % 360.0
            ),
            "true_anomaly_deg": float(np.degrees(self.true_anomaly_rad) % 360.0),
        }


@dataclass(frozen=True)
class CartesianState:
    """One Cartesian position and velocity state."""

    epoch_utc: str
    frame: str
    position_km: np.ndarray
    velocity_km_s: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.position_km, dtype=float)
        velocity = np.asarray(self.velocity_km_s, dtype=float)
        if position.shape != (3,) or velocity.shape != (3,):
            raise ValueError("Cartesian position and velocity must each have shape (3,).")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            raise ValueError("Cartesian state contains non-finite values.")
        object.__setattr__(self, "position_km", position)
        object.__setattr__(self, "velocity_km_s", velocity)

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch_utc": self.epoch_utc,
            "frame": self.frame,
            "position_km": self.position_km.tolist(),
            "velocity_km_s": self.velocity_km_s.tolist(),
            "position_magnitude_km": float(np.linalg.norm(self.position_km)),
            "velocity_magnitude_km_s": float(np.linalg.norm(self.velocity_km_s)),
        }


@dataclass(frozen=True)
class StateHistory:
    """State history produced by one propagation model."""

    model_name: str
    frame: str
    epoch_utc: str
    elapsed_seconds: np.ndarray
    timestamps_utc: tuple[str, ...]
    positions_km: np.ndarray
    velocities_km_s: np.ndarray
    runtime_seconds: float
    solver_status: str
    function_evaluations: int | None = None

    def __post_init__(self) -> None:
        times = np.asarray(self.elapsed_seconds, dtype=float)
        positions = np.asarray(self.positions_km, dtype=float)
        velocities = np.asarray(self.velocities_km_s, dtype=float)
        if times.ndim != 1:
            raise ValueError("elapsed_seconds must be one-dimensional.")
        if positions.shape != (times.size, 3):
            raise ValueError("positions_km must have shape (N, 3).")
        if velocities.shape != (times.size, 3):
            raise ValueError("velocities_km_s must have shape (N, 3).")
        if len(self.timestamps_utc) != times.size:
            raise ValueError("timestamps_utc length must match elapsed_seconds.")
        if times.size == 0:
            raise ValueError("State history cannot be empty.")
        if not np.all(np.isfinite(times)):
            raise ValueError("State-history times contain non-finite values.")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise ValueError("State history contains non-finite values.")
        if np.any(np.diff(times) < 0.0):
            raise ValueError("State-history times must be non-decreasing.")
        object.__setattr__(self, "elapsed_seconds", times)
        object.__setattr__(self, "positions_km", positions)
        object.__setattr__(self, "velocities_km_s", velocities)
