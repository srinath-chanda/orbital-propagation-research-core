"""Analytical Keplerian two-body propagation."""

from __future__ import annotations

import math
from time import perf_counter

import numpy as np

from ..data_models import ClassicalElements, StateHistory
from ..orbital_elements import (
    eccentric_to_true_anomaly,
    elements_to_cartesian,
    solve_kepler_elliptic,
    true_to_eccentric_anomaly,
)
from ..time_utils import timestamps_from_epoch


def propagate_analytical_two_body(
    elements: ClassicalElements,
    gravitational_parameter_km3_s2: float,
    elapsed_seconds: np.ndarray,
    *,
    epoch_utc: str,
    frame: str,
) -> StateHistory:
    """Propagate a fixed Keplerian ellipse using mean-anomaly evolution."""
    times = np.asarray(elapsed_seconds, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("elapsed_seconds must be a non-empty one-dimensional array.")
    if np.any(times < 0.0) or np.any(np.diff(times) < 0.0):
        raise ValueError("elapsed_seconds must be non-negative and non-decreasing.")

    mu = float(gravitational_parameter_km3_s2)
    a = float(elements.semi_major_axis_km)
    e = float(elements.eccentricity)
    mean_motion_rad_s = math.sqrt(mu / (a**3))
    eccentric_anomaly_0 = true_to_eccentric_anomaly(elements.true_anomaly_rad, e)
    mean_anomaly_0 = eccentric_anomaly_0 - e * math.sin(eccentric_anomaly_0)

    positions = np.empty((times.size, 3), dtype=float)
    velocities = np.empty((times.size, 3), dtype=float)

    started = perf_counter()
    for index, elapsed in enumerate(times):
        mean_anomaly = mean_anomaly_0 + mean_motion_rad_s * float(elapsed)
        eccentric_anomaly = solve_kepler_elliptic(mean_anomaly, e)
        true_anomaly = eccentric_to_true_anomaly(eccentric_anomaly, e)
        propagated_elements = ClassicalElements(
            semi_major_axis_km=elements.semi_major_axis_km,
            eccentricity=elements.eccentricity,
            inclination_rad=elements.inclination_rad,
            raan_rad=elements.raan_rad,
            argument_of_perigee_rad=elements.argument_of_perigee_rad,
            true_anomaly_rad=true_anomaly,
        )
        positions[index], velocities[index] = elements_to_cartesian(
            propagated_elements,
            mu,
        )
    runtime = perf_counter() - started

    return StateHistory(
        model_name="analytical_two_body",
        frame=frame,
        epoch_utc=epoch_utc,
        elapsed_seconds=times,
        timestamps_utc=timestamps_from_epoch(epoch_utc, times),
        positions_km=positions,
        velocities_km_s=velocities,
        runtime_seconds=runtime,
        solver_status="analytical_kepler_solution_completed",
        function_evaluations=None,
    )
