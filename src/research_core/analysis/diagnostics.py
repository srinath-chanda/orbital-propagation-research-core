"""Orbit summaries and conservation diagnostics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..data_models import CartesianState, ClassicalElements, StateHistory


def specific_orbital_energy(
    positions_km: np.ndarray,
    velocities_km_s: np.ndarray,
    gravitational_parameter_km3_s2: float,
) -> np.ndarray:
    positions = np.asarray(positions_km, dtype=float)
    velocities = np.asarray(velocities_km_s, dtype=float)
    radius = np.linalg.norm(positions, axis=1)
    speed_squared = np.sum(velocities * velocities, axis=1)
    return 0.5 * speed_squared - float(gravitational_parameter_km3_s2) / radius


def angular_momentum_vectors(
    positions_km: np.ndarray,
    velocities_km_s: np.ndarray,
) -> np.ndarray:
    return np.cross(
        np.asarray(positions_km, dtype=float),
        np.asarray(velocities_km_s, dtype=float),
    )


def conservation_diagnostics(
    history: StateHistory,
    gravitational_parameter_km3_s2: float,
) -> dict[str, np.ndarray | float | str | int | None]:
    """Calculate two-body energy and angular-momentum drift."""
    energies = specific_orbital_energy(
        history.positions_km,
        history.velocities_km_s,
        gravitational_parameter_km3_s2,
    )
    h_vectors = angular_momentum_vectors(history.positions_km, history.velocities_km_s)
    h_magnitudes = np.linalg.norm(h_vectors, axis=1)
    energy_scale = abs(float(energies[0]))
    h_scale = abs(float(h_magnitudes[0]))
    relative_energy_drift = (energies - energies[0]) / energy_scale
    relative_h_drift = (h_magnitudes - h_magnitudes[0]) / h_scale

    return {
        "model_name": history.model_name,
        "elapsed_seconds": history.elapsed_seconds,
        "timestamps_utc": history.timestamps_utc,
        "specific_energy_km2_s2": energies,
        "relative_energy_drift": relative_energy_drift,
        "angular_momentum_magnitude_km2_s": h_magnitudes,
        "relative_angular_momentum_drift": relative_h_drift,
        "maximum_absolute_relative_energy_drift": float(
            np.max(np.abs(relative_energy_drift))
        ),
        "maximum_absolute_relative_angular_momentum_drift": float(
            np.max(np.abs(relative_h_drift))
        ),
        "runtime_seconds": history.runtime_seconds,
        "function_evaluations": history.function_evaluations,
        "solver_status": history.solver_status,
    }


def create_orbit_summary(
    elements: ClassicalElements,
    initial_state: CartesianState,
    gravitational_parameter_km3_s2: float,
    earth_equatorial_radius_km: float,
) -> dict[str, Any]:
    """Create a human-readable initial orbit summary."""
    mu = float(gravitational_parameter_km3_s2)
    a = float(elements.semi_major_axis_km)
    e = float(elements.eccentricity)
    radius = float(np.linalg.norm(initial_state.position_km))
    speed = float(np.linalg.norm(initial_state.velocity_km_s))
    period_seconds = 2.0 * math.pi * math.sqrt((a**3) / mu)
    perigee_radius = a * (1.0 - e)
    apogee_radius = a * (1.0 + e)
    specific_energy = 0.5 * speed * speed - mu / radius
    h_vector = np.cross(initial_state.position_km, initial_state.velocity_km_s)
    h_magnitude = float(np.linalg.norm(h_vector))
    vis_viva_speed = math.sqrt(mu * (2.0 / radius - 1.0 / a))

    return {
        "epoch_utc": initial_state.epoch_utc,
        "frame": initial_state.frame,
        "element_type": "osculating",
        "semi_major_axis_km": a,
        "eccentricity": e,
        "inclination_deg": float(math.degrees(elements.inclination_rad)),
        "raan_deg": float(math.degrees(elements.raan_rad) % 360.0),
        "argument_of_perigee_deg": float(
            math.degrees(elements.argument_of_perigee_rad) % 360.0
        ),
        "true_anomaly_deg": float(math.degrees(elements.true_anomaly_rad) % 360.0),
        "initial_radius_km": radius,
        "initial_speed_km_s": speed,
        "vis_viva_speed_km_s": vis_viva_speed,
        "vis_viva_absolute_difference_km_s": abs(speed - vis_viva_speed),
        "orbital_period_seconds": period_seconds,
        "orbital_period_minutes": period_seconds / 60.0,
        "perigee_radius_km": perigee_radius,
        "apogee_radius_km": apogee_radius,
        "perigee_altitude_km": perigee_radius - earth_equatorial_radius_km,
        "apogee_altitude_km": apogee_radius - earth_equatorial_radius_km,
        "specific_orbital_energy_km2_s2": specific_energy,
        "angular_momentum_vector_km2_s": h_vector.tolist(),
        "angular_momentum_magnitude_km2_s": h_magnitude,
        "physical_models_in_this_release": [
            "analytical point-mass two-body",
            "numerical point-mass two-body",
        ],
        "excluded_perturbations": [
            "J2 and higher gravity harmonics",
            "atmospheric drag",
            "third-body gravity",
            "solar radiation pressure",
            "manoeuvres",
        ],
    }
