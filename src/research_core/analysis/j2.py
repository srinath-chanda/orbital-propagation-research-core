"""J2-specific orbital-element, secular-rate, RTN, and conservation analysis."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..data_models import ClassicalElements, StateHistory
from ..orbital_elements import cartesian_to_elements
from ..propagators.numerical_j2 import (
    central_gravity_acceleration,
    j2_perturbing_acceleration,
)


def analytical_j2_raan_rate_rad_s(
    elements: ClassicalElements,
    gravitational_parameter_km3_s2: float,
    earth_equatorial_radius_km: float,
    j2: float,
) -> float:
    """Return the first-order secular J2 RAAN rate in rad/s."""
    mu = float(gravitational_parameter_km3_s2)
    radius = float(earth_equatorial_radius_km)
    coefficient = float(j2)
    a = float(elements.semi_major_axis_km)
    e = float(elements.eccentricity)
    inclination = float(elements.inclination_rad)
    if mu <= 0.0 or radius <= 0.0 or coefficient < 0.0 or a <= 0.0:
        raise ValueError("J2 secular-rate inputs must be physically valid.")
    if not (0.0 <= e < 1.0):
        raise ValueError("The J2 secular-rate formula requires 0 <= e < 1.")
    mean_motion = math.sqrt(mu / a**3)
    semi_latus_rectum = a * (1.0 - e * e)
    return (
        -1.5
        * coefficient
        * mean_motion
        * (radius / semi_latus_rectum) ** 2
        * math.cos(inclination)
    )


def create_osculating_element_history(
    history: StateHistory,
    gravitational_parameter_km3_s2: float,
) -> dict[str, Any]:
    """Convert every Cartesian state into osculating classical elements."""
    semi_major_axis: list[float] = []
    eccentricity: list[float] = []
    inclination: list[float] = []
    raan: list[float] = []
    argument_of_perigee: list[float] = []
    true_anomaly: list[float] = []

    for position, velocity in zip(history.positions_km, history.velocities_km_s):
        elements = cartesian_to_elements(
            position,
            velocity,
            gravitational_parameter_km3_s2,
        )
        if elements["raan_deg"] is None:
            raise ValueError(
                "RAAN is undefined for an equatorial orbit; this J2 analysis "
                "requires a non-equatorial benchmark."
            )
        if elements["argument_of_perigee_deg"] is None:
            raise ValueError(
                "Argument of perigee is undefined; use a small non-zero "
                "eccentricity for this classical-element benchmark."
            )
        if elements["true_anomaly_deg"] is None:
            raise ValueError(
                "True anomaly is undefined; use a non-circular benchmark orbit."
            )
        semi_major_axis.append(float(elements["semi_major_axis_km"]))
        eccentricity.append(float(elements["eccentricity"]))
        inclination.append(float(elements["inclination_deg"]))
        raan.append(float(elements["raan_deg"]))
        argument_of_perigee.append(float(elements["argument_of_perigee_deg"]))
        true_anomaly.append(float(elements["true_anomaly_deg"]))

    wrapped_raan_deg = np.asarray(raan, dtype=float)
    unwrapped_raan_rad = np.unwrap(np.radians(wrapped_raan_deg))
    unwrapped_raan_deg = np.degrees(unwrapped_raan_rad)

    return {
        "model_name": history.model_name,
        "frame": history.frame,
        "epoch_utc": history.epoch_utc,
        "elapsed_seconds": history.elapsed_seconds,
        "timestamps_utc": history.timestamps_utc,
        "semi_major_axis_km": np.asarray(semi_major_axis, dtype=float),
        "eccentricity": np.asarray(eccentricity, dtype=float),
        "inclination_deg": np.asarray(inclination, dtype=float),
        "raan_deg": wrapped_raan_deg,
        "raan_unwrapped_deg": np.asarray(unwrapped_raan_deg, dtype=float),
        "argument_of_perigee_deg": np.asarray(
            argument_of_perigee,
            dtype=float,
        ),
        "true_anomaly_deg": np.asarray(true_anomaly, dtype=float),
    }


def fit_raan_rate(
    element_history: dict[str, Any],
    analytical_rate_rad_s: float,
) -> dict[str, float]:
    """Fit a linear secular RAAN trend to an osculating element history."""
    elapsed = np.asarray(element_history["elapsed_seconds"], dtype=float)
    raan_unwrapped_rad = np.radians(
        np.asarray(element_history["raan_unwrapped_deg"], dtype=float)
    )
    if elapsed.size < 2 or float(elapsed[-1] - elapsed[0]) <= 0.0:
        raise ValueError("At least two distinct time points are required for RAAN fitting.")

    fitted_rate_rad_s, fitted_intercept_rad = np.polyfit(
        elapsed,
        raan_unwrapped_rad,
        1,
    )
    analytical = float(analytical_rate_rad_s)
    absolute_difference = abs(float(fitted_rate_rad_s) - analytical)
    relative_difference = (
        absolute_difference / abs(analytical)
        if abs(analytical) > np.finfo(float).eps
        else math.inf
    )
    seconds_per_day = 86400.0
    return {
        "analytical_raan_rate_rad_s": analytical,
        "analytical_raan_rate_deg_day": math.degrees(analytical) * seconds_per_day,
        "fitted_raan_rate_rad_s": float(fitted_rate_rad_s),
        "fitted_raan_rate_deg_day": math.degrees(float(fitted_rate_rad_s))
        * seconds_per_day,
        "fitted_intercept_rad": float(fitted_intercept_rad),
        "absolute_rate_difference_rad_s": absolute_difference,
        "relative_rate_difference": relative_difference,
    }


def compare_in_reference_rtn(
    reference: StateHistory,
    comparison: StateHistory,
) -> dict[str, Any]:
    """Project state differences into the reference trajectory's RTN frame."""
    if reference.frame != comparison.frame:
        raise ValueError("RTN comparison requires the same reference frame.")
    if reference.epoch_utc != comparison.epoch_utc:
        raise ValueError("RTN comparison requires the same initial epoch.")
    if reference.elapsed_seconds.shape != comparison.elapsed_seconds.shape:
        raise ValueError("RTN comparison requires matching time-grid shapes.")
    if not np.allclose(
        reference.elapsed_seconds,
        comparison.elapsed_seconds,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("RTN comparison requires identical output epochs.")

    r_reference = reference.positions_km
    v_reference = reference.velocities_km_s
    delta_r_km = comparison.positions_km - r_reference
    delta_v_km_s = comparison.velocities_km_s - v_reference

    radial_hat = r_reference / np.linalg.norm(r_reference, axis=1)[:, None]
    normal = np.cross(r_reference, v_reference)
    normal_hat = normal / np.linalg.norm(normal, axis=1)[:, None]
    transverse_hat = np.cross(normal_hat, radial_hat)

    def project(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
        return np.einsum("ij,ij->i", vectors, basis)

    return {
        "reference_model": reference.model_name,
        "comparison_model": comparison.model_name,
        "frame": reference.frame,
        "elapsed_seconds": reference.elapsed_seconds,
        "timestamps_utc": reference.timestamps_utc,
        "radial_position_difference_m": project(delta_r_km, radial_hat) * 1000.0,
        "along_track_position_difference_m": project(
            delta_r_km,
            transverse_hat,
        )
        * 1000.0,
        "cross_track_position_difference_m": project(delta_r_km, normal_hat)
        * 1000.0,
        "radial_velocity_difference_mm_s": project(delta_v_km_s, radial_hat)
        * 1.0e6,
        "along_track_velocity_difference_mm_s": project(
            delta_v_km_s,
            transverse_hat,
        )
        * 1.0e6,
        "cross_track_velocity_difference_mm_s": project(delta_v_km_s, normal_hat)
        * 1.0e6,
    }


def j2_specific_potential_km2_s2(
    positions_km: np.ndarray,
    gravitational_parameter_km3_s2: float,
    earth_equatorial_radius_km: float,
    j2: float,
) -> np.ndarray:
    """Return the point-mass plus J2 specific potential in km²/s²."""
    positions = np.asarray(positions_km, dtype=float)
    radius = np.linalg.norm(positions, axis=1)
    z = positions[:, 2]
    mu = float(gravitational_parameter_km3_s2)
    earth_radius = float(earth_equatorial_radius_km)
    coefficient = float(j2)
    legendre_p2 = 0.5 * (3.0 * (z / radius) ** 2 - 1.0)
    return -mu / radius + mu * coefficient * earth_radius**2 / radius**3 * legendre_p2


def j2_conservation_diagnostics(
    history: StateHistory,
    gravitational_parameter_km3_s2: float,
    earth_equatorial_radius_km: float,
    j2: float,
) -> dict[str, Any]:
    """Check conserved total energy and z-angular momentum for static J2 gravity."""
    speed_squared = np.sum(history.velocities_km_s**2, axis=1)
    total_energy = 0.5 * speed_squared + j2_specific_potential_km2_s2(
        history.positions_km,
        gravitational_parameter_km3_s2,
        earth_equatorial_radius_km,
        j2,
    )
    angular_momentum = np.cross(history.positions_km, history.velocities_km_s)
    h_z = angular_momentum[:, 2]
    energy_scale = abs(float(total_energy[0]))
    h_z_scale = abs(float(h_z[0]))
    relative_energy_drift = (total_energy - total_energy[0]) / energy_scale
    relative_h_z_drift = (h_z - h_z[0]) / h_z_scale
    return {
        "model_name": history.model_name,
        "elapsed_seconds": history.elapsed_seconds,
        "timestamps_utc": history.timestamps_utc,
        "total_specific_energy_km2_s2": total_energy,
        "relative_total_energy_drift": relative_energy_drift,
        "angular_momentum_z_km2_s": h_z,
        "relative_angular_momentum_z_drift": relative_h_z_drift,
        "maximum_absolute_relative_total_energy_drift": float(
            np.max(np.abs(relative_energy_drift))
        ),
        "maximum_absolute_relative_angular_momentum_z_drift": float(
            np.max(np.abs(relative_h_z_drift))
        ),
    }


def create_j2_validation_summary(
    *,
    initial_position_km: np.ndarray,
    elements: ClassicalElements,
    element_history: dict[str, Any],
    history: StateHistory,
    gravitational_parameter_km3_s2: float,
    earth_equatorial_radius_km: float,
    j2: float,
) -> dict[str, Any]:
    """Create the measured quantities used by J2 validation checks."""
    analytical_rate = analytical_j2_raan_rate_rad_s(
        elements,
        gravitational_parameter_km3_s2,
        earth_equatorial_radius_km,
        j2,
    )
    raan_fit = fit_raan_rate(element_history, analytical_rate)
    conservation = j2_conservation_diagnostics(
        history,
        gravitational_parameter_km3_s2,
        earth_equatorial_radius_km,
        j2,
    )
    central = central_gravity_acceleration(
        initial_position_km,
        gravitational_parameter_km3_s2,
    )
    perturbing = j2_perturbing_acceleration(
        initial_position_km,
        gravitational_parameter_km3_s2,
        earth_equatorial_radius_km,
        j2,
    )
    acceleration_ratio = float(np.linalg.norm(perturbing) / np.linalg.norm(central))
    final_raan_change = float(
        element_history["raan_unwrapped_deg"][-1]
        - element_history["raan_unwrapped_deg"][0]
    )
    return {
        **raan_fit,
        "duration_hours": float(history.elapsed_seconds[-1]) / 3600.0,
        "initial_j2_acceleration_magnitude_km_s2": float(
            np.linalg.norm(perturbing)
        ),
        "initial_central_acceleration_magnitude_km_s2": float(
            np.linalg.norm(central)
        ),
        "initial_j2_to_central_acceleration_ratio": acceleration_ratio,
        "final_raan_change_deg": final_raan_change,
        "maximum_absolute_relative_total_energy_drift": conservation[
            "maximum_absolute_relative_total_energy_drift"
        ],
        "maximum_absolute_relative_angular_momentum_z_drift": conservation[
            "maximum_absolute_relative_angular_momentum_z_drift"
        ],
        "conservation_diagnostics": conservation,
    }
