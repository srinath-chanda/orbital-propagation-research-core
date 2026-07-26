"""Numerical propagation with point-mass gravity, J2, and simplified drag."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp

from ..data_models import CartesianState, StateHistory
from ..time_utils import timestamps_from_epoch
from .numerical_j2 import total_j2_acceleration


def exponential_atmospheric_density_kg_m3(
    position_km: np.ndarray,
    earth_equatorial_radius_km: float,
    reference_altitude_km: float,
    reference_density_kg_m3: float,
    scale_height_km: float,
) -> float:
    """Return density from a single-scale-height exponential atmosphere.

    The model is intentionally simple and is intended only for sensitivity
    analysis. It does not include latitude, local time, season, composition,
    solar activity, geomagnetic activity, or atmospheric winds.
    """
    position = np.asarray(position_km, dtype=float)
    if position.shape != (3,):
        raise ValueError("Position must have shape (3,).")
    if not np.all(np.isfinite(position)):
        raise ValueError("Position contains non-finite values.")

    earth_radius = float(earth_equatorial_radius_km)
    reference_altitude = float(reference_altitude_km)
    reference_density = float(reference_density_kg_m3)
    scale_height = float(scale_height_km)
    if earth_radius <= 0.0 or not np.isfinite(earth_radius):
        raise ValueError("Earth radius must be positive and finite.")
    if reference_altitude < 0.0 or not np.isfinite(reference_altitude):
        raise ValueError("Reference altitude must be non-negative and finite.")
    if reference_density < 0.0 or not np.isfinite(reference_density):
        raise ValueError("Reference density must be non-negative and finite.")
    if scale_height <= 0.0 or not np.isfinite(scale_height):
        raise ValueError("Scale height must be positive and finite.")

    altitude_km = float(np.linalg.norm(position) - earth_radius)
    exponent = -(altitude_km - reference_altitude) / scale_height
    exponent = float(np.clip(exponent, -700.0, 700.0))
    density = reference_density * float(np.exp(exponent))
    if not np.isfinite(density):
        raise FloatingPointError("Atmospheric-density calculation became non-finite.")
    return density


def atmospheric_velocity_km_s(
    position_km: np.ndarray,
    earth_rotation_rate_rad_s: float,
    *,
    co_rotating_atmosphere: bool,
) -> np.ndarray:
    """Return the simplified atmospheric inertial velocity in km/s."""
    position = np.asarray(position_km, dtype=float)
    if position.shape != (3,):
        raise ValueError("Position must have shape (3,).")
    omega = float(earth_rotation_rate_rad_s)
    if omega < 0.0 or not np.isfinite(omega):
        raise ValueError("Earth rotation rate must be non-negative and finite.")
    if not co_rotating_atmosphere:
        return np.zeros(3, dtype=float)
    return np.cross(np.array([0.0, 0.0, omega], dtype=float), position)


def atmospheric_relative_velocity_km_s(
    position_km: np.ndarray,
    inertial_velocity_km_s: np.ndarray,
    earth_rotation_rate_rad_s: float,
    *,
    co_rotating_atmosphere: bool,
) -> np.ndarray:
    """Return spacecraft velocity relative to the simplified atmosphere."""
    velocity = np.asarray(inertial_velocity_km_s, dtype=float)
    if velocity.shape != (3,):
        raise ValueError("Velocity must have shape (3,).")
    return velocity - atmospheric_velocity_km_s(
        position_km,
        earth_rotation_rate_rad_s,
        co_rotating_atmosphere=co_rotating_atmosphere,
    )


def drag_acceleration_km_s2(
    position_km: np.ndarray,
    inertial_velocity_km_s: np.ndarray,
    *,
    earth_equatorial_radius_km: float,
    earth_rotation_rate_rad_s: float,
    mass_kg: float,
    cross_sectional_area_m2: float,
    drag_coefficient: float,
    reference_altitude_km: float,
    reference_density_kg_m3: float,
    scale_height_km: float,
    co_rotating_atmosphere: bool,
) -> np.ndarray:
    """Return simplified aerodynamic drag acceleration in km/s²."""
    mass = float(mass_kg)
    area = float(cross_sectional_area_m2)
    coefficient = float(drag_coefficient)
    if mass <= 0.0 or not np.isfinite(mass):
        raise ValueError("Spacecraft mass must be positive and finite.")
    if area < 0.0 or not np.isfinite(area):
        raise ValueError("Cross-sectional area must be non-negative and finite.")
    if coefficient < 0.0 or not np.isfinite(coefficient):
        raise ValueError("Drag coefficient must be non-negative and finite.")

    density = exponential_atmospheric_density_kg_m3(
        position_km,
        earth_equatorial_radius_km,
        reference_altitude_km,
        reference_density_kg_m3,
        scale_height_km,
    )
    relative_velocity_km_s = atmospheric_relative_velocity_km_s(
        position_km,
        inertial_velocity_km_s,
        earth_rotation_rate_rad_s,
        co_rotating_atmosphere=co_rotating_atmosphere,
    )
    relative_speed_km_s = float(np.linalg.norm(relative_velocity_km_s))
    if relative_speed_km_s == 0.0 or density == 0.0 or area == 0.0 or coefficient == 0.0:
        return np.zeros(3, dtype=float)

    relative_velocity_m_s = relative_velocity_km_s * 1000.0
    relative_speed_m_s = relative_speed_km_s * 1000.0
    acceleration_m_s2 = (
        -0.5
        * density
        * coefficient
        * area
        / mass
        * relative_speed_m_s
        * relative_velocity_m_s
    )
    return acceleration_m_s2 / 1000.0


def j2_drag_derivative(
    _elapsed_seconds: float,
    state_vector: np.ndarray,
    gravitational_parameter_km3_s2: float,
    earth_equatorial_radius_km: float,
    j2: float,
    earth_rotation_rate_rad_s: float,
    mass_kg: float,
    cross_sectional_area_m2: float,
    drag_coefficient: float,
    reference_altitude_km: float,
    reference_density_kg_m3: float,
    scale_height_km: float,
    co_rotating_atmosphere: bool,
) -> np.ndarray:
    """Return the six-dimensional point-mass + J2 + drag derivative."""
    state = np.asarray(state_vector, dtype=float)
    if state.shape != (6,):
        raise ValueError("State vector must have shape (6,).")
    position = state[:3]
    velocity = state[3:]
    derivative = np.empty(6, dtype=float)
    derivative[:3] = velocity
    derivative[3:] = total_j2_acceleration(
        position,
        gravitational_parameter_km3_s2,
        earth_equatorial_radius_km,
        j2,
    ) + drag_acceleration_km_s2(
        position,
        velocity,
        earth_equatorial_radius_km=earth_equatorial_radius_km,
        earth_rotation_rate_rad_s=earth_rotation_rate_rad_s,
        mass_kg=mass_kg,
        cross_sectional_area_m2=cross_sectional_area_m2,
        drag_coefficient=drag_coefficient,
        reference_altitude_km=reference_altitude_km,
        reference_density_kg_m3=reference_density_kg_m3,
        scale_height_km=scale_height_km,
        co_rotating_atmosphere=co_rotating_atmosphere,
    )
    return derivative


def propagate_numerical_j2_drag(
    initial_state: CartesianState,
    gravitational_parameter_km3_s2: float,
    earth_equatorial_radius_km: float,
    j2: float,
    earth_rotation_rate_rad_s: float,
    elapsed_seconds: np.ndarray,
    *,
    mass_kg: float,
    cross_sectional_area_m2: float,
    drag_coefficient: float,
    reference_altitude_km: float,
    reference_density_kg_m3: float,
    scale_height_km: float,
    co_rotating_atmosphere: bool,
    method: str,
    relative_tolerance: float,
    absolute_tolerance: float,
    maximum_step_seconds: float,
) -> StateHistory:
    """Propagate a state with point-mass gravity, J2, and simplified drag."""
    times = np.asarray(elapsed_seconds, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("elapsed_seconds must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError("elapsed_seconds must contain finite non-negative values.")
    if times[0] != 0.0:
        raise ValueError("The numerical J2+drag time grid must begin at zero.")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("elapsed_seconds must be non-decreasing.")

    final_time = float(times[-1])
    initial_vector = np.concatenate(
        (initial_state.position_km, initial_state.velocity_km_s)
    )

    if final_time == 0.0:
        states = initial_vector[:, np.newaxis]
        runtime = 0.0
        function_evaluations = 0
        message = "No integration required for zero-duration history."
    else:
        started = perf_counter()
        solution = solve_ivp(
            fun=j2_drag_derivative,
            t_span=(0.0, final_time),
            y0=initial_vector,
            method=method,
            t_eval=times,
            rtol=float(relative_tolerance),
            atol=float(absolute_tolerance),
            max_step=float(maximum_step_seconds),
            args=(
                float(gravitational_parameter_km3_s2),
                float(earth_equatorial_radius_km),
                float(j2),
                float(earth_rotation_rate_rad_s),
                float(mass_kg),
                float(cross_sectional_area_m2),
                float(drag_coefficient),
                float(reference_altitude_km),
                float(reference_density_kg_m3),
                float(scale_height_km),
                bool(co_rotating_atmosphere),
            ),
        )
        runtime = perf_counter() - started
        if not solution.success:
            raise RuntimeError(
                f"Numerical J2+drag integration failed: {solution.message}"
            )
        if solution.y.shape != (6, times.size):
            raise RuntimeError(
                "Numerical J2+drag integrator returned an unexpected state-history shape."
            )
        states = solution.y
        function_evaluations = int(solution.nfev)
        message = str(solution.message)

    positions = states[:3].T
    velocities = states[3:].T
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise FloatingPointError("J2+drag propagation produced non-finite states.")

    return StateHistory(
        model_name="numerical_j2_drag",
        frame=initial_state.frame,
        epoch_utc=initial_state.epoch_utc,
        elapsed_seconds=times,
        timestamps_utc=timestamps_from_epoch(initial_state.epoch_utc, times),
        positions_km=positions,
        velocities_km_s=velocities,
        runtime_seconds=runtime,
        solver_status=message,
        function_evaluations=function_evaluations,
    )
