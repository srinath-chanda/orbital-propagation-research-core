"""Numerical propagation with point-mass gravity and the J2 perturbation."""

from __future__ import annotations

from functools import partial
from time import perf_counter
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from ..data_models import CartesianState, StateHistory
from ..earth_orientation import (
    SUPPORTED_POLE_MODELS,
    earth_pole_unit_vector,
    iau_1976_1980_true_pole_unit_vector,
)
from ..gmat_eop import GmatEopDataset, gmat_validated_eop_pole_unit_vector
from ..time_utils import timestamps_from_epoch


def central_gravity_acceleration(
    position_km: np.ndarray,
    gravitational_parameter_km3_s2: float,
) -> np.ndarray:
    """Return point-mass gravitational acceleration in km/s²."""
    position = np.asarray(position_km, dtype=float)
    if position.shape != (3,):
        raise ValueError("Position must have shape (3,).")
    radius = float(np.linalg.norm(position))
    if radius <= 0.0 or not np.isfinite(radius):
        raise ValueError("Position magnitude must be positive and finite.")
    mu = float(gravitational_parameter_km3_s2)
    if mu <= 0.0 or not np.isfinite(mu):
        raise ValueError("Gravitational parameter must be positive and finite.")
    return -mu * position / radius**3


def j2_perturbing_acceleration(
    position_km: np.ndarray,
    gravitational_parameter_km3_s2: float,
    earth_equatorial_radius_km: float,
    j2: float,
) -> np.ndarray:
    """Return only the J2 perturbing acceleration in km/s².

    The expression assumes an Earth-centred inertial frame whose z-axis is
    aligned with Earth's mean rotation axis.
    """
    position = np.asarray(position_km, dtype=float)
    if position.shape != (3,):
        raise ValueError("Position must have shape (3,).")
    if not np.all(np.isfinite(position)):
        raise ValueError("Position contains non-finite values.")

    mu = float(gravitational_parameter_km3_s2)
    radius_equatorial = float(earth_equatorial_radius_km)
    coefficient_j2 = float(j2)
    if mu <= 0.0 or not np.isfinite(mu):
        raise ValueError("Gravitational parameter must be positive and finite.")
    if radius_equatorial <= 0.0 or not np.isfinite(radius_equatorial):
        raise ValueError("Earth equatorial radius must be positive and finite.")
    if coefficient_j2 < 0.0 or not np.isfinite(coefficient_j2):
        raise ValueError("J2 must be non-negative and finite.")

    x, y, z = position
    radius_squared = float(np.dot(position, position))
    if radius_squared <= 0.0:
        raise ValueError("Position magnitude must be non-zero.")
    radius = float(np.sqrt(radius_squared))
    z_ratio_squared = (z * z) / radius_squared
    factor = 1.5 * coefficient_j2 * mu * radius_equatorial**2 / radius**5

    return factor * np.array(
        [
            x * (5.0 * z_ratio_squared - 1.0),
            y * (5.0 * z_ratio_squared - 1.0),
            z * (5.0 * z_ratio_squared - 3.0),
        ],
        dtype=float,
    )


def j2_perturbing_acceleration_about_axis(
    position_km: np.ndarray,
    symmetry_axis_unit_vector: np.ndarray,
    gravitational_parameter_km3_s2: float,
    earth_equatorial_radius_km: float,
    j2: float,
) -> np.ndarray:
    """Return J2 acceleration about an arbitrary inertial symmetry axis.

    This vector form reduces exactly to ``j2_perturbing_acceleration`` when
    ``symmetry_axis_unit_vector`` is ``[0, 0, 1]``.
    """
    position = np.asarray(position_km, dtype=float)
    axis = np.asarray(symmetry_axis_unit_vector, dtype=float)
    if position.shape != (3,) or axis.shape != (3,):
        raise ValueError("Position and symmetry axis must each have shape (3,).")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(axis)):
        raise ValueError("Position and symmetry axis must contain finite values.")

    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 0.0:
        raise ValueError("The J2 symmetry axis must be non-zero.")
    axis = axis / axis_norm

    mu = float(gravitational_parameter_km3_s2)
    radius_equatorial = float(earth_equatorial_radius_km)
    coefficient_j2 = float(j2)
    if mu <= 0.0 or not np.isfinite(mu):
        raise ValueError("Gravitational parameter must be positive and finite.")
    if radius_equatorial <= 0.0 or not np.isfinite(radius_equatorial):
        raise ValueError("Earth equatorial radius must be positive and finite.")
    if coefficient_j2 < 0.0 or not np.isfinite(coefficient_j2):
        raise ValueError("J2 must be non-negative and finite.")

    radius_squared = float(np.dot(position, position))
    if radius_squared <= 0.0:
        raise ValueError("Position magnitude must be non-zero.")
    radius = float(np.sqrt(radius_squared))
    projection = float(np.dot(position, axis))
    projection_ratio_squared = projection * projection / radius_squared
    factor = 1.5 * coefficient_j2 * mu * radius_equatorial**2 / radius**5
    return factor * (
        (5.0 * projection_ratio_squared - 1.0) * position
        - 2.0 * projection * axis
    )


def j2_perturbing_acceleration_gmat_matched(
    position_km: np.ndarray,
    epoch_utc: str,
    elapsed_seconds: float,
    gravitational_parameter_km3_s2: float,
    earth_equatorial_radius_km: float,
    j2: float,
) -> np.ndarray:
    """Return the pole-aware J2 term targeted at GMAT degree 2/order 0."""
    axis = iau_1976_1980_true_pole_unit_vector(epoch_utc, elapsed_seconds)
    return j2_perturbing_acceleration_about_axis(
        position_km,
        axis,
        gravitational_parameter_km3_s2,
        earth_equatorial_radius_km,
        j2,
    )


def j2_perturbing_acceleration_orientation_model(
    position_km: np.ndarray,
    epoch_utc: str,
    elapsed_seconds: float,
    gravitational_parameter_km3_s2: float,
    earth_equatorial_radius_km: float,
    j2: float,
    orientation_model: str,
) -> np.ndarray:
    """Return J2 acceleration for a named Earth-pole realization."""
    axis = earth_pole_unit_vector(epoch_utc, elapsed_seconds, orientation_model)
    return j2_perturbing_acceleration_about_axis(
        position_km,
        axis,
        gravitational_parameter_km3_s2,
        earth_equatorial_radius_km,
        j2,
    )


def total_j2_acceleration(
    position_km: np.ndarray,
    gravitational_parameter_km3_s2: float,
    earth_equatorial_radius_km: float,
    j2: float,
) -> np.ndarray:
    """Return point-mass plus J2 acceleration in km/s²."""
    return central_gravity_acceleration(
        position_km,
        gravitational_parameter_km3_s2,
    ) + j2_perturbing_acceleration(
        position_km,
        gravitational_parameter_km3_s2,
        earth_equatorial_radius_km,
        j2,
    )


def j2_derivative(
    _elapsed_seconds: float,
    state_vector: np.ndarray,
    gravitational_parameter_km3_s2: float,
    earth_equatorial_radius_km: float,
    j2: float,
) -> np.ndarray:
    """Return the six-dimensional derivative for point-mass plus J2 motion."""
    state = np.asarray(state_vector, dtype=float)
    if state.shape != (6,):
        raise ValueError("State vector must have shape (6,).")
    derivative = np.empty(6, dtype=float)
    derivative[:3] = state[3:]
    derivative[3:] = total_j2_acceleration(
        state[:3],
        gravitational_parameter_km3_s2,
        earth_equatorial_radius_km,
        j2,
    )
    return derivative


def j2_gmat_matched_derivative(
    elapsed_seconds: float,
    state_vector: np.ndarray,
    epoch_utc: str,
    gravitational_parameter_km3_s2: float,
    earth_equatorial_radius_km: float,
    j2: float,
) -> np.ndarray:
    """Return the derivative for point mass plus pole-aware J2 motion."""
    state = np.asarray(state_vector, dtype=float)
    if state.shape != (6,):
        raise ValueError("State vector must have shape (6,).")
    derivative = np.empty(6, dtype=float)
    derivative[:3] = state[3:]
    derivative[3:] = central_gravity_acceleration(
        state[:3],
        gravitational_parameter_km3_s2,
    ) + j2_perturbing_acceleration_gmat_matched(
        state[:3],
        epoch_utc,
        elapsed_seconds,
        gravitational_parameter_km3_s2,
        earth_equatorial_radius_km,
        j2,
    )
    return derivative


def j2_orientation_model_derivative(
    elapsed_seconds: float,
    state_vector: np.ndarray,
    epoch_utc: str,
    gravitational_parameter_km3_s2: float,
    earth_equatorial_radius_km: float,
    j2: float,
    orientation_model: str,
) -> np.ndarray:
    """Return point-mass plus J2 derivative for a named pole realization."""
    state = np.asarray(state_vector, dtype=float)
    if state.shape != (6,):
        raise ValueError("State vector must have shape (6,).")
    derivative = np.empty(6, dtype=float)
    derivative[:3] = state[3:]
    derivative[3:] = central_gravity_acceleration(
        state[:3], gravitational_parameter_km3_s2
    ) + j2_perturbing_acceleration_orientation_model(
        state[:3],
        epoch_utc,
        elapsed_seconds,
        gravitational_parameter_km3_s2,
        earth_equatorial_radius_km,
        j2,
        orientation_model,
    )
    return derivative


def j2_pole_provider_derivative(
    elapsed_seconds: float,
    state_vector: np.ndarray,
    epoch_utc: str,
    gravitational_parameter_km3_s2: float,
    earth_equatorial_radius_km: float,
    j2: float,
    pole_provider: Callable[[str, float], np.ndarray],
) -> np.ndarray:
    """Return point-mass plus J2 derivative for an external pole provider."""
    state = np.asarray(state_vector, dtype=float)
    if state.shape != (6,):
        raise ValueError("State vector must have shape (6,).")
    axis = np.asarray(pole_provider(epoch_utc, elapsed_seconds), dtype=float)
    derivative = np.empty(6, dtype=float)
    derivative[:3] = state[3:]
    derivative[3:] = central_gravity_acceleration(
        state[:3], gravitational_parameter_km3_s2
    ) + j2_perturbing_acceleration_about_axis(
        state[:3],
        axis,
        gravitational_parameter_km3_s2,
        earth_equatorial_radius_km,
        j2,
    )
    return derivative


def propagate_numerical_j2(
    initial_state: CartesianState,
    gravitational_parameter_km3_s2: float,
    earth_equatorial_radius_km: float,
    j2: float,
    elapsed_seconds: np.ndarray,
    *,
    method: str,
    relative_tolerance: float,
    absolute_tolerance: float,
    maximum_step_seconds: float,
) -> StateHistory:
    """Propagate one state using point-mass gravity plus J2."""
    times = np.asarray(elapsed_seconds, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("elapsed_seconds must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError("elapsed_seconds must contain finite non-negative values.")
    if times[0] != 0.0:
        raise ValueError("The numerical J2 time grid must begin at zero.")
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
            fun=j2_derivative,
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
            ),
        )
        runtime = perf_counter() - started
        if not solution.success:
            raise RuntimeError(f"Numerical J2 integration failed: {solution.message}")
        if solution.y.shape != (6, times.size):
            raise RuntimeError(
                "Numerical J2 integrator returned an unexpected state-history shape."
            )
        states = solution.y
        function_evaluations = int(solution.nfev)
        message = str(solution.message)

    return StateHistory(
        model_name="numerical_j2",
        frame=initial_state.frame,
        epoch_utc=initial_state.epoch_utc,
        elapsed_seconds=times,
        timestamps_utc=timestamps_from_epoch(initial_state.epoch_utc, times),
        positions_km=states[:3].T,
        velocities_km_s=states[3:].T,
        runtime_seconds=runtime,
        solver_status=message,
        function_evaluations=function_evaluations,
    )


def propagate_numerical_j2_gmat_matched(
    initial_state: CartesianState,
    gravitational_parameter_km3_s2: float,
    earth_equatorial_radius_km: float,
    j2: float,
    elapsed_seconds: np.ndarray,
    *,
    method: str,
    relative_tolerance: float,
    absolute_tolerance: float,
    maximum_step_seconds: float,
) -> StateHistory:
    """Propagate the historical IAU-1976/1980 GMAT-comparison baseline.

    This path is retained unchanged so Research Core 1A.8–1C.2 evidence stays
    reproducible.  New GMAT-matched work should use
    :func:`propagate_numerical_j2_gmat_validated`, which includes the
    independently validated full-EOP realization.
    """
    if initial_state.frame != "EarthMJ2000Eq":
        raise ValueError(
            "The GMAT-matched J2 propagator currently requires EarthMJ2000Eq."
        )
    times = np.asarray(elapsed_seconds, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("elapsed_seconds must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError("elapsed_seconds must contain finite non-negative values.")
    if times[0] != 0.0:
        raise ValueError("The numerical J2 time grid must begin at zero.")
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
            fun=j2_gmat_matched_derivative,
            t_span=(0.0, final_time),
            y0=initial_vector,
            method=method,
            t_eval=times,
            rtol=float(relative_tolerance),
            atol=float(absolute_tolerance),
            max_step=float(maximum_step_seconds),
            args=(
                initial_state.epoch_utc,
                float(gravitational_parameter_km3_s2),
                float(earth_equatorial_radius_km),
                float(j2),
            ),
        )
        runtime = perf_counter() - started
        if not solution.success:
            raise RuntimeError(
                f"GMAT-matched numerical J2 integration failed: {solution.message}"
            )
        if solution.y.shape != (6, times.size):
            raise RuntimeError(
                "GMAT-matched J2 integrator returned an unexpected state-history shape."
            )
        states = solution.y
        function_evaluations = int(solution.nfev)
        message = str(solution.message)

    return StateHistory(
        model_name="numerical_j2_gmat_matched",
        frame=initial_state.frame,
        epoch_utc=initial_state.epoch_utc,
        elapsed_seconds=times,
        timestamps_utc=timestamps_from_epoch(initial_state.epoch_utc, times),
        positions_km=states[:3].T,
        velocities_km_s=states[3:].T,
        runtime_seconds=runtime,
        solver_status=message,
        function_evaluations=function_evaluations,
    )


def propagate_numerical_j2_orientation_model(
    initial_state: CartesianState,
    gravitational_parameter_km3_s2: float,
    earth_equatorial_radius_km: float,
    j2: float,
    elapsed_seconds: np.ndarray,
    *,
    orientation_model: str,
    method: str,
    relative_tolerance: float,
    absolute_tolerance: float,
    maximum_step_seconds: float,
) -> StateHistory:
    """Propagate J2 gravity using one supported Earth-pole realization."""
    if initial_state.frame != "EarthMJ2000Eq":
        raise ValueError(
            "Orientation-model J2 propagation currently requires EarthMJ2000Eq."
        )
    if orientation_model not in SUPPORTED_POLE_MODELS:
        raise ValueError(
            f"Unsupported pole model {orientation_model!r}; expected one of "
            f"{', '.join(SUPPORTED_POLE_MODELS)}."
        )
    times = np.asarray(elapsed_seconds, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("elapsed_seconds must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError("elapsed_seconds must contain finite non-negative values.")
    if times[0] != 0.0:
        raise ValueError("The orientation-model J2 time grid must begin at zero.")
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
            fun=j2_orientation_model_derivative,
            t_span=(0.0, final_time),
            y0=initial_vector,
            method=method,
            t_eval=times,
            rtol=float(relative_tolerance),
            atol=float(absolute_tolerance),
            max_step=float(maximum_step_seconds),
            args=(
                initial_state.epoch_utc,
                float(gravitational_parameter_km3_s2),
                float(earth_equatorial_radius_km),
                float(j2),
                orientation_model,
            ),
        )
        runtime = perf_counter() - started
        if not solution.success:
            raise RuntimeError(
                f"Orientation-model J2 integration failed: {solution.message}"
            )
        if solution.y.shape != (6, times.size):
            raise RuntimeError(
                "Orientation-model J2 integrator returned an unexpected history."
            )
        states = solution.y
        function_evaluations = int(solution.nfev)
        message = str(solution.message)

    return StateHistory(
        model_name=f"numerical_j2_{orientation_model}",
        frame=initial_state.frame,
        epoch_utc=initial_state.epoch_utc,
        elapsed_seconds=times,
        timestamps_utc=timestamps_from_epoch(initial_state.epoch_utc, times),
        positions_km=states[:3].T,
        velocities_km_s=states[3:].T,
        runtime_seconds=runtime,
        solver_status=message,
        function_evaluations=function_evaluations,
    )


def propagate_numerical_j2_pole_provider(
    initial_state: CartesianState,
    gravitational_parameter_km3_s2: float,
    earth_equatorial_radius_km: float,
    j2: float,
    elapsed_seconds: np.ndarray,
    *,
    pole_provider: Callable[[str, float], np.ndarray],
    model_name: str,
    method: str,
    relative_tolerance: float,
    absolute_tolerance: float,
    maximum_step_seconds: float,
) -> StateHistory:
    """Propagate J2 gravity using a traceable external pole provider."""
    if initial_state.frame != "EarthMJ2000Eq":
        raise ValueError(
            "Pole-provider J2 propagation currently requires EarthMJ2000Eq."
        )
    if not callable(pole_provider):
        raise TypeError("pole_provider must be callable.")
    name = str(model_name).strip()
    if not name:
        raise ValueError("model_name must not be empty.")
    times = np.asarray(elapsed_seconds, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("elapsed_seconds must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError("elapsed_seconds must contain finite non-negative values.")
    if times[0] != 0.0:
        raise ValueError("The pole-provider J2 time grid must begin at zero.")
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
            fun=j2_pole_provider_derivative,
            t_span=(0.0, final_time),
            y0=initial_vector,
            method=method,
            t_eval=times,
            rtol=float(relative_tolerance),
            atol=float(absolute_tolerance),
            max_step=float(maximum_step_seconds),
            args=(
                initial_state.epoch_utc,
                float(gravitational_parameter_km3_s2),
                float(earth_equatorial_radius_km),
                float(j2),
                pole_provider,
            ),
        )
        runtime = perf_counter() - started
        if not solution.success:
            raise RuntimeError(
                f"Pole-provider numerical J2 integration failed: {solution.message}"
            )
        if solution.y.shape != (6, times.size):
            raise RuntimeError(
                "Pole-provider J2 integrator returned an unexpected history."
            )
        states = solution.y
        function_evaluations = int(solution.nfev)
        message = str(solution.message)

    return StateHistory(
        model_name=name,
        frame=initial_state.frame,
        epoch_utc=initial_state.epoch_utc,
        elapsed_seconds=times,
        timestamps_utc=timestamps_from_epoch(initial_state.epoch_utc, times),
        positions_km=states[:3].T,
        velocities_km_s=states[3:].T,
        runtime_seconds=runtime,
        solver_status=message,
        function_evaluations=function_evaluations,
    )


def propagate_numerical_j2_gmat_validated(
    initial_state: CartesianState,
    gravitational_parameter_km3_s2: float,
    earth_equatorial_radius_km: float,
    j2: float,
    elapsed_seconds: np.ndarray,
    *,
    eop_dataset: GmatEopDataset,
    method: str,
    relative_tolerance: float,
    absolute_tolerance: float,
    maximum_step_seconds: float,
) -> StateHistory:
    """Propagate J2 with the independently validated full GMAT R2026a EOP pole."""
    if not isinstance(eop_dataset, GmatEopDataset):
        raise TypeError("eop_dataset must be a GmatEopDataset.")
    provider = partial(
        gmat_validated_eop_pole_unit_vector,
        dataset=eop_dataset,
    )
    return propagate_numerical_j2_pole_provider(
        initial_state,
        gravitational_parameter_km3_s2,
        earth_equatorial_radius_km,
        j2,
        elapsed_seconds,
        pole_provider=provider,
        model_name="numerical_j2_gmat_validated_eop",
        method=method,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step_seconds=maximum_step_seconds,
    )
