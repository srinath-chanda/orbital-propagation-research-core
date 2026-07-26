"""Numerical propagation with normalized spherical-harmonic Earth gravity."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from astropy.time import Time, TimeDelta
from scipy.integrate import solve_ivp

from ..data_models import CartesianState, StateHistory
from ..gmat_eop import GmatEopDataset, gmat_r2026a_inertial_to_fixed_matrix
from ..gravity_harmonics import CofGravityField, gravity_acceleration_inertial_km_s2
from ..time_utils import timestamps_from_epoch


def spherical_harmonic_derivative(
    elapsed_seconds: float,
    state_vector: np.ndarray,
    epoch_utc: Time,
    gravity_field: CofGravityField,
    eop_dataset: GmatEopDataset,
    degree: int,
    order: int,
) -> np.ndarray:
    """Return the EarthMJ2000Eq derivative for one gravity truncation."""
    state = np.asarray(state_vector, dtype=float)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("state_vector must contain six finite values.")
    elapsed = float(elapsed_seconds)
    if not np.isfinite(elapsed):
        raise ValueError("elapsed_seconds must be finite.")
    evaluation_time = epoch_utc + TimeDelta(elapsed, format="sec")
    sample = eop_dataset.sample(evaluation_time.utc)
    rotation = gmat_r2026a_inertial_to_fixed_matrix(evaluation_time.utc, sample)
    acceleration = gravity_acceleration_inertial_km_s2(
        state[:3],
        rotation,
        gravity_field,
        degree=int(degree),
        order=int(order),
    )
    return np.concatenate((state[3:], acceleration))


def propagate_spherical_harmonic_gravity(
    initial_state: CartesianState,
    gravity_field: CofGravityField,
    eop_dataset: GmatEopDataset,
    elapsed_seconds: np.ndarray,
    *,
    degree: int,
    order: int,
    method: str,
    relative_tolerance: float,
    absolute_tolerance: float,
    maximum_step_seconds: float,
) -> StateHistory:
    """Propagate one normalized gravity truncation on an explicit output grid."""
    if initial_state.frame != "EarthMJ2000Eq":
        raise ValueError("Spherical-harmonic propagation requires EarthMJ2000Eq.")
    if not isinstance(gravity_field, CofGravityField):
        raise TypeError("gravity_field must be a CofGravityField.")
    if not isinstance(eop_dataset, GmatEopDataset):
        raise TypeError("eop_dataset must be a GmatEopDataset.")
    model_degree = int(degree)
    model_order = int(order)
    if model_degree < 0 or model_degree > gravity_field.maximum_degree:
        raise ValueError("Requested degree is outside the gravity-field limits.")
    if model_order < 0 or model_order > min(model_degree, gravity_field.maximum_order):
        raise ValueError("Requested order is outside the gravity-field limits.")
    times = np.asarray(elapsed_seconds, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("elapsed_seconds must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(times)) or times[0] != 0.0:
        raise ValueError("The output grid must be finite and begin at zero.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("The output grid must increase strictly.")
    rtol = float(relative_tolerance)
    atol = float(absolute_tolerance)
    max_step = float(maximum_step_seconds)
    if rtol <= 0.0 or atol <= 0.0 or max_step <= 0.0:
        raise ValueError("Integrator tolerances and maximum step must be positive.")

    initial_vector = np.concatenate(
        (initial_state.position_km, initial_state.velocity_km_s)
    )
    epoch = Time(initial_state.epoch_utc, scale="utc")
    started = perf_counter()
    solution = solve_ivp(
        fun=spherical_harmonic_derivative,
        t_span=(0.0, float(times[-1])),
        y0=initial_vector,
        method=str(method),
        t_eval=times,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        args=(
            epoch,
            gravity_field,
            eop_dataset,
            model_degree,
            model_order,
        ),
    )
    runtime = perf_counter() - started
    if not solution.success:
        raise RuntimeError(f"Spherical-harmonic integration failed: {solution.message}")
    if solution.y.shape != (6, times.size):
        raise RuntimeError("Spherical-harmonic integrator returned an unexpected history.")
    return StateHistory(
        model_name=f"numerical_gravity_{model_degree}x{model_order}",
        frame=initial_state.frame,
        epoch_utc=initial_state.epoch_utc,
        elapsed_seconds=times,
        timestamps_utc=timestamps_from_epoch(initial_state.epoch_utc, times),
        positions_km=solution.y[:3].T,
        velocities_km_s=solution.y[3:].T,
        runtime_seconds=runtime,
        solver_status=str(solution.message),
        function_evaluations=int(solution.nfev),
    )
