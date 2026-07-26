"""Numerical point-mass two-body propagation."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp

from ..data_models import CartesianState, StateHistory
from ..time_utils import timestamps_from_epoch


def two_body_derivative(
    _time_seconds: float,
    state: np.ndarray,
    gravitational_parameter_km3_s2: float,
) -> np.ndarray:
    """Return [velocity, point-mass gravitational acceleration]."""
    position = state[:3]
    velocity = state[3:]
    radius = float(np.linalg.norm(position))
    if radius <= 0.0 or not np.isfinite(radius):
        raise ValueError("Numerical state has an invalid position magnitude.")
    acceleration = -gravitational_parameter_km3_s2 * position / (radius**3)
    return np.concatenate((velocity, acceleration))


def propagate_numerical_two_body(
    initial_state: CartesianState,
    gravitational_parameter_km3_s2: float,
    elapsed_seconds: np.ndarray,
    *,
    method: str,
    relative_tolerance: float,
    absolute_tolerance: float,
    maximum_step_seconds: float,
) -> StateHistory:
    """Numerically integrate point-mass two-body motion with SciPy."""
    times = np.asarray(elapsed_seconds, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("elapsed_seconds must be a non-empty one-dimensional array.")
    if times[0] != 0.0:
        raise ValueError("The numerical time grid must begin at zero.")
    if np.any(times < 0.0) or np.any(np.diff(times) < 0.0):
        raise ValueError("elapsed_seconds must be non-negative and non-decreasing.")

    initial_vector = np.concatenate(
        (initial_state.position_km, initial_state.velocity_km_s)
    )
    final_time = float(times[-1])
    if final_time <= 0.0:
        raise ValueError("The numerical propagation duration must be positive.")

    started = perf_counter()
    solution = solve_ivp(
        fun=two_body_derivative,
        t_span=(0.0, final_time),
        y0=initial_vector,
        method=method,
        t_eval=times,
        rtol=float(relative_tolerance),
        atol=float(absolute_tolerance),
        max_step=float(maximum_step_seconds),
        args=(float(gravitational_parameter_km3_s2),),
    )
    runtime = perf_counter() - started

    if not solution.success:
        raise RuntimeError(f"Numerical two-body integration failed: {solution.message}")
    if solution.y.shape != (6, times.size):
        raise RuntimeError(
            "Numerical integrator returned an unexpected state-history shape."
        )

    return StateHistory(
        model_name="numerical_two_body",
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
