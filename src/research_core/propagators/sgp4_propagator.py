"""SGP4 propagation from a frozen TLE, including TEME→GCRS conversion."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from astropy.time import Time, TimeDelta
import astropy.units as u
from sgp4.api import SGP4_ERRORS, Satrec

from ..data_models import StateHistory
from ..frames import teme_to_gcrs
from ..tle import FrozenTLE
from ..time_utils import timestamps_from_epoch


def propagate_sgp4_frozen_tle(
    tle: FrozenTLE,
    elapsed_seconds: np.ndarray,
) -> tuple[StateHistory, StateHistory, dict[str, Any]]:
    """Propagate a frozen TLE in TEME and transform the history to GCRS."""
    times = np.asarray(elapsed_seconds, dtype=float)
    if times.ndim != 1 or times.size == 0 or times[0] != 0.0:
        raise ValueError("SGP4 elapsed_seconds must be non-empty and begin at zero.")
    if np.any(~np.isfinite(times)) or np.any(times < 0.0) or np.any(np.diff(times) < 0.0):
        raise ValueError("SGP4 elapsed_seconds must be finite, non-negative, and ordered.")

    satellite = Satrec.twoline2rv(tle.line1, tle.line2)
    epoch = Time(tle.epoch_utc, scale="utc")
    astropy_times = epoch + TimeDelta(times * u.s)
    timestamps = timestamps_from_epoch(tle.epoch_utc, times)

    started = perf_counter()
    errors, positions_teme, velocities_teme = satellite.sgp4_array(
        astropy_times.jd1,
        astropy_times.jd2,
    )
    sgp4_runtime = perf_counter() - started
    errors = np.asarray(errors, dtype=int)
    positions_teme = np.asarray(positions_teme, dtype=float)
    velocities_teme = np.asarray(velocities_teme, dtype=float)

    unique_errors, counts = np.unique(errors, return_counts=True)
    error_counts = {
        str(int(code)): {
            "count": int(count),
            "message": SGP4_ERRORS.get(int(code), "unknown SGP4 error"),
        }
        for code, count in zip(unique_errors, counts)
    }
    nonzero = np.flatnonzero(errors != 0)
    if nonzero.size:
        first = int(nonzero[0])
        code = int(errors[first])
        raise RuntimeError(
            f"SGP4 failed at index {first}, t={times[first]} s, code={code}: "
            f"{SGP4_ERRORS.get(code, 'unknown error')}"
        )
    if not np.all(np.isfinite(positions_teme)) or not np.all(np.isfinite(velocities_teme)):
        raise FloatingPointError("SGP4 returned non-finite state values.")

    transform_started = perf_counter()
    positions_gcrs, velocities_gcrs, transform_warnings = teme_to_gcrs(
        positions_teme,
        velocities_teme,
        timestamps,
    )
    transform_runtime = perf_counter() - transform_started

    teme_history = StateHistory(
        model_name="sgp4_teme",
        frame="TEME",
        epoch_utc=tle.epoch_utc,
        elapsed_seconds=times,
        timestamps_utc=timestamps,
        positions_km=positions_teme,
        velocities_km_s=velocities_teme,
        runtime_seconds=sgp4_runtime,
        solver_status="SGP4 propagation completed with zero error codes.",
        function_evaluations=None,
    )
    gcrs_history = StateHistory(
        model_name="sgp4",
        frame="GCRS_ASTROPY_FROM_TEME",
        epoch_utc=tle.epoch_utc,
        elapsed_seconds=times,
        timestamps_utc=timestamps,
        positions_km=positions_gcrs,
        velocities_km_s=velocities_gcrs,
        runtime_seconds=sgp4_runtime + transform_runtime,
        solver_status="SGP4 TEME propagation and Astropy GCRS transformation completed.",
        function_evaluations=None,
    )
    diagnostics = {
        "sgp4_runtime_seconds": sgp4_runtime,
        "teme_to_gcrs_runtime_seconds": transform_runtime,
        "total_runtime_seconds": sgp4_runtime + transform_runtime,
        "state_count": int(times.size),
        "error_counts": error_counts,
        "nonzero_error_count": int(np.count_nonzero(errors)),
        "astropy_transform_warnings": transform_warnings,
    }
    return teme_history, gcrs_history, diagnostics
