"""Earth-orientation helpers for GMAT-compatible zonal gravity.

GMAT's Earth body orientation uses the IAU-1976/FK5 precession model with
IAU-1980 nutation.  ERFA's ``pnm80`` routine implements the same named
precession-nutation theory.  Only the inertial direction of the true pole is
needed for an axisymmetric degree-2/order-0 gravity field.  Rotation about the
pole therefore cancels from the J2 acceleration.
"""

from __future__ import annotations

import numpy as np
from astropy.time import Time, TimeDelta
import erfa


SUPPORTED_POLE_MODELS = (
    "j2000_fixed",
    "iau1976_precession",
    "iau1976_1980",
    "iau2000a",
    "iau2006_2000a",
)


POLE_MODEL_DESCRIPTIONS = {
    "j2000_fixed": "Static J2000 z-axis diagnostic",
    "iau1976_precession": "IAU 1976 precession-only mean pole",
    "iau1976_1980": "IAU 1976 precession plus IAU 1980 nutation true pole",
    "iau2000a": "IAU 2000A bias-precession-nutation pole",
    "iau2006_2000a": "IAU 2006 precession plus IAU 2000A nutation pole",
}


def _as_utc_time(epoch_utc: str) -> Time:
    epoch = Time(str(epoch_utc), scale="utc")
    if not np.isfinite(float(epoch.utc.jd)):
        raise ValueError("The Earth-orientation epoch must be finite.")
    return epoch


def _orientation_matrix(
    epoch_utc: str,
    elapsed_seconds: float,
    model: str,
) -> np.ndarray:
    """Return the selected J2000-to-date Earth-orientation matrix."""
    if model not in SUPPORTED_POLE_MODELS:
        raise ValueError(
            f"Unsupported pole model {model!r}; expected one of "
            f"{', '.join(SUPPORTED_POLE_MODELS)}."
        )
    elapsed = float(elapsed_seconds)
    if not np.isfinite(elapsed):
        raise ValueError("elapsed_seconds must be finite.")
    if model == "j2000_fixed":
        return np.identity(3, dtype=float)
    epoch = _as_utc_time(epoch_utc)
    evaluation_time = epoch + TimeDelta(elapsed, format="sec")
    tt1 = evaluation_time.tt.jd1
    tt2 = evaluation_time.tt.jd2
    if model == "iau1976_precession":
        matrix = erfa.pmat76(tt1, tt2)
    elif model == "iau1976_1980":
        matrix = erfa.pnm80(tt1, tt2)
    elif model == "iau2000a":
        matrix = erfa.pnm00a(tt1, tt2)
    else:
        matrix = erfa.pnm06a(tt1, tt2)
    return np.asarray(matrix, dtype=float)


def earth_pole_unit_vector(
    epoch_utc: str,
    elapsed_seconds: float,
    model: str,
) -> np.ndarray:
    """Return an Earth pole direction for a named orientation realization.

    Each ERFA matrix maps J2000/GCRS input axes into the corresponding
    mean/true equator-of-date axes.  Its third row is therefore that model's
    pole expressed in the input inertial frame.  These models remain available
    for diagnostics and historical reproduction.  Research Core 1C.3 adopts
    the separate full-EOP path in :mod:`research_core.gmat_eop` as the
    validated GMAT-matched baseline.
    """
    matrix = _orientation_matrix(epoch_utc, elapsed_seconds, model)
    axis = matrix[2, :].copy()
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError(f"Pole model {model!r} returned an invalid axis.")
    return axis / norm


def pole_angular_separation_arcsec(
    first_axis: np.ndarray,
    second_axis: np.ndarray,
) -> float:
    """Return the unsigned angular separation between two axes in arcseconds."""
    first = np.asarray(first_axis, dtype=float)
    second = np.asarray(second_axis, dtype=float)
    if first.shape != (3,) or second.shape != (3,):
        raise ValueError("Pole axes must each have shape (3,).")
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 0.0 or second_norm <= 0.0:
        raise ValueError("Pole axes must be non-zero.")
    cosine = float(np.dot(first / first_norm, second / second_norm))
    angle_rad = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return float(np.degrees(angle_rad) * 3600.0)


def iau_1976_1980_true_pole_unit_vector(
    epoch_utc: str,
    elapsed_seconds: float,
) -> np.ndarray:
    """Return the true-of-date Earth pole in J2000 inertial coordinates.

    ``erfa.pnm80`` returns the matrix that transforms a vector from the J2000
    celestial frame to the true equator and equinox of date.  Its third row is
    therefore the true pole expressed in the input inertial coordinates.

    UTC is converted to TT before evaluating precession and nutation.  Polar
    motion and EOP corrections are intentionally not included in this legacy
    compatibility helper.  The adopted 1C.3 baseline is provided separately by
    ``gmat_validated_eop_pole_unit_vector``.
    """
    return earth_pole_unit_vector(epoch_utc, elapsed_seconds, "iau1976_1980")


def iau_1976_1980_precession_nutation_matrix(
    epoch_utc: str,
    elapsed_seconds: float,
) -> np.ndarray:
    """Return the J2000-to-true-of-date IAU-1976/1980 rotation matrix."""
    return _orientation_matrix(epoch_utc, elapsed_seconds, "iau1976_1980")
