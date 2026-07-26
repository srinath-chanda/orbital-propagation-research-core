"""Reference-frame transformations used by the SGP4 experiment."""

from __future__ import annotations

import warnings
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import CartesianDifferential, CartesianRepresentation, GCRS, TEME
from astropy.time import Time
from astropy.utils import iers

# Reproducible offline behavior. The installed astropy-iers-data package is recorded
# in environment metadata. No run silently downloads newer Earth-orientation data.
iers.conf.auto_download = False
iers.conf.iers_degraded_accuracy = "warn"


def teme_to_gcrs(
    positions_teme_km: np.ndarray,
    velocities_teme_km_s: np.ndarray,
    timestamps_utc: tuple[str, ...] | list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Transform TEME position/velocity arrays to GCRS using Astropy."""
    positions = np.asarray(positions_teme_km, dtype=float)
    velocities = np.asarray(velocities_teme_km_s, dtype=float)
    if positions.ndim == 1:
        positions = positions[np.newaxis, :]
    if velocities.ndim == 1:
        velocities = velocities[np.newaxis, :]
    if positions.shape != velocities.shape or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("TEME positions and velocities must both have shape (N, 3).")
    if len(timestamps_utc) != positions.shape[0]:
        raise ValueError("Timestamp count must match the number of TEME states.")

    times = Time(list(timestamps_utc), scale="utc")
    representation = CartesianRepresentation(
        positions.T * u.km,
        differentials=CartesianDifferential(velocities.T * u.km / u.s),
    )
    frame = TEME(representation, obstime=times)
    captured: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        transformed = frame.transform_to(GCRS(obstime=times))
        captured = [str(item.message) for item in caught]

    position_gcrs = transformed.cartesian.xyz.to_value(u.km).T
    velocity_gcrs = transformed.cartesian.differentials["s"].d_xyz.to_value(u.km / u.s).T
    return position_gcrs, velocity_gcrs, captured


def gcrs_to_teme(
    positions_gcrs_km: np.ndarray,
    velocities_gcrs_km_s: np.ndarray,
    timestamps_utc: tuple[str, ...] | list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Transform GCRS position/velocity arrays to TEME using Astropy."""
    positions = np.asarray(positions_gcrs_km, dtype=float)
    velocities = np.asarray(velocities_gcrs_km_s, dtype=float)
    if positions.ndim == 1:
        positions = positions[np.newaxis, :]
    if velocities.ndim == 1:
        velocities = velocities[np.newaxis, :]
    if positions.shape != velocities.shape or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("GCRS positions and velocities must both have shape (N, 3).")
    if len(timestamps_utc) != positions.shape[0]:
        raise ValueError("Timestamp count must match the number of GCRS states.")

    times = Time(list(timestamps_utc), scale="utc")
    representation = CartesianRepresentation(
        positions.T * u.km,
        differentials=CartesianDifferential(velocities.T * u.km / u.s),
    )
    frame = GCRS(representation, obstime=times)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        transformed = frame.transform_to(TEME(obstime=times))
        captured = [str(item.message) for item in caught]
    position_teme = transformed.cartesian.xyz.to_value(u.km).T
    velocity_teme = transformed.cartesian.differentials["s"].d_xyz.to_value(u.km / u.s).T
    return position_teme, velocity_teme, captured


def frame_roundtrip_error(
    position_teme_km: np.ndarray,
    velocity_teme_km_s: np.ndarray,
    timestamp_utc: str,
) -> dict[str, Any]:
    """Measure one-state TEME→GCRS→TEME numerical round-trip residual."""
    gcrs_r, gcrs_v, warnings_forward = teme_to_gcrs(
        position_teme_km,
        velocity_teme_km_s,
        [timestamp_utc],
    )
    back_r, back_v, warnings_reverse = gcrs_to_teme(gcrs_r, gcrs_v, [timestamp_utc])
    position_error_m = float(np.linalg.norm(back_r[0] - np.asarray(position_teme_km)) * 1000.0)
    velocity_error_mm_s = float(
        np.linalg.norm(back_v[0] - np.asarray(velocity_teme_km_s)) * 1.0e6
    )
    return {
        "position_roundtrip_error_m": position_error_m,
        "velocity_roundtrip_error_mm_s": velocity_error_mm_s,
        "astropy_warnings": warnings_forward + warnings_reverse,
    }
