"""Earth-fixed coordinate and ground-track analysis for Research Core 1A.5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import astropy.units as u
import numpy as np
from astropy.coordinates import (
    CartesianDifferential,
    CartesianRepresentation,
    EarthLocation,
    GCRS,
    ITRS,
)
from astropy.time import Time
from astropy.utils import iers

from .data_models import StateHistory

# Reproducible offline behaviour. No run silently downloads newer EOP data.
iers.conf.auto_download = False
iers.conf.iers_degraded_accuracy = "warn"


@dataclass(frozen=True)
class GroundTrackHistory:
    """Earth-fixed and WGS-84 geodetic history for one propagation model."""

    model_name: str
    source_frame: str
    earth_fixed_frame: str
    ellipsoid: str
    epoch_utc: str
    elapsed_seconds: np.ndarray
    timestamps_utc: tuple[str, ...]
    positions_itrs_km: np.ndarray
    velocities_itrs_km_s: np.ndarray
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    altitude_km: np.ndarray
    transform_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        elapsed = np.asarray(self.elapsed_seconds, dtype=float)
        positions = np.asarray(self.positions_itrs_km, dtype=float)
        velocities = np.asarray(self.velocities_itrs_km_s, dtype=float)
        latitude = np.asarray(self.latitude_deg, dtype=float)
        longitude = np.asarray(self.longitude_deg, dtype=float)
        altitude = np.asarray(self.altitude_km, dtype=float)
        n = elapsed.size
        if elapsed.ndim != 1 or n == 0:
            raise ValueError("Ground-track elapsed_seconds must be a non-empty vector.")
        if positions.shape != (n, 3) or velocities.shape != (n, 3):
            raise ValueError("Ground-track ITRS states must have shape (N, 3).")
        for name, values in (
            ("latitude_deg", latitude),
            ("longitude_deg", longitude),
            ("altitude_km", altitude),
        ):
            if values.shape != (n,):
                raise ValueError(f"Ground-track {name} must have shape (N,).")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Ground-track {name} contains non-finite values.")
        if len(self.timestamps_utc) != n:
            raise ValueError("Ground-track timestamp count must match state count.")
        if np.any(latitude < -90.0) or np.any(latitude > 90.0):
            raise ValueError("Ground-track latitude is outside [-90, 90] degrees.")
        if np.any(longitude < -180.0) or np.any(longitude > 180.0):
            raise ValueError("Ground-track longitude is outside [-180, 180] degrees.")
        object.__setattr__(self, "elapsed_seconds", elapsed)
        object.__setattr__(self, "positions_itrs_km", positions)
        object.__setattr__(self, "velocities_itrs_km_s", velocities)
        object.__setattr__(self, "latitude_deg", latitude)
        object.__setattr__(self, "longitude_deg", longitude)
        object.__setattr__(self, "altitude_km", altitude)


def wrap_longitude_deg(longitude_deg: np.ndarray | float) -> np.ndarray:
    """Wrap longitude to the closed interval [-180, 180]."""
    values = np.asarray(longitude_deg, dtype=float)
    wrapped = (values + 180.0) % 360.0 - 180.0
    # Preserve +180 for positive inputs that map exactly to the anti-meridian.
    wrapped = np.where((wrapped == -180.0) & (values > 0.0), 180.0, wrapped)
    return wrapped


def gcrs_state_history_to_ground_track(
    history: StateHistory,
    *,
    ellipsoid: str = "WGS84",
) -> GroundTrackHistory:
    """Transform a GCRS state history to ITRS and WGS-84 geodetic coordinates."""
    if not history.frame.startswith("GCRS"):
        raise ValueError(
            f"Ground-track conversion requires a GCRS history, received {history.frame!r}."
        )
    times = Time(list(history.timestamps_utc), scale="utc")
    representation = CartesianRepresentation(
        history.positions_km.T * u.km,
        differentials=CartesianDifferential(history.velocities_km_s.T * u.km / u.s),
    )
    gcrs = GCRS(representation, obstime=times)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        itrs = gcrs.transform_to(ITRS(obstime=times))
    captured = tuple(str(item.message) for item in caught)

    positions_itrs_km = itrs.cartesian.xyz.to_value(u.km).T
    velocities_itrs_km_s = (
        itrs.cartesian.differentials["s"].d_xyz.to_value(u.km / u.s).T
    )
    locations = EarthLocation.from_geocentric(
        positions_itrs_km[:, 0] * u.km,
        positions_itrs_km[:, 1] * u.km,
        positions_itrs_km[:, 2] * u.km,
    )
    longitude, latitude, height = locations.to_geodetic(ellipsoid=ellipsoid)
    return GroundTrackHistory(
        model_name=history.model_name,
        source_frame=history.frame,
        earth_fixed_frame="ITRS_ASTROPY",
        ellipsoid=ellipsoid,
        epoch_utc=history.epoch_utc,
        elapsed_seconds=history.elapsed_seconds,
        timestamps_utc=history.timestamps_utc,
        positions_itrs_km=positions_itrs_km,
        velocities_itrs_km_s=velocities_itrs_km_s,
        latitude_deg=latitude.to_value(u.deg),
        longitude_deg=wrap_longitude_deg(longitude.to_value(u.deg)),
        altitude_km=height.to_value(u.km),
        transform_warnings=captured,
    )


def geodetic_roundtrip_error(
    track: GroundTrackHistory,
    *,
    sample_count: int = 25,
) -> dict[str, Any]:
    """Measure geodetic→ITRS reconstruction residuals at representative samples."""
    n = track.elapsed_seconds.size
    count = max(1, min(int(sample_count), n))
    indices = np.unique(np.linspace(0, n - 1, count, dtype=int))
    locations = EarthLocation.from_geodetic(
        lon=track.longitude_deg[indices] * u.deg,
        lat=track.latitude_deg[indices] * u.deg,
        height=track.altitude_km[indices] * u.km,
        ellipsoid=track.ellipsoid,
    )
    reconstructed = np.column_stack(
        [
            locations.x.to_value(u.km),
            locations.y.to_value(u.km),
            locations.z.to_value(u.km),
        ]
    )
    residual_m = np.linalg.norm(
        reconstructed - track.positions_itrs_km[indices], axis=1
    ) * 1000.0
    return {
        "sample_count": int(indices.size),
        "sample_indices": indices.tolist(),
        "maximum_position_residual_m": float(np.max(residual_m)),
        "rms_position_residual_m": float(np.sqrt(np.mean(residual_m**2))),
        "residual_m": residual_m.tolist(),
    }


def split_at_antimeridian(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    *,
    threshold_deg: float = 180.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a longitude/latitude line where it crosses the anti-meridian."""
    longitude = np.asarray(longitude_deg, dtype=float)
    latitude = np.asarray(latitude_deg, dtype=float)
    if longitude.shape != latitude.shape or longitude.ndim != 1:
        raise ValueError("Longitude and latitude must be matching one-dimensional arrays.")
    if longitude.size == 0:
        return []
    break_indices = np.where(np.abs(np.diff(longitude)) > float(threshold_deg))[0] + 1
    longitude_segments = np.split(longitude, break_indices)
    latitude_segments = np.split(latitude, break_indices)
    return [
        (lon_segment, lat_segment)
        for lon_segment, lat_segment in zip(longitude_segments, latitude_segments)
        if lon_segment.size > 0
    ]


def compare_ground_tracks(
    reference: GroundTrackHistory,
    comparison: GroundTrackHistory,
    *,
    surface_radius_km: float,
) -> dict[str, Any]:
    """Compare two synchronized ground tracks using spherical central-angle distance."""
    if reference.timestamps_utc != comparison.timestamps_utc:
        raise ValueError("Ground-track timestamps must match exactly for comparison.")
    if reference.elapsed_seconds.shape != comparison.elapsed_seconds.shape or not np.allclose(
        reference.elapsed_seconds, comparison.elapsed_seconds, rtol=0.0, atol=1.0e-9
    ):
        raise ValueError("Ground-track elapsed-time grids must match exactly.")
    lat1 = np.radians(reference.latitude_deg)
    lat2 = np.radians(comparison.latitude_deg)
    dlat = lat2 - lat1
    dlon = np.radians(
        wrap_longitude_deg(comparison.longitude_deg - reference.longitude_deg)
    )
    haversine = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    central_angle = 2.0 * np.arctan2(
        np.sqrt(np.clip(haversine, 0.0, 1.0)),
        np.sqrt(np.clip(1.0 - haversine, 0.0, 1.0)),
    )
    surface_separation_km = float(surface_radius_km) * central_angle
    altitude_difference_km = comparison.altitude_km - reference.altitude_km
    itrs_position_separation_km = np.linalg.norm(
        comparison.positions_itrs_km - reference.positions_itrs_km,
        axis=1,
    )
    longitude_difference_deg = wrap_longitude_deg(
        comparison.longitude_deg - reference.longitude_deg
    )
    latitude_difference_deg = comparison.latitude_deg - reference.latitude_deg
    return {
        "reference_model": reference.model_name,
        "comparison_model": comparison.model_name,
        "frame": reference.earth_fixed_frame,
        "ellipsoid": reference.ellipsoid,
        "surface_distance_model": "spherical_central_angle",
        "surface_radius_km": float(surface_radius_km),
        "elapsed_seconds": reference.elapsed_seconds,
        "timestamps_utc": reference.timestamps_utc,
        "latitude_difference_deg": latitude_difference_deg,
        "longitude_difference_deg": longitude_difference_deg,
        "altitude_difference_km": altitude_difference_km,
        "surface_separation_km": surface_separation_km,
        "itrs_position_separation_km": itrs_position_separation_km,
        "maximum_surface_separation_km": float(np.max(surface_separation_km)),
        "final_surface_separation_km": float(surface_separation_km[-1]),
        "rms_surface_separation_km": float(
            np.sqrt(np.mean(surface_separation_km**2))
        ),
        "maximum_absolute_altitude_difference_km": float(
            np.max(np.abs(altitude_difference_km))
        ),
        "final_altitude_difference_km": float(altitude_difference_km[-1]),
        "maximum_itrs_position_separation_km": float(
            np.max(itrs_position_separation_km)
        ),
    }


def ground_track_summary(
    reference: GroundTrackHistory,
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a compact ground-track summary for JSON and reporting."""
    return {
        "reference_model": reference.model_name,
        "frame": reference.earth_fixed_frame,
        "ellipsoid": reference.ellipsoid,
        "duration_hours": float(reference.elapsed_seconds[-1] / 3600.0),
        "sample_count": int(reference.elapsed_seconds.size),
        "reference_minimum_altitude_km": float(np.min(reference.altitude_km)),
        "reference_maximum_altitude_km": float(np.max(reference.altitude_km)),
        "reference_latitude_range_deg": [
            float(np.min(reference.latitude_deg)),
            float(np.max(reference.latitude_deg)),
        ],
        "models": {
            item["comparison_model"]: {
                "final_surface_separation_km": item[
                    "final_surface_separation_km"
                ],
                "maximum_surface_separation_km": item[
                    "maximum_surface_separation_km"
                ],
                "rms_surface_separation_km": item[
                    "rms_surface_separation_km"
                ],
                "final_altitude_difference_km": item[
                    "final_altitude_difference_km"
                ],
                "maximum_absolute_altitude_difference_km": item[
                    "maximum_absolute_altitude_difference_km"
                ],
            }
            for item in comparisons
        },
    }
