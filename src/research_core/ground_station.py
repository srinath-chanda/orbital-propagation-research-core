"""Ground-station visibility and pass analysis for Research Core 1A.6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, minimize_scalar

from .ground_track import GroundTrackHistory
from .time_utils import format_utc_timestamp, parse_utc_timestamp


@dataclass(frozen=True)
class GroundStation:
    """One fixed WGS-84 ground station."""

    station_id: str
    name: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    minimum_elevation_deg: float

    def __post_init__(self) -> None:
        if not self.station_id.strip():
            raise ValueError("Ground-station ID must not be empty.")
        if not self.name.strip():
            raise ValueError("Ground-station name must not be empty.")
        if not -90.0 <= float(self.latitude_deg) <= 90.0:
            raise ValueError("Ground-station latitude must be in [-90, 90] degrees.")
        if not -180.0 <= float(self.longitude_deg) <= 180.0:
            raise ValueError("Ground-station longitude must be in [-180, 180] degrees.")
        if not -5.0 <= float(self.minimum_elevation_deg) <= 90.0:
            raise ValueError("Ground-station minimum elevation must be in [-5, 90] degrees.")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "GroundStation":
        return cls(
            station_id=str(value["station_id"]),
            name=str(value["name"]),
            latitude_deg=float(value["latitude_deg"]),
            longitude_deg=float(value["longitude_deg"]),
            altitude_m=float(value["altitude_m"]),
            minimum_elevation_deg=float(value["minimum_elevation_deg"]),
        )

    def earth_location(self) -> EarthLocation:
        return EarthLocation.from_geodetic(
            lon=self.longitude_deg * u.deg,
            lat=self.latitude_deg * u.deg,
            height=self.altitude_m * u.m,
            ellipsoid="WGS84",
        )

    def itrs_position_km(self) -> np.ndarray:
        location = self.earth_location()
        return np.array(
            [
                location.x.to_value(u.km),
                location.y.to_value(u.km),
                location.z.to_value(u.km),
            ],
            dtype=float,
        )

    def station_roundtrip_error_m(self) -> float:
        position = self.itrs_position_km()
        recovered = EarthLocation.from_geocentric(
            position[0] * u.km,
            position[1] * u.km,
            position[2] * u.km,
        )
        longitude, latitude, height = recovered.to_geodetic(ellipsoid="WGS84")
        reconstructed = EarthLocation.from_geodetic(
            longitude,
            latitude,
            height,
            ellipsoid="WGS84",
        )
        residual_km = np.linalg.norm(
            np.array(
                [
                    reconstructed.x.to_value(u.km),
                    reconstructed.y.to_value(u.km),
                    reconstructed.z.to_value(u.km),
                ]
            )
            - position
        )
        return float(residual_km * 1000.0)


@dataclass(frozen=True)
class VisibilityHistory:
    """Topocentric azimuth, elevation, range and range-rate history."""

    model_name: str
    station: GroundStation
    epoch_utc: str
    elapsed_seconds: np.ndarray
    timestamps_utc: tuple[str, ...]
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    range_km: np.ndarray
    range_rate_km_s: np.ndarray

    def __post_init__(self) -> None:
        elapsed = np.asarray(self.elapsed_seconds, dtype=float)
        azimuth = np.asarray(self.azimuth_deg, dtype=float)
        elevation = np.asarray(self.elevation_deg, dtype=float)
        range_km = np.asarray(self.range_km, dtype=float)
        range_rate = np.asarray(self.range_rate_km_s, dtype=float)
        n = elapsed.size
        if elapsed.ndim != 1 or n < 2:
            raise ValueError("Visibility history requires at least two time samples.")
        if np.any(np.diff(elapsed) <= 0.0):
            raise ValueError("Visibility-history times must be strictly increasing.")
        for name, values in (
            ("azimuth_deg", azimuth),
            ("elevation_deg", elevation),
            ("range_km", range_km),
            ("range_rate_km_s", range_rate),
        ):
            if values.shape != (n,):
                raise ValueError(f"Visibility-history {name} must have shape (N,).")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Visibility-history {name} contains non-finite values.")
        if len(self.timestamps_utc) != n:
            raise ValueError("Visibility-history timestamps must match the time grid.")
        if np.any(azimuth < 0.0) or np.any(azimuth >= 360.0):
            raise ValueError("Azimuth must be in [0, 360) degrees.")
        if np.any(elevation < -90.0) or np.any(elevation > 90.0):
            raise ValueError("Elevation must be in [-90, 90] degrees.")
        if np.any(range_km <= 0.0):
            raise ValueError("Topocentric range must be positive.")
        object.__setattr__(self, "elapsed_seconds", elapsed)
        object.__setattr__(self, "azimuth_deg", azimuth)
        object.__setattr__(self, "elevation_deg", elevation)
        object.__setattr__(self, "range_km", range_km)
        object.__setattr__(self, "range_rate_km_s", range_rate)


@dataclass(frozen=True)
class GroundStationPass:
    """One refined geometric ground-station pass."""

    pass_id: str
    model_name: str
    station_id: str
    station_name: str
    minimum_elevation_deg: float
    aos_elapsed_seconds: float
    aos_utc: str
    maximum_elevation_elapsed_seconds: float
    maximum_elevation_utc: str
    los_elapsed_seconds: float
    los_utc: str
    duration_seconds: float
    maximum_elevation_deg: float
    azimuth_at_aos_deg: float
    azimuth_at_maximum_deg: float
    azimuth_at_los_deg: float
    range_at_maximum_elevation_km: float
    closest_range_km: float
    closest_range_elapsed_seconds: float
    closest_range_utc: str
    partial_at_start: bool
    partial_at_end: bool
    aos_threshold_residual_deg: float
    los_threshold_residual_deg: float
    refinement_method: str

    def as_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


def station_topocentric_from_itrs(
    positions_itrs_km: np.ndarray,
    velocities_itrs_km_s: np.ndarray,
    station: GroundStation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert ITRS states to station-centred azimuth, elevation, range and range rate."""
    positions = np.asarray(positions_itrs_km, dtype=float)
    velocities = np.asarray(velocities_itrs_km_s, dtype=float)
    if positions.ndim == 1:
        positions = positions.reshape(1, 3)
    if velocities.ndim == 1:
        velocities = velocities.reshape(1, 3)
    if positions.shape != velocities.shape or positions.shape[1:] != (3,):
        raise ValueError("ITRS positions and velocities must have matching shape (N, 3).")

    station_position = station.itrs_position_km()
    relative = positions - station_position

    latitude = np.radians(station.latitude_deg)
    longitude = np.radians(station.longitude_deg)
    sin_lat, cos_lat = np.sin(latitude), np.cos(latitude)
    sin_lon, cos_lon = np.sin(longitude), np.cos(longitude)

    rotation = np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=float,
    )
    enu = relative @ rotation.T
    east = enu[:, 0]
    north = enu[:, 1]
    up = enu[:, 2]
    horizontal = np.hypot(east, north)
    range_km = np.linalg.norm(relative, axis=1)
    elevation_deg = np.degrees(np.arctan2(up, horizontal))
    azimuth_deg = np.degrees(np.arctan2(east, north)) % 360.0
    range_rate_km_s = np.einsum("ij,ij->i", relative, velocities) / range_km
    return azimuth_deg, elevation_deg, range_km, range_rate_km_s


def visibility_from_ground_track(
    track: GroundTrackHistory,
    station: GroundStation,
) -> VisibilityHistory:
    """Create topocentric visibility history from an Earth-fixed state history."""
    azimuth, elevation, range_km, range_rate = station_topocentric_from_itrs(
        track.positions_itrs_km,
        track.velocities_itrs_km_s,
        station,
    )
    return VisibilityHistory(
        model_name=track.model_name,
        station=station,
        epoch_utc=track.epoch_utc,
        elapsed_seconds=track.elapsed_seconds,
        timestamps_utc=track.timestamps_utc,
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        range_km=range_km,
        range_rate_km_s=range_rate,
    )


def _timestamp(epoch_utc: str, elapsed_seconds: float) -> str:
    from datetime import timedelta

    return format_utc_timestamp(
        parse_utc_timestamp(epoch_utc) + timedelta(seconds=float(elapsed_seconds))
    )


def _root_between(
    interpolator: PchipInterpolator,
    start: float,
    end: float,
    threshold: float,
    tolerance_seconds: float,
) -> float:
    function = lambda value: float(interpolator(value) - threshold)
    f_start = function(start)
    f_end = function(end)
    if abs(f_start) <= 1.0e-12:
        return float(start)
    if abs(f_end) <= 1.0e-12:
        return float(end)
    if f_start * f_end > 0.0:
        raise ValueError("Pass-boundary interval does not bracket the elevation threshold.")
    return float(
        brentq(
            function,
            float(start),
            float(end),
            xtol=min(float(tolerance_seconds), 1.0e-6),
            rtol=4.0 * np.finfo(float).eps,
            maxiter=100,
        )
    )


def _bounded_extreme(
    interpolator: PchipInterpolator,
    start: float,
    end: float,
    *,
    maximum: bool,
    tolerance_seconds: float,
) -> tuple[float, float]:
    sign = -1.0 if maximum else 1.0
    result = minimize_scalar(
        lambda value: sign * float(interpolator(value)),
        bounds=(float(start), float(end)),
        method="bounded",
        options={"xatol": float(tolerance_seconds)},
    )
    candidates = [float(start), float(end), float(result.x)]
    values = [float(interpolator(value)) for value in candidates]
    index = int(np.argmax(values) if maximum else np.argmin(values))
    return candidates[index], values[index]


def detect_passes(
    visibility: VisibilityHistory,
    *,
    refinement_tolerance_seconds: float,
    calculate_closest_range: bool = True,
) -> list[GroundStationPass]:
    """Detect and refine all geometric passes above the station elevation mask."""
    tolerance = float(refinement_tolerance_seconds)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Pass refinement tolerance must be positive and finite.")

    time = visibility.elapsed_seconds
    elevation = visibility.elevation_deg
    threshold = float(visibility.station.minimum_elevation_deg)
    visible = elevation >= threshold

    starts = np.where(visible & np.concatenate(([True], ~visible[:-1])))[0]
    ends = np.where(visible & np.concatenate((~visible[1:], [True])))[0]
    if starts.size != ends.size:
        raise RuntimeError("Internal pass-group detection produced mismatched boundaries.")

    elevation_interp = PchipInterpolator(time, elevation, extrapolate=False)
    range_interp = PchipInterpolator(time, visibility.range_km, extrapolate=False)
    azimuth_unwrapped = np.unwrap(np.radians(visibility.azimuth_deg))
    azimuth_interp = PchipInterpolator(time, azimuth_unwrapped, extrapolate=False)

    passes: list[GroundStationPass] = []
    for sequence, (start_index, end_index) in enumerate(zip(starts, ends), start=1):
        partial_start = int(start_index) == 0
        partial_end = int(end_index) == time.size - 1

        if partial_start:
            aos = float(time[0])
        else:
            aos = _root_between(
                elevation_interp,
                float(time[start_index - 1]),
                float(time[start_index]),
                threshold,
                tolerance,
            )

        if partial_end:
            los = float(time[-1])
        else:
            los = _root_between(
                elevation_interp,
                float(time[end_index]),
                float(time[end_index + 1]),
                threshold,
                tolerance,
            )

        max_time, max_elevation = _bounded_extreme(
            elevation_interp,
            aos,
            los,
            maximum=True,
            tolerance_seconds=tolerance,
        )
        if calculate_closest_range:
            closest_time, closest_range = _bounded_extreme(
                range_interp,
                aos,
                los,
                maximum=False,
                tolerance_seconds=tolerance,
            )
        else:
            closest_time = max_time
            closest_range = float(range_interp(max_time))

        azimuth = lambda value: float(np.degrees(azimuth_interp(value)) % 360.0)
        pass_id = (
            f"{visibility.station.station_id}-{visibility.model_name.upper()}-"
            f"P{sequence:03d}"
        )
        aos_residual = float(elevation_interp(aos) - threshold)
        los_residual = float(elevation_interp(los) - threshold)
        passes.append(
            GroundStationPass(
                pass_id=pass_id,
                model_name=visibility.model_name,
                station_id=visibility.station.station_id,
                station_name=visibility.station.name,
                minimum_elevation_deg=threshold,
                aos_elapsed_seconds=aos,
                aos_utc=_timestamp(visibility.epoch_utc, aos),
                maximum_elevation_elapsed_seconds=max_time,
                maximum_elevation_utc=_timestamp(visibility.epoch_utc, max_time),
                los_elapsed_seconds=los,
                los_utc=_timestamp(visibility.epoch_utc, los),
                duration_seconds=float(los - aos),
                maximum_elevation_deg=max_elevation,
                azimuth_at_aos_deg=azimuth(aos),
                azimuth_at_maximum_deg=azimuth(max_time),
                azimuth_at_los_deg=azimuth(los),
                range_at_maximum_elevation_km=float(range_interp(max_time)),
                closest_range_km=closest_range,
                closest_range_elapsed_seconds=closest_time,
                closest_range_utc=_timestamp(visibility.epoch_utc, closest_time),
                partial_at_start=partial_start,
                partial_at_end=partial_end,
                aos_threshold_residual_deg=aos_residual,
                los_threshold_residual_deg=los_residual,
                refinement_method="PCHIP elevation with Brent root; bounded scalar extrema",
            )
        )
    return passes


def match_passes(
    reference_passes: list[GroundStationPass],
    comparison_passes: list[GroundStationPass],
    *,
    maximum_time_difference_seconds: float,
) -> list[dict[str, Any]]:
    """Greedily match passes by closest maximum-elevation epoch."""
    window = float(maximum_time_difference_seconds)
    if not np.isfinite(window) or window <= 0.0:
        raise ValueError("Pass-matching time window must be positive and finite.")

    candidates: list[tuple[float, int, int]] = []
    for reference_index, reference in enumerate(reference_passes):
        for comparison_index, comparison in enumerate(comparison_passes):
            difference = abs(
                comparison.maximum_elevation_elapsed_seconds
                - reference.maximum_elevation_elapsed_seconds
            )
            if difference <= window:
                candidates.append((difference, reference_index, comparison_index))
    candidates.sort(key=lambda item: item[0])

    matched_reference: set[int] = set()
    matched_comparison: set[int] = set()
    pairs: dict[int, int] = {}
    for _, reference_index, comparison_index in candidates:
        if reference_index in matched_reference or comparison_index in matched_comparison:
            continue
        matched_reference.add(reference_index)
        matched_comparison.add(comparison_index)
        pairs[reference_index] = comparison_index

    rows: list[dict[str, Any]] = []
    comparison_model = (
        comparison_passes[0].model_name if comparison_passes else "unknown"
    )
    for reference_index, reference in enumerate(reference_passes):
        if reference_index not in pairs:
            rows.append(
                {
                    "match_status": "reference_unmatched",
                    "reference_model": reference.model_name,
                    "comparison_model": comparison_model,
                    "reference_pass_id": reference.pass_id,
                    "comparison_pass_id": None,
                    "reference_aos_utc": reference.aos_utc,
                    "comparison_aos_utc": None,
                    "aos_difference_seconds": None,
                    "maximum_time_difference_seconds": None,
                    "los_difference_seconds": None,
                    "duration_difference_seconds": None,
                    "maximum_elevation_difference_deg": None,
                    "closest_range_difference_km": None,
                }
            )
            continue
        comparison = comparison_passes[pairs[reference_index]]
        rows.append(
            {
                "match_status": "matched",
                "reference_model": reference.model_name,
                "comparison_model": comparison.model_name,
                "reference_pass_id": reference.pass_id,
                "comparison_pass_id": comparison.pass_id,
                "reference_aos_utc": reference.aos_utc,
                "comparison_aos_utc": comparison.aos_utc,
                "aos_difference_seconds": comparison.aos_elapsed_seconds
                - reference.aos_elapsed_seconds,
                "maximum_time_difference_seconds": comparison.maximum_elevation_elapsed_seconds
                - reference.maximum_elevation_elapsed_seconds,
                "los_difference_seconds": comparison.los_elapsed_seconds
                - reference.los_elapsed_seconds,
                "duration_difference_seconds": comparison.duration_seconds
                - reference.duration_seconds,
                "maximum_elevation_difference_deg": comparison.maximum_elevation_deg
                - reference.maximum_elevation_deg,
                "closest_range_difference_km": comparison.closest_range_km
                - reference.closest_range_km,
            }
        )

    for comparison_index, comparison in enumerate(comparison_passes):
        if comparison_index in matched_comparison:
            continue
        rows.append(
            {
                "match_status": "comparison_unmatched",
                "reference_model": (
                    reference_passes[0].model_name if reference_passes else "unknown"
                ),
                "comparison_model": comparison.model_name,
                "reference_pass_id": None,
                "comparison_pass_id": comparison.pass_id,
                "reference_aos_utc": None,
                "comparison_aos_utc": comparison.aos_utc,
                "aos_difference_seconds": None,
                "maximum_time_difference_seconds": None,
                "los_difference_seconds": None,
                "duration_difference_seconds": None,
                "maximum_elevation_difference_deg": None,
                "closest_range_difference_km": None,
            }
        )
    return rows


def pass_analysis_summary(
    station: GroundStation,
    passes_by_model: dict[str, list[GroundStationPass]],
    comparisons_by_model: dict[str, list[dict[str, Any]]],
    *,
    coarse_step_seconds: float,
    refinement_tolerance_seconds: float,
) -> dict[str, Any]:
    """Create compact pass and timing-comparison statistics."""
    model_summaries: dict[str, Any] = {}
    for model, passes in passes_by_model.items():
        if passes:
            model_summaries[model] = {
                "pass_count": len(passes),
                "total_visible_time_seconds": float(
                    sum(item.duration_seconds for item in passes)
                ),
                "maximum_elevation_deg": float(
                    max(item.maximum_elevation_deg for item in passes)
                ),
                "minimum_closest_range_km": float(
                    min(item.closest_range_km for item in passes)
                ),
                "partial_pass_count": sum(
                    int(item.partial_at_start or item.partial_at_end)
                    for item in passes
                ),
            }
        else:
            model_summaries[model] = {
                "pass_count": 0,
                "total_visible_time_seconds": 0.0,
                "maximum_elevation_deg": None,
                "minimum_closest_range_km": None,
                "partial_pass_count": 0,
            }

    comparison_summaries: dict[str, Any] = {}
    for model, rows in comparisons_by_model.items():
        matched = [row for row in rows if row["match_status"] == "matched"]
        reference_unmatched = sum(
            row["match_status"] == "reference_unmatched" for row in rows
        )
        comparison_unmatched = sum(
            row["match_status"] == "comparison_unmatched" for row in rows
        )
        if matched:
            aos = np.asarray([row["aos_difference_seconds"] for row in matched], dtype=float)
            maximum = np.asarray(
                [row["maximum_time_difference_seconds"] for row in matched],
                dtype=float,
            )
            los = np.asarray([row["los_difference_seconds"] for row in matched], dtype=float)
            comparison_summaries[model] = {
                "matched_pass_count": len(matched),
                "reference_unmatched_count": int(reference_unmatched),
                "comparison_unmatched_count": int(comparison_unmatched),
                "maximum_absolute_aos_difference_seconds": float(np.max(np.abs(aos))),
                "maximum_absolute_maximum_time_difference_seconds": float(
                    np.max(np.abs(maximum))
                ),
                "maximum_absolute_los_difference_seconds": float(np.max(np.abs(los))),
                "rms_aos_difference_seconds": float(np.sqrt(np.mean(aos**2))),
                "rms_los_difference_seconds": float(np.sqrt(np.mean(los**2))),
            }
        else:
            comparison_summaries[model] = {
                "matched_pass_count": 0,
                "reference_unmatched_count": int(reference_unmatched),
                "comparison_unmatched_count": int(comparison_unmatched),
                "maximum_absolute_aos_difference_seconds": None,
                "maximum_absolute_maximum_time_difference_seconds": None,
                "maximum_absolute_los_difference_seconds": None,
                "rms_aos_difference_seconds": None,
                "rms_los_difference_seconds": None,
            }

    return {
        "station": {
            "station_id": station.station_id,
            "name": station.name,
            "latitude_deg": station.latitude_deg,
            "longitude_deg": station.longitude_deg,
            "altitude_m": station.altitude_m,
            "minimum_elevation_deg": station.minimum_elevation_deg,
        },
        "visibility_type": "geometric line of sight above elevation mask",
        "terrain_mask_included": False,
        "atmospheric_refraction_included": False,
        "coarse_step_seconds": float(coarse_step_seconds),
        "refinement_tolerance_seconds": float(refinement_tolerance_seconds),
        "models": model_summaries,
        "comparisons_against_sgp4": comparison_summaries,
    }
