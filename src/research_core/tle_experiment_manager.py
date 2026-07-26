"""Integrated fixed-TLE research pipeline for Research Core 1A.7."""

from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import BUILD_MARKER, RESEARCH_CORE_VERSION
from .analysis.comparison import compare_state_histories
from .analysis.j2 import compare_in_reference_rtn
from .analysis.sgp4 import create_sgp4_model_summary, initial_state_differences_m, tle_age_report
from .configuration import load_and_validate_config
from .data_models import CartesianState, ClassicalElements, StateHistory
from .frames import frame_roundtrip_error
from .ground_track import (
    compare_ground_tracks,
    geodetic_roundtrip_error,
    gcrs_state_history_to_ground_track,
    ground_track_summary,
)
from .ground_track_outputs import (
    create_ground_track_figures,
    write_ground_track_comparison_csv,
    write_ground_track_csv,
    write_ground_track_summary_csv,
    write_ground_track_technical_summary,
)
from .ground_station import (
    GroundStation,
    detect_passes,
    match_passes,
    pass_analysis_summary,
    visibility_from_ground_track,
)
from .logging_utils import close_run_logger, create_run_logger
from .metadata import collect_environment_metadata, utc_now_iso, write_json
from .pass_outputs import (
    create_pass_figures,
    write_pass_comparison_csv,
    write_pass_summary_csv,
    write_pass_technical_summary,
    write_passes_csv,
    write_visibility_history_csv,
)
from .orbital_elements import cartesian_to_elements
from .research_report import (
    write_final_validation_summary,
    write_run_manifest,
    write_tle_research_report,
)
from .outputs import write_comparison_csv, write_rtn_comparison_csv, write_state_history_csv
from .propagators import (
    propagate_analytical_two_body,
    propagate_numerical_j2,
    propagate_numerical_j2_drag,
    propagate_numerical_two_body,
    propagate_sgp4_frozen_tle,
)
from .sgp4_outputs import (
    create_sgp4_figures,
    write_sgp4_model_summary_csv,
    write_technical_summary,
    write_tle_age_csv,
)
from .time_utils import build_time_grid, parse_utc_timestamp
from .tle import load_frozen_tle, tle_parameter_summary


@dataclass(frozen=True)
class TLEExperimentRunResult:
    experiment_id: str
    result_directory: Path
    warnings: tuple[str, ...]
    created_files: tuple[Path, ...]
    validation_status: str
    tle_epoch_utc: str
    end_tle_age_hours: float
    frame_roundtrip_position_error_m: float
    maximum_separation_km_by_model: dict[str, float]
    final_separation_km_by_model: dict[str, float]
    nonzero_sgp4_error_count: int
    maximum_ground_track_separation_km_by_model: dict[str, float]
    final_ground_track_separation_km_by_model: dict[str, float]
    geodetic_roundtrip_position_error_m: float
    pass_station_id: str
    pass_station_name: str
    pass_minimum_elevation_deg: float
    pass_count_by_model: dict[str, int]
    matched_pass_count_by_model: dict[str, int]
    maximum_absolute_aos_difference_seconds_by_model: dict[str, float | None]
    maximum_absolute_los_difference_seconds_by_model: dict[str, float | None]
    research_report_path: Path
    run_manifest_path: Path


def _safe(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not cleaned:
        raise ValueError("Experiment ID cannot be converted into a safe folder name.")
    return cleaned


def _run_directory(results_root: Path, experiment_id: str) -> Path:
    parent = results_root / _safe(experiment_id)
    for _ in range(10):
        candidate = parent / datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("Could not create a unique TLE result directory.")


def _resolve(path_value: str, config_path: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _classical_from_state(state: CartesianState, mu: float) -> ClassicalElements:
    values = cartesian_to_elements(state.position_km, state.velocity_km_s, mu)
    required = ("raan_deg", "argument_of_perigee_deg", "true_anomaly_deg")
    if any(values[name] is None for name in required):
        raise ValueError("The TLE-derived common state is singular in classical elements.")
    return ClassicalElements(
        semi_major_axis_km=float(values["semi_major_axis_km"]),
        eccentricity=float(values["eccentricity"]),
        inclination_rad=math.radians(float(values["inclination_deg"])),
        raan_rad=math.radians(float(values["raan_deg"])),
        argument_of_perigee_rad=math.radians(float(values["argument_of_perigee_deg"])),
        true_anomaly_rad=math.radians(float(values["true_anomaly_deg"])),
    )


def _validation(
    *,
    experiment_id: str,
    warnings: list[str],
    provenance: dict[str, Any],
    sgp4_diagnostics: dict[str, Any],
    roundtrip: dict[str, Any],
    initial_differences: dict[str, Any],
    comparisons: list[dict[str, Any]],
    age: dict[str, Any],
    ground_tracks: list[Any],
    ground_roundtrip: dict[str, Any],
    ground_comparisons: list[dict[str, Any]],
    pass_sgp4_diagnostics: dict[str, Any],
    stations: list[GroundStation],
    visibility_by_station: dict[str, dict[str, Any]],
    passes_by_station: dict[str, dict[str, list[Any]]],
    pass_comparisons_by_station: dict[str, dict[str, list[dict[str, Any]]]],
    pass_summaries: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(identifier: str, name: str, measured: Any, criterion: str, passed: bool) -> None:
        checks.append({
            "validation_id": identifier,
            "name": name,
            "measured_value": measured,
            "criterion": criterion,
            "status": "passed" if passed else "failed",
        })

    add("VAL-TLE-001", "TLE line 1 checksum", provenance["tle_line1_checksum_valid"], "must be true", bool(provenance["tle_line1_checksum_valid"]))
    add("VAL-TLE-002", "TLE line 2 checksum", provenance["tle_line2_checksum_valid"], "must be true", bool(provenance["tle_line2_checksum_valid"]))
    expected_catalog = int(config["initial_state"]["norad_catalog_number"])
    add("VAL-TLE-003", "NORAD catalog identity", provenance["norad_catalog_number"], f"must equal {expected_catalog}", int(provenance["norad_catalog_number"]) == expected_catalog)
    config_epoch = config["initial_state"]["epoch_utc"]
    epoch_difference = abs((parse_utc_timestamp(config_epoch) - parse_utc_timestamp(provenance["tle_epoch_utc"])).total_seconds())
    add("VAL-TLE-004", "Configuration epoch matches TLE epoch", epoch_difference, "<= 1e-6 s", epoch_difference <= 1.0e-6)
    add("VAL-SGP4-001", "SGP4 nonzero error codes", sgp4_diagnostics["nonzero_error_count"], "must equal 0", int(sgp4_diagnostics["nonzero_error_count"]) == 0)
    add("VAL-FRAME-001", "TEME/GCRS roundtrip position", roundtrip["position_roundtrip_error_m"], "<= 1e-6 m", float(roundtrip["position_roundtrip_error_m"]) <= 1.0e-6)
    add("VAL-FRAME-002", "TEME/GCRS roundtrip velocity", roundtrip["velocity_roundtrip_error_mm_s"], "<= 1e-5 mm/s", float(roundtrip["velocity_roundtrip_error_mm_s"]) <= 1.0e-5)
    for model, values in initial_differences.items():
        position_limit = 1.0e-3 if model == "analytical_two_body" else 1.0e-9
        velocity_limit = 1.0e-3 if model == "analytical_two_body" else 1.0e-9
        add(f"VAL-COMMON-{model}-R", f"{model} common initial position", values["position_difference_m"], f"<= {position_limit} m", float(values["position_difference_m"]) <= position_limit)
        add(f"VAL-COMMON-{model}-V", f"{model} common initial velocity", values["velocity_difference_mm_s"], f"<= {velocity_limit} mm/s", float(values["velocity_difference_mm_s"]) <= velocity_limit)
    finite = all(np.all(np.isfinite(item["position_difference_m"])) and np.all(np.isfinite(item["velocity_difference_mm_s"])) for item in comparisons)
    add("VAL-SGP4-002", "All model separations finite", finite, "must be true", finite)
    add("VAL-TLE-AGE-001", "TLE start age", age["start_age_hours"], "absolute value <= 1e-9 h", abs(float(age["start_age_hours"])) <= 1.0e-9)
    expected_end = float(config["propagation"]["default_duration_hours"])
    add("VAL-TLE-AGE-002", "TLE end age", age["end_age_hours"], f"must equal {expected_end} h", abs(float(age["end_age_hours"]) - expected_end) <= 1.0e-9)
    finite_ground = all(
        np.all(np.isfinite(track.latitude_deg))
        and np.all(np.isfinite(track.longitude_deg))
        and np.all(np.isfinite(track.altitude_km))
        for track in ground_tracks
    )
    add("VAL-GT-001", "Ground-track coordinates finite", finite_ground, "must be true", finite_ground)
    latitude_valid = all(
        np.all((track.latitude_deg >= -90.0) & (track.latitude_deg <= 90.0))
        for track in ground_tracks
    )
    longitude_valid = all(
        np.all((track.longitude_deg >= -180.0) & (track.longitude_deg <= 180.0))
        for track in ground_tracks
    )
    add("VAL-GT-002", "Geodetic latitude range", latitude_valid, "all values in [-90, 90] deg", latitude_valid)
    add("VAL-GT-003", "Wrapped longitude range", longitude_valid, "all values in [-180, 180] deg", longitude_valid)
    altitude_plausible = all(
        np.all((track.altitude_km > 100.0) & (track.altitude_km < 2000.0))
        for track in ground_tracks
    )
    add("VAL-GT-004", "LEO geodetic altitude range", altitude_plausible, "all values in (100, 2000) km", altitude_plausible)
    add(
        "VAL-GT-005",
        "WGS-84 geodetic/ITRS reconstruction",
        ground_roundtrip["maximum_position_residual_m"],
        "<= 1e-6 m",
        float(ground_roundtrip["maximum_position_residual_m"]) <= 1.0e-6,
    )
    ground_comparisons_finite = all(
        np.all(np.isfinite(item["surface_separation_km"]))
        and np.all(np.isfinite(item["altitude_difference_km"]))
        for item in ground_comparisons
    )
    add("VAL-GT-006", "Ground-track model comparisons finite", ground_comparisons_finite, "must be true", ground_comparisons_finite)
    initial_ground_equality = all(
        float(item["surface_separation_km"][0]) <= 1.0e-6
        and abs(float(item["altitude_difference_km"][0])) <= 1.0e-6
        for item in ground_comparisons
    )
    add("VAL-GT-007", "Common initial sub-satellite point", initial_ground_equality, "surface and altitude differences <= 1e-6 km", initial_ground_equality)

    add(
        "VAL-PASS-SGP4-001",
        "Pass-grid SGP4 nonzero error codes",
        pass_sgp4_diagnostics["nonzero_error_count"],
        "must equal 0",
        int(pass_sgp4_diagnostics["nonzero_error_count"]) == 0,
    )
    pass_config = config["pass_analysis"]
    expected_coarse_step = float(pass_config["coarse_step_seconds"])
    for station in stations:
        station_id = station.station_id
        visibility_map = visibility_by_station[station_id]
        pass_map = passes_by_station[station_id]
        comparison_map = pass_comparisons_by_station[station_id]
        station_roundtrip = station.station_roundtrip_error_m()
        add(
            f"VAL-PASS-{station_id}-001",
            f"{station.name} station WGS-84/ITRS roundtrip",
            station_roundtrip,
            "<= 1e-6 m",
            station_roundtrip <= 1.0e-6,
        )
        visibility_finite = all(
            np.all(np.isfinite(item.azimuth_deg))
            and np.all(np.isfinite(item.elevation_deg))
            and np.all(np.isfinite(item.range_km))
            and np.all(np.isfinite(item.range_rate_km_s))
            for item in visibility_map.values()
        )
        add(
            f"VAL-PASS-{station_id}-002",
            f"{station.name} visibility histories finite",
            visibility_finite,
            "must be true",
            visibility_finite,
        )
        angular_ranges = all(
            np.all((item.azimuth_deg >= 0.0) & (item.azimuth_deg < 360.0))
            and np.all((item.elevation_deg >= -90.0) & (item.elevation_deg <= 90.0))
            and np.all(item.range_km > 0.0)
            for item in visibility_map.values()
        )
        add(
            f"VAL-PASS-{station_id}-003",
            f"{station.name} topocentric coordinate ranges",
            angular_ranges,
            "azimuth [0,360), elevation [-90,90], range > 0",
            angular_ranges,
        )
        reference_visibility = visibility_map["sgp4"]
        common_initial = all(
            abs(float(item.azimuth_deg[0] - reference_visibility.azimuth_deg[0])) <= 1.0e-8
            and abs(float(item.elevation_deg[0] - reference_visibility.elevation_deg[0])) <= 1.0e-8
            and abs(float(item.range_km[0] - reference_visibility.range_km[0])) <= 1.0e-8
            for item in visibility_map.values()
        )
        add(
            f"VAL-PASS-{station_id}-004",
            f"{station.name} common initial topocentric geometry",
            common_initial,
            "azimuth, elevation and range agree within 1e-8",
            common_initial,
        )
        all_passes = [item for values in pass_map.values() for item in values]
        ordered = all(
            item.aos_elapsed_seconds <= item.maximum_elevation_elapsed_seconds <= item.los_elapsed_seconds
            and item.duration_seconds >= 0.0
            for item in all_passes
        )
        add(
            f"VAL-PASS-{station_id}-005",
            f"{station.name} pass-event ordering",
            ordered,
            "AOS <= maximum elevation <= LOS",
            ordered,
        )
        threshold_residuals = [
            abs(item.aos_threshold_residual_deg)
            for item in all_passes
            if not item.partial_at_start
        ] + [
            abs(item.los_threshold_residual_deg)
            for item in all_passes
            if not item.partial_at_end
        ]
        max_threshold_residual = max(threshold_residuals, default=0.0)
        add(
            f"VAL-PASS-{station_id}-006",
            f"{station.name} refined AOS/LOS threshold residual",
            max_threshold_residual,
            "<= 1e-6 deg",
            max_threshold_residual <= 1.0e-6,
        )
        maxima_valid = all(
            item.maximum_elevation_deg + 1.0e-9 >= station.minimum_elevation_deg
            and item.closest_range_km > 0.0
            for item in all_passes
        )
        add(
            f"VAL-PASS-{station_id}-007",
            f"{station.name} pass maxima and ranges",
            maxima_valid,
            "maximum elevation >= mask and range > 0",
            maxima_valid,
        )
        comparison_values = [
            value
            for rows in comparison_map.values()
            for row in rows
            if row["match_status"] == "matched"
            for key, value in row.items()
            if key.endswith("difference_seconds") or key.endswith("difference_deg") or key.endswith("difference_km")
        ]
        comparisons_finite = all(value is not None and np.isfinite(float(value)) for value in comparison_values)
        add(
            f"VAL-PASS-{station_id}-008",
            f"{station.name} matched-pass differences finite",
            comparisons_finite,
            "all matched differences finite",
            comparisons_finite,
        )
        time_grid = reference_visibility.elapsed_seconds
        actual_steps = np.diff(time_grid)
        coarse_grid_valid = bool(
            actual_steps.size > 0
            and np.all(actual_steps > 0.0)
            and np.all(actual_steps <= expected_coarse_step + 1.0e-9)
        )
        add(
            f"VAL-PASS-{station_id}-009",
            f"{station.name} coarse pass-search time grid",
            float(np.max(actual_steps)) if actual_steps.size else None,
            f"maximum step <= {expected_coarse_step} s",
            coarse_grid_valid,
        )
        if pass_summaries[station_id]["models"]["sgp4"]["pass_count"] == 0:
            warnings.append(
                f"No SGP4 pass above the elevation mask was found for {station.name} in this run window."
            )

    failed = [check for check in checks if check["status"] == "failed"]
    overall = "failed" if failed else ("passed_with_warnings" if warnings else "passed")
    return {
        "research_core_version": RESEARCH_CORE_VERSION,
        "experiment_id": experiment_id,
        "created_utc": utc_now_iso(),
        "overall_status": overall,
        "stage": "Research Core 1A.7 complete research pipeline and HTML report",
        "checks": checks,
        "warnings": warnings,
        "failed_check_count": len(failed),
        "scientific_interpretation": "State, ground-track and pass-time differences from SGP4 are model separations, not measured orbit errors.",
    }


def run_tle_experiment(config_path: str | Path, *, project_root: str | Path) -> TLEExperimentRunResult:
    source = Path(config_path).expanduser().resolve()
    config, warnings = load_and_validate_config(source)
    if config["initial_state"]["source_type"] != "fixed_tle":
        raise ValueError("run_tle_experiment requires initial_state.source_type='fixed_tle'.")

    initial = config["initial_state"]
    tle = load_frozen_tle(
        _resolve(initial["tle_file"], source),
        _resolve(initial["tle_metadata_file"], source),
        expected_catalog_number=int(initial["norad_catalog_number"]),
    )
    if config["initial_state"]["epoch_utc"] != tle.epoch_utc:
        raise ValueError(
            f"Configuration epoch {config['initial_state']['epoch_utc']} must exactly match frozen TLE epoch {tle.epoch_utc}."
        )

    outputs = config["outputs"]
    results_root = Path(outputs["results_root"]).expanduser()
    if not results_root.is_absolute():
        results_root = Path(project_root).resolve() / results_root
    result_directory = _run_directory(results_root.resolve(), config["experiment"]["experiment_id"])
    log_path = result_directory / "run_log.txt"
    logger = create_run_logger(log_path)
    created: list[Path] = [log_path]
    started_utc = utc_now_iso()

    try:
        logger.info("Research Core version: %s", RESEARCH_CORE_VERSION)
        logger.info("Build marker: %s", BUILD_MARKER)
        logger.info("Fixed TLE: %s", tle.file_path)
        logger.info("TLE epoch: %s", tle.epoch_utc)
        for warning in warnings:
            logger.warning("%s", warning)

        duration = float(config["propagation"]["default_duration_hours"])
        step = float(config["propagation"]["output_step_seconds"])
        elapsed = build_time_grid(duration, step)
        logger.info("Propagating SGP4 for %.3f h at %.3f s output spacing.", duration, step)
        sgp4_teme, sgp4_gcrs, sgp4_diagnostics = propagate_sgp4_frozen_tle(tle, elapsed)
        logger.info("SGP4 propagation and TEME→GCRS transformation completed.")

        common_state = CartesianState(
            epoch_utc=tle.epoch_utc,
            frame=sgp4_gcrs.frame,
            position_km=sgp4_gcrs.positions_km[0],
            velocity_km_s=sgp4_gcrs.velocities_km_s[0],
        )
        earth = config["earth_model"]
        mu = float(earth["gravitational_parameter_km3_s2"])
        elements = _classical_from_state(common_state, mu)
        integrator = config["integrator"]
        kwargs = {
            "method": integrator["method"],
            "relative_tolerance": float(integrator["relative_tolerance"]),
            "absolute_tolerance": float(integrator["absolute_tolerance"]),
            "maximum_step_seconds": float(integrator["maximum_step_seconds"]),
        }

        analytical = propagate_analytical_two_body(
            elements, mu, elapsed, epoch_utc=tle.epoch_utc, frame=common_state.frame
        )
        numerical_two_body = propagate_numerical_two_body(common_state, mu, elapsed, **kwargs)
        numerical_j2 = propagate_numerical_j2(
            common_state, mu, float(earth["equatorial_radius_km"]), float(earth["j2"]), elapsed, **kwargs
        )
        drag = config["drag"]
        numerical_drag = propagate_numerical_j2_drag(
            common_state,
            mu,
            float(earth["equatorial_radius_km"]),
            float(earth["j2"]),
            float(earth["earth_rotation_rate_rad_s"]),
            elapsed,
            mass_kg=float(drag["mass_kg"]),
            cross_sectional_area_m2=float(drag["cross_sectional_area_m2"]),
            drag_coefficient=float(drag["drag_coefficient"]),
            reference_altitude_km=float(drag["reference_altitude_km"]),
            reference_density_kg_m3=float(drag["reference_density_kg_m3"]),
            scale_height_km=float(drag["scale_height_km"]),
            co_rotating_atmosphere=bool(drag["co_rotating_atmosphere"]),
            **kwargs,
        )
        comparison_histories = [analytical, numerical_two_body, numerical_j2, numerical_drag]
        comparisons = [compare_state_histories(sgp4_gcrs, history) for history in comparison_histories]
        rtn = [compare_in_reference_rtn(sgp4_gcrs, history) for history in comparison_histories]
        summary = create_sgp4_model_summary(comparisons, [sgp4_gcrs, *comparison_histories])
        initial_differences = initial_state_differences_m(sgp4_gcrs, comparison_histories)
        age = tle_age_report(tle.epoch_utc, sgp4_gcrs.timestamps_utc, elapsed)
        roundtrip = frame_roundtrip_error(
            sgp4_teme.positions_km[0], sgp4_teme.velocities_km_s[0], tle.epoch_utc
        )
        provenance = tle_parameter_summary(tle)

        ground_config = config.get("ground_track_analysis", {})
        ellipsoid = str(ground_config.get("earth_ellipsoid", "WGS84"))
        surface_radius_km = float(
            ground_config.get(
                "surface_radius_km", earth["equatorial_radius_km"]
            )
        )
        ground_tracks = [
            gcrs_state_history_to_ground_track(history, ellipsoid=ellipsoid)
            for history in [sgp4_gcrs, *comparison_histories]
        ]
        reference_ground_track = ground_tracks[0]
        ground_comparisons = [
            compare_ground_tracks(
                reference_ground_track,
                track,
                surface_radius_km=surface_radius_km,
            )
            for track in ground_tracks[1:]
        ]
        ground_summary = ground_track_summary(
            reference_ground_track, ground_comparisons
        )
        ground_roundtrip = geodetic_roundtrip_error(reference_ground_track)
        ground_transform_warnings = [
            warning
            for track in ground_tracks
            for warning in track.transform_warnings
        ]
        if ground_transform_warnings:
            warnings.append(
                "Astropy issued GCRS/ITRS or Earth-orientation warnings; see ground_track_diagnostics.json."
            )

        pass_config = config["pass_analysis"]
        if not bool(pass_config["enabled"]):
            raise ValueError("Research Core 1A.6 fixed-TLE runs require pass_analysis.enabled=true.")
        pass_step = float(pass_config["coarse_step_seconds"])
        pass_tolerance = float(pass_config["refinement_tolerance_seconds"])
        pass_elapsed = build_time_grid(duration, pass_step)
        logger.info(
            "Running geometric ground-station pass analysis at %.3f s coarse spacing.",
            pass_step,
        )
        pass_sgp4_teme, pass_sgp4_gcrs, pass_sgp4_diagnostics = propagate_sgp4_frozen_tle(
            tle, pass_elapsed
        )
        pass_analytical = propagate_analytical_two_body(
            elements,
            mu,
            pass_elapsed,
            epoch_utc=tle.epoch_utc,
            frame=common_state.frame,
        )
        pass_numerical_two_body = propagate_numerical_two_body(
            common_state, mu, pass_elapsed, **kwargs
        )
        pass_numerical_j2 = propagate_numerical_j2(
            common_state,
            mu,
            float(earth["equatorial_radius_km"]),
            float(earth["j2"]),
            pass_elapsed,
            **kwargs,
        )
        pass_numerical_drag = propagate_numerical_j2_drag(
            common_state,
            mu,
            float(earth["equatorial_radius_km"]),
            float(earth["j2"]),
            float(earth["earth_rotation_rate_rad_s"]),
            pass_elapsed,
            mass_kg=float(drag["mass_kg"]),
            cross_sectional_area_m2=float(drag["cross_sectional_area_m2"]),
            drag_coefficient=float(drag["drag_coefficient"]),
            reference_altitude_km=float(drag["reference_altitude_km"]),
            reference_density_kg_m3=float(drag["reference_density_kg_m3"]),
            scale_height_km=float(drag["scale_height_km"]),
            co_rotating_atmosphere=bool(drag["co_rotating_atmosphere"]),
            **kwargs,
        )
        pass_histories = [
            pass_sgp4_gcrs,
            pass_analytical,
            pass_numerical_two_body,
            pass_numerical_j2,
            pass_numerical_drag,
        ]
        pass_ground_tracks = [
            gcrs_state_history_to_ground_track(history, ellipsoid=ellipsoid)
            for history in pass_histories
        ]
        station_mappings = {
            str(item["station_id"]): item for item in config["ground_stations"]
        }
        requested_station_ids = pass_config.get("station_ids") or list(station_mappings)
        stations = [
            GroundStation.from_mapping(station_mappings[station_id])
            for station_id in requested_station_ids
        ]
        visibility_by_station: dict[str, dict[str, Any]] = {}
        passes_by_station: dict[str, dict[str, list[Any]]] = {}
        pass_comparisons_by_station: dict[str, dict[str, list[dict[str, Any]]]] = {}
        pass_summaries: dict[str, dict[str, Any]] = {}
        for station in stations:
            visibility_map = {
                track.model_name: visibility_from_ground_track(track, station)
                for track in pass_ground_tracks
            }
            passes_map = {
                model: detect_passes(
                    visibility,
                    refinement_tolerance_seconds=pass_tolerance,
                    calculate_closest_range=bool(pass_config["calculate_closest_range"]),
                )
                for model, visibility in visibility_map.items()
            }
            comparison_map = {
                model: match_passes(
                    passes_map["sgp4"],
                    model_passes,
                    maximum_time_difference_seconds=float(
                        pass_config["match_passes_within_seconds"]
                    ),
                )
                for model, model_passes in passes_map.items()
                if model != "sgp4"
            }
            summary_for_station = pass_analysis_summary(
                station,
                passes_map,
                comparison_map,
                coarse_step_seconds=pass_step,
                refinement_tolerance_seconds=pass_tolerance,
            )
            visibility_by_station[station.station_id] = visibility_map
            passes_by_station[station.station_id] = passes_map
            pass_comparisons_by_station[station.station_id] = comparison_map
            pass_summaries[station.station_id] = summary_for_station
            logger.info(
                "%s SGP4 passes above %.1f deg: %d",
                station.name,
                station.minimum_elevation_deg,
                len(passes_map["sgp4"]),
            )
        pass_transform_warnings = [
            warning
            for track in pass_ground_tracks
            for warning in track.transform_warnings
        ]
        if pass_transform_warnings:
            warnings.append(
                "Astropy issued pass-grid GCRS/ITRS warnings; see pass_analysis_diagnostics.json."
            )
        warnings.append(
            "Ground-station passes are geometric only; terrain, refraction, antenna and link-budget constraints are excluded."
        )

        if sgp4_diagnostics["astropy_transform_warnings"]:
            warnings.append("Astropy issued frame/Earth-orientation warnings; see sgp4_diagnostics.json.")
        warnings.append("The frozen TLE and SGP4 are comparison references, not measured orbit truth.")
        validation = _validation(
            experiment_id=config["experiment"]["experiment_id"],
            warnings=warnings,
            provenance=provenance,
            sgp4_diagnostics=sgp4_diagnostics,
            roundtrip=roundtrip,
            initial_differences=initial_differences,
            comparisons=comparisons,
            age=age,
            ground_tracks=ground_tracks,
            ground_roundtrip=ground_roundtrip,
            ground_comparisons=ground_comparisons,
            pass_sgp4_diagnostics=pass_sgp4_diagnostics,
            stations=stations,
            visibility_by_station=visibility_by_station,
            passes_by_station=passes_by_station,
            pass_comparisons_by_station=pass_comparisons_by_station,
            pass_summaries=pass_summaries,
            config=config,
        )

        resolved = deepcopy(config)
        resolved["_runtime"] = {
            "research_core_version": RESEARCH_CORE_VERSION,
            "build_marker": BUILD_MARKER,
            "run_started_utc": started_utc,
            "configuration_source": str(source),
            "result_directory": str(result_directory),
            "integrated_run": True,
            "implemented_models": ["sgp4", "analytical_two_body", "numerical_two_body", "numerical_j2", "numerical_j2_drag"],
            "comparison_frame": sgp4_gcrs.frame,
            "raw_sgp4_frame": "TEME",
            "earth_fixed_frame": reference_ground_track.earth_fixed_frame,
            "geodetic_ellipsoid": reference_ground_track.ellipsoid,
            "ground_track_analysis_performed": True,
            "ground_station_pass_analysis_performed": True,
            "combined_html_report_generated": True,
            "run_manifest_generated": True,
            "final_validation_summary_generated": True,
            "pass_coarse_step_seconds": pass_step,
            "pass_refinement_tolerance_seconds": pass_tolerance,
            "pass_station_ids": [station.station_id for station in stations],
        }

        files: list[tuple[str, Any]] = []
        for name, data in [
            ("experiment_configuration.json", resolved),
            ("environment_metadata.json", collect_environment_metadata(config_path=source, result_directory=result_directory, run_started_utc=started_utc)),
            ("tle_provenance.json", provenance),
            ("sgp4_diagnostics.json", sgp4_diagnostics),
            ("frame_roundtrip_validation.json", roundtrip),
            ("common_initial_state.json", {**common_state.as_dict(), "source": "SGP4 TEME state transformed to GCRS at TLE epoch"}),
            ("common_initial_state_differences.json", initial_differences),
            ("sgp4_model_error_summary.json", summary),
            ("ground_track_summary.json", ground_summary),
            ("ground_track_roundtrip_validation.json", ground_roundtrip),
            ("ground_track_diagnostics.json", {
                "earth_fixed_frame": reference_ground_track.earth_fixed_frame,
                "ellipsoid": reference_ground_track.ellipsoid,
                "surface_distance_model": "spherical_central_angle",
                "surface_radius_km": surface_radius_km,
                "astropy_transform_warnings": ground_transform_warnings,
                "map_background_used_for_calculation": False,
            }),
            ("pass_analysis_summary.json", {
                "reference_model": "sgp4",
                "stations": pass_summaries,
            }),
            ("pass_analysis_diagnostics.json", {
                "pass_grid_sample_count": int(pass_elapsed.size),
                "coarse_step_seconds": pass_step,
                "refinement_tolerance_seconds": pass_tolerance,
                "refinement_method": "PCHIP elevation with Brent root; bounded scalar extrema",
                "sgp4_nonzero_error_count": pass_sgp4_diagnostics["nonzero_error_count"],
                "astropy_transform_warnings": pass_transform_warnings,
                "visibility_type": "geometric line of sight above elevation mask",
                "terrain_mask_included": False,
                "atmospheric_refraction_included": False,
                "antenna_constraints_included": False,
                "link_budget_included": False,
            }),
            ("validation_status.json", validation),
        ]:
            path = result_directory / name
            write_json(data, path)
            created.append(path)

        age_path = result_directory / "tle_age_report.csv"
        write_tle_age_csv(age_path, age)
        created.append(age_path)
        summary_csv = result_directory / "sgp4_model_error_summary.csv"
        write_sgp4_model_summary_csv(summary_csv, summary)
        created.append(summary_csv)

        histories = [sgp4_teme, sgp4_gcrs, *comparison_histories]
        for history in histories:
            path = result_directory / f"{history.model_name}_states.csv"
            write_state_history_csv(path, history)
            created.append(path)
        for comparison in comparisons:
            model = comparison["comparison_model"]
            path = result_directory / f"sgp4_vs_{model}.csv"
            write_comparison_csv(path, comparison)
            created.append(path)
        for item in rtn:
            model = item["comparison_model"]
            path = result_directory / f"sgp4_vs_{model}_rtn.csv"
            write_rtn_comparison_csv(path, item)
            created.append(path)

        for track in ground_tracks:
            path = result_directory / f"{track.model_name}_ground_track.csv"
            write_ground_track_csv(path, track)
            created.append(path)
        for comparison in ground_comparisons:
            model = comparison["comparison_model"]
            path = result_directory / f"sgp4_vs_{model}_ground_track.csv"
            write_ground_track_comparison_csv(path, comparison)
            created.append(path)
        ground_summary_csv = result_directory / "ground_track_summary.csv"
        write_ground_track_summary_csv(ground_summary_csv, ground_summary)
        created.append(ground_summary_csv)

        for station in stations:
            station_stem = _safe(station.station_id).lower().replace("-", "_")
            visibility_map = visibility_by_station[station.station_id]
            passes_map = passes_by_station[station.station_id]
            comparison_map = pass_comparisons_by_station[station.station_id]
            for model, visibility in visibility_map.items():
                visibility_path = result_directory / f"{station_stem}_{model}_visibility.csv"
                write_visibility_history_csv(visibility_path, visibility)
                created.append(visibility_path)
            for model, model_passes in passes_map.items():
                passes_path = result_directory / f"{station_stem}_{model}_passes.csv"
                write_passes_csv(passes_path, model_passes)
                created.append(passes_path)
            for model, rows in comparison_map.items():
                comparison_path = result_directory / f"{station_stem}_sgp4_vs_{model}_passes.csv"
                write_pass_comparison_csv(comparison_path, rows)
                created.append(comparison_path)
            pass_summary_csv = result_directory / f"{station_stem}_pass_summary.csv"
            write_pass_summary_csv(pass_summary_csv, pass_summaries[station.station_id])
            created.append(pass_summary_csv)

        figure_paths = create_sgp4_figures(
            result_directory / "figures", sgp4_gcrs, comparisons, rtn, age,
            save_png=bool(outputs["save_png"]), save_pdf=bool(outputs["save_pdf"]),
        )
        created.extend(figure_paths)

        background_value = ground_config.get("map_background_file")
        background_path = (
            _resolve(str(background_value), source)
            if background_value
            else None
        )
        ground_figure_paths = create_ground_track_figures(
            result_directory / "figures",
            ground_tracks,
            ground_comparisons,
            background_file=background_path,
            save_png=bool(outputs["save_png"]),
            save_pdf=bool(outputs["save_pdf"]),
        )
        created.extend(ground_figure_paths)

        for station in stations:
            pass_figure_paths = create_pass_figures(
                result_directory / "figures",
                visibility_by_station[station.station_id],
                passes_by_station[station.station_id],
                pass_comparisons_by_station[station.station_id],
                save_png=bool(outputs["save_png"]),
                save_pdf=bool(outputs["save_pdf"]),
            )
            created.extend(pass_figure_paths)
            pass_technical = result_directory / f"{_safe(station.station_id).upper()}_PASS_TECHNICAL_SUMMARY.md"
            write_pass_technical_summary(
                pass_technical,
                summary=pass_summaries[station.station_id],
                validation_status=validation,
            )
            created.append(pass_technical)

        ground_technical = result_directory / "GROUND_TRACK_TECHNICAL_SUMMARY.md"
        write_ground_track_technical_summary(
            ground_technical,
            summary=ground_summary,
            roundtrip=ground_roundtrip,
            validation_status=validation,
        )
        created.append(ground_technical)

        technical = result_directory / "SGP4_TECHNICAL_SUMMARY.md"
        write_technical_summary(
            technical,
            provenance=provenance,
            summary=summary,
            initial_differences=initial_differences,
            frame_roundtrip=roundtrip,
            age_report=age,
            validation_status=validation,
        )
        created.append(technical)

        final_validation_path = result_directory / "FINAL_VALIDATION_SUMMARY.json"
        write_final_validation_summary(
            final_validation_path,
            config=config,
            validation=validation,
            warnings=warnings,
        )
        created.append(final_validation_path)

        research_report_path = result_directory / "RESEARCH_REPORT.html"
        write_tle_research_report(
            research_report_path,
            config=config,
            provenance=provenance,
            model_summary=summary,
            ground_summary=ground_summary,
            pass_summaries=pass_summaries,
            passes_by_station=passes_by_station,
            validation=validation,
            warnings=warnings,
            age_report=age,
            frame_roundtrip=roundtrip,
            geodetic_roundtrip=ground_roundtrip,
            created_files=created,
        )
        created.append(research_report_path)

        run_manifest_path = result_directory / "RUN_MANIFEST.json"
        write_run_manifest(
            run_manifest_path,
            result_directory=result_directory,
            config=config,
            validation=validation,
            warnings=warnings,
        )
        created.append(run_manifest_path)

        logger.info("Combined HTML report: %s", research_report_path)
        logger.info("Run manifest: %s", run_manifest_path)
        logger.info("Created %d files.", len(created))
        logger.info("Validation status: %s", validation["overall_status"])
    except Exception:
        logger.exception("Fixed-TLE experiment failed.")
        raise
    finally:
        close_run_logger(logger)

    maximums = {name: float(value["maximum_position_difference_km"]) for name, value in summary["models"].items()}
    finals = {name: float(value["final_position_difference_km"]) for name, value in summary["models"].items()}
    ground_maximums = {
        name: float(value["maximum_surface_separation_km"])
        for name, value in ground_summary["models"].items()
    }
    ground_finals = {
        name: float(value["final_surface_separation_km"])
        for name, value in ground_summary["models"].items()
    }
    primary_station = stations[0]
    primary_pass_summary = pass_summaries[primary_station.station_id]
    pass_counts = {
        model: int(values["pass_count"])
        for model, values in primary_pass_summary["models"].items()
    }
    matched_counts = {
        model: int(values["matched_pass_count"])
        for model, values in primary_pass_summary["comparisons_against_sgp4"].items()
    }
    maximum_aos_differences = {
        model: (
            None
            if values["maximum_absolute_aos_difference_seconds"] is None
            else float(values["maximum_absolute_aos_difference_seconds"])
        )
        for model, values in primary_pass_summary["comparisons_against_sgp4"].items()
    }
    maximum_los_differences = {
        model: (
            None
            if values["maximum_absolute_los_difference_seconds"] is None
            else float(values["maximum_absolute_los_difference_seconds"])
        )
        for model, values in primary_pass_summary["comparisons_against_sgp4"].items()
    }
    return TLEExperimentRunResult(
        experiment_id=config["experiment"]["experiment_id"],
        result_directory=result_directory,
        warnings=tuple(warnings),
        created_files=tuple(created),
        validation_status=validation["overall_status"],
        tle_epoch_utc=tle.epoch_utc,
        end_tle_age_hours=float(age["end_age_hours"]),
        frame_roundtrip_position_error_m=float(roundtrip["position_roundtrip_error_m"]),
        maximum_separation_km_by_model=maximums,
        final_separation_km_by_model=finals,
        nonzero_sgp4_error_count=int(sgp4_diagnostics["nonzero_error_count"]),
        maximum_ground_track_separation_km_by_model=ground_maximums,
        final_ground_track_separation_km_by_model=ground_finals,
        geodetic_roundtrip_position_error_m=float(
            ground_roundtrip["maximum_position_residual_m"]
        ),
        pass_station_id=primary_station.station_id,
        pass_station_name=primary_station.name,
        pass_minimum_elevation_deg=primary_station.minimum_elevation_deg,
        pass_count_by_model=pass_counts,
        matched_pass_count_by_model=matched_counts,
        maximum_absolute_aos_difference_seconds_by_model=maximum_aos_differences,
        maximum_absolute_los_difference_seconds_by_model=maximum_los_differences,
        research_report_path=research_report_path,
        run_manifest_path=run_manifest_path,
    )
