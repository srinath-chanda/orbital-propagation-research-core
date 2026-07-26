"""Configuration loading and validation for Research Core 1A.0."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from .constants import (
    SUPPORTED_INITIAL_STATE_SOURCES,
    SUPPORTED_INTEGRATORS,
    SUPPORTED_MODELS,
    SUPPORTED_SCHEMA_VERSIONS,
)


class ConfigValidationError(ValueError):
    """Raised when an experiment configuration is invalid."""


def _require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ConfigValidationError(f"'{key}' must be a JSON object.")
    return value


def _require_non_empty_string(parent: dict[str, Any], key: str, path: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"'{path}.{key}' must be a non-empty string.")
    return value.strip()


def _require_bool(parent: dict[str, Any], key: str, path: str) -> bool:
    value = parent.get(key)
    if not isinstance(value, bool):
        raise ConfigValidationError(f"'{path}.{key}' must be true or false.")
    return value


def _require_finite_number(
    parent: dict[str, Any],
    key: str,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_greater_than: float | None = None,
) -> float:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValidationError(f"'{path}.{key}' must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigValidationError(f"'{path}.{key}' must be finite.")
    if minimum is not None and number < minimum:
        raise ConfigValidationError(
            f"'{path}.{key}' must be greater than or equal to {minimum}."
        )
    if maximum is not None and number > maximum:
        raise ConfigValidationError(
            f"'{path}.{key}' must be less than or equal to {maximum}."
        )
    if strictly_greater_than is not None and number <= strictly_greater_than:
        raise ConfigValidationError(
            f"'{path}.{key}' must be greater than {strictly_greater_than}."
        )
    return number


def _parse_utc_timestamp(value: str, field_path: str) -> datetime:
    if not value.endswith("Z"):
        raise ConfigValidationError(
            f"'{field_path}' must be an ISO 8601 UTC timestamp ending in 'Z'."
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ConfigValidationError(
            f"'{field_path}' is not a valid ISO 8601 timestamp: {value!r}."
        ) from exc
    return parsed


def load_json_config(config_path: str | Path) -> dict[str, Any]:
    """Load a JSON experiment configuration from disk."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ConfigValidationError(
            f"Invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise ConfigValidationError("The configuration root must be a JSON object.")
    return data


def validate_config(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
) -> list[str]:
    """Validate a configuration and return non-fatal scientific warnings."""
    warnings: list[str] = []

    schema_version = _require_non_empty_string(config, "schema_version", "root")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        allowed = ", ".join(sorted(SUPPORTED_SCHEMA_VERSIONS))
        raise ConfigValidationError(
            f"Unsupported schema_version {schema_version!r}. Supported: {allowed}."
        )

    experiment = _require_mapping(config, "experiment")
    _require_non_empty_string(experiment, "experiment_id", "experiment")
    _require_non_empty_string(experiment, "case_id", "experiment")
    _require_non_empty_string(experiment, "title", "experiment")
    _require_non_empty_string(experiment, "description", "experiment")
    _require_non_empty_string(experiment, "author", "experiment")
    _require_bool(experiment, "reproducible_run", "experiment")

    initial_state = _require_mapping(config, "initial_state")
    source_type = _require_non_empty_string(
        initial_state, "source_type", "initial_state"
    )
    if source_type not in SUPPORTED_INITIAL_STATE_SOURCES:
        allowed = ", ".join(sorted(SUPPORTED_INITIAL_STATE_SOURCES))
        raise ConfigValidationError(
            f"Unsupported initial_state.source_type {source_type!r}. "
            f"Supported: {allowed}."
        )
    epoch = _require_non_empty_string(initial_state, "epoch_utc", "initial_state")
    _parse_utc_timestamp(epoch, "initial_state.epoch_utc")
    frame = _require_non_empty_string(initial_state, "frame", "initial_state")

    if source_type == "fixed_tle":
        _require_finite_number(
            initial_state,
            "norad_catalog_number",
            "initial_state",
            strictly_greater_than=0.0,
        )
        tle_file_value = _require_non_empty_string(
            initial_state, "tle_file", "initial_state"
        )
        tle_metadata_value = _require_non_empty_string(
            initial_state, "tle_metadata_file", "initial_state"
        )
        if config_path is not None:
            config_parent = Path(config_path).expanduser().resolve().parent
            for label, value in (("tle_file", tle_file_value), ("tle_metadata_file", tle_metadata_value)):
                candidate = Path(value).expanduser()
                if not candidate.is_absolute():
                    candidate = config_parent / candidate
                if not candidate.is_file():
                    raise ConfigValidationError(
                        f"'initial_state.{label}' was not found: {candidate.resolve()}"
                    )
        if frame != "GCRS_ASTROPY_FROM_TEME":
            raise ConfigValidationError(
                "Fixed-TLE experiments must use frame 'GCRS_ASTROPY_FROM_TEME'."
            )

    if source_type == "classical_elements":
        semi_major_axis = _require_finite_number(
            initial_state,
            "semi_major_axis_km",
            "initial_state",
            strictly_greater_than=0.0,
        )
        _require_finite_number(
            initial_state,
            "eccentricity",
            "initial_state",
            minimum=0.0,
            maximum=0.999999999999,
        )
        _require_finite_number(
            initial_state,
            "inclination_deg",
            "initial_state",
            minimum=0.0,
            maximum=180.0,
        )
        for angle_name in (
            "raan_deg",
            "argument_of_perigee_deg",
            "true_anomaly_deg",
        ):
            _require_finite_number(initial_state, angle_name, "initial_state")
    else:
        semi_major_axis = None

    earth = _require_mapping(config, "earth_model")
    mu = _require_finite_number(
        earth,
        "gravitational_parameter_km3_s2",
        "earth_model",
        strictly_greater_than=0.0,
    )
    radius = _require_finite_number(
        earth,
        "equatorial_radius_km",
        "earth_model",
        strictly_greater_than=0.0,
    )
    _require_finite_number(
        earth, "j2", "earth_model", minimum=0.0, maximum=0.01
    )
    _require_finite_number(
        earth,
        "earth_rotation_rate_rad_s",
        "earth_model",
        minimum=0.0,
    )
    if mu < 100_000.0:
        warnings.append(
            "Earth gravitational parameter is unusually small for kilometres-based units."
        )
    if semi_major_axis is not None and semi_major_axis <= radius:
        raise ConfigValidationError(
            "initial_state.semi_major_axis_km must be larger than "
            "earth_model.equatorial_radius_km for the selected LEO case."
        )
    constants_reference = str(earth.get("constants_reference", "")).strip()
    if not constants_reference or constants_reference.startswith("TO_BE_"):
        warnings.append(
            "Earth constants still require an authoritative citation before paper runs."
        )

    propagation = _require_mapping(config, "propagation")
    duration_hours = _require_finite_number(
        propagation,
        "default_duration_hours",
        "propagation",
        strictly_greater_than=0.0,
    )
    output_step = _require_finite_number(
        propagation,
        "output_step_seconds",
        "propagation",
        strictly_greater_than=0.0,
    )
    if output_step > duration_hours * 3600.0:
        warnings.append(
            "The output step is longer than the default propagation duration."
        )
    models = propagation.get("models")
    if not isinstance(models, list) or not models:
        raise ConfigValidationError(
            "'propagation.models' must be a non-empty JSON array."
        )
    if any(not isinstance(model, str) or not model for model in models):
        raise ConfigValidationError(
            "Every entry in 'propagation.models' must be a non-empty string."
        )
    unsupported = sorted(set(models) - SUPPORTED_MODELS)
    if unsupported:
        allowed = ", ".join(sorted(SUPPORTED_MODELS))
        raise ConfigValidationError(
            f"Unsupported propagation model(s): {', '.join(unsupported)}. "
            f"Supported: {allowed}."
        )
    if len(models) != len(set(models)):
        raise ConfigValidationError("'propagation.models' contains duplicates.")
    reference_model = _require_non_empty_string(
        propagation, "comparison_reference_model", "propagation"
    )
    if reference_model not in models:
        raise ConfigValidationError(
            "'propagation.comparison_reference_model' must also appear in "
            "'propagation.models'."
        )
    _require_finite_number(
        propagation,
        "stop_below_altitude_km",
        "propagation",
        minimum=0.0,
    )
    _require_bool(
        propagation,
        "save_state_at_initial_epoch",
        "propagation",
    )

    batch_durations = propagation.get("batch_durations_hours")
    if not isinstance(batch_durations, list) or not batch_durations:
        raise ConfigValidationError(
            "'propagation.batch_durations_hours' must be a non-empty array."
        )
    for index, value in enumerate(batch_durations):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigValidationError(
                f"'propagation.batch_durations_hours[{index}]' must be numeric."
            )
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ConfigValidationError(
                f"'propagation.batch_durations_hours[{index}]' must be positive."
            )

    integrator = _require_mapping(config, "integrator")
    method = _require_non_empty_string(integrator, "method", "integrator")
    if method not in SUPPORTED_INTEGRATORS:
        allowed = ", ".join(sorted(SUPPORTED_INTEGRATORS))
        raise ConfigValidationError(
            f"Unsupported integrator method {method!r}. Supported: {allowed}."
        )
    _require_finite_number(
        integrator,
        "relative_tolerance",
        "integrator",
        strictly_greater_than=0.0,
    )
    _require_finite_number(
        integrator,
        "absolute_tolerance",
        "integrator",
        strictly_greater_than=0.0,
    )
    _require_finite_number(
        integrator,
        "maximum_step_seconds",
        "integrator",
        strictly_greater_than=0.0,
    )
    if str(integrator.get("settings_status", "")).startswith("provisional"):
        warnings.append(
            "Integrator settings are provisional until the convergence study is complete."
        )

    drag = _require_mapping(config, "drag")
    drag_enabled = _require_bool(drag, "enabled", "drag")
    if drag_enabled:
        for key in ("mass_kg", "cross_sectional_area_m2", "drag_coefficient"):
            _require_finite_number(
                drag,
                key,
                "drag",
                strictly_greater_than=0.0,
            )
        _require_finite_number(
            drag,
            "reference_altitude_km",
            "drag",
            minimum=0.0,
        )
        _require_finite_number(
            drag,
            "reference_density_kg_m3",
            "drag",
            minimum=0.0,
        )
        _require_finite_number(
            drag,
            "scale_height_km",
            "drag",
            strictly_greater_than=0.0,
        )
        _require_bool(drag, "co_rotating_atmosphere", "drag")
        if str(drag.get("parameter_status", "")).startswith("illustrative"):
            warnings.append(
                "Drag parameters are illustrative sensitivity values, not paper-ready atmosphere data."
            )
        sensitivity = drag.get("sensitivity")
        if sensitivity is not None:
            if not isinstance(sensitivity, dict):
                raise ConfigValidationError("'drag.sensitivity' must be a JSON object.")
            sensitivity_enabled = _require_bool(
                sensitivity, "enabled", "drag.sensitivity"
            )
            if sensitivity_enabled:
                multipliers = _require_mapping(sensitivity, "multipliers")
                for parameter in (
                    "mass_kg",
                    "cross_sectional_area_m2",
                    "drag_coefficient",
                    "reference_density_kg_m3",
                    "scale_height_km",
                ):
                    values = multipliers.get(parameter)
                    if not isinstance(values, list) or not values:
                        raise ConfigValidationError(
                            f"'drag.sensitivity.multipliers.{parameter}' must be a non-empty array."
                        )
                    for index, value in enumerate(values):
                        if isinstance(value, bool) or not isinstance(value, (int, float)):
                            raise ConfigValidationError(
                                f"'drag.sensitivity.multipliers.{parameter}[{index}]' must be numeric."
                            )
                        number = float(value)
                        if not math.isfinite(number) or number <= 0.0:
                            raise ConfigValidationError(
                                f"'drag.sensitivity.multipliers.{parameter}[{index}]' must be positive and finite."
                            )

    stations = config.get("ground_stations")
    if not isinstance(stations, list) or not stations:
        raise ConfigValidationError(
            "'ground_stations' must be a non-empty JSON array."
        )
    station_ids: set[str] = set()
    for index, station in enumerate(stations):
        if not isinstance(station, dict):
            raise ConfigValidationError(
                f"'ground_stations[{index}]' must be a JSON object."
            )
        station_path = f"ground_stations[{index}]"
        station_id = _require_non_empty_string(
            station, "station_id", station_path
        )
        if station_id in station_ids:
            raise ConfigValidationError(
                f"Duplicate ground-station ID: {station_id!r}."
            )
        station_ids.add(station_id)
        _require_non_empty_string(station, "name", station_path)
        _require_finite_number(
            station,
            "latitude_deg",
            station_path,
            minimum=-90.0,
            maximum=90.0,
        )
        _require_finite_number(
            station,
            "longitude_deg",
            station_path,
            minimum=-180.0,
            maximum=180.0,
        )
        _require_finite_number(station, "altitude_m", station_path)
        _require_finite_number(
            station,
            "minimum_elevation_deg",
            station_path,
            minimum=-5.0,
            maximum=90.0,
        )

    ground_track = _require_mapping(config, "ground_track_analysis")
    ground_track_enabled = _require_bool(
        ground_track, "enabled", "ground_track_analysis"
    )
    _require_non_empty_string(
        ground_track, "earth_fixed_frame", "ground_track_analysis"
    )
    ellipsoid = _require_non_empty_string(
        ground_track, "earth_ellipsoid", "ground_track_analysis"
    )
    if ellipsoid not in {"WGS84", "WGS72", "GRS80"}:
        raise ConfigValidationError(
            "'ground_track_analysis.earth_ellipsoid' must be WGS84, WGS72, or GRS80."
        )
    distance_model = _require_non_empty_string(
        ground_track, "surface_distance_model", "ground_track_analysis"
    )
    if distance_model != "spherical_central_angle":
        raise ConfigValidationError(
            "Research Core 1A.6 supports only the spherical_central_angle surface-distance model."
        )
    _require_finite_number(
        ground_track,
        "surface_radius_km",
        "ground_track_analysis",
        strictly_greater_than=0.0,
    )
    _require_finite_number(
        ground_track,
        "date_line_split_threshold_deg",
        "ground_track_analysis",
        minimum=90.0,
        maximum=360.0,
    )
    background = ground_track.get("map_background_file")
    if background is not None and (not isinstance(background, str) or not background.strip()):
        raise ConfigValidationError(
            "'ground_track_analysis.map_background_file' must be null or a non-empty string."
        )
    if ground_track_enabled and source_type != "fixed_tle":
        warnings.append(
            "Ground-track analysis is currently implemented only for fixed-TLE GCRS experiments."
        )

    pass_analysis = _require_mapping(config, "pass_analysis")
    pass_enabled = _require_bool(pass_analysis, "enabled", "pass_analysis")
    coarse_step = _require_finite_number(
        pass_analysis,
        "coarse_step_seconds",
        "pass_analysis",
        strictly_greater_than=0.0,
    )
    refinement_tolerance = _require_finite_number(
        pass_analysis,
        "refinement_tolerance_seconds",
        "pass_analysis",
        strictly_greater_than=0.0,
    )
    if refinement_tolerance > coarse_step:
        raise ConfigValidationError(
            "'pass_analysis.refinement_tolerance_seconds' must not exceed the coarse step."
        )
    _require_bool(
        pass_analysis, "calculate_closest_range", "pass_analysis"
    )
    _require_bool(
        pass_analysis, "save_visibility_history", "pass_analysis"
    )
    match_window = _require_finite_number(
        pass_analysis,
        "match_passes_within_seconds",
        "pass_analysis",
        strictly_greater_than=0.0,
    )
    if match_window < coarse_step:
        warnings.append(
            "Pass-matching window is shorter than the coarse access-search step."
        )
    refinement_method = _require_non_empty_string(
        pass_analysis, "refinement_method", "pass_analysis"
    )
    if refinement_method != "pchip_brentq":
        raise ConfigValidationError(
            "Research Core 1A.6 supports only pass_analysis.refinement_method='pchip_brentq'."
        )
    station_selection = pass_analysis.get("station_ids")
    if not isinstance(station_selection, list) or not station_selection:
        raise ConfigValidationError(
            "'pass_analysis.station_ids' must be a non-empty JSON array."
        )
    if any(not isinstance(value, str) or not value.strip() for value in station_selection):
        raise ConfigValidationError(
            "Every entry in 'pass_analysis.station_ids' must be a non-empty string."
        )
    if len(station_selection) != len(set(station_selection)):
        raise ConfigValidationError("'pass_analysis.station_ids' contains duplicates.")
    unknown_stations = sorted(set(station_selection) - station_ids)
    if unknown_stations:
        raise ConfigValidationError(
            "Unknown pass-analysis station ID(s): " + ", ".join(unknown_stations)
        )
    if pass_enabled and source_type != "fixed_tle":
        warnings.append(
            "Ground-station pass analysis is currently implemented only for fixed-TLE GCRS experiments."
        )
    if pass_enabled and not ground_track_enabled:
        raise ConfigValidationError(
            "Ground-station pass analysis requires ground_track_analysis.enabled=true."
        )

    convergence = _require_mapping(config, "convergence")
    _require_bool(convergence, "enabled", "convergence")
    _require_finite_number(
        convergence,
        "duration_hours",
        "convergence",
        strictly_greater_than=0.0,
    )
    _require_finite_number(
        convergence,
        "output_step_seconds",
        "convergence",
        strictly_greater_than=0.0,
    )
    repetitions = convergence.get("runtime_repetitions")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int):
        raise ConfigValidationError(
            "'convergence.runtime_repetitions' must be a positive integer."
        )
    if repetitions <= 0 or repetitions > 20:
        raise ConfigValidationError(
            "'convergence.runtime_repetitions' must be between 1 and 20."
        )
    convergence_method = _require_non_empty_string(
        convergence, "method", "convergence"
    )
    if convergence_method not in SUPPORTED_INTEGRATORS:
        allowed = ", ".join(sorted(SUPPORTED_INTEGRATORS))
        raise ConfigValidationError(
            f"Unsupported convergence method {convergence_method!r}. "
            f"Supported: {allowed}."
        )
    for key in ("relative_tolerances", "absolute_tolerances", "maximum_steps_seconds"):
        values = convergence.get(key)
        if not isinstance(values, list) or not values:
            raise ConfigValidationError(
                f"'convergence.{key}' must be a non-empty array."
            )
        converted: list[float] = []
        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigValidationError(
                    f"'convergence.{key}[{index}]' must be numeric."
                )
            number = float(value)
            if not math.isfinite(number) or number <= 0.0:
                raise ConfigValidationError(
                    f"'convergence.{key}[{index}]' must be finite and positive."
                )
            converted.append(number)
        if len(converted) != len(set(converted)):
            raise ConfigValidationError(
                f"'convergence.{key}' contains duplicate values."
            )
    _require_finite_number(
        convergence,
        "reference_relative_tolerance",
        "convergence",
        strictly_greater_than=0.0,
    )
    _require_finite_number(
        convergence,
        "reference_absolute_tolerance",
        "convergence",
        strictly_greater_than=0.0,
    )
    _require_finite_number(
        convergence,
        "reference_maximum_step_seconds",
        "convergence",
        strictly_greater_than=0.0,
    )

    external = _require_mapping(config, "external_validation")
    external_enabled = _require_bool(external, "enabled", "external_validation")
    if external_enabled:
        base = (
            Path(config_path).expanduser().resolve().parent
            if config_path is not None
            else Path.cwd()
        )
        for key in (
            "state_reference_file",
            "state_reference_metadata_file",
        ):
            value = external.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ConfigValidationError(
                    f"'external_validation.{key}' is required when external validation is enabled."
                )
            reference_path = Path(value)
            if not reference_path.is_absolute():
                reference_path = base / reference_path
            if not reference_path.is_file():
                raise ConfigValidationError(
                    f"External reference file not found: {reference_path.resolve()}"
                )

    validation = _require_mapping(config, "validation")
    _require_bool(
        validation,
        "fail_run_on_validation_failure",
        "validation",
    )
    if str(validation.get("threshold_status", "")).startswith("provisional"):
        warnings.append(
            "Validation thresholds remain provisional until external validation."
        )

    outputs = _require_mapping(config, "outputs")
    _require_non_empty_string(outputs, "results_root", "outputs")
    for key in (
        "save_resolved_configuration",
        "save_initial_state",
        "save_states_csv",
        "save_elements_csv",
        "save_rtn_errors_csv",
        "save_passes_csv",
        "save_runtime_csv",
        "save_validation_status",
        "save_run_log",
        "save_png",
        "save_pdf",
        "open_figures_automatically",
        "open_report_automatically",
    ):
        _require_bool(outputs, key, "outputs")

    return warnings


def load_and_validate_config(
    config_path: str | Path,
) -> tuple[dict[str, Any], list[str]]:
    """Load and validate a JSON configuration."""
    path = Path(config_path).expanduser().resolve()
    config = load_json_config(path)
    warnings = validate_config(config, config_path=path)
    return deepcopy(config), warnings
