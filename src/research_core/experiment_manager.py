"""Integrated controlled-orbit research pipeline for Research Core 1A.7."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import BUILD_MARKER, RESEARCH_CORE_VERSION
from .analysis.comparison import compare_state_histories, create_error_summary
from .analysis.diagnostics import conservation_diagnostics, create_orbit_summary
from .analysis.drag import (
    create_drag_diagnostics,
    create_drag_validation_summary,
    drag_parameter_dict,
    run_drag_sensitivity,
    sensitivity_direction_checks,
)
from .analysis.j2 import (
    compare_in_reference_rtn,
    create_j2_validation_summary,
    create_osculating_element_history,
)
from .configuration import load_and_validate_config
from .data_models import CartesianState, StateHistory
from .logging_utils import close_run_logger, create_run_logger
from .metadata import collect_environment_metadata, utc_now_iso, write_json
from .research_report import (
    write_controlled_research_report,
    write_final_validation_summary,
    write_run_manifest,
)
from .orbital_elements import (
    cartesian_to_elements,
    elements_from_config,
    elements_to_cartesian,
)
from .outputs import (
    create_drag_figures,
    create_j2_figures,
    create_two_body_figures,
    write_comparison_csv,
    write_conservation_csv,
    write_drag_diagnostics_csv,
    write_drag_sensitivity_csv,
    write_drag_validation_csv,
    write_element_history_csv,
    write_initial_conditions_csv,
    write_j2_conservation_csv,
    write_j2_validation_csv,
    write_model_error_summary_csv,
    write_rtn_comparison_csv,
    write_state_history_csv,
)
from .propagators import (
    propagate_analytical_two_body,
    propagate_numerical_j2,
    propagate_numerical_j2_drag,
    propagate_numerical_two_body,
)
from .time_utils import build_time_grid


@dataclass(frozen=True)
class ExperimentRunResult:
    """Summary of a completed integrated two-body, J2, and drag run."""

    experiment_id: str
    result_directory: Path
    warnings: tuple[str, ...]
    created_files: tuple[Path, ...]
    validation_status: str
    maximum_position_difference_m: float
    maximum_velocity_difference_mm_s: float
    numerical_maximum_relative_energy_drift: float
    numerical_maximum_relative_angular_momentum_drift: float
    maximum_j2_two_body_position_difference_km: float
    analytical_raan_rate_deg_day: float
    fitted_raan_rate_deg_day: float
    raan_rate_relative_difference: float
    j2_maximum_relative_total_energy_drift: float
    j2_maximum_relative_angular_momentum_z_drift: float
    maximum_drag_j2_position_difference_km: float
    final_drag_semi_major_axis_difference_vs_j2_m: float
    drag_total_specific_energy_loss_km2_s2: float
    initial_drag_acceleration_m_s2: float
    zero_density_maximum_position_difference_m: float
    sensitivity_case_count: int
    sensitivity_direction_checks_passed: bool | None


def _safe_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("Experiment ID cannot be converted into a safe folder name.")
    return cleaned


def _timestamp_folder_name() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")


def _resolve_results_root(configured_root: str, *, project_root: str | Path) -> Path:
    path = Path(configured_root).expanduser()
    if not path.is_absolute():
        path = Path(project_root).expanduser().resolve() / path
    return path.resolve()


def _create_unique_run_directory(*, results_root: Path, experiment_id: str) -> Path:
    experiment_directory = results_root / _safe_path_component(experiment_id)
    for _ in range(10):
        candidate = experiment_directory / _timestamp_folder_name()
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("Could not create a unique result directory.")


def _resolved_configuration(
    config: dict[str, Any],
    *,
    source_path: Path,
    result_directory: Path,
    run_started_utc: str,
) -> dict[str, Any]:
    resolved = deepcopy(config)
    resolved["_runtime"] = {
        "research_core_version": RESEARCH_CORE_VERSION,
        "build_marker": BUILD_MARKER,
        "run_started_utc": run_started_utc,
        "configuration_source": str(source_path),
        "result_directory": str(result_directory),
        "foundation_only": False,
        "orbit_propagation_performed": True,
        "integrated_run": True,
        "implemented_models": [
            "analytical_two_body",
            "numerical_two_body",
            "numerical_j2",
            "numerical_j2_drag",
        ],
    }
    return resolved


def _check(
    validation_id: str,
    name: str,
    measured_value: float | bool | None,
    criterion: str,
    passed: bool | None,
) -> dict[str, Any]:
    if passed is None:
        status = "not_evaluated"
    else:
        status = "passed" if passed else "failed"
    return {
        "validation_id": validation_id,
        "name": name,
        "measured_value": measured_value,
        "criterion": criterion,
        "status": status,
    }


def _create_validation_status(
    *,
    experiment_id: str,
    warnings: list[str],
    orbit_summary: dict[str, Any],
    two_body_error_summary: dict[str, Any],
    two_body_diagnostics: list[dict[str, Any]],
    j2_validation: dict[str, Any],
    initial_j2_state_difference_km: float,
    drag_validation: dict[str, Any],
    initial_drag_state_difference_km: float,
    zero_density_maximum_position_difference_m: float,
    sensitivity_checks: dict[str, bool | None],
    validation_config: dict[str, Any],
) -> dict[str, Any]:
    diagnostic_map = {item["model_name"]: item for item in two_body_diagnostics}
    maximum_position = float(
        two_body_error_summary["position_difference_m"]["maximum_absolute"]
    )
    maximum_velocity = float(
        two_body_error_summary["velocity_difference_mm_s"]["maximum_absolute"]
    )
    numerical_energy = float(
        diagnostic_map["numerical_two_body"][
            "maximum_absolute_relative_energy_drift"
        ]
    )
    numerical_h = float(
        diagnostic_map["numerical_two_body"][
            "maximum_absolute_relative_angular_momentum_drift"
        ]
    )

    position_limit = float(
        validation_config["two_body_maximum_position_difference_m"]
    )
    velocity_limit = float(
        validation_config["two_body_maximum_velocity_difference_mm_s"]
    )
    energy_limit = float(
        validation_config["two_body_maximum_relative_energy_drift"]
    )
    h_limit = float(
        validation_config["two_body_maximum_relative_angular_momentum_drift"]
    )
    raan_limit = float(
        validation_config["j2_raan_rate_maximum_relative_difference"]
    )
    j2_energy_limit = float(
        validation_config["j2_maximum_relative_total_energy_drift"]
    )
    j2_hz_limit = float(
        validation_config["j2_maximum_relative_angular_momentum_z_drift"]
    )
    raan_minimum_duration_hours = float(
        validation_config.get("j2_raan_validation_minimum_duration_hours", 6.0)
    )
    raan_duration_sufficient = (
        float(j2_validation["duration_hours"]) >= raan_minimum_duration_hours
    )

    analytical_raan_rate = float(j2_validation["analytical_raan_rate_rad_s"])
    fitted_raan_rate = float(j2_validation["fitted_raan_rate_rad_s"])
    raan_sign_correct = (
        np.sign(analytical_raan_rate) == np.sign(fitted_raan_rate)
        and analytical_raan_rate != 0.0
    )
    acceleration_ratio = float(
        j2_validation["initial_j2_to_central_acceleration_ratio"]
    )
    j2_energy_drift = float(
        j2_validation["maximum_absolute_relative_total_energy_drift"]
    )
    j2_hz_drift = float(
        j2_validation[
            "maximum_absolute_relative_angular_momentum_z_drift"
        ]
    )

    zero_density_limit = float(
        validation_config["drag_zero_density_maximum_position_difference_m"]
    )
    drag_acceleration_ratio_limit = float(
        validation_config[
            "drag_maximum_initial_to_central_acceleration_ratio"
        ]
    )
    drag_minimum_energy_loss = float(
        validation_config["drag_minimum_energy_loss_km2_s2"]
    )
    drag_minimum_a_reduction = float(
        validation_config[
            "drag_minimum_semi_major_axis_reduction_vs_j2_m"
        ]
    )
    drag_direction_cosine_limit = float(
        validation_config["drag_direction_cosine_maximum"]
    )

    evaluated_sensitivity = [
        value for value in sensitivity_checks.values() if value is not None
    ]
    sensitivity_passed: bool | None = (
        all(evaluated_sensitivity) if evaluated_sensitivity else None
    )

    checks = [
        _check(
            "VAL-CONFIG-001",
            "Valid configuration loaded",
            None,
            "Configuration loads and passes schema/range checks",
            True,
        ),
        _check(
            "VAL-STATE-004",
            "Initial state satisfies vis-viva",
            float(orbit_summary["vis_viva_absolute_difference_km_s"]),
            "Absolute speed difference <= 1e-10 km/s",
            float(orbit_summary["vis_viva_absolute_difference_km_s"])
            <= 1e-10,
        ),
        _check(
            "VAL-NTB-002",
            "Numerical two-body agrees with analytical two-body",
            maximum_position,
            f"Maximum position difference <= {position_limit} m",
            maximum_position <= position_limit,
        ),
        _check(
            "VAL-NTB-002-V",
            "Numerical two-body velocity agrees with analytical two-body",
            maximum_velocity,
            f"Maximum velocity difference <= {velocity_limit} mm/s",
            maximum_velocity <= velocity_limit,
        ),
        _check(
            "VAL-NTB-003",
            "Numerical two-body conserves specific energy",
            numerical_energy,
            f"Maximum relative energy drift <= {energy_limit}",
            numerical_energy <= energy_limit,
        ),
        _check(
            "VAL-NTB-004",
            "Numerical two-body conserves angular momentum",
            numerical_h,
            f"Maximum relative angular-momentum drift <= {h_limit}",
            numerical_h <= h_limit,
        ),
        _check(
            "VAL-J2-001",
            "J2 and two-body models start from the same Cartesian state",
            initial_j2_state_difference_km,
            "Initial six-state Euclidean difference <= 1e-12 in km-based units",
            initial_j2_state_difference_km <= 1e-12,
        ),
        _check(
            "VAL-J2-002",
            "J2 acceleration is a small perturbation of central gravity",
            acceleration_ratio,
            "0 < initial J2/central acceleration ratio < 0.01",
            0.0 < acceleration_ratio < 0.01,
        ),
        _check(
            "VAL-J2-003",
            "Numerical RAAN drift has the analytical J2 direction",
            fitted_raan_rate,
            (
                "Fitted and analytical RAAN rates have the same non-zero sign; "
                f"requires duration >= {raan_minimum_duration_hours} h"
            ),
            bool(raan_sign_correct) if raan_duration_sufficient else None,
        ),
        _check(
            "VAL-J2-004",
            "Numerical RAAN trend agrees with first-order secular J2 theory",
            float(j2_validation["relative_rate_difference"]),
            (
                f"Relative RAAN-rate difference <= {raan_limit}; "
                f"requires duration >= {raan_minimum_duration_hours} h"
            ),
            (
                float(j2_validation["relative_rate_difference"]) <= raan_limit
                if raan_duration_sufficient
                else None
            ),
        ),
        _check(
            "VAL-J2-005",
            "Numerical J2 conserves total specific energy",
            j2_energy_drift,
            f"Maximum relative total-energy drift <= {j2_energy_limit}",
            j2_energy_drift <= j2_energy_limit,
        ),
        _check(
            "VAL-J2-006",
            "Numerical J2 conserves z-angular momentum",
            j2_hz_drift,
            f"Maximum relative h_z drift <= {j2_hz_limit}",
            j2_hz_drift <= j2_hz_limit,
        ),
        _check(
            "VAL-DRAG-001",
            "J2+drag and J2 models start from the same Cartesian state",
            initial_drag_state_difference_km,
            "Initial six-state Euclidean difference <= 1e-12 in km-based units",
            initial_drag_state_difference_km <= 1e-12,
        ),
        _check(
            "VAL-DRAG-002",
            "Zero-density drag model reduces to numerical J2",
            zero_density_maximum_position_difference_m,
            f"Maximum position difference <= {zero_density_limit} m",
            zero_density_maximum_position_difference_m <= zero_density_limit,
        ),
        _check(
            "VAL-DRAG-003",
            "Drag acceleration opposes atmospheric-relative velocity",
            float(
                drag_validation[
                    "initial_drag_relative_velocity_direction_cosine"
                ]
            ),
            f"Direction cosine <= {drag_direction_cosine_limit}",
            float(
                drag_validation[
                    "initial_drag_relative_velocity_direction_cosine"
                ]
            )
            <= drag_direction_cosine_limit,
        ),
        _check(
            "VAL-DRAG-004",
            "Drag acceleration is small relative to central gravity",
            float(
                drag_validation[
                    "initial_drag_to_central_acceleration_ratio"
                ]
            ),
            (
                "0 < initial drag/central acceleration ratio <= "
                f"{drag_acceleration_ratio_limit}"
            ),
            0.0
            < float(
                drag_validation[
                    "initial_drag_to_central_acceleration_ratio"
                ]
            )
            <= drag_acceleration_ratio_limit,
        ),
        _check(
            "VAL-DRAG-005",
            "Simplified drag dissipates total specific mechanical energy",
            float(drag_validation["total_specific_energy_loss_km2_s2"]),
            f"Total specific-energy loss >= {drag_minimum_energy_loss} km²/s²",
            float(drag_validation["total_specific_energy_loss_km2_s2"])
            >= drag_minimum_energy_loss,
        ),
        _check(
            "VAL-DRAG-006",
            "Simplified drag reduces semi-major axis relative to J2",
            float(
                drag_validation[
                    "final_semi_major_axis_difference_vs_j2_m"
                ]
            ),
            (
                "Final J2+drag minus J2 semi-major axis <= -"
                f"{drag_minimum_a_reduction} m"
            ),
            float(
                drag_validation[
                    "final_semi_major_axis_difference_vs_j2_m"
                ]
            )
            <= -drag_minimum_a_reduction,
        ),
        _check(
            "VAL-DRAG-007",
            "One-at-a-time drag sensitivity directions are physically consistent",
            sensitivity_passed,
            (
                "Lower mass and higher area, Cd, and density increase the "
                "magnitude of semi-major-axis reduction"
            ),
            sensitivity_passed,
        ),
    ]
    failed = [check for check in checks if check["status"] == "failed"]
    overall = "failed" if failed else ("passed_with_warnings" if warnings else "passed")
    return {
        "research_core_version": RESEARCH_CORE_VERSION,
        "experiment_id": experiment_id,
        "created_utc": utc_now_iso(),
        "overall_status": overall,
        "threshold_status": validation_config.get("threshold_status", "unknown"),
        "stage": "Research Core 1A.3 integrated two-body, J2, and simplified drag",
        "orbit_propagation_performed": True,
        "checks": checks,
        "warnings": warnings,
        "failed_validation_ids": [check["validation_id"] for check in failed],
        "sensitivity_direction_checks": sensitivity_checks,
        "not_yet_implemented": [
            "High-fidelity atmosphere and real space-weather inputs",
            "Attitude-dependent projected area and drag coefficient",
            "SGP4 and TLE propagation",
            "General multi-model RTN statistics",
            "Ground-track calculation",
            "Ground-station pass calculation",
            "External GMAT/STK/Orekit validation",
        ],
    }


def _technical_summary_markdown(
    *,
    config: dict[str, Any],
    orbit_summary: dict[str, Any],
    two_body_error_summary: dict[str, Any],
    j2_error_summary: dict[str, Any],
    drag_error_summary: dict[str, Any],
    histories: list[StateHistory],
    two_body_diagnostics: list[dict[str, Any]],
    j2_validation: dict[str, Any],
    j2_rtn_comparison: dict[str, Any],
    drag_validation: dict[str, Any],
    drag_rtn_comparison: dict[str, Any],
    zero_density_maximum_position_difference_m: float,
    sensitivity_results: list[dict[str, Any]],
    sensitivity_checks: dict[str, bool | None],
    validation_status: dict[str, Any],
) -> str:
    history_map = {history.model_name: history for history in histories}
    diagnostic_map = {item["model_name"]: item for item in two_body_diagnostics}
    max_j2_cross_track_km = float(
        np.max(np.abs(j2_rtn_comparison["cross_track_position_difference_m"]))
        / 1000.0
    )
    max_drag_along_track_km = float(
        np.max(np.abs(drag_rtn_comparison["along_track_position_difference_m"]))
        / 1000.0
    )
    sensitivity_lines = "\n".join(
        f"- `{item['case_id']}`: final Δa versus J2 = "
        f"{item['final_semi_major_axis_difference_vs_j2_m']:.6f} m"
        for item in sensitivity_results
    ) or "- Sensitivity matrix disabled."
    return f"""# Research Core 1A.3 Integrated Two-Body, J2, and Drag Summary

## Experiment

- Experiment ID: `{config['experiment']['experiment_id']}`
- Case ID: `{config['experiment']['case_id']}`
- Epoch: `{orbit_summary['epoch_utc']}`
- Frame: `{orbit_summary['frame']}`
- Duration: {config['propagation']['default_duration_hours']} hours
- Output interval: {config['propagation']['output_step_seconds']} seconds
- Integrated models: analytical two-body, numerical two-body, numerical J2, numerical J2 + simplified drag

## Initial orbit

- Semi-major axis: {orbit_summary['semi_major_axis_km']:.6f} km
- Eccentricity: {orbit_summary['eccentricity']:.9f}
- Inclination: {orbit_summary['inclination_deg']:.6f} deg
- Perigee altitude: {orbit_summary['perigee_altitude_km']:.6f} km
- Apogee altitude: {orbit_summary['apogee_altitude_km']:.6f} km
- Orbital period: {orbit_summary['orbital_period_minutes']:.6f} min

## Numerical two-body verification

- Maximum analytical–numerical position difference: {two_body_error_summary['position_difference_m']['maximum_absolute']:.9e} m
- Maximum analytical–numerical velocity difference: {two_body_error_summary['velocity_difference_mm_s']['maximum_absolute']:.9e} mm/s
- Numerical maximum relative energy drift: {diagnostic_map['numerical_two_body']['maximum_absolute_relative_energy_drift']:.9e}
- Numerical maximum relative angular-momentum drift: {diagnostic_map['numerical_two_body']['maximum_absolute_relative_angular_momentum_drift']:.9e}

## J2 physical-model comparison

- Maximum numerical J2–two-body position separation: {j2_error_summary['position_difference_m']['maximum_absolute'] / 1000.0:.6f} km
- Maximum absolute cross-track separation: {max_j2_cross_track_km:.6f} km
- Analytical secular RAAN rate: {j2_validation['analytical_raan_rate_deg_day']:.9f} deg/day
- Fitted numerical RAAN rate: {j2_validation['fitted_raan_rate_deg_day']:.9f} deg/day
- Relative RAAN-rate difference: {j2_validation['relative_rate_difference']:.9e}
- Maximum relative J2 total-energy drift: {j2_validation['maximum_absolute_relative_total_energy_drift']:.9e}
- Maximum relative J2 h_z drift: {j2_validation['maximum_absolute_relative_angular_momentum_z_drift']:.9e}

## Simplified drag model

- Spacecraft mass: {config['drag']['mass_kg']:.6f} kg
- Cross-sectional area: {config['drag']['cross_sectional_area_m2']:.6f} m²
- Drag coefficient: {config['drag']['drag_coefficient']:.6f}
- Ballistic coefficient: {drag_validation['ballistic_coefficient_kg_m2']:.6f} kg/m²
- Reference density: {config['drag']['reference_density_kg_m3']:.9e} kg/m³ at {config['drag']['reference_altitude_km']:.3f} km
- Scale height: {config['drag']['scale_height_km']:.3f} km
- Co-rotating atmosphere: {config['drag']['co_rotating_atmosphere']}
- Initial drag acceleration: {drag_validation['initial_drag_acceleration_m_s2']:.9e} m/s²
- Initial drag/central acceleration ratio: {drag_validation['initial_drag_to_central_acceleration_ratio']:.9e}

## Drag effects relative to J2

- Maximum J2+drag–J2 position separation: {drag_error_summary['position_difference_m']['maximum_absolute'] / 1000.0:.9f} km
- Final J2+drag–J2 position separation: {drag_error_summary['position_difference_m']['final'] / 1000.0:.9f} km
- Maximum absolute along-track separation: {max_drag_along_track_km:.9f} km
- Final semi-major-axis difference versus J2: {drag_validation['final_semi_major_axis_difference_vs_j2_m']:.9f} m
- Minimum semi-major-axis difference versus J2: {drag_validation['minimum_semi_major_axis_difference_vs_j2_m']:.9f} m
- Total specific mechanical-energy loss: {drag_validation['total_specific_energy_loss_km2_s2']:.9e} km²/s²
- Zero-density limiting-case maximum difference from J2: {zero_density_maximum_position_difference_m:.9e} m

## One-at-a-time sensitivity cases

{sensitivity_lines}

Sensitivity direction checks: `{sensitivity_checks}`

## Runtime

- Analytical two-body: {history_map['analytical_two_body'].runtime_seconds:.6f} s
- Numerical two-body: {history_map['numerical_two_body'].runtime_seconds:.6f} s
- Numerical J2: {history_map['numerical_j2'].runtime_seconds:.6f} s
- Numerical J2 + drag: {history_map['numerical_j2_drag'].runtime_seconds:.6f} s

## Validation status

**{validation_status['overall_status']}**

## Assumptions

The drag model uses one exponential scale height, a constant reference density,
constant mass, constant projected area, constant drag coefficient, and optional
rigid atmospheric co-rotation. It excludes real solar and geomagnetic activity,
latitude/local-time/seasonal density variation, winds, atmospheric composition,
spacecraft attitude, variable area, and gas-surface interaction.

## Interpretation

The zero-density limiting case verifies that the implementation reduces to J2
when aerodynamic drag is removed. Energy loss and semi-major-axis reduction show
the expected dissipative direction. The numerical values are sensitivity results
for the configured illustrative atmosphere and are not real-date orbital-decay
forecasts or measured-satellite accuracy claims.
"""


def _scalar_j2_summary(j2_validation: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in j2_validation.items()
        if key != "conservation_diagnostics"
    }


def _scalar_drag_summary(drag_validation: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "comparison_error_summary",
        "semi_major_axis_difference_vs_j2_m",
    }
    return {
        key: value
        for key, value in drag_validation.items()
        if key not in excluded
    }


def run_experiment(
    config_path: str | Path,
    *,
    project_root: str | Path,
    console_logging: bool = True,
) -> ExperimentRunResult:
    """Execute the integrated Research Core 1A.3 benchmark."""
    source_path = Path(config_path).expanduser().resolve()
    config, warnings = load_and_validate_config(source_path)

    requested_models = set(config["propagation"]["models"])
    required_models = {
        "analytical_two_body",
        "numerical_two_body",
        "numerical_j2",
        "numerical_j2_drag",
    }
    missing = required_models - requested_models
    if missing:
        raise ValueError(
            "Research Core 1A.7 requires these models in propagation.models: "
            + ", ".join(sorted(missing))
        )
    future_models = requested_models - required_models
    if future_models:
        warnings.append(
            "Models configured but not executed in 1A.3: "
            + ", ".join(sorted(future_models))
        )
    if not config["drag"]["enabled"]:
        raise ValueError("Research Core 1A.3 requires drag.enabled = true.")

    experiment_id = config["experiment"]["experiment_id"]
    results_root = _resolve_results_root(
        config["outputs"]["results_root"],
        project_root=project_root,
    )
    result_directory = _create_unique_run_directory(
        results_root=results_root,
        experiment_id=experiment_id,
    )

    log_path = result_directory / "run_log.txt"
    logger = create_run_logger(log_path, console=console_logging)
    created_files: list[Path] = [log_path]
    run_started_utc = utc_now_iso()

    try:
        logger.info("Research Core version: %s", RESEARCH_CORE_VERSION)
        logger.info("Build marker: %s", BUILD_MARKER)
        logger.info("Configuration: %s", source_path)
        logger.info("Experiment ID: %s", experiment_id)
        logger.info("Result directory: %s", result_directory)
        logger.info("Configuration validation: PASSED")
        for warning in warnings:
            logger.warning("%s", warning)

        resolved_config_path = result_directory / "experiment_configuration.json"
        write_json(
            _resolved_configuration(
                config,
                source_path=source_path,
                result_directory=result_directory,
                run_started_utc=run_started_utc,
            ),
            resolved_config_path,
        )
        created_files.append(resolved_config_path)

        metadata_path = result_directory / "environment_metadata.json"
        write_json(
            collect_environment_metadata(
                config_path=source_path,
                result_directory=result_directory,
                run_started_utc=run_started_utc,
            ),
            metadata_path,
        )
        created_files.append(metadata_path)

        initial_config = config["initial_state"]
        earth = config["earth_model"]
        elements = elements_from_config(initial_config)
        position, velocity = elements_to_cartesian(
            elements,
            earth["gravitational_parameter_km3_s2"],
        )
        initial_state = CartesianState(
            epoch_utc=initial_config["epoch_utc"],
            frame=initial_config["frame"],
            position_km=position,
            velocity_km_s=velocity,
        )
        reconstructed = cartesian_to_elements(
            position,
            velocity,
            earth["gravitational_parameter_km3_s2"],
        )
        orbit_summary = create_orbit_summary(
            elements,
            initial_state,
            earth["gravitational_parameter_km3_s2"],
            earth["equatorial_radius_km"],
        )
        orbit_summary["physical_models_in_this_release"] = [
            "analytical point-mass two-body",
            "numerical point-mass two-body",
            "numerical point-mass plus J2",
            "numerical point-mass plus J2 plus simplified atmospheric drag",
        ]
        orbit_summary["excluded_perturbations"] = [
            "J3 and higher gravity harmonics",
            "high-fidelity atmosphere and space weather",
            "atmospheric winds and attitude-dependent area",
            "third-body gravity",
            "solar radiation pressure",
            "manoeuvres",
        ]
        logger.info(
            "Initial state created: |r|=%.6f km, |v|=%.9f km/s",
            np.linalg.norm(position),
            np.linalg.norm(velocity),
        )

        initial_state_path = result_directory / "initial_state.json"
        write_json(
            {
                **initial_state.as_dict(),
                "source": "osculating classical elements",
                "configured_elements": elements.as_degrees_dict(),
                "reconstructed_elements": reconstructed,
            },
            initial_state_path,
        )
        created_files.append(initial_state_path)

        initial_csv_path = result_directory / "initial_conditions.csv"
        write_initial_conditions_csv(
            initial_csv_path,
            elements,
            initial_state,
            reconstructed,
        )
        created_files.append(initial_csv_path)

        orbit_summary_path = result_directory / "orbit_summary.json"
        write_json(orbit_summary, orbit_summary_path)
        created_files.append(orbit_summary_path)

        time_grid = build_time_grid(
            config["propagation"]["default_duration_hours"],
            config["propagation"]["output_step_seconds"],
        )
        logger.info(
            "Time grid: %d points from %.1f to %.1f seconds",
            time_grid.size,
            time_grid[0],
            time_grid[-1],
        )

        integrator = config["integrator"]
        analytical = propagate_analytical_two_body(
            elements,
            earth["gravitational_parameter_km3_s2"],
            time_grid,
            epoch_utc=initial_state.epoch_utc,
            frame=initial_state.frame,
        )
        numerical = propagate_numerical_two_body(
            initial_state,
            earth["gravitational_parameter_km3_s2"],
            time_grid,
            method=integrator["method"],
            relative_tolerance=integrator["relative_tolerance"],
            absolute_tolerance=integrator["absolute_tolerance"],
            maximum_step_seconds=integrator["maximum_step_seconds"],
        )
        numerical_j2 = propagate_numerical_j2(
            initial_state,
            earth["gravitational_parameter_km3_s2"],
            earth["equatorial_radius_km"],
            earth["j2"],
            time_grid,
            method=integrator["method"],
            relative_tolerance=integrator["relative_tolerance"],
            absolute_tolerance=integrator["absolute_tolerance"],
            maximum_step_seconds=integrator["maximum_step_seconds"],
        )
        drag_parameters = drag_parameter_dict(config["drag"])
        numerical_j2_drag = propagate_numerical_j2_drag(
            initial_state,
            earth["gravitational_parameter_km3_s2"],
            earth["equatorial_radius_km"],
            earth["j2"],
            earth["earth_rotation_rate_rad_s"],
            time_grid,
            method=integrator["method"],
            relative_tolerance=integrator["relative_tolerance"],
            absolute_tolerance=integrator["absolute_tolerance"],
            maximum_step_seconds=integrator["maximum_step_seconds"],
            **drag_parameters,
        )
        histories = [analytical, numerical, numerical_j2, numerical_j2_drag]
        for history in histories:
            logger.info(
                "%s completed in %.6f seconds with %s function evaluations",
                history.model_name,
                history.runtime_seconds,
                history.function_evaluations,
            )

        state_paths = {
            "analytical_two_body": result_directory
            / "analytical_two_body_states.csv",
            "numerical_two_body": result_directory / "numerical_two_body_states.csv",
            "numerical_j2": result_directory / "numerical_j2_states.csv",
            "numerical_j2_drag": result_directory
            / "numerical_j2_drag_states.csv",
        }
        for history in histories:
            write_state_history_csv(state_paths[history.model_name], history)
            created_files.append(state_paths[history.model_name])

        two_body_comparison = compare_state_histories(analytical, numerical)
        two_body_error_summary = create_error_summary(two_body_comparison)
        two_body_comparison_path = result_directory / "two_body_comparison.csv"
        write_comparison_csv(two_body_comparison_path, two_body_comparison)
        created_files.append(two_body_comparison_path)

        j2_comparison = compare_state_histories(numerical, numerical_j2)
        j2_error_summary = create_error_summary(j2_comparison)
        j2_comparison_path = result_directory / "j2_two_body_comparison.csv"
        write_comparison_csv(j2_comparison_path, j2_comparison)
        created_files.append(j2_comparison_path)

        j2_rtn_comparison = compare_in_reference_rtn(numerical, numerical_j2)
        j2_rtn_path = result_directory / "j2_rtn_comparison.csv"
        write_rtn_comparison_csv(j2_rtn_path, j2_rtn_comparison)
        created_files.append(j2_rtn_path)

        drag_comparison = compare_state_histories(numerical_j2, numerical_j2_drag)
        drag_error_summary = create_error_summary(drag_comparison)
        drag_comparison_path = result_directory / "drag_j2_comparison.csv"
        write_comparison_csv(drag_comparison_path, drag_comparison)
        created_files.append(drag_comparison_path)

        drag_rtn_comparison = compare_in_reference_rtn(
            numerical_j2,
            numerical_j2_drag,
        )
        drag_rtn_path = result_directory / "drag_rtn_comparison.csv"
        write_rtn_comparison_csv(drag_rtn_path, drag_rtn_comparison)
        created_files.append(drag_rtn_path)

        two_body_diagnostics = [
            conservation_diagnostics(
                analytical,
                earth["gravitational_parameter_km3_s2"],
            ),
            conservation_diagnostics(
                numerical,
                earth["gravitational_parameter_km3_s2"],
            ),
        ]
        conservation_path = result_directory / "conservation_diagnostics.csv"
        write_conservation_csv(conservation_path, two_body_diagnostics)
        created_files.append(conservation_path)

        j2_element_history = create_osculating_element_history(
            numerical_j2,
            earth["gravitational_parameter_km3_s2"],
        )
        j2_elements_path = result_directory / "j2_orbital_elements.csv"
        write_element_history_csv(j2_elements_path, j2_element_history)
        created_files.append(j2_elements_path)

        drag_element_history = create_osculating_element_history(
            numerical_j2_drag,
            earth["gravitational_parameter_km3_s2"],
        )
        drag_elements_path = result_directory / "j2_drag_orbital_elements.csv"
        write_element_history_csv(drag_elements_path, drag_element_history)
        created_files.append(drag_elements_path)

        j2_validation = create_j2_validation_summary(
            initial_position_km=initial_state.position_km,
            elements=elements,
            element_history=j2_element_history,
            history=numerical_j2,
            gravitational_parameter_km3_s2=earth[
                "gravitational_parameter_km3_s2"
            ],
            earth_equatorial_radius_km=earth["equatorial_radius_km"],
            j2=earth["j2"],
        )
        j2_conservation_path = (
            result_directory / "j2_conservation_diagnostics.csv"
        )
        write_j2_conservation_csv(
            j2_conservation_path,
            j2_validation["conservation_diagnostics"],
        )
        created_files.append(j2_conservation_path)

        j2_validation_csv_path = result_directory / "j2_validation.csv"
        write_j2_validation_csv(j2_validation_csv_path, j2_validation)
        created_files.append(j2_validation_csv_path)

        j2_validation_json_path = result_directory / "j2_validation_summary.json"
        write_json(_scalar_j2_summary(j2_validation), j2_validation_json_path)
        created_files.append(j2_validation_json_path)

        drag_diagnostics = create_drag_diagnostics(
            numerical_j2_drag,
            gravitational_parameter_km3_s2=earth[
                "gravitational_parameter_km3_s2"
            ],
            earth_equatorial_radius_km=earth["equatorial_radius_km"],
            j2=earth["j2"],
            earth_rotation_rate_rad_s=earth["earth_rotation_rate_rad_s"],
            drag_config=config["drag"],
        )
        drag_diagnostics_path = result_directory / "drag_diagnostics.csv"
        write_drag_diagnostics_csv(drag_diagnostics_path, drag_diagnostics)
        created_files.append(drag_diagnostics_path)

        drag_validation = create_drag_validation_summary(
            initial_state=initial_state,
            j2_history=numerical_j2,
            drag_history=numerical_j2_drag,
            j2_element_history=j2_element_history,
            drag_element_history=drag_element_history,
            drag_diagnostics=drag_diagnostics,
            gravitational_parameter_km3_s2=earth[
                "gravitational_parameter_km3_s2"
            ],
            earth_equatorial_radius_km=earth["equatorial_radius_km"],
            earth_rotation_rate_rad_s=earth["earth_rotation_rate_rad_s"],
            drag_config=config["drag"],
        )
        drag_validation_csv_path = result_directory / "drag_validation.csv"
        write_drag_validation_csv(drag_validation_csv_path, drag_validation)
        created_files.append(drag_validation_csv_path)

        drag_validation_json_path = result_directory / "drag_validation_summary.json"
        write_json(_scalar_drag_summary(drag_validation), drag_validation_json_path)
        created_files.append(drag_validation_json_path)

        zero_density_parameters = deepcopy(drag_parameters)
        zero_density_parameters["reference_density_kg_m3"] = 0.0
        zero_density_history = propagate_numerical_j2_drag(
            initial_state,
            earth["gravitational_parameter_km3_s2"],
            earth["equatorial_radius_km"],
            earth["j2"],
            earth["earth_rotation_rate_rad_s"],
            time_grid,
            method=integrator["method"],
            relative_tolerance=integrator["relative_tolerance"],
            absolute_tolerance=integrator["absolute_tolerance"],
            maximum_step_seconds=integrator["maximum_step_seconds"],
            **zero_density_parameters,
        )
        zero_density_comparison = compare_state_histories(
            numerical_j2,
            zero_density_history,
        )
        zero_density_error_summary = create_error_summary(
            zero_density_comparison
        )
        zero_density_maximum_position_difference_m = float(
            zero_density_error_summary["position_difference_m"][
                "maximum_absolute"
            ]
        )
        zero_density_path = result_directory / "drag_zero_density_limit.json"
        write_json(
            {
                "reference_model": "numerical_j2",
                "comparison_model": "numerical_j2_drag_zero_density",
                "maximum_position_difference_m": zero_density_maximum_position_difference_m,
                "maximum_velocity_difference_mm_s": float(
                    zero_density_error_summary["velocity_difference_mm_s"][
                        "maximum_absolute"
                    ]
                ),
                "runtime_seconds": zero_density_history.runtime_seconds,
                "function_evaluations": zero_density_history.function_evaluations,
            },
            zero_density_path,
        )
        created_files.append(zero_density_path)

        sensitivity_results = run_drag_sensitivity(
            initial_state=initial_state,
            j2_history=numerical_j2,
            j2_element_history=j2_element_history,
            elapsed_seconds=time_grid,
            gravitational_parameter_km3_s2=earth[
                "gravitational_parameter_km3_s2"
            ],
            earth_equatorial_radius_km=earth["equatorial_radius_km"],
            j2=earth["j2"],
            earth_rotation_rate_rad_s=earth["earth_rotation_rate_rad_s"],
            drag_config=config["drag"],
            integrator_config=integrator,
        )
        sensitivity_checks = sensitivity_direction_checks(sensitivity_results)
        sensitivity_csv_path = result_directory / "drag_sensitivity.csv"
        write_drag_sensitivity_csv(sensitivity_csv_path, sensitivity_results)
        created_files.append(sensitivity_csv_path)
        sensitivity_json_path = result_directory / "drag_sensitivity_summary.json"
        write_json(
            {
                "method": "one_at_a_time_multipliers",
                "case_count": len(sensitivity_results),
                "direction_checks": sensitivity_checks,
                "results": sensitivity_results,
            },
            sensitivity_json_path,
        )
        created_files.append(sensitivity_json_path)

        summary_csv_path = result_directory / "model_error_summary.csv"
        write_model_error_summary_csv(
            summary_csv_path,
            two_body_error_summary,
            [analytical, numerical],
            two_body_diagnostics,
        )
        created_files.append(summary_csv_path)

        error_summary_path = result_directory / "model_error_summary.json"
        write_json(
            {
                "two_body_verification": two_body_error_summary,
                "j2_physical_model_comparison": j2_error_summary,
                "drag_physical_model_comparison": drag_error_summary,
                "runtime_seconds": {
                    history.model_name: history.runtime_seconds
                    for history in histories
                },
                "function_evaluations": {
                    history.model_name: history.function_evaluations
                    for history in histories
                },
            },
            error_summary_path,
        )
        created_files.append(error_summary_path)

        initial_j2_state_difference_km = float(
            np.linalg.norm(numerical_j2.positions_km[0] - numerical.positions_km[0])
            + np.linalg.norm(
                numerical_j2.velocities_km_s[0] - numerical.velocities_km_s[0]
            )
        )
        initial_drag_state_difference_km = float(
            np.linalg.norm(
                numerical_j2_drag.positions_km[0] - numerical_j2.positions_km[0]
            )
            + np.linalg.norm(
                numerical_j2_drag.velocities_km_s[0]
                - numerical_j2.velocities_km_s[0]
            )
        )
        validation_status = _create_validation_status(
            experiment_id=experiment_id,
            warnings=warnings,
            orbit_summary=orbit_summary,
            two_body_error_summary=two_body_error_summary,
            two_body_diagnostics=two_body_diagnostics,
            j2_validation=j2_validation,
            initial_j2_state_difference_km=initial_j2_state_difference_km,
            drag_validation=drag_validation,
            initial_drag_state_difference_km=initial_drag_state_difference_km,
            zero_density_maximum_position_difference_m=zero_density_maximum_position_difference_m,
            sensitivity_checks=sensitivity_checks,
            validation_config=config["validation"],
        )
        validation_path = result_directory / "validation_status.json"
        write_json(validation_status, validation_path)
        created_files.append(validation_path)

        figures = create_two_body_figures(
            result_directory / "figures",
            two_body_comparison,
            two_body_diagnostics,
            save_png=config["outputs"]["save_png"],
            save_pdf=config["outputs"]["save_pdf"],
        )
        figures.extend(
            create_j2_figures(
                result_directory / "figures",
                j2_comparison=j2_comparison,
                rtn_comparison=j2_rtn_comparison,
                element_history=j2_element_history,
                j2_validation=j2_validation,
                save_png=config["outputs"]["save_png"],
                save_pdf=config["outputs"]["save_pdf"],
            )
        )
        figures.extend(
            create_drag_figures(
                result_directory / "figures",
                j2_history=numerical_j2,
                drag_history=numerical_j2_drag,
                drag_comparison=drag_comparison,
                rtn_comparison=drag_rtn_comparison,
                j2_element_history=j2_element_history,
                drag_element_history=drag_element_history,
                drag_diagnostics=drag_diagnostics,
                sensitivity_results=sensitivity_results,
                save_png=config["outputs"]["save_png"],
                save_pdf=config["outputs"]["save_pdf"],
            )
        )
        created_files.extend(figures)

        technical_summary_path = result_directory / "TECHNICAL_SUMMARY.md"
        technical_summary_path.write_text(
            _technical_summary_markdown(
                config=config,
                orbit_summary=orbit_summary,
                two_body_error_summary=two_body_error_summary,
                j2_error_summary=j2_error_summary,
                drag_error_summary=drag_error_summary,
                histories=histories,
                two_body_diagnostics=two_body_diagnostics,
                j2_validation=j2_validation,
                j2_rtn_comparison=j2_rtn_comparison,
                drag_validation=drag_validation,
                drag_rtn_comparison=drag_rtn_comparison,
                zero_density_maximum_position_difference_m=zero_density_maximum_position_difference_m,
                sensitivity_results=sensitivity_results,
                sensitivity_checks=sensitivity_checks,
                validation_status=validation_status,
            ),
            encoding="utf-8",
            newline="\n",
        )
        created_files.append(technical_summary_path)

        final_validation_path = result_directory / "FINAL_VALIDATION_SUMMARY.json"
        write_final_validation_summary(
            final_validation_path,
            config=config,
            validation=validation_status,
            warnings=warnings,
        )
        created_files.append(final_validation_path)

        research_report_path = result_directory / "RESEARCH_REPORT.html"
        write_controlled_research_report(
            research_report_path,
            config=config,
            orbit_summary=orbit_summary,
            two_body_summary=two_body_error_summary,
            j2_validation=j2_validation,
            drag_validation=drag_validation,
            validation=validation_status,
            warnings=warnings,
            created_files=created_files,
        )
        created_files.append(research_report_path)

        run_manifest_path = result_directory / "RUN_MANIFEST.json"
        write_run_manifest(
            run_manifest_path,
            result_directory=result_directory,
            config=config,
            validation=validation_status,
            warnings=warnings,
        )
        created_files.append(run_manifest_path)

        logger.info("Combined HTML report: %s", research_report_path)
        logger.info("Run manifest: %s", run_manifest_path)
        logger.info(
            "Two-body maximum position difference: %.9e m",
            two_body_error_summary["position_difference_m"]["maximum_absolute"],
        )
        logger.info(
            "J2 versus two-body maximum separation: %.9f km",
            j2_error_summary["position_difference_m"]["maximum_absolute"]
            / 1000.0,
        )
        logger.info(
            "J2+drag versus J2 maximum separation: %.9f km",
            drag_error_summary["position_difference_m"]["maximum_absolute"]
            / 1000.0,
        )
        logger.info(
            "Final drag semi-major-axis difference versus J2: %.9f m",
            drag_validation["final_semi_major_axis_difference_vs_j2_m"],
        )
        logger.info(
            "Drag total specific-energy loss: %.9e km^2/s^2",
            drag_validation["total_specific_energy_loss_km2_s2"],
        )
        logger.info(
            "Zero-density limiting-case maximum difference: %.9e m",
            zero_density_maximum_position_difference_m,
        )
        logger.info(
            "Drag sensitivity cases completed: %d",
            len(sensitivity_results),
        )
        logger.info("Validation status: %s", validation_status["overall_status"])
        logger.info("Research Core 1A.7 controlled-orbit pipeline completed successfully.")

        if (
            validation_status["overall_status"] == "failed"
            and config["validation"]["fail_run_on_validation_failure"]
        ):
            raise RuntimeError(
                "One or more scientific validation checks failed. See validation_status.json."
            )
    except Exception:
        logger.exception("Research Core 1A.7 controlled-orbit pipeline failed.")
        raise
    finally:
        close_run_logger(logger)

    evaluated_sensitivity = [
        value for value in sensitivity_checks.values() if value is not None
    ]
    sensitivity_passed = (
        all(evaluated_sensitivity) if evaluated_sensitivity else None
    )
    return ExperimentRunResult(
        experiment_id=experiment_id,
        result_directory=result_directory,
        warnings=tuple(warnings),
        created_files=tuple(created_files),
        validation_status=validation_status["overall_status"],
        maximum_position_difference_m=float(
            two_body_error_summary["position_difference_m"]["maximum_absolute"]
        ),
        maximum_velocity_difference_mm_s=float(
            two_body_error_summary["velocity_difference_mm_s"]["maximum_absolute"]
        ),
        numerical_maximum_relative_energy_drift=float(
            two_body_diagnostics[1]["maximum_absolute_relative_energy_drift"]
        ),
        numerical_maximum_relative_angular_momentum_drift=float(
            two_body_diagnostics[1][
                "maximum_absolute_relative_angular_momentum_drift"
            ]
        ),
        maximum_j2_two_body_position_difference_km=float(
            j2_error_summary["position_difference_m"]["maximum_absolute"]
            / 1000.0
        ),
        analytical_raan_rate_deg_day=float(
            j2_validation["analytical_raan_rate_deg_day"]
        ),
        fitted_raan_rate_deg_day=float(
            j2_validation["fitted_raan_rate_deg_day"]
        ),
        raan_rate_relative_difference=float(
            j2_validation["relative_rate_difference"]
        ),
        j2_maximum_relative_total_energy_drift=float(
            j2_validation["maximum_absolute_relative_total_energy_drift"]
        ),
        j2_maximum_relative_angular_momentum_z_drift=float(
            j2_validation[
                "maximum_absolute_relative_angular_momentum_z_drift"
            ]
        ),
        maximum_drag_j2_position_difference_km=float(
            drag_error_summary["position_difference_m"]["maximum_absolute"]
            / 1000.0
        ),
        final_drag_semi_major_axis_difference_vs_j2_m=float(
            drag_validation["final_semi_major_axis_difference_vs_j2_m"]
        ),
        drag_total_specific_energy_loss_km2_s2=float(
            drag_validation["total_specific_energy_loss_km2_s2"]
        ),
        initial_drag_acceleration_m_s2=float(
            drag_validation["initial_drag_acceleration_m_s2"]
        ),
        zero_density_maximum_position_difference_m=zero_density_maximum_position_difference_m,
        sensitivity_case_count=len(sensitivity_results),
        sensitivity_direction_checks_passed=sensitivity_passed,
    )


run_foundation_experiment = run_experiment
