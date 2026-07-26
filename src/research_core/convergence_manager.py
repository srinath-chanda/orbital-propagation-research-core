"""Orchestration for the Research Core 1A.1C convergence study."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from . import BUILD_MARKER, RESEARCH_CORE_VERSION
from .analysis.comparison import compare_state_histories, create_error_summary
from .analysis.convergence import (
    ConvergenceCase,
    generate_convergence_cases,
    passing_rows,
    reference_case,
    select_recommendations,
)
from .analysis.diagnostics import conservation_diagnostics
from .configuration import load_and_validate_config
from .convergence_outputs import (
    convergence_summary_markdown,
    create_convergence_figures,
    write_convergence_csv,
)
from .data_models import CartesianState, StateHistory
from .logging_utils import close_run_logger, create_run_logger
from .metadata import collect_environment_metadata, utc_now_iso, write_json
from .orbital_elements import elements_from_config, elements_to_cartesian
from .propagators import (
    propagate_analytical_two_body,
    propagate_numerical_two_body,
)
from .time_utils import build_time_grid


@dataclass(frozen=True)
class ConvergenceRunResult:
    experiment_id: str
    result_directory: Path
    matrix_candidate_count: int
    evaluated_setting_count: int
    passing_candidate_count: int
    balanced_case_id: str | None
    validation_status: str
    created_files: tuple[Path, ...]
    warnings: tuple[str, ...]


def _safe_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("Experiment ID cannot be converted into a safe folder name.")
    return cleaned


def _timestamp_folder_name() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")


def _create_result_directory(
    *, project_root: str | Path, results_root: str, experiment_id: str
) -> Path:
    base = Path(results_root).expanduser()
    if not base.is_absolute():
        base = Path(project_root).expanduser().resolve() / base
    convergence_root = (
        base.resolve() / _safe_path_component(experiment_id) / "convergence"
    )
    for _ in range(10):
        candidate = convergence_root / _timestamp_folder_name()
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("Could not create a unique convergence result directory.")


def _run_candidate_repeated(
    case: ConvergenceCase,
    *,
    repetitions: int,
    initial_state: CartesianState,
    mu: float,
    time_grid: np.ndarray,
) -> tuple[StateHistory, list[float]]:
    histories: list[StateHistory] = []
    runtimes: list[float] = []
    for _ in range(repetitions):
        history = propagate_numerical_two_body(
            initial_state,
            mu,
            time_grid,
            method=case.method,
            relative_tolerance=case.relative_tolerance,
            absolute_tolerance=case.absolute_tolerance,
            maximum_step_seconds=case.maximum_step_seconds,
        )
        histories.append(history)
        runtimes.append(float(history.runtime_seconds))
    representative = histories[0]
    return representative, runtimes


def _row_for_case(
    case: ConvergenceCase,
    *,
    history: StateHistory,
    runtimes: list[float],
    analytical: StateHistory,
    numerical_reference: StateHistory,
    mu: float,
    validation_config: dict[str, Any],
    current_integrator: dict[str, Any],
    row_role: str,
) -> dict[str, Any]:
    vs_analytical = create_error_summary(
        compare_state_histories(analytical, history)
    )
    vs_reference = create_error_summary(
        compare_state_histories(numerical_reference, history)
    )
    diagnostics = conservation_diagnostics(history, mu)
    row = {
        "row_role": row_role,
        **case.as_dict(),
        "runtime_repetitions": len(runtimes),
        "minimum_runtime_seconds": float(min(runtimes)),
        "median_runtime_seconds": float(median(runtimes)),
        "maximum_runtime_seconds": float(max(runtimes)),
        "function_evaluations": int(history.function_evaluations or 0),
        "maximum_position_difference_vs_analytical_m": float(
            vs_analytical["position_difference_m"]["maximum_absolute"]
        ),
        "final_position_difference_vs_analytical_m": float(
            vs_analytical["position_difference_m"]["final"]
        ),
        "rms_position_difference_vs_analytical_m": float(
            vs_analytical["position_difference_m"]["rms"]
        ),
        "maximum_velocity_difference_vs_analytical_mm_s": float(
            vs_analytical["velocity_difference_mm_s"]["maximum_absolute"]
        ),
        "maximum_position_difference_vs_reference_m": float(
            vs_reference["position_difference_m"]["maximum_absolute"]
        ),
        "maximum_velocity_difference_vs_reference_mm_s": float(
            vs_reference["velocity_difference_mm_s"]["maximum_absolute"]
        ),
        "maximum_absolute_relative_energy_drift": float(
            diagnostics["maximum_absolute_relative_energy_drift"]
        ),
        "maximum_absolute_relative_angular_momentum_drift": float(
            diagnostics[
                "maximum_absolute_relative_angular_momentum_drift"
            ]
        ),
        "is_current_configuration": bool(
            case.method == current_integrator["method"]
            and np.isclose(
                case.relative_tolerance,
                float(current_integrator["relative_tolerance"]),
                rtol=0.0,
                atol=0.0,
            )
            and np.isclose(
                case.absolute_tolerance,
                float(current_integrator["absolute_tolerance"]),
                rtol=0.0,
                atol=0.0,
            )
            and np.isclose(
                case.maximum_step_seconds,
                float(current_integrator["maximum_step_seconds"]),
                rtol=0.0,
                atol=0.0,
            )
        ),
    }
    passed = passing_rows([row], validation_config)
    row["passes_provisional_thresholds"] = bool(passed)
    return row


def _reference_summary(
    case: ConvergenceCase,
    history: StateHistory,
    analytical: StateHistory,
    mu: float,
) -> dict[str, Any]:
    error = create_error_summary(compare_state_histories(analytical, history))
    diagnostics = conservation_diagnostics(history, mu)
    return {
        **case.as_dict(),
        "runtime_seconds": float(history.runtime_seconds),
        "function_evaluations": int(history.function_evaluations or 0),
        "maximum_position_difference_vs_analytical_m": float(
            error["position_difference_m"]["maximum_absolute"]
        ),
        "maximum_velocity_difference_vs_analytical_mm_s": float(
            error["velocity_difference_mm_s"]["maximum_absolute"]
        ),
        "maximum_absolute_relative_energy_drift": float(
            diagnostics["maximum_absolute_relative_energy_drift"]
        ),
        "maximum_absolute_relative_angular_momentum_drift": float(
            diagnostics[
                "maximum_absolute_relative_angular_momentum_drift"
            ]
        ),
    }


def run_convergence_study(
    config_path: str | Path,
    *,
    project_root: str | Path,
    console_logging: bool = True,
) -> ConvergenceRunResult:
    source_path = Path(config_path).expanduser().resolve()
    config, warnings = load_and_validate_config(source_path)
    convergence = config["convergence"]
    if not convergence["enabled"]:
        raise ValueError(
            "The convergence study is disabled in the configuration."
        )

    experiment_id = config["experiment"]["experiment_id"]
    result_directory = _create_result_directory(
        project_root=project_root,
        results_root=config["outputs"]["results_root"],
        experiment_id=experiment_id,
    )
    logger = create_run_logger(
        result_directory / "run_log.txt", console=console_logging
    )
    created_files: list[Path] = [result_directory / "run_log.txt"]
    run_started_utc = utc_now_iso()

    try:
        logger.info("Research Core version: %s", RESEARCH_CORE_VERSION)
        logger.info("Build marker: %s", BUILD_MARKER)
        logger.info("Convergence configuration: %s", source_path)
        logger.info("Result directory: %s", result_directory)
        for warning in warnings:
            logger.warning("%s", warning)

        resolved = json.loads(json.dumps(config))
        resolved["_runtime"] = {
            "research_core_version": RESEARCH_CORE_VERSION,
            "build_marker": BUILD_MARKER,
            "run_started_utc": run_started_utc,
            "configuration_source": str(source_path),
            "result_directory": str(result_directory),
            "study_type": "numerical_two_body_convergence",
        }
        resolved_path = result_directory / "experiment_configuration.json"
        write_json(resolved, resolved_path)
        created_files.append(resolved_path)

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

        elements = elements_from_config(config["initial_state"])
        mu = float(config["earth_model"]["gravitational_parameter_km3_s2"])
        position, velocity = elements_to_cartesian(elements, mu)
        initial_state = CartesianState(
            epoch_utc=config["initial_state"]["epoch_utc"],
            frame=config["initial_state"]["frame"],
            position_km=position,
            velocity_km_s=velocity,
        )
        time_grid = build_time_grid(
            convergence["duration_hours"],
            convergence["output_step_seconds"],
        )
        analytical = propagate_analytical_two_body(
            elements,
            mu,
            time_grid,
            epoch_utc=initial_state.epoch_utc,
            frame=initial_state.frame,
        )

        reference_settings = reference_case(convergence)
        logger.info(
            "Running numerical reference: rtol=%g atol=%g max_step=%g s",
            reference_settings.relative_tolerance,
            reference_settings.absolute_tolerance,
            reference_settings.maximum_step_seconds,
        )
        numerical_reference = propagate_numerical_two_body(
            initial_state,
            mu,
            time_grid,
            method=reference_settings.method,
            relative_tolerance=reference_settings.relative_tolerance,
            absolute_tolerance=reference_settings.absolute_tolerance,
            maximum_step_seconds=reference_settings.maximum_step_seconds,
        )
        reference_summary = _reference_summary(
            reference_settings,
            numerical_reference,
            analytical,
            mu,
        )
        reference_path = result_directory / "numerical_reference_summary.json"
        write_json(reference_summary, reference_path)
        created_files.append(reference_path)

        cases = generate_convergence_cases(convergence)
        logger.info(
            "Running %d candidate settings with %d runtime repetitions each.",
            len(cases),
            convergence["runtime_repetitions"],
        )
        rows: list[dict[str, Any]] = []
        for index, case in enumerate(cases, start=1):
            logger.info(
                "Candidate %d/%d: %s",
                index,
                len(cases),
                case.case_id,
            )
            history, runtimes = _run_candidate_repeated(
                case,
                repetitions=int(convergence["runtime_repetitions"]),
                initial_state=initial_state,
                mu=mu,
                time_grid=time_grid,
            )
            rows.append(
                _row_for_case(
                    case,
                    history=history,
                    runtimes=runtimes,
                    analytical=analytical,
                    numerical_reference=numerical_reference,
                    mu=mu,
                    validation_config=config["validation"],
                    current_integrator=config["integrator"],
                    row_role="matrix_candidate",
                )
            )

        current_case = ConvergenceCase(
            case_id="CURRENT_CONFIGURATION",
            method=str(config["integrator"]["method"]),
            relative_tolerance=float(
                config["integrator"]["relative_tolerance"]
            ),
            absolute_tolerance=float(
                config["integrator"]["absolute_tolerance"]
            ),
            maximum_step_seconds=float(
                config["integrator"]["maximum_step_seconds"]
            ),
        )
        duplicate_current = next(
            (
                row
                for row in rows
                if row["method"] == current_case.method
                and row["relative_tolerance"]
                == current_case.relative_tolerance
                and row["absolute_tolerance"]
                == current_case.absolute_tolerance
                and row["maximum_step_seconds"]
                == current_case.maximum_step_seconds
            ),
            None,
        )
        if duplicate_current is None:
            logger.info(
                "Running current production configuration as an additional baseline."
            )
            current_history, current_runtimes = _run_candidate_repeated(
                current_case,
                repetitions=int(convergence["runtime_repetitions"]),
                initial_state=initial_state,
                mu=mu,
                time_grid=time_grid,
            )
            current_row = _row_for_case(
                current_case,
                history=current_history,
                runtimes=current_runtimes,
                analytical=analytical,
                numerical_reference=numerical_reference,
                mu=mu,
                validation_config=config["validation"],
                current_integrator=config["integrator"],
                row_role="current_configuration",
            )
            rows.append(current_row)
        else:
            duplicate_current["row_role"] = "matrix_candidate_and_current_configuration"
            current_row = duplicate_current

        current_path = result_directory / "current_configuration_summary.json"
        write_json(current_row, current_path)
        created_files.append(current_path)

        results_csv = result_directory / "convergence_results.csv"
        write_convergence_csv(results_csv, rows)
        created_files.append(results_csv)

        results_json = result_directory / "convergence_results.json"
        write_json(
            {
                "research_core_version": RESEARCH_CORE_VERSION,
                "experiment_id": experiment_id,
                "matrix_candidate_count": len(cases),
                "evaluated_setting_count": len(rows),
                "rows": rows,
            },
            results_json,
        )
        created_files.append(results_json)

        selection = select_recommendations(rows, config["validation"])
        selection.update(
            {
                "research_core_version": RESEARCH_CORE_VERSION,
                "experiment_id": experiment_id,
                "created_utc": utc_now_iso(),
                "threshold_status": config["validation"]["threshold_status"],
                "recommendation_status": (
                    "provisional_until_external_validation_and_review"
                ),
                "current_configuration": config["integrator"],
            }
        )
        selected_path = result_directory / "selected_integrator_settings.json"
        write_json(selection, selected_path)
        created_files.append(selected_path)

        passing_count = int(selection["passing_case_count"])
        reference_position_limit = float(
            config["validation"]["two_body_maximum_position_difference_m"]
        )
        checks = [
            {
                "validation_id": "VAL-CONV-001",
                "name": "High-accuracy numerical reference completed",
                "status": "passed"
                if reference_summary[
                    "maximum_position_difference_vs_analytical_m"
                ]
                <= reference_position_limit
                else "failed",
                "measured_value": reference_summary[
                    "maximum_position_difference_vs_analytical_m"
                ],
                "criterion": (
                    f"Reference maximum position difference <= "
                    f"{reference_position_limit} m"
                ),
            },
            {
                "validation_id": "VAL-CONV-002",
                "name": "Full candidate matrix completed",
                "status": "passed"
                if len(
                    [row for row in rows if row["row_role"].startswith("matrix_candidate")]
                )
                == len(cases)
                else "failed",
                "measured_value": len(
                    [row for row in rows if row["row_role"].startswith("matrix_candidate")]
                ),
                "criterion": f"Exactly {len(cases)} candidate rows",
            },
            {
                "validation_id": "VAL-CONV-003",
                "name": "Runtime and function evaluations recorded",
                "status": "passed"
                if all(
                    row["median_runtime_seconds"] > 0.0
                    and row["function_evaluations"] > 0
                    for row in rows
                )
                else "failed",
                "measured_value": None,
                "criterion": "Positive runtime and function evaluations for every case",
            },
            {
                "validation_id": "VAL-CONV-004",
                "name": "Provisional production recommendation available",
                "status": "passed"
                if selection["balanced_recommendation"] is not None
                else "failed",
                "measured_value": passing_count,
                "criterion": "At least one candidate passes provisional thresholds",
            },
        ]
        failed = [check for check in checks if check["status"] == "failed"]
        validation_status = {
            "research_core_version": RESEARCH_CORE_VERSION,
            "experiment_id": experiment_id,
            "created_utc": utc_now_iso(),
            "stage": "Research Core 1A.1C numerical convergence study",
            "overall_status": "failed"
            if failed
            else ("passed_with_warnings" if warnings else "passed"),
            "checks": checks,
            "warnings": warnings,
            "failed_validation_ids": [
                check["validation_id"] for check in failed
            ],
            "scientific_caution": (
                "This convergence study validates numerical integration of the "
                "point-mass two-body model. It does not validate real-orbit accuracy."
            ),
        }
        validation_path = result_directory / "convergence_validation_status.json"
        write_json(validation_status, validation_path)
        created_files.append(validation_path)

        figures = create_convergence_figures(
            result_directory / "figures",
            rows,
            selection,
            save_png=config["outputs"]["save_png"],
            save_pdf=config["outputs"]["save_pdf"],
        )
        created_files.extend(figures)

        summary_path = result_directory / "CONVERGENCE_SUMMARY.md"
        summary_path.write_text(
            convergence_summary_markdown(
                config=config,
                rows=rows,
                reference_summary=reference_summary,
                selection=selection,
                validation_status=validation_status,
            ),
            encoding="utf-8",
            newline="\n",
        )
        created_files.append(summary_path)

        balanced = selection.get("balanced_recommendation")
        logger.info("Passing candidates: %d/%d", passing_count, len(rows))
        if balanced:
            logger.info(
                "Balanced recommendation: %s, rtol=%g, atol=%g, max_step=%g s",
                balanced["case_id"],
                balanced["relative_tolerance"],
                balanced["absolute_tolerance"],
                balanced["maximum_step_seconds"],
            )
        logger.info("Validation status: %s", validation_status["overall_status"])
        logger.info("Convergence study completed successfully.")
    except Exception:
        logger.exception("Convergence study failed.")
        raise
    finally:
        close_run_logger(logger)

    balanced = selection.get("balanced_recommendation")
    return ConvergenceRunResult(
        experiment_id=experiment_id,
        result_directory=result_directory,
        matrix_candidate_count=len(cases),
        evaluated_setting_count=len(rows),
        passing_candidate_count=passing_count,
        balanced_case_id=balanced["case_id"] if balanced else None,
        validation_status=validation_status["overall_status"],
        created_files=tuple(created_files),
        warnings=tuple(warnings),
    )
