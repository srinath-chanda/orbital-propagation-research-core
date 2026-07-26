"""Frozen Paper 1 production matrix, aggregation, and review packaging."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import re
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .configuration import load_and_validate_config
from .convergence_manager import run_convergence_study
from .experiment_manager import run_experiment
from .gmat_eop_closure import verify_gmat_eop_adoption
from .gmat_gravity_multicase_closure import verify_gravity_multicase_closure
from .tle_experiment_manager import run_tle_experiment


SCHEMA_VERSION = "P1.0"
EXPECTED_CASES = (
    (
        "CASE-LEO400",
        (6, 24, 72, 168),
        (
            "analytical_two_body",
            "numerical_two_body",
            "numerical_j2",
            "numerical_j2_drag",
        ),
    ),
    (
        "CASE-SSO700",
        (6, 24, 72, 168),
        ("analytical_two_body", "numerical_two_body", "numerical_j2"),
    ),
    (
        "CASE-ISS-TLE",
        (6, 24, 72),
        (
            "sgp4",
            "analytical_two_body",
            "numerical_two_body",
            "numerical_j2",
            "numerical_j2_drag",
        ),
    ),
)


@dataclass(frozen=True)
class Paper1BaselineVerification:
    closure_id: str
    evidence_count: int
    drag_scenario_count: int
    drag_check_count: int
    maximum_drag_time_residual_seconds: float
    decision: str


@dataclass(frozen=True)
class Paper1Preparation:
    matrix_id: str
    experiment_count: int
    primary_model_run_count: int
    executed_model_run_count: int
    configuration_directory: Path
    manifest_path: Path


@dataclass(frozen=True)
class Paper1ProductionResult:
    matrix_id: str
    status: str
    completed_experiment_count: int
    expected_experiment_count: int
    failed_experiment_count: int
    result_directory: Path
    summary_path: Path
    report_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def _project_path(root: Path, value: str, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"{label} must be project-relative.")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must remain inside the project.") from exc
    return resolved


def _safe(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not cleaned:
        raise ValueError("Identifier cannot be converted into a safe path component.")
    return cleaned


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")


def load_paper1_matrix(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Paper 1 production schema must be {SCHEMA_VERSION!r}.")
    observed = tuple(
        (
            str(case["case_id"]),
            tuple(int(value) for value in case["durations_hours"]),
            tuple(str(value) for value in case["primary_models"]),
        )
        for case in payload.get("cases", [])
    )
    if observed != EXPECTED_CASES:
        raise ValueError("The Paper 1 case, duration, or primary-model matrix changed.")
    experiment_count = sum(len(item[1]) for item in EXPECTED_CASES)
    model_runs = sum(len(item[1]) * len(item[2]) for item in EXPECTED_CASES)
    executed_runs = sum(
        len(case["durations_hours"]) * len(case["executed_models"])
        for case in payload["cases"]
    )
    if int(payload["primary_experiment_count"]) != experiment_count:
        raise ValueError("Paper 1 primary experiment count must remain 11.")
    if int(payload["primary_model_run_count"]) != model_runs:
        raise ValueError("Paper 1 primary model-run count must remain 43.")
    if int(payload["executed_model_run_count"]) != executed_runs:
        raise ValueError("Paper 1 executed model-run count is inconsistent.")
    expected_integrator = {
        "method": "DOP853",
        "relative_tolerance": 1e-11,
        "absolute_tolerance": 1e-13,
        "maximum_step_seconds": 60.0,
    }
    if payload.get("integrator_freeze") != expected_integrator:
        raise ValueError("The frozen Paper 1 integrator settings changed.")
    if float(payload["output_step_seconds"]) != 60.0:
        raise ValueError("Paper 1 output spacing must remain 60 seconds.")
    rules = payload["production_rules"]
    if rules.get("modify_force_models") or rules.get("modify_acceptance_thresholds"):
        raise ValueError("Paper 1 production may not modify physics or thresholds.")
    return payload


def _verify_result_manifest(result_directory: Path) -> int:
    manifest_path = result_directory / "RUN_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("files", [])
    if not records:
        raise ValueError(f"Result manifest is empty: {manifest_path}")
    for record in records:
        member = (result_directory / record["path"]).resolve()
        try:
            member.relative_to(result_directory.resolve())
        except ValueError as exc:
            raise ValueError("Result manifest member escapes its result directory.") from exc
        if not member.is_file() or _sha256(member) != record["sha256"]:
            raise ValueError(f"Result manifest checksum failed: {member}")
    return len(records)


def verify_paper1_baseline(
    closure_path: str | Path, *, project_root: str | Path
) -> Paper1BaselineVerification:
    root = Path(project_root).resolve()
    closure_file = Path(closure_path).resolve()
    closure = json.loads(closure_file.read_text(encoding="utf-8"))
    if closure.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Paper 1 closure schema is invalid.")
    if closure.get("next_authorized_activity") != "paper1_production_matrix_only":
        raise ValueError("Paper 1 production is not authorized by the closure record.")
    evidence = closure.get("required_evidence", {})
    if set(evidence) != {
        "eop_adoption",
        "gravity_closure",
        "drag_acceleration_summary",
        "fixed_tle",
        "fixed_tle_metadata",
    }:
        raise ValueError("Paper 1 closure evidence set is incomplete.")
    resolved: dict[str, Path] = {}
    for key, record in evidence.items():
        source = _project_path(root, str(record["path"]), key)
        if not source.is_file():
            raise FileNotFoundError(f"Paper 1 prerequisite is missing: {source}")
        if _sha256(source) != str(record["sha256"]):
            raise ValueError(f"Paper 1 prerequisite checksum failed: {key}")
        resolved[key] = source

    eop = verify_gmat_eop_adoption(resolved["eop_adoption"], project_root=root)
    if eop.adoption_decision != "adopt_gmat_r2026a_eop_full_as_validated_baseline":
        raise ValueError("The validated full-EOP baseline is not adopted.")
    gravity = verify_gravity_multicase_closure(
        resolved["gravity_closure"], project_root=root
    )
    if gravity.case_count != 6 or gravity.model_run_count != 24 or gravity.check_count != 96:
        raise ValueError("The closed higher-order-gravity evidence is incomplete.")

    drag = json.loads(resolved["drag_acceleration_summary"].read_text(encoding="utf-8"))
    if (
        drag.get("status") != "passed_with_warnings"
        or int(drag.get("passed_scenario_count", 0)) != 4
        or int(drag.get("passed_check_count", 0)) != 25
        or int(drag.get("failed_check_count", 1)) != 0
    ):
        raise ValueError("The GMAT drag-acceleration gate is not closed.")
    maximum_time = 0.0
    for scenario in drag["scenarios"]:
        source = _project_path(root, scenario["source_report"], "drag source report")
        if _sha256(source) != scenario["source_report_sha256"]:
            raise ValueError(f"Drag source report checksum failed: {source}")
        grid = scenario["time_grid"]
        residual = float(grid["maximum_absolute_raw_time_residual_seconds"])
        tolerance = float(grid["synchronization_tolerance_seconds"])
        if residual > tolerance:
            raise ValueError("A drag time-grid residual exceeds its closed limit.")
        maximum_time = max(maximum_time, residual)
    _verify_result_manifest(resolved["drag_acceleration_summary"].parent)
    return Paper1BaselineVerification(
        closure_id=str(closure["closure_id"]),
        evidence_count=len(evidence),
        drag_scenario_count=int(drag["scenario_count"]),
        drag_check_count=int(drag["check_count"]),
        maximum_drag_time_residual_seconds=maximum_time,
        decision="paper1_production_matrix_authorized",
    )


def _rewrite_config_paths(
    config: dict[str, Any], *, source: Path, destination: Path
) -> None:
    path_fields = [
        (config.get("initial_state", {}), "tle_file"),
        (config.get("initial_state", {}), "tle_metadata_file"),
        (config.get("ground_track_analysis", {}), "map_background_file"),
    ]
    for container, key in path_fields:
        value = container.get(key)
        if not value:
            continue
        absolute = (source.parent / value).resolve()
        container[key] = Path(os.path.relpath(absolute, destination.parent)).as_posix()


def prepare_paper1_production(
    matrix_path: str | Path,
    *,
    project_root: str | Path,
    output_directory: str | Path | None = None,
) -> Paper1Preparation:
    root = Path(project_root).resolve()
    matrix_file = Path(matrix_path).resolve()
    matrix = load_paper1_matrix(matrix_file)
    closure = _project_path(root, matrix["baseline_closure"], "baseline_closure")
    baseline = verify_paper1_baseline(closure, project_root=root)
    destination = (
        Path(output_directory).resolve()
        if output_directory is not None
        else root / "configs" / "paper1_runs"
    )
    destination.mkdir(parents=True, exist_ok=True)
    integrator = matrix["integrator_freeze"]
    records: list[dict[str, Any]] = []
    for case in matrix["cases"]:
        source = _project_path(root, case["base_configuration"], "base_configuration")
        base = json.loads(source.read_text(encoding="utf-8"))
        if base["experiment"]["case_id"] != case["case_id"]:
            raise ValueError(f"Base configuration case mismatch: {source}")
        for duration in case["durations_hours"]:
            token = f"{int(duration):03d}H"
            generated = deepcopy(base)
            generated["experiment"]["experiment_id"] = (
                f"EXP-PAPER1-{case['case_id'].removeprefix('CASE-')}-{token}"
            )
            generated["experiment"]["title"] = (
                f"{base['experiment']['title']} — Paper 1 {int(duration)} h production run"
            )
            generated["experiment"]["description"] = (
                f"Frozen Paper 1 production run from matrix {matrix['matrix_id']}; "
                f"duration {int(duration)} hours and output spacing 60 seconds."
            )
            generated["propagation"]["default_duration_hours"] = int(duration)
            generated["propagation"]["output_step_seconds"] = 60
            generated["propagation"]["models"] = list(case["executed_models"])
            generated["integrator"].update(integrator)
            generated["integrator"]["settings_status"] = (
                "paper1_frozen_after_production_convergence"
            )
            generated["convergence"]["enabled"] = False
            generated["earth_model"]["constants_reference"] = (
                "Frozen Research Core Paper 1 benchmark values recorded explicitly in this "
                "configuration; bibliographic source resolution remains a manuscript-QA item."
            )
            sensitivity = generated.get("drag", {}).get("sensitivity")
            if sensitivity is not None:
                sensitivity["enabled"] = bool(
                    case["case_id"] == "CASE-LEO400" and int(duration) == 24
                )
            generated["production"] = {
                "matrix_id": matrix["matrix_id"],
                "baseline_closure_id": baseline.closure_id,
                "case_id": case["case_id"],
                "duration_hours": int(duration),
                "primary_models": list(case["primary_models"]),
                "executed_models": list(case["executed_models"]),
                "paper_role": case["paper_role"],
            }
            path = destination / f"{case['case_id']}_{token}.json"
            _rewrite_config_paths(generated, source=source, destination=path)
            text = json.dumps(generated, indent=2) + "\n"
            if path.exists() and path.read_text(encoding="utf-8") != text:
                raise FileExistsError(
                    f"A different generated production config already exists: {path}"
                )
            path.write_text(text, encoding="utf-8", newline="\n")
            load_and_validate_config(path)
            records.append(
                {
                    "case_id": case["case_id"],
                    "duration_hours": int(duration),
                    "configuration": str(path),
                    "configuration_sha256": _sha256(path),
                    "primary_model_count": len(case["primary_models"]),
                    "executed_model_count": len(case["executed_models"]),
                }
            )
    manifest = destination / "PAPER1_PRODUCTION_PREPARATION.json"
    _write_json(
        manifest,
        {
            "schema_version": SCHEMA_VERSION,
            "matrix_id": matrix["matrix_id"],
            "matrix_sha256": _sha256(matrix_file),
            "baseline_closure_id": baseline.closure_id,
            "baseline_decision": baseline.decision,
            "experiment_count": len(records),
            "primary_model_run_count": sum(r["primary_model_count"] for r in records),
            "executed_model_run_count": sum(r["executed_model_count"] for r in records),
            "runs": records,
        },
    )
    return Paper1Preparation(
        matrix_id=str(matrix["matrix_id"]),
        experiment_count=len(records),
        primary_model_run_count=sum(r["primary_model_count"] for r in records),
        executed_model_run_count=sum(r["executed_model_count"] for r in records),
        configuration_directory=destination,
        manifest_path=manifest,
    )


def _runtime_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    result = Path(record["result_directory"])
    rows: list[dict[str, Any]] = []
    if record["run_kind"] == "controlled":
        summary = json.loads((result / "model_error_summary.json").read_text(encoding="utf-8"))
        for model, runtime in summary["runtime_seconds"].items():
            rows.append(
                {
                    "case_id": record["case_id"],
                    "duration_hours": record["duration_hours"],
                    "model": model,
                    "runtime_seconds": runtime,
                    "function_evaluations": summary["function_evaluations"].get(model),
                }
            )
    else:
        summary = json.loads(
            (result / "sgp4_model_error_summary.json").read_text(encoding="utf-8")
        )
        diagnostics = json.loads((result / "sgp4_diagnostics.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "case_id": record["case_id"],
                "duration_hours": record["duration_hours"],
                "model": "sgp4",
                "runtime_seconds": diagnostics["sgp4_runtime_seconds"],
                "function_evaluations": None,
            }
        )
        for model, values in summary["models"].items():
            rows.append(
                {
                    "case_id": record["case_id"],
                    "duration_hours": record["duration_hours"],
                    "model": model,
                    "runtime_seconds": values["runtime_seconds"],
                    "function_evaluations": values["function_evaluations"],
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _create_figures(
    result_directory: Path,
    records: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
) -> list[Path]:
    created: list[Path] = []
    figure, axis = plt.subplots(figsize=(9, 5.5))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in runtime_rows:
        groups.setdefault((row["case_id"], row["model"]), []).append(row)
    for (case_id, model), rows in sorted(groups.items()):
        rows.sort(key=lambda item: item["duration_hours"])
        axis.plot(
            [item["duration_hours"] for item in rows],
            [item["runtime_seconds"] for item in rows],
            marker="o",
            label=f"{case_id.removeprefix('CASE-')} — {model}",
        )
    axis.set_xlabel("Propagation duration (hours)")
    axis.set_ylabel("Wall-clock runtime (seconds)")
    axis.set_yscale("log")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        path = result_directory / f"paper1_runtime_vs_duration.{suffix}"
        figure.savefig(path, dpi=220)
        created.append(path)
    plt.close(figure)

    controlled = [item for item in records if item["run_kind"] == "controlled"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for case_id in ("CASE-LEO400", "CASE-SSO700"):
        rows = sorted(
            (item for item in controlled if item["case_id"] == case_id),
            key=lambda item: item["duration_hours"],
        )
        axes[0].plot(
            [item["duration_hours"] for item in rows],
            [item["maximum_j2_two_body_position_difference_km"] for item in rows],
            marker="o",
            label=case_id,
        )
        axes[1].plot(
            [item["duration_hours"] for item in rows],
            [item["maximum_drag_j2_position_difference_km"] for item in rows],
            marker="o",
            label=case_id,
        )
    axes[0].set_title("J2 separation from two-body")
    axes[1].set_title("Simplified-drag separation from J2")
    for axis in axes:
        axis.set_xlabel("Propagation duration (hours)")
        axis.set_ylabel("Maximum position separation (km)")
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        path = result_directory / f"paper1_controlled_effects_vs_duration.{suffix}"
        figure.savefig(path, dpi=220)
        created.append(path)
    plt.close(figure)

    tle = sorted(
        (item for item in records if item["run_kind"] == "fixed_tle"),
        key=lambda item: item["duration_hours"],
    )
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    if tle:
        models = sorted(tle[0]["maximum_separation_km_by_model"])
        for model in models:
            axis.plot(
                [item["duration_hours"] for item in tle],
                [item["maximum_separation_km_by_model"][model] for item in tle],
                marker="o",
                label=model,
            )
    axis.set_xlabel("Propagation duration from fixed TLE epoch (hours)")
    axis.set_ylabel("Maximum position separation from SGP4 (km)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        path = result_directory / f"paper1_sgp4_separation_vs_duration.{suffix}"
        figure.savefig(path, dpi=220)
        created.append(path)
    plt.close(figure)
    return created


def _report_html(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['case_id'])}</td>"
        f"<td>{item['duration_hours']}</td>"
        f"<td>{html.escape(item['run_kind'])}</td>"
        f"<td>{html.escape(item['validation_status'])}</td>"
        f"<td><code>{html.escape(item['result_directory'])}</code></td>"
        "</tr>"
        for item in records
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Paper 1 Production</title>
<style>body{{font-family:Arial,sans-serif;max-width:1200px;margin:2rem auto;line-height:1.45}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:.45rem;text-align:left}}
code{{font-size:.85em}}img{{max-width:100%;height:auto}}</style></head><body>
<h1>Paper 1 Production Matrix</h1>
<p><strong>Status:</strong> {html.escape(summary['status'])}</p>
<p><strong>Completed experiments:</strong> {summary['completed_experiment_count']}/{summary['expected_experiment_count']}</p>
<p>This production run uses the frozen Paper 1 model scope. Simplified drag is a sensitivity model, and SGP4 is not measured-orbit truth.</p>
<table><thead><tr><th>Case</th><th>Hours</th><th>Kind</th><th>Status</th><th>Result folder</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Aggregate figures</h2>
<img src="paper1_runtime_vs_duration.png" alt="Runtime versus duration">
<img src="paper1_controlled_effects_vs_duration.png" alt="Controlled model effects">
<img src="paper1_sgp4_separation_vs_duration.png" alt="SGP4 separation">
</body></html>"""


def run_paper1_production(
    matrix_path: str | Path,
    *,
    project_root: str | Path,
    progress: Callable[[str], None] | None = None,
) -> Paper1ProductionResult:
    root = Path(project_root).resolve()
    matrix_file = Path(matrix_path).resolve()
    matrix = load_paper1_matrix(matrix_file)
    preparation = prepare_paper1_production(matrix_file, project_root=root)
    manifest = json.loads(preparation.manifest_path.read_text(encoding="utf-8"))
    for record in manifest["runs"]:
        path = Path(record["configuration"])
        if not path.is_file() or _sha256(path) != record["configuration_sha256"]:
            raise ValueError(f"Prepared production configuration changed: {path}")

    result_directory = root / "results" / _safe(matrix["matrix_id"]) / _timestamp()
    result_directory.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    convergence_record: dict[str, Any] | None = None

    if matrix["run_convergence_first"]:
        if progress:
            progress("Running the frozen CASE-LEO400 convergence study...")
        convergence_config = _project_path(
            root, matrix["convergence_configuration"], "convergence_configuration"
        )
        convergence = run_convergence_study(
            convergence_config, project_root=root, console_logging=False
        )
        convergence_record = {
            "validation_status": convergence.validation_status,
            "result_directory": str(convergence.result_directory),
            "evaluated_setting_count": convergence.evaluated_setting_count,
            "passing_candidate_count": convergence.passing_candidate_count,
            "balanced_case_id": convergence.balanced_case_id,
        }
        _write_json(result_directory / "paper1_convergence_record.json", convergence_record)

    for index, prepared in enumerate(manifest["runs"], start=1):
        config_path = Path(prepared["configuration"])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        case_id = str(prepared["case_id"])
        duration = int(prepared["duration_hours"])
        if progress:
            progress(
                f"Running production experiment {index}/{manifest['experiment_count']}: "
                f"{case_id}, {duration} h"
            )
        try:
            if config["initial_state"]["source_type"] == "fixed_tle":
                result = run_tle_experiment(
                    config_path, project_root=root
                )
                row = {
                    "case_id": case_id,
                    "duration_hours": duration,
                    "run_kind": "fixed_tle",
                    "validation_status": result.validation_status,
                    "result_directory": str(result.result_directory),
                    "maximum_separation_km_by_model": result.maximum_separation_km_by_model,
                    "final_separation_km_by_model": result.final_separation_km_by_model,
                    "pass_count_by_model": result.pass_count_by_model,
                    "matched_pass_count_by_model": result.matched_pass_count_by_model,
                    "maximum_absolute_aos_difference_seconds_by_model": result.maximum_absolute_aos_difference_seconds_by_model,
                    "maximum_absolute_los_difference_seconds_by_model": result.maximum_absolute_los_difference_seconds_by_model,
                }
            else:
                result = run_experiment(
                    config_path, project_root=root, console_logging=False
                )
                row = {
                    "case_id": case_id,
                    "duration_hours": duration,
                    "run_kind": "controlled",
                    "validation_status": result.validation_status,
                    "result_directory": str(result.result_directory),
                    "maximum_two_body_position_difference_m": result.maximum_position_difference_m,
                    "maximum_two_body_velocity_difference_mm_s": result.maximum_velocity_difference_mm_s,
                    "maximum_j2_two_body_position_difference_km": result.maximum_j2_two_body_position_difference_km,
                    "analytical_raan_rate_deg_day": result.analytical_raan_rate_deg_day,
                    "fitted_raan_rate_deg_day": result.fitted_raan_rate_deg_day,
                    "raan_rate_relative_difference": result.raan_rate_relative_difference,
                    "maximum_drag_j2_position_difference_km": result.maximum_drag_j2_position_difference_km,
                    "final_drag_semi_major_axis_difference_vs_j2_m": result.final_drag_semi_major_axis_difference_vs_j2_m,
                    "drag_total_specific_energy_loss_km2_s2": result.drag_total_specific_energy_loss_km2_s2,
                }
            records.append(row)
        except Exception as exc:
            failures.append(
                {
                    "case_id": case_id,
                    "duration_hours": duration,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            _write_json(result_directory / "paper1_failures.json", failures)
            break
        _write_json(result_directory / "paper1_progress.json", records)

    runtime_rows = [row for record in records for row in _runtime_rows(record)]
    _write_csv(result_directory / "paper1_runtime_summary.csv", runtime_rows)
    flat_rows = [
        {
            "case_id": item["case_id"],
            "duration_hours": item["duration_hours"],
            "run_kind": item["run_kind"],
            "validation_status": item["validation_status"],
            "result_directory": item["result_directory"],
        }
        for item in records
    ]
    _write_csv(result_directory / "paper1_production_runs.csv", flat_rows)
    complete = len(records) == int(manifest["experiment_count"]) and not failures
    statuses_ok = all(
        item["validation_status"] in {"passed", "passed_with_warnings"}
        for item in records
    )
    convergence_ok = convergence_record is None or convergence_record[
        "validation_status"
    ] in {"passed", "passed_with_warnings"}
    status = (
        "passed_with_warnings"
        if complete and statuses_ok and convergence_ok
        else "failed_or_incomplete"
    )
    if complete:
        _create_figures(result_directory, records, runtime_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "matrix_id": matrix["matrix_id"],
        "paper_title": matrix["paper_title"],
        "status": status,
        "completed_experiment_count": len(records),
        "expected_experiment_count": int(manifest["experiment_count"]),
        "failed_experiment_count": len(failures),
        "primary_model_run_count": int(manifest["primary_model_run_count"]),
        "executed_model_run_count": int(manifest["executed_model_run_count"]),
        "convergence": convergence_record,
        "runs": records,
        "failures": failures,
        "scope_decision": "paper1_models_frozen_no_additional_force_models",
    }
    summary_path = result_directory / "paper1_production_summary.json"
    _write_json(summary_path, summary)
    report_path = result_directory / "PAPER1_PRODUCTION_REPORT.html"
    report_path.write_text(
        _report_html(summary, records), encoding="utf-8", newline="\n"
    )
    files = [
        path for path in result_directory.rglob("*") if path.is_file() and path.name != "RUN_MANIFEST.json"
    ]
    _write_json(
        result_directory / "RUN_MANIFEST.json",
        {
            "files": [
                {
                    "path": path.relative_to(result_directory).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in sorted(files)
            ]
        },
    )
    return Paper1ProductionResult(
        matrix_id=str(matrix["matrix_id"]),
        status=status,
        completed_experiment_count=len(records),
        expected_experiment_count=int(manifest["experiment_count"]),
        failed_experiment_count=len(failures),
        result_directory=result_directory,
        summary_path=summary_path,
        report_path=report_path,
    )


def package_paper1_production_results(
    matrix_path: str | Path,
    *,
    project_root: str | Path,
    output_path: str | Path,
) -> Path:
    root = Path(project_root).resolve()
    matrix_file = Path(matrix_path).resolve()
    matrix = load_paper1_matrix(matrix_file)
    result_root = root / "results" / _safe(matrix["matrix_id"])
    completed = sorted(
        path
        for path in result_root.glob("*")
        if path.is_dir() and (path / "paper1_production_summary.json").is_file()
    )
    if not completed:
        raise FileNotFoundError("No Paper 1 production result is available to package.")
    official = completed[-1]
    summary = json.loads(
        (official / "paper1_production_summary.json").read_text(encoding="utf-8")
    )
    if summary["status"] != "passed_with_warnings":
        raise ValueError("The latest Paper 1 production matrix is not complete and passing.")
    members: set[Path] = {
        matrix_file,
        _project_path(root, matrix["baseline_closure"], "baseline_closure"),
    }
    generated = root / "configs" / "paper1_runs"
    members.update(path for path in generated.rglob("*") if path.is_file())
    members.update(path for path in official.rglob("*") if path.is_file())
    selected_names = {
        "experiment_configuration.json",
        "environment_metadata.json",
        "initial_conditions.csv",
        "orbit_summary.json",
        "model_error_summary.json",
        "j2_validation_summary.json",
        "drag_validation_summary.json",
        "drag_sensitivity.csv",
        "drag_sensitivity_summary.json",
        "sgp4_model_error_summary.json",
        "sgp4_model_error_summary.csv",
        "ground_track_summary.json",
        "pass_analysis_summary.json",
        "tle_age_report.csv",
        "validation_status.json",
        "FINAL_VALIDATION_SUMMARY.md",
        "RUN_MANIFEST.json",
    }
    for run in summary["runs"]:
        result = Path(run["result_directory"])
        for name in selected_names:
            candidate = result / name
            if candidate.is_file():
                members.add(candidate)
    archive = Path(output_path).resolve()
    if archive.exists():
        raise FileExistsError(f"Move or rename the existing package first: {archive}")
    temporary = archive.with_name(f".{archive.name}.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
        for member in sorted(members):
            try:
                name = member.relative_to(root).as_posix()
            except ValueError:
                name = f"external_results/{member.parent.name}/{member.name}"
            stream.write(member, name)
    os.replace(temporary, archive)
    with zipfile.ZipFile(archive) as stream:
        bad = stream.testzip()
        if bad:
            raise RuntimeError(f"Paper 1 package failed ZIP verification at {bad}.")
    return archive
