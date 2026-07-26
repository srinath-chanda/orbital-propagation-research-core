"""Research Core 1C.2 independent GMAT validation of the full-EOP candidate."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import shutil
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time, TimeDelta

from .analysis.comparison import compare_state_histories, create_error_summary
from .analysis.j2 import compare_in_reference_rtn
from .earth_orientation import earth_pole_unit_vector
from .external_validation import (
    _canonicalize_nominal_output_grid,
    _initial_difference,
    _maximum_step_deviation_seconds,
    build_gmat_script,
    initial_state_from_config,
    parse_stk_time_pos_vel,
)
from .gmat_eop import GmatEopDataset, gmat_r2026a_eop_pole_unit_vector
from .gmat_multicase import build_gmat_multicase_master_script
from .outputs import (
    write_comparison_csv,
    write_json,
    write_rtn_comparison_csv,
    write_state_history_csv,
)
from .propagators.numerical_j2 import propagate_numerical_j2_pole_provider
from .propagators.numerical_two_body import propagate_numerical_two_body


SCHEMA_VERSION = "1C.2"
CANDIDATE_MODEL = "gmat_r2026a_eop_full"
CLOSED_BASELINE_MODEL = "iau1976_1980"


@dataclass(frozen=True)
class PreparedIndependentMatrix:
    matrix_id: str
    reference_root: Path
    manifest_path: Path
    master_script: Path
    run_order_path: Path
    case_count: int
    expected_output_count: int
    archived_outputs: tuple[Path, ...]


@dataclass(frozen=True)
class IndependentValidationResult:
    matrix_id: str
    result_directory: Path
    validation_status: str
    adoption_decision: str
    case_count: int
    passed_case_count: int
    failed_case_count: int
    incomplete_case_count: int
    summary_json: Path
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
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _resolve_project_path(value: str, root: Path, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"{label} must be project-relative.")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must remain inside the project root.") from exc
    return resolved


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _thresholds_for_duration(
    spec: dict[str, Any], duration_hours: float
) -> tuple[float, float]:
    for tier in spec["threshold_policy"]["candidate_duration_tiers"]:
        if duration_hours <= float(tier["maximum_duration_hours"]):
            return (
                float(tier["maximum_position_difference_m"]),
                float(tier["maximum_velocity_difference_mm_s"]),
            )
    raise ValueError(
        f"No candidate threshold tier covers {duration_hours:g} hours."
    )


def load_independent_matrix_spec(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Independent matrix schema must be {SCHEMA_VERSION!r}.")
    if payload.get("candidate_model") != CANDIDATE_MODEL:
        raise ValueError(f"candidate_model must be {CANDIDATE_MODEL!r}.")
    if payload.get("closed_baseline_model") != CLOSED_BASELINE_MODEL:
        raise ValueError(f"closed_baseline_model must be {CLOSED_BASELINE_MODEL!r}.")
    if payload.get("preregistration_status") != "frozen_before_any_1c2_gmat_output":
        raise ValueError("The independent matrix must be explicitly preregistered.")
    expected_hash = str(payload.get("eop_expected_sha256", ""))
    if len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash):
        raise ValueError("eop_expected_sha256 must be a lowercase SHA-256 digest.")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) < 3:
        raise ValueError("The independent matrix requires at least three cases.")
    required = {
        "case_id",
        "factor",
        "epoch_utc",
        "altitude_km",
        "eccentricity",
        "inclination_deg",
        "raan_deg",
        "argument_of_perigee_deg",
        "true_anomaly_deg",
        "duration_hours",
    }
    seen: set[str] = set()
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Independent case {index} must be an object.")
        missing = sorted(required - set(case))
        if missing:
            raise ValueError(f"Independent case {index} is missing: {', '.join(missing)}.")
        case_id = str(case["case_id"])
        if not case_id or not case_id[0].isalpha() or not case_id.replace("_", "").isalnum():
            raise ValueError(f"Invalid independent case_id: {case_id!r}.")
        if case_id in seen:
            raise ValueError(f"Duplicate independent case_id: {case_id}.")
        seen.add(case_id)
        altitude = float(case["altitude_km"])
        eccentricity = float(case["eccentricity"])
        inclination = float(case["inclination_deg"])
        duration = float(case["duration_hours"])
        if altitude <= 100.0:
            raise ValueError(f"{case_id} altitude must exceed 100 km.")
        if not 0.0 <= eccentricity < 0.2:
            raise ValueError(f"{case_id} eccentricity must be in [0, 0.2).")
        if not 0.0 <= inclination <= 180.0:
            raise ValueError(f"{case_id} inclination must be in [0, 180] degrees.")
        if duration <= 0.0:
            raise ValueError(f"{case_id} duration must be positive.")
        for angle in ("raan_deg", "argument_of_perigee_deg", "true_anomaly_deg"):
            if not 0.0 <= float(case[angle]) < 360.0:
                raise ValueError(f"{case_id} {angle} must be in [0, 360).")
        datetime.fromisoformat(str(case["epoch_utc"]).replace("Z", "+00:00"))
        _thresholds_for_duration(payload, duration)
    policy = payload.get("adoption_rule", {})
    if not all(
        bool(policy.get(name))
        for name in (
            "all_cases_must_pass",
            "missing_or_checksum_invalid_evidence_prohibits_adoption",
            "thresholds_must_not_be_changed_after_output_inspection",
            "candidate_is_adopted_only_after_complete_independent_pass",
        )
    ):
        raise ValueError("All independent adoption safeguards must be enabled.")
    return payload


def _case_configuration(
    baseline: dict[str, Any], spec: dict[str, Any], case: dict[str, Any]
) -> dict[str, Any]:
    config = deepcopy(baseline)
    case_id = str(case["case_id"])
    duration_hours = float(case["duration_hours"])
    earth_radius = float(config["earth_model"]["equatorial_radius_km"])
    position_gate, velocity_gate = _thresholds_for_duration(spec, duration_hours)
    policy = spec["threshold_policy"]
    config["experiment"].update(
        {
            "experiment_id": f"EXP-GMAT-EOP-1C2-{case_id.replace('_', '-')}",
            "case_id": f"CASE-GMAT-EOP-1C2-{case_id.replace('_', '-')}",
            "title": f"Independent full-EOP GMAT validation: {case_id}",
            "description": (
                "Preregistered Research Core 1C.2 holdout comparison against "
                f"GMAT R2026a; design role={case['factor']}."
            ),
        }
    )
    config["initial_state"].update(
        {
            "epoch_utc": str(case["epoch_utc"]),
            "semi_major_axis_km": earth_radius + float(case["altitude_km"]),
            "eccentricity": float(case["eccentricity"]),
            "inclination_deg": float(case["inclination_deg"]),
            "raan_deg": float(case["raan_deg"]),
            "argument_of_perigee_deg": float(case["argument_of_perigee_deg"]),
            "true_anomaly_deg": float(case["true_anomaly_deg"]),
            "notes": (
                f"Independent Research Core 1C.2 holdout {case_id}; all elements "
                "were frozen before new GMAT output."
            ),
        }
    )
    config["propagation"].update(
        {
            "default_duration_hours": duration_hours,
            "output_step_seconds": float(spec["output_step_seconds"]),
            "comparison_reference_model": "numerical_j2",
        }
    )
    config["external_validation"].update(
        {
            "duration_seconds": duration_hours * 3600.0,
            "output_step_seconds": float(spec["output_step_seconds"]),
            "threshold_status": "preregistered_before_1c2_gmat_execution",
            "python_models": [
                "numerical_two_body",
                CLOSED_BASELINE_MODEL,
                CANDIDATE_MODEL,
            ],
            "thresholds": {
                "initial_position_difference_m": float(policy["initial_position_difference_m"]),
                "initial_velocity_difference_mm_s": float(policy["initial_velocity_difference_mm_s"]),
                "two_body_maximum_position_difference_m": float(
                    policy["two_body_maximum_position_difference_m"]
                ),
                "two_body_maximum_velocity_difference_mm_s": float(
                    policy["two_body_maximum_velocity_difference_mm_s"]
                ),
                "j2_maximum_position_difference_m": position_gate,
                "j2_maximum_velocity_difference_mm_s": velocity_gate,
            },
            "candidate_model": CANDIDATE_MODEL,
            "closed_baseline_model": CLOSED_BASELINE_MODEL,
            "eop_file": str(spec["eop_file"]),
            "eop_expected_sha256": str(spec["eop_expected_sha256"]),
        }
    )
    config["external_validation"]["acceleration_diagnostic"]["enabled"] = False
    config["external_validation"]["short_arc"]["enabled"] = False
    config["validation"]["threshold_status"] = "gmat_eop_1c2_execution_pending"
    config["scientific_cautions"] = [
        *config["scientific_cautions"],
        "This is a preregistered independent holdout case for the selected full-EOP candidate.",
        "Thresholds must not be changed after any 1C.2 GMAT output is inspected.",
        "The closed baseline is diagnostic; only the full-EOP candidate controls adoption.",
    ]
    return config


def _archive_outputs(output_dir: Path, expected_names: set[str]) -> tuple[Path, ...]:
    existing = [path for path in output_dir.glob("*.e") if path.name in expected_names]
    if not existing:
        return ()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")
    archive_dir = output_dir.parent / "archive" / stamp
    archive_dir.mkdir(parents=True, exist_ok=False)
    archived = []
    for source in existing:
        destination = archive_dir / source.name
        shutil.move(str(source), str(destination))
        archived.append(destination)
    return tuple(archived)


def prepare_independent_matrix(
    matrix_path: str | Path, *, project_root: str | Path
) -> PreparedIndependentMatrix:
    root = Path(project_root).resolve()
    matrix_file = Path(matrix_path).resolve()
    spec = load_independent_matrix_spec(matrix_file)
    baseline_path = _resolve_project_path(
        str(spec["baseline_configuration"]), root, "baseline_configuration"
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    reference_root = _resolve_project_path(
        str(spec["reference_root"]), root, "reference_root"
    )
    eop_path = _resolve_project_path(str(spec["eop_file"]), root, "eop_file")
    dataset = GmatEopDataset.from_file(
        eop_path, expected_sha256=str(spec["eop_expected_sha256"])
    )
    scripts_dir = reference_root / "scripts"
    output_dir = reference_root / "output"
    cases_dir = reference_root / "cases"
    for directory in (scripts_dir, output_dir, cases_dir):
        directory.mkdir(parents=True, exist_ok=True)
    expected_names = {
        f"{case['case_id']}_{suffix}.e"
        for case in spec["cases"]
        for suffix in ("TWO_BODY", "J2")
    }
    archived = _archive_outputs(output_dir, expected_names)
    prepared_cases: list[dict[str, Any]] = []
    master_cases: list[tuple[dict[str, Any], Path, Path]] = []
    for case in spec["cases"]:
        case_id = str(case["case_id"])
        config = _case_configuration(baseline, spec, case)
        config_path = cases_dir / f"{case_id}.json"
        two_body_output = output_dir / f"{case_id}_TWO_BODY.e"
        j2_output = output_dir / f"{case_id}_J2.e"
        two_body_script = scripts_dir / f"{case_id}_TWO_BODY.script"
        j2_script = scripts_dir / f"{case_id}_J2.script"
        _write_json(config_path, config)
        two_body_script.write_text(
            build_gmat_script(
                config,
                model="two_body",
                output_ephemeris=two_body_output,
                stage_label="1C2_independent_holdout",
                generated_release="1C.2",
            ),
            encoding="utf-8",
            newline="\n",
        )
        j2_script.write_text(
            build_gmat_script(
                config,
                model="j2",
                output_ephemeris=j2_output,
                stage_label="1C2_independent_holdout",
                generated_release="1C.2",
            ),
            encoding="utf-8",
            newline="\n",
        )
        master_cases.append((config, two_body_output, j2_output))
        start = Time(str(case["epoch_utc"]), scale="utc")
        end = start + TimeDelta(float(case["duration_hours"]) * 3600.0, format="sec")
        start_sample = dataset.sample(start)
        end_sample = dataset.sample(end)
        if "clamped" in start_sample.coverage_status or "clamped" in end_sample.coverage_status:
            raise ValueError(f"{case_id} lies outside the frozen GMAT EOP coverage.")
        prepared_cases.append(
            {
                **case,
                "configuration": _relative(config_path, root),
                "configuration_sha256": _sha256(config_path),
                "two_body_script": _relative(two_body_script, root),
                "two_body_script_sha256": _sha256(two_body_script),
                "j2_script": _relative(j2_script, root),
                "j2_script_sha256": _sha256(j2_script),
                "two_body_output": _relative(two_body_output, root),
                "j2_output": _relative(j2_output, root),
                "preregistered_thresholds": config["external_validation"]["thresholds"],
                "eop_start_coverage": start_sample.coverage_status,
                "eop_end_coverage": end_sample.coverage_status,
            }
        )
    master_script = scripts_dir / "RUN_ALL_CASES_1C2.script"
    master_script.write_text(
        build_gmat_multicase_master_script(
            master_cases,
            tool_version=str(spec["tool_version"]),
            script_title="Research Core 1C.2 independent full-EOP validation master script",
        ),
        encoding="utf-8",
        newline="\n",
    )
    run_order = reference_root / "RUN_ORDER_1C2.txt"
    run_order.write_text(
        "\n".join(
            [
                "RESEARCH CORE 1C.2 INDEPENDENT GMAT RUN ORDER",
                "",
                "Preferred: open scripts/RUN_ALL_CASES_1C2.script in GMAT R2026a and run once.",
                "Fallback: run each TWO_BODY script followed by its matching J2 script.",
                "",
                *[
                    f"{index:02d}. {case['case_id']}: TWO_BODY, then J2"
                    for index, case in enumerate(spec["cases"], start=1)
                ],
                "",
                f"Expected ephemeris files: {len(spec['cases']) * 2}",
                "Do not edit cases, thresholds, the EOP file, or outputs.",
                "Retain all outputs even if a later Python gate fails.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "research_core_version": "1C.2",
        "matrix_id": spec["matrix_id"],
        "status": "scripts_prepared_independent_gmat_execution_pending",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "preregistration_status": spec["preregistration_status"],
        "matrix_source": _relative(matrix_file, root),
        "matrix_source_sha256": _sha256(matrix_file),
        "baseline_configuration": _relative(baseline_path, root),
        "baseline_configuration_sha256": _sha256(baseline_path),
        "eop_file": _relative(eop_path, root),
        "eop_file_sha256": dataset.source_sha256,
        "eop_first_mjd": dataset.first_mjd_utc,
        "eop_last_mjd": dataset.last_mjd_utc,
        "candidate_model": CANDIDATE_MODEL,
        "closed_baseline_model": CLOSED_BASELINE_MODEL,
        "case_count": len(prepared_cases),
        "expected_output_count": len(prepared_cases) * 2,
        "archived_previous_output_count": len(archived),
        "master_script": _relative(master_script, root),
        "master_script_sha256": _sha256(master_script),
        "run_order": _relative(run_order, root),
        "threshold_policy": spec["threshold_policy"],
        "adoption_rule": spec["adoption_rule"],
        "cases": prepared_cases,
    }
    manifest_path = reference_root / "GMAT_1C2_MATRIX_MANIFEST.json"
    _write_json(manifest_path, manifest)
    return PreparedIndependentMatrix(
        matrix_id=str(spec["matrix_id"]),
        reference_root=reference_root,
        manifest_path=manifest_path,
        master_script=master_script,
        run_order_path=run_order,
        case_count=len(prepared_cases),
        expected_output_count=len(prepared_cases) * 2,
        archived_outputs=archived,
    )


def _check(identifier: str, name: str, value: float | str, criterion: str, passed: bool) -> dict[str, Any]:
    return {
        "validation_id": identifier,
        "name": name,
        "measured_value": value,
        "criterion": criterion,
        "status": "passed" if passed else "failed",
    }


def _run_case(
    *,
    case: dict[str, Any],
    config_path: Path,
    two_body_path: Path,
    j2_path: Path,
    dataset: GmatEopDataset,
    root: Path,
    case_result_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frame = str(config["external_validation"]["frame"])
    expected_step = float(config["external_validation"]["output_step_seconds"])
    duration = float(config["external_validation"]["duration_seconds"])
    gmat_tb = parse_stk_time_pos_vel(two_body_path, model_name="gmat_two_body", frame=frame)
    gmat_j2 = parse_stk_time_pos_vel(j2_path, model_name="gmat_j2", frame=frame)
    if gmat_tb.epoch_utc != gmat_j2.epoch_utc:
        raise ValueError("GMAT two-body and J2 files have different epochs.")
    gmat_tb, tb_grid = _canonicalize_nominal_output_grid(
        gmat_tb,
        expected_step_seconds=expected_step,
        expected_duration_seconds=duration,
        tolerance_seconds=1.0e-6,
    )
    gmat_j2, j2_grid = _canonicalize_nominal_output_grid(
        gmat_j2,
        expected_step_seconds=expected_step,
        expected_duration_seconds=duration,
        tolerance_seconds=1.0e-6,
    )
    if not np.array_equal(gmat_tb.elapsed_seconds, gmat_j2.elapsed_seconds):
        raise ValueError("Canonical GMAT two-body and J2 grids differ.")
    state = initial_state_from_config(config)
    if gmat_tb.epoch_utc != state.epoch_utc:
        raise ValueError("GMAT ScenarioEpoch does not match the frozen configuration.")
    earth = config["earth_model"]
    integrator = config["integrator"]
    times = gmat_tb.elapsed_seconds
    mu = float(earth["gravitational_parameter_km3_s2"])
    radius = float(earth["equatorial_radius_km"])
    j2 = float(earth["j2"])
    kwargs = {
        "method": str(integrator["method"]),
        "relative_tolerance": float(integrator["relative_tolerance"]),
        "absolute_tolerance": float(integrator["absolute_tolerance"]),
        "maximum_step_seconds": float(integrator["maximum_step_seconds"]),
    }
    python_tb = propagate_numerical_two_body(state, mu, times, **kwargs)
    baseline_provider = partial(earth_pole_unit_vector, model=CLOSED_BASELINE_MODEL)
    candidate_provider = partial(
        gmat_r2026a_eop_pole_unit_vector,
        dataset=dataset,
        model=CANDIDATE_MODEL,
    )
    python_baseline = propagate_numerical_j2_pole_provider(
        state,
        mu,
        radius,
        j2,
        times,
        pole_provider=baseline_provider,
        model_name=f"numerical_j2_{CLOSED_BASELINE_MODEL}",
        **kwargs,
    )
    python_candidate = propagate_numerical_j2_pole_provider(
        state,
        mu,
        radius,
        j2,
        times,
        pole_provider=candidate_provider,
        model_name=f"numerical_j2_{CANDIDATE_MODEL}",
        **kwargs,
    )
    histories = {
        "gmat_two_body": gmat_tb,
        "gmat_j2": gmat_j2,
        "python_two_body": python_tb,
        "python_closed_baseline": python_baseline,
        "python_full_eop_candidate": python_candidate,
    }
    comparisons = {
        "two_body": compare_state_histories(gmat_tb, python_tb),
        "closed_baseline": compare_state_histories(gmat_j2, python_baseline),
        "full_eop_candidate": compare_state_histories(gmat_j2, python_candidate),
    }
    rtn = {
        "two_body": compare_in_reference_rtn(gmat_tb, python_tb),
        "closed_baseline": compare_in_reference_rtn(gmat_j2, python_baseline),
        "full_eop_candidate": compare_in_reference_rtn(gmat_j2, python_candidate),
    }
    summaries = {name: create_error_summary(value) for name, value in comparisons.items()}
    thresholds = config["external_validation"]["thresholds"]
    tb_initial_pos, tb_initial_vel = _initial_difference(state, gmat_tb)
    j2_initial_pos, j2_initial_vel = _initial_difference(state, gmat_j2)
    start = Time(state.epoch_utc, scale="utc")
    end = start + TimeDelta(duration, format="sec")
    start_sample = dataset.sample(start)
    end_sample = dataset.sample(end)
    candidate_position = float(
        summaries["full_eop_candidate"]["position_difference_m"]["maximum_absolute"]
    )
    candidate_velocity = float(
        summaries["full_eop_candidate"]["velocity_difference_mm_s"]["maximum_absolute"]
    )
    tb_position = float(summaries["two_body"]["position_difference_m"]["maximum_absolute"])
    tb_velocity = float(summaries["two_body"]["velocity_difference_mm_s"]["maximum_absolute"])
    checks = [
        _check("1C2-001", "Two-body initial position", tb_initial_pos, f"<= {thresholds['initial_position_difference_m']} m", tb_initial_pos <= float(thresholds["initial_position_difference_m"])),
        _check("1C2-002", "Two-body initial velocity", tb_initial_vel, f"<= {thresholds['initial_velocity_difference_mm_s']} mm/s", tb_initial_vel <= float(thresholds["initial_velocity_difference_mm_s"])),
        _check("1C2-003", "J2 initial position", j2_initial_pos, f"<= {thresholds['initial_position_difference_m']} m", j2_initial_pos <= float(thresholds["initial_position_difference_m"])),
        _check("1C2-004", "J2 initial velocity", j2_initial_vel, f"<= {thresholds['initial_velocity_difference_mm_s']} mm/s", j2_initial_vel <= float(thresholds["initial_velocity_difference_mm_s"])),
        _check("1C2-005", "Output duration", float(times[-1]), f"abs(final-{duration}) <= 1e-6 s", abs(float(times[-1]) - duration) <= 1.0e-6),
        _check("1C2-006", "Output step", _maximum_step_deviation_seconds(times, expected_step), "<= 1e-6 s", _maximum_step_deviation_seconds(times, expected_step) <= 1.0e-6),
        _check("1C2-007", "Two-body raw epoch grid", float(tb_grid["maximum_absolute_raw_time_residual_seconds"]), "<= 1e-6 s", float(tb_grid["maximum_absolute_raw_time_residual_seconds"]) <= 1.0e-6),
        _check("1C2-008", "J2 raw epoch grid", float(j2_grid["maximum_absolute_raw_time_residual_seconds"]), "<= 1e-6 s", float(j2_grid["maximum_absolute_raw_time_residual_seconds"]) <= 1.0e-6),
        _check("1C2-009", "Two-body maximum position", tb_position, f"<= {thresholds['two_body_maximum_position_difference_m']} m", tb_position <= float(thresholds["two_body_maximum_position_difference_m"])),
        _check("1C2-010", "Two-body maximum velocity", tb_velocity, f"<= {thresholds['two_body_maximum_velocity_difference_mm_s']} mm/s", tb_velocity <= float(thresholds["two_body_maximum_velocity_difference_mm_s"])),
        _check("1C2-011", "Full-EOP candidate maximum position", candidate_position, f"<= {thresholds['j2_maximum_position_difference_m']} m", candidate_position <= float(thresholds["j2_maximum_position_difference_m"])),
        _check("1C2-012", "Full-EOP candidate maximum velocity", candidate_velocity, f"<= {thresholds['j2_maximum_velocity_difference_mm_s']} mm/s", candidate_velocity <= float(thresholds["j2_maximum_velocity_difference_mm_s"])),
        _check("1C2-013", "Frozen EOP checksum", dataset.source_sha256, f"== {config['external_validation']['eop_expected_sha256']}", dataset.source_sha256 == str(config["external_validation"]["eop_expected_sha256"])),
        _check("1C2-014", "EOP interval remains inside tagged coverage", f"{start_sample.coverage_status};{end_sample.coverage_status}", "neither endpoint is clamped", "clamped" not in start_sample.coverage_status and "clamped" not in end_sample.coverage_status),
    ]
    failed = [item for item in checks if item["status"] == "failed"]
    status = "passed_with_warnings" if not failed else "failed_validation"
    case_result_dir.mkdir(parents=True, exist_ok=False)
    for name, history in histories.items():
        write_state_history_csv(case_result_dir / f"{name}_states.csv", history)
    for name, comparison in comparisons.items():
        write_comparison_csv(case_result_dir / f"python_vs_gmat_{name}_cartesian.csv", comparison)
    for name, comparison in rtn.items():
        write_rtn_comparison_csv(case_result_dir / f"python_vs_gmat_{name}_rtn.csv", comparison)
    payload = {
        "research_core_version": "1C.2",
        "case_id": case["case_id"],
        "status": status,
        "models": summaries,
        "checks": checks,
        "failed_check_count": len(failed),
        "time_grid": {"two_body": tb_grid, "j2": j2_grid},
        "eop": {
            "source_sha256": dataset.source_sha256,
            "start_sample": start_sample.__dict__,
            "end_sample": end_sample.__dict__,
        },
        "source_files": {
            "configuration": _relative(config_path, root),
            "configuration_sha256": _sha256(config_path),
            "gmat_two_body": _relative(two_body_path, root),
            "gmat_two_body_sha256": _sha256(two_body_path),
            "gmat_j2": _relative(j2_path, root),
            "gmat_j2_sha256": _sha256(j2_path),
        },
    }
    _write_json(case_result_dir / "case_validation_summary.json", payload)
    manifest_files = []
    for path in sorted(case_result_dir.rglob("*")):
        if path.is_file() and path.name != "RUN_MANIFEST.json":
            manifest_files.append(
                {
                    "path": path.relative_to(case_result_dir).as_posix(),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    _write_json(case_result_dir / "RUN_MANIFEST.json", {"files": manifest_files})
    baseline_position = float(
        summaries["closed_baseline"]["position_difference_m"]["maximum_absolute"]
    )
    baseline_velocity = float(
        summaries["closed_baseline"]["velocity_difference_mm_s"]["maximum_absolute"]
    )
    return {
        **case,
        "status": status,
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "two_body_maximum_position_difference_m": tb_position,
        "two_body_maximum_velocity_difference_mm_s": tb_velocity,
        "closed_baseline_maximum_position_difference_m": baseline_position,
        "closed_baseline_maximum_velocity_difference_mm_s": baseline_velocity,
        "candidate_maximum_position_difference_m": candidate_position,
        "candidate_maximum_velocity_difference_mm_s": candidate_velocity,
        "candidate_position_gate_m": float(thresholds["j2_maximum_position_difference_m"]),
        "candidate_velocity_gate_mm_s": float(thresholds["j2_maximum_velocity_difference_mm_s"]),
        "result_directory": _relative(case_result_dir, root),
        "error": None,
    }


def _aggregate_report(matrix_id: str, status: str, decision: str, records: list[dict[str, Any]]) -> str:
    rows = []
    for record in records:
        def number(field: str) -> str:
            value = record.get(field)
            return "—" if value is None else f"{float(value):.9g}"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(record['case_id']))}</td>"
            f"<td>{number('duration_hours')}</td>"
            f"<td>{html.escape(str(record['status']))}</td>"
            f"<td>{number('two_body_maximum_position_difference_m')}</td>"
            f"<td>{number('closed_baseline_maximum_position_difference_m')}</td>"
            f"<td>{number('candidate_maximum_position_difference_m')}</td>"
            f"<td>{number('candidate_position_gate_m')}</td>"
            "</tr>"
        )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Research Core 1C.2 Independent GMAT Validation</title>
<style>body{{font-family:Arial,sans-serif;margin:2rem;color:#172033}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccd3df;padding:.45rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#edf2f8}}</style>
</head><body><h1>Research Core 1C.2 Independent GMAT Validation</h1>
<p><strong>Matrix:</strong> {html.escape(matrix_id)}<br><strong>Status:</strong> {html.escape(status)}<br>
<strong>Adoption decision:</strong> {html.escape(decision)}</p>
<table><thead><tr><th>Case</th><th>Hours</th><th>Status</th><th>Two-body m</th>
<th>Closed baseline m</th><th>Full-EOP candidate m</th><th>Candidate gate m</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p>Every case is a preregistered holdout not used for 1C.1 candidate selection.
No adoption claim is made if any case, evidence, or checksum gate fails.</p>
<img src="figures/independent_candidate_position.png" style="max-width:100%" alt="Independent residuals">
</body></html>"""


def run_independent_validation(
    matrix_path: str | Path, *, project_root: str | Path, allow_missing: bool = False
) -> IndependentValidationResult:
    root = Path(project_root).resolve()
    matrix_file = Path(matrix_path).resolve()
    spec = load_independent_matrix_spec(matrix_file)
    reference_root = _resolve_project_path(str(spec["reference_root"]), root, "reference_root")
    eop_path = _resolve_project_path(str(spec["eop_file"]), root, "eop_file")
    dataset = GmatEopDataset.from_file(eop_path, expected_sha256=str(spec["eop_expected_sha256"]))
    cases_dir = reference_root / "cases"
    output_dir = reference_root / "output"
    missing = []
    for case in spec["cases"]:
        case_id = str(case["case_id"])
        for path in (cases_dir / f"{case_id}.json", output_dir / f"{case_id}_TWO_BODY.e", output_dir / f"{case_id}_J2.e"):
            if not path.is_file():
                missing.append(path)
    if missing and not allow_missing:
        shown = "\n".join(f"  - {_relative(path, root)}" for path in missing)
        raise FileNotFoundError(
            f"GMAT 1C.2 independent matrix is incomplete; {len(missing)} files are missing:\n{shown}\nRun RUN_ALL_CASES_1C2.script in GMAT, then retry."
        )
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")
    result_dir = root / "results" / str(spec["matrix_id"]) / stamp
    cases_result_dir = result_dir / "cases"
    cases_result_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for case in spec["cases"]:
        case_id = str(case["case_id"])
        config_path = cases_dir / f"{case_id}.json"
        two_body_path = output_dir / f"{case_id}_TWO_BODY.e"
        j2_path = output_dir / f"{case_id}_J2.e"
        if not all(path.is_file() for path in (config_path, two_body_path, j2_path)):
            records.append({**case, "status": "incomplete", "error": "Required configuration or GMAT output is missing."})
            continue
        try:
            records.append(
                _run_case(
                    case=case,
                    config_path=config_path,
                    two_body_path=two_body_path,
                    j2_path=j2_path,
                    dataset=dataset,
                    root=root,
                    case_result_dir=cases_result_dir / case_id,
                )
            )
        except Exception as exc:
            records.append({**case, "status": "failed_validation", "error": f"{type(exc).__name__}: {exc}"})
    passed = sum(item["status"] == "passed_with_warnings" for item in records)
    incomplete = sum(item["status"] == "incomplete" for item in records)
    failed = len(records) - passed - incomplete
    if incomplete:
        status = "incomplete"
        decision = "candidate_not_adopted_incomplete_evidence"
    elif failed:
        status = "failed_validation"
        decision = "candidate_not_adopted_independent_gate_failed"
    else:
        status = "passed_with_warnings"
        decision = "adopt_gmat_r2026a_eop_full_as_validated_baseline"
    fieldnames = list(records[0].keys())
    for record in records[1:]:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    csv_path = result_dir / "gmat_eop_1c2_matrix_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "research_core_version": "1C.2",
        "matrix_id": spec["matrix_id"],
        "validation_status": status,
        "adoption_decision": decision,
        "candidate_model": CANDIDATE_MODEL,
        "closed_baseline_model": CLOSED_BASELINE_MODEL,
        "case_count": len(records),
        "passed_case_count": passed,
        "failed_case_count": failed,
        "incomplete_case_count": incomplete,
        "thresholds_preregistered": True,
        "thresholds_relaxed_after_results": False,
        "matrix_source": _relative(matrix_file, root),
        "matrix_source_sha256": _sha256(matrix_file),
        "eop_source": _relative(eop_path, root),
        "eop_source_sha256": dataset.source_sha256,
        "cases": records,
        "warnings": [
            "This validates agreement with configured GMAT R2026a software, not measured orbit truth.",
            "The frozen tagged EOP realization is release-specific and is not a continuously updated best-estimate series.",
            "Higher-degree gravity, drag, third bodies, SRP, tides, and relativity remain outside this gate.",
        ],
    }
    summary_json = result_dir / "gmat_eop_1c2_matrix_summary.json"
    _write_json(summary_json, summary)
    figures_dir = result_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    complete = [item for item in records if item.get("candidate_maximum_position_difference_m") is not None]
    if complete:
        x = np.arange(len(complete))
        figure, axis = plt.subplots(figsize=(11, 5))
        axis.bar(x - 0.2, [item["closed_baseline_maximum_position_difference_m"] for item in complete], 0.4, label="Closed baseline")
        axis.bar(x + 0.2, [item["candidate_maximum_position_difference_m"] for item in complete], 0.4, label="Full-EOP candidate")
        axis.set_yscale("log")
        axis.set_xticks(x, [item["case_id"].split("_")[0] for item in complete])
        axis.set_ylabel("Maximum position difference (m, log scale)")
        axis.set_title("Independent GMAT holdout residuals")
        axis.grid(True, which="both", axis="y", alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(figures_dir / "independent_candidate_position.png", dpi=180)
        figure.savefig(figures_dir / "independent_candidate_position.pdf")
        plt.close(figure)
    report = result_dir / "GMAT_EOP_1C2_INDEPENDENT_REPORT.html"
    report.write_text(_aggregate_report(str(spec["matrix_id"]), status, decision, records), encoding="utf-8", newline="\n")
    manifest_files = []
    for path in sorted(result_dir.rglob("*")):
        if path.is_file() and path.name != "RUN_MANIFEST.json":
            manifest_files.append({"path": path.relative_to(result_dir).as_posix(), "sha256": _sha256(path), "size_bytes": path.stat().st_size})
    _write_json(result_dir / "RUN_MANIFEST.json", {"files": manifest_files})
    return IndependentValidationResult(
        matrix_id=str(spec["matrix_id"]),
        result_directory=result_dir,
        validation_status=status,
        adoption_decision=decision,
        case_count=len(records),
        passed_case_count=passed,
        failed_case_count=failed,
        incomplete_case_count=incomplete,
        summary_json=summary_json,
        report_path=report,
    )


def verify_gmat_eop_install(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    supplied = Path(path).expanduser().resolve()
    if supplied.is_file():
        candidates = [supplied]
    elif supplied.is_dir():
        direct = supplied / "application" / "data" / "planetary_coeff" / "eopc04_08.62-now"
        candidates = [direct] if direct.is_file() else list(supplied.glob("**/eopc04_08.62-now"))
    else:
        raise FileNotFoundError(f"GMAT path does not exist: {supplied}")
    if not candidates:
        raise FileNotFoundError(f"No eopc04_08.62-now file found beneath: {supplied}")
    if len(candidates) > 1:
        raise ValueError("More than one GMAT EOP file was found; provide the exact file path.")
    source = candidates[0]
    source_bytes = source.read_bytes()
    actual = hashlib.sha256(source_bytes).hexdigest()
    canonical_lf = source_bytes.replace(b"\r\n", b"\n")
    canonical_lf_sha256 = hashlib.sha256(canonical_lf).hexdigest()
    raw_exact = actual == expected_sha256
    line_ending_equivalent = not raw_exact and canonical_lf_sha256 == expected_sha256
    return {
        "path": str(source),
        "sha256": actual,
        "canonical_lf_sha256": canonical_lf_sha256,
        "expected_sha256": expected_sha256,
        "byte_exact_match": raw_exact,
        "line_ending_equivalent": line_ending_equivalent,
        "matches_gmat_r2026a_tag": raw_exact or line_ending_equivalent,
        "size_bytes": source.stat().st_size,
    }


def package_independent_results(
    matrix_path: str | Path, *, project_root: str | Path, output_path: str | Path
) -> Path:
    root = Path(project_root).resolve()
    spec = load_independent_matrix_spec(matrix_path)
    reference_root = _resolve_project_path(str(spec["reference_root"]), root, "reference_root")
    expected = [
        reference_root / "output" / f"{case['case_id']}_{suffix}.e"
        for case in spec["cases"]
        for suffix in ("TWO_BODY", "J2")
    ]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot package 1C.2 results: {len(missing)} GMAT files are missing.")
    result_root = root / "results" / str(spec["matrix_id"])
    result_dirs = sorted(
        path
        for path in result_root.glob("*")
        if path.is_dir()
        and (path / "gmat_eop_1c2_matrix_summary.json").is_file()
        and (path / "GMAT_EOP_1C2_INDEPENDENT_REPORT.html").is_file()
        and (path / "RUN_MANIFEST.json").is_file()
    )
    if not result_dirs:
        raise FileNotFoundError(
            "No completed 1C.2 result folder is available for packaging. "
            "Interrupted folders are intentionally ignored."
        )
    latest = result_dirs[-1]
    archive = Path(output_path).resolve()
    if archive.exists():
        raise FileExistsError(f"Move or rename the existing ZIP first: {archive}")
    members = [Path(matrix_path).resolve(), reference_root / "GMAT_1C2_MATRIX_MANIFEST.json", reference_root / "RUN_ORDER_1C2.txt"]
    members.extend((reference_root / "cases").glob("*.json"))
    members.extend((reference_root / "scripts").glob("*.script"))
    members.extend(expected)
    members.extend(path for path in latest.rglob("*") if path.is_file())
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(set(member.resolve() for member in members)):
            bundle.write(path, arcname=_relative(path, root))
    with zipfile.ZipFile(archive, "r") as bundle:
        bad = bundle.testzip()
        if bad is not None:
            raise RuntimeError(f"Created 1C.2 results ZIP failed integrity at {bad}.")
    return archive
