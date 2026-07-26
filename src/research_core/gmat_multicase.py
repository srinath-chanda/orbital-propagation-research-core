"""Research Core 1B multi-case GMAT preparation and validation."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import shutil
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .external_validation import (
    ExternalValidationResult,
    build_gmat_script,
    initial_state_from_config,
    run_gmat_external_validation,
)


MATRIX_SCHEMA_VERSION = "1B.0"


@dataclass(frozen=True)
class PreparedGmatMatrix:
    matrix_id: str
    reference_root: Path
    manifest_path: Path
    master_script: Path
    run_order_path: Path
    case_count: int
    expected_output_count: int
    archived_outputs: tuple[Path, ...]


@dataclass(frozen=True)
class MultiCaseValidationResult:
    matrix_id: str
    result_directory: Path
    validation_status: str
    total_case_count: int
    passed_case_count: int
    failed_case_count: int
    incomplete_case_count: int
    summary_csv: Path
    summary_json: Path
    report_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _project_relative(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _resolve_project_path(value: str, project_root: Path, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"{label} must be project-relative, not absolute.")
    resolved = (project_root / candidate).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must remain inside the project root.") from exc
    return resolved


def load_gmat_matrix_spec(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != MATRIX_SCHEMA_VERSION:
        raise ValueError(
            f"GMAT matrix schema must be {MATRIX_SCHEMA_VERSION!r}."
        )
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("GMAT matrix must contain at least one case.")
    required = {
        "case_id",
        "factor",
        "epoch_utc",
        "altitude_km",
        "inclination_deg",
        "duration_hours",
    }
    seen: set[str] = set()
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Matrix case {index} must be an object.")
        missing = sorted(required - set(case))
        if missing:
            raise ValueError(
                f"Matrix case {index} is missing: {', '.join(missing)}."
            )
        case_id = str(case["case_id"])
        if not case_id.replace("_", "").isalnum() or not case_id[0].isalpha():
            raise ValueError(
                f"Matrix case_id {case_id!r} must contain letters, digits, and underscores."
            )
        if case_id in seen:
            raise ValueError(f"Duplicate matrix case_id: {case_id}.")
        seen.add(case_id)
        altitude = float(case["altitude_km"])
        inclination = float(case["inclination_deg"])
        duration = float(case["duration_hours"])
        if altitude <= 100.0:
            raise ValueError(f"{case_id} altitude must exceed 100 km.")
        if not 0.0 <= inclination <= 180.0:
            raise ValueError(f"{case_id} inclination must be in [0, 180] degrees.")
        if duration <= 0.0:
            raise ValueError(f"{case_id} duration must be positive.")
        datetime.fromisoformat(str(case["epoch_utc"]).replace("Z", "+00:00"))
    return payload


def _duration_thresholds(
    spec: dict[str, Any], duration_hours: float
) -> tuple[float, float]:
    policy = spec["threshold_policy"]
    tiers = policy["j2_duration_tiers"]
    for tier in tiers:
        if duration_hours <= float(tier["maximum_duration_hours"]):
            return (
                float(tier["maximum_position_difference_m"]),
                float(tier["maximum_velocity_difference_mm_s"]),
            )
    raise ValueError(
        f"No preregistered J2 threshold tier covers {duration_hours:g} hours."
    )


def _case_configuration(
    baseline: dict[str, Any],
    spec: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    config = deepcopy(baseline)
    case_id = str(case["case_id"])
    duration_hours = float(case["duration_hours"])
    earth_radius = float(config["earth_model"]["equatorial_radius_km"])
    j2_position, j2_velocity = _duration_thresholds(spec, duration_hours)
    policy = spec["threshold_policy"]

    config["experiment"].update(
        {
            "experiment_id": f"EXP-GMAT-1B-{case_id.replace('_', '-')}",
            "case_id": f"CASE-GMAT-1B-{case_id.replace('_', '-')}",
            "title": f"Research Core 1B GMAT multi-case validation: {case_id}",
            "description": (
                "Preregistered multi-case degree-2/order-0 comparison against "
                f"GMAT R2026a; factor={case['factor']}."
            ),
        }
    )
    config["initial_state"].update(
        {
            "epoch_utc": str(case["epoch_utc"]),
            "semi_major_axis_km": earth_radius + float(case["altitude_km"]),
            "inclination_deg": float(case["inclination_deg"]),
            "raan_deg": float(case.get("raan_deg", 20.0)),
            "argument_of_perigee_deg": float(
                case.get("argument_of_perigee_deg", 30.0)
            ),
            "true_anomaly_deg": float(case.get("true_anomaly_deg", 0.0)),
            "notes": (
                f"Research Core 1B matrix case {case_id}; nominal altitude "
                "defines semi-major axis relative to the matched JGM2 radius."
            ),
        }
    )
    config["propagation"]["default_duration_hours"] = duration_hours
    config["external_validation"].update(
        {
            "duration_seconds": duration_hours * 3600.0,
            "output_step_seconds": float(spec["output_step_seconds"]),
            "threshold_status": "preregistered_before_gmat_1b_execution",
            "thresholds": {
                "initial_position_difference_m": float(
                    policy["initial_position_difference_m"]
                ),
                "initial_velocity_difference_mm_s": float(
                    policy["initial_velocity_difference_mm_s"]
                ),
                "two_body_maximum_position_difference_m": float(
                    policy["two_body_maximum_position_difference_m"]
                ),
                "two_body_maximum_velocity_difference_mm_s": float(
                    policy["two_body_maximum_velocity_difference_mm_s"]
                ),
                "j2_maximum_position_difference_m": j2_position,
                "j2_maximum_velocity_difference_mm_s": j2_velocity,
            },
        }
    )
    config["external_validation"]["acceleration_diagnostic"]["enabled"] = False
    config["external_validation"]["short_arc"]["enabled"] = False
    config["validation"]["threshold_status"] = (
        "gmat_1b_multicase_execution_pending"
    )
    config["scientific_cautions"] = [
        *config["scientific_cautions"],
        "This configuration belongs to the preregistered Research Core 1B matrix.",
        "A matrix threshold must not be changed after inspecting GMAT results.",
    ]
    return config


def _gmat_epoch(value: str) -> str:
    epoch = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)
    return epoch.astimezone(timezone.utc).strftime("%d %b %Y %H:%M:%S.%f")[:-3]


def _portable_gmat_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _master_resource_block(
    config: dict[str, Any],
    *,
    case_number: int,
    model: str,
    output_ephemeris: Path,
) -> tuple[str, str]:
    state = initial_state_from_config(config)
    ext = config["external_validation"]
    suffix = "TB" if model == "two_body" else "J2"
    prefix = f"C{case_number:02d}{suffix}"
    degree = 0 if model == "two_body" else 2
    duration = float(ext["duration_seconds"])
    output_step = float(ext["output_step_seconds"])
    initial_step = min(float(ext["gmat_initial_step_seconds"]), output_step)
    max_step = min(float(ext["gmat_maximum_step_seconds"]), output_step)
    resource = f"""
% {config['experiment']['case_id']} {model}
Create Spacecraft {prefix}Sat;
{prefix}Sat.DateFormat = UTCGregorian;
{prefix}Sat.Epoch = '{_gmat_epoch(state.epoch_utc)}';
{prefix}Sat.CoordinateSystem = EarthMJ2000Eq;
{prefix}Sat.DisplayStateType = Cartesian;
{prefix}Sat.X = {state.position_km[0]:.15f};
{prefix}Sat.Y = {state.position_km[1]:.15f};
{prefix}Sat.Z = {state.position_km[2]:.15f};
{prefix}Sat.VX = {state.velocity_km_s[0]:.15f};
{prefix}Sat.VY = {state.velocity_km_s[1]:.15f};
{prefix}Sat.VZ = {state.velocity_km_s[2]:.15f};
{prefix}Sat.DryMass = 500;
{prefix}Sat.Cd = 2.2;
{prefix}Sat.Cr = 1.0;
{prefix}Sat.DragArea = 4;
{prefix}Sat.SRPArea = 4;

Create ForceModel {prefix}FM;
{prefix}FM.CentralBody = Earth;
{prefix}FM.PrimaryBodies = {{Earth}};
{prefix}FM.Drag = None;
{prefix}FM.SRP = Off;
{prefix}FM.RelativisticCorrection = Off;
{prefix}FM.ErrorControl = RSSStep;
{prefix}FM.GravityField.Earth.Degree = {degree};
{prefix}FM.GravityField.Earth.Order = 0;
{prefix}FM.GravityField.Earth.PotentialFile = '{ext['gravity_file']}';
{prefix}FM.GravityField.Earth.TideModel = 'None';

Create Propagator {prefix}Prop;
{prefix}Prop.FM = {prefix}FM;
{prefix}Prop.Type = {ext['gmat_integrator']};
{prefix}Prop.InitialStepSize = {initial_step:.15g};
{prefix}Prop.Accuracy = {float(ext['gmat_accuracy']):.15g};
{prefix}Prop.MinStep = 1e-6;
{prefix}Prop.MaxStep = {max_step:.15g};
{prefix}Prop.MaxStepAttempts = 50;
{prefix}Prop.StopIfAccuracyIsViolated = true;

Create EphemerisFile {prefix}Eph;
{prefix}Eph.Spacecraft = {prefix}Sat;
{prefix}Eph.Filename = '{_portable_gmat_path(output_ephemeris)}';
{prefix}Eph.FileFormat = STK-TimePosVel;
{prefix}Eph.EpochFormat = UTCGregorian;
{prefix}Eph.InitialEpoch = InitialSpacecraftEpoch;
{prefix}Eph.FinalEpoch = FinalSpacecraftEpoch;
{prefix}Eph.StepSize = {output_step:.15g};
{prefix}Eph.Interpolator = Lagrange;
{prefix}Eph.InterpolationOrder = 7;
{prefix}Eph.CoordinateSystem = EarthMJ2000Eq;
{prefix}Eph.WriteEphemeris = true;
"""
    mission = (
        f"Propagate {prefix}Prop({prefix}Sat) "
        f"{{{prefix}Sat.ElapsedSecs = {duration:.15g}}};"
    )
    return resource, mission


def build_gmat_multicase_master_script(
    cases: Iterable[tuple[dict[str, Any], Path, Path]],
    *,
    tool_version: str,
    script_title: str = "Research Core 1C.0 GMAT multi-case master script",
) -> str:
    resources: list[str] = []
    missions: list[str] = []
    for index, (config, two_body_output, j2_output) in enumerate(cases, start=1):
        for model, output in (("two_body", two_body_output), ("j2", j2_output)):
            resource, mission = _master_resource_block(
                config,
                case_number=index,
                model=model,
                output_ephemeris=output,
            )
            resources.append(resource)
            missions.append(mission)
    return (
        "%\n"
        f"% {script_title}\n"
        f"% Target GMAT release: {tool_version}\n"
        "% Preregistered degree-2/order-0 software-model comparison.\n"
        "% Regenerate locally before running so output paths are correct.\n"
        "%\n"
        + "".join(resources)
        + "\nBeginMissionSequence;\n"
        + "\n".join(missions)
        + "\n"
    )


def _archive_existing_outputs(output_dir: Path, expected_names: set[str]) -> tuple[Path, ...]:
    existing = [path for path in output_dir.glob("*.e") if path.name in expected_names]
    if not existing:
        return ()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")
    archive_dir = output_dir.parent / "archive" / stamp
    archive_dir.mkdir(parents=True, exist_ok=False)
    archived: list[Path] = []
    for source in existing:
        destination = archive_dir / source.name
        shutil.move(str(source), str(destination))
        archived.append(destination)
    return tuple(archived)


def prepare_gmat_multicase_files(
    matrix_path: str | Path,
    *,
    project_root: str | Path,
) -> PreparedGmatMatrix:
    root = Path(project_root).resolve()
    matrix_file = Path(matrix_path).resolve()
    spec = load_gmat_matrix_spec(matrix_file)
    baseline_path = _resolve_project_path(
        str(spec["baseline_configuration"]), root, "baseline_configuration"
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    reference_root = _resolve_project_path(
        str(spec["reference_root"]), root, "reference_root"
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
    archived = _archive_existing_outputs(output_dir, expected_names)

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
        _write_json(config, config_path)
        two_body_script.write_text(
            build_gmat_script(
                config,
                model="two_body",
                output_ephemeris=two_body_output,
                stage_label="1B_multicase",
            ),
            encoding="utf-8",
            newline="\n",
        )
        j2_script.write_text(
            build_gmat_script(
                config,
                model="j2",
                output_ephemeris=j2_output,
                stage_label="1B_multicase",
            ),
            encoding="utf-8",
            newline="\n",
        )
        master_cases.append((config, two_body_output, j2_output))
        prepared_cases.append(
            {
                **case,
                "configuration": _project_relative(config_path, root),
                "configuration_sha256": _sha256(config_path),
                "two_body_script": _project_relative(two_body_script, root),
                "two_body_script_sha256": _sha256(two_body_script),
                "j2_script": _project_relative(j2_script, root),
                "j2_script_sha256": _sha256(j2_script),
                "two_body_output": _project_relative(two_body_output, root),
                "j2_output": _project_relative(j2_output, root),
                "preregistered_thresholds": config["external_validation"][
                    "thresholds"
                ],
            }
        )

    master_script = scripts_dir / "RUN_ALL_CASES_1B.script"
    master_script.write_text(
        build_gmat_multicase_master_script(
            master_cases,
            tool_version=str(spec["tool_version"]),
        ),
        encoding="utf-8",
        newline="\n",
    )
    run_order_path = reference_root / "RUN_ORDER_1B.txt"
    run_order_path.write_text(
        "\n".join(
            [
                "RESEARCH CORE 1C.0 GMAT RUN ORDER",
                "",
                "Preferred: run scripts/RUN_ALL_CASES_1B.script once in GMAT R2026a.",
                "Fallback: if the master script does not interpret, run each numbered",
                "TWO_BODY script followed by its matching J2 script.",
                "",
                *[
                    f"{index:02d}. {case['case_id']}: TWO_BODY, then J2"
                    for index, case in enumerate(spec["cases"], start=1)
                ],
                "",
                f"Expected ephemeris files: {len(spec['cases']) * 2}",
                "Do not edit thresholds after inspecting results.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "research_core_version": "1C.0",
        "matrix_schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_id": spec["matrix_id"],
        "status": "scripts_prepared_gmat_execution_pending",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "tool": "GMAT",
        "tool_version": spec["tool_version"],
        "matrix_source": _project_relative(matrix_file, root),
        "matrix_source_sha256": _sha256(matrix_file),
        "baseline_configuration": _project_relative(baseline_path, root),
        "baseline_configuration_sha256": _sha256(baseline_path),
        "master_script": _project_relative(master_script, root),
        "master_script_sha256": _sha256(master_script),
        "run_order": _project_relative(run_order_path, root),
        "case_count": len(prepared_cases),
        "expected_output_count": len(prepared_cases) * 2,
        "archived_previous_output_count": len(archived),
        "threshold_policy": spec["threshold_policy"],
        "cases": prepared_cases,
    }
    manifest_path = reference_root / "GMAT_1B_MATRIX_MANIFEST.json"
    _write_json(manifest, manifest_path)
    return PreparedGmatMatrix(
        matrix_id=str(spec["matrix_id"]),
        reference_root=reference_root,
        manifest_path=manifest_path,
        master_script=master_script,
        run_order_path=run_order_path,
        case_count=len(prepared_cases),
        expected_output_count=len(prepared_cases) * 2,
        archived_outputs=archived,
    )


def _result_record(
    case: dict[str, Any], result: ExternalValidationResult, project_root: Path
) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "factor": case["factor"],
        "epoch_utc": case["epoch_utc"],
        "altitude_km": float(case["altitude_km"]),
        "inclination_deg": float(case["inclination_deg"]),
        "duration_hours": float(case["duration_hours"]),
        "status": result.validation_status,
        "two_body_maximum_position_difference_m": result.two_body_maximum_position_difference_m,
        "two_body_maximum_velocity_difference_mm_s": result.two_body_maximum_velocity_difference_mm_s,
        "fixed_axis_maximum_position_difference_m": result.j2_maximum_position_difference_m,
        "fixed_axis_maximum_velocity_difference_mm_s": result.j2_maximum_velocity_difference_mm_s,
        "pole_aware_maximum_position_difference_m": result.j2_gmat_matched_maximum_position_difference_m,
        "pole_aware_maximum_velocity_difference_mm_s": result.j2_gmat_matched_maximum_velocity_difference_mm_s,
        "result_directory": _project_relative(result.result_directory, project_root),
        "report": _project_relative(result.report_path, project_root),
        "error": None,
    }


def _write_matrix_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "case_id",
        "factor",
        "epoch_utc",
        "altitude_km",
        "inclination_deg",
        "duration_hours",
        "status",
        "two_body_maximum_position_difference_m",
        "two_body_maximum_velocity_difference_mm_s",
        "fixed_axis_maximum_position_difference_m",
        "fixed_axis_maximum_velocity_difference_mm_s",
        "pole_aware_maximum_position_difference_m",
        "pole_aware_maximum_velocity_difference_mm_s",
        "result_directory",
        "report",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _matrix_report_html(
    matrix_id: str,
    status: str,
    records: list[dict[str, Any]],
) -> str:
    rows = []
    for record in records:
        def number(name: str) -> str:
            value = record.get(name)
            return "—" if value is None else f"{float(value):.6g}"

        rows.append(
            "<tr>"
            f"<td>{html.escape(str(record['case_id']))}</td>"
            f"<td>{html.escape(str(record['factor']))}</td>"
            f"<td>{number('altitude_km')}</td>"
            f"<td>{number('inclination_deg')}</td>"
            f"<td>{number('duration_hours')}</td>"
            f"<td>{html.escape(str(record['status']))}</td>"
            f"<td>{number('two_body_maximum_position_difference_m')}</td>"
            f"<td>{number('pole_aware_maximum_position_difference_m')}</td>"
            f"<td>{number('pole_aware_maximum_velocity_difference_mm_s')}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Research Core 1B GMAT Multi-Case Validation</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 2rem; color: #172033; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccd3df; padding: .45rem; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
th {{ background: #edf2f8; }}
</style></head><body>
<h1>Research Core 1B GMAT Multi-Case Validation</h1>
<p><strong>Matrix:</strong> {html.escape(matrix_id)}</p>
<p><strong>Status:</strong> {html.escape(status)}</p>
<table><thead><tr><th>Case</th><th>Factor</th><th>Altitude km</th>
<th>Inclination deg</th><th>Duration h</th><th>Status</th>
<th>Two-body position m</th><th>Pole-aware position m</th>
<th>Pole-aware velocity mm/s</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p>This is an independent software-model comparison, not measured-orbit truth
or flight qualification.</p></body></html>
"""


def run_gmat_multicase_validation(
    matrix_path: str | Path,
    *,
    project_root: str | Path,
    allow_missing: bool = False,
) -> MultiCaseValidationResult:
    root = Path(project_root).resolve()
    spec = load_gmat_matrix_spec(matrix_path)
    reference_root = _resolve_project_path(
        str(spec["reference_root"]), root, "reference_root"
    )
    cases_dir = reference_root / "cases"
    output_dir = reference_root / "output"
    missing: list[Path] = []
    for case in spec["cases"]:
        case_id = str(case["case_id"])
        for path in (
            cases_dir / f"{case_id}.json",
            output_dir / f"{case_id}_TWO_BODY.e",
            output_dir / f"{case_id}_J2.e",
        ):
            if not path.is_file():
                missing.append(path)
    if missing and not allow_missing:
        shown = "\n".join(f"  - {_project_relative(path, root)}" for path in missing)
        raise FileNotFoundError(
            f"GMAT 1B matrix is incomplete; {len(missing)} required files are missing:\n"
            f"{shown}\nRun the prepared GMAT master script, then retry."
        )

    records: list[dict[str, Any]] = []
    for case in spec["cases"]:
        case_id = str(case["case_id"])
        config_path = cases_dir / f"{case_id}.json"
        two_body = output_dir / f"{case_id}_TWO_BODY.e"
        j2 = output_dir / f"{case_id}_J2.e"
        if not all(path.is_file() for path in (config_path, two_body, j2)):
            records.append(
                {
                    **case,
                    "status": "incomplete",
                    "two_body_maximum_position_difference_m": None,
                    "two_body_maximum_velocity_difference_mm_s": None,
                    "fixed_axis_maximum_position_difference_m": None,
                    "fixed_axis_maximum_velocity_difference_mm_s": None,
                    "pole_aware_maximum_position_difference_m": None,
                    "pole_aware_maximum_velocity_difference_mm_s": None,
                    "result_directory": None,
                    "report": None,
                    "error": "Required generated configuration or GMAT ephemeris is missing.",
                }
            )
            continue
        try:
            result = run_gmat_external_validation(
                config_path,
                two_body,
                j2,
                project_root=root,
            )
            records.append(_result_record(case, result, root))
        except Exception as exc:  # preserve other cases and aggregate the failure
            records.append(
                {
                    **case,
                    "status": "failed_validation",
                    "two_body_maximum_position_difference_m": None,
                    "two_body_maximum_velocity_difference_mm_s": None,
                    "fixed_axis_maximum_position_difference_m": None,
                    "fixed_axis_maximum_velocity_difference_mm_s": None,
                    "pole_aware_maximum_position_difference_m": None,
                    "pole_aware_maximum_velocity_difference_mm_s": None,
                    "result_directory": None,
                    "report": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    passed = sum(record["status"] == "passed_with_warnings" for record in records)
    incomplete = sum(record["status"] == "incomplete" for record in records)
    failed = len(records) - passed - incomplete
    if incomplete:
        status = "incomplete"
    elif failed:
        status = "failed_validation"
    else:
        status = "passed_with_warnings"

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")
    result_dir = root / "results" / str(spec["matrix_id"]) / stamp
    result_dir.mkdir(parents=True, exist_ok=False)
    summary_csv = result_dir / "gmat_1b_matrix_summary.csv"
    _write_matrix_summary_csv(summary_csv, records)
    summary_payload = {
        "research_core_version": "1C.0",
        "matrix_id": spec["matrix_id"],
        "validation_status": status,
        "total_case_count": len(records),
        "passed_case_count": passed,
        "failed_case_count": failed,
        "incomplete_case_count": incomplete,
        "thresholds_preregistered": True,
        "matrix_source": _project_relative(Path(matrix_path).resolve(), root),
        "matrix_source_sha256": _sha256(Path(matrix_path).resolve()),
        "cases": records,
        "claim": (
            "Pole-aware degree-2/order-0 agreement passed the preregistered "
            "Research Core 1B multi-case matrix."
            if status == "passed_with_warnings"
            else "No multi-case validation claim is made until every case passes."
        ),
    }
    summary_json = result_dir / "gmat_1b_matrix_summary.json"
    _write_json(summary_payload, summary_json)
    report = result_dir / "GMAT_1B_MATRIX_REPORT.html"
    report.write_text(
        _matrix_report_html(str(spec["matrix_id"]), status, records),
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "files": [
            {
                "path": path.name,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (summary_csv, summary_json, report)
        ]
    }
    _write_json(manifest, result_dir / "RUN_MANIFEST.json")
    return MultiCaseValidationResult(
        matrix_id=str(spec["matrix_id"]),
        result_directory=result_dir,
        validation_status=status,
        total_case_count=len(records),
        passed_case_count=passed,
        failed_case_count=failed,
        incomplete_case_count=incomplete,
        summary_csv=summary_csv,
        summary_json=summary_json,
        report_path=report,
    )


def package_gmat_multicase_results(
    matrix_path: str | Path,
    *,
    project_root: str | Path,
    output_path: str | Path,
) -> Path:
    root = Path(project_root).resolve()
    spec = load_gmat_matrix_spec(matrix_path)
    reference_root = _resolve_project_path(
        str(spec["reference_root"]), root, "reference_root"
    )
    expected = [
        reference_root / "output" / f"{case['case_id']}_{suffix}.e"
        for case in spec["cases"]
        for suffix in ("TWO_BODY", "J2")
    ]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Cannot package results: {len(missing)} GMAT ephemerides are missing."
        )
    matrix_results = root / "results" / str(spec["matrix_id"])
    aggregate_dirs = sorted(path for path in matrix_results.glob("*") if path.is_dir())
    if not aggregate_dirs:
        raise FileNotFoundError("Run the Python 1B matrix validation before packaging.")
    latest_aggregate = aggregate_dirs[-1]
    archive = Path(output_path).resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        raise FileExistsError(f"Move or rename the existing results ZIP first: {archive}")

    members: list[Path] = [Path(matrix_path).resolve()]
    members.extend(path for path in reference_root.iterdir() if path.is_file())
    for current_directory in ("cases", "scripts", "output"):
        members.extend(
            path
            for path in (reference_root / current_directory).rglob("*")
            if path.is_file()
        )
    members.extend(path for path in latest_aggregate.rglob("*") if path.is_file())
    unique_members = sorted(set(members))
    inventory = [
        {
            "path": _project_relative(path, root),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in unique_members
    ]
    temporary = archive.with_name(f".{archive.name}.tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as stream:
        for path in unique_members:
            stream.write(path, _project_relative(path, root))
        stream.writestr(
            "GMAT_1B_RESULTS_PACKAGE_MANIFEST.json",
            json.dumps(
                {
                    "matrix_id": spec["matrix_id"],
                    "created_at_utc": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "files": inventory,
                },
                indent=2,
            )
            + "\n",
        )
    os.replace(temporary, archive)
    with zipfile.ZipFile(archive, "r") as stream:
        bad = stream.testzip()
        if bad is not None:
            raise RuntimeError(f"Created results ZIP failed integrity check at {bad}.")
    return archive
