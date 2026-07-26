"""Research Core 1D.1 higher-degree/order GMAT short-arc validation."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .analysis.comparison import compare_state_histories, create_error_summary
from .external_validation import (
    _canonicalize_nominal_output_grid,
    initial_state_from_config,
    parse_stk_time_pos_vel,
)
from .gmat_eop import GMAT_R2026A_EOP_SHA256, GmatEopDataset
from .gmat_gravity_closure import verify_gravity_ladder_closure
from .gravity_harmonics import CofGravityField
from .outputs import write_comparison_csv, write_state_history_csv
from .propagators.numerical_gravity import propagate_spherical_harmonic_gravity


SCHEMA_VERSION = "1D.1"
EXPECTED_MODELS = (("G20", 2, 0), ("G44", 4, 4), ("G88", 8, 8), ("G2020", 20, 20))


@dataclass(frozen=True)
class PreparedGravityShortArc:
    experiment_id: str
    model_count: int
    expected_output_count: int
    master_script: Path
    run_order: Path
    manifest: Path
    archived_outputs: tuple[Path, ...]


@dataclass(frozen=True)
class GravityShortArcResult:
    experiment_id: str
    status: str
    decision: str
    model_count: int
    passed_model_count: int
    check_count: int
    maximum_position_difference_m: float
    maximum_velocity_difference_mm_s: float
    result_directory: Path
    report_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


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


def load_gravity_short_arc_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Short-arc schema must be {SCHEMA_VERSION!r}.")
    models = tuple(
        (str(item["model_id"]), int(item["degree"]), int(item["order"]))
        for item in payload.get("models", [])
    )
    if models != EXPECTED_MODELS:
        raise ValueError(f"The preregistered short-arc models must be {EXPECTED_MODELS!r}.")
    if payload.get("threshold_status") != "preregistered_before_first_1d1_gmat_run":
        raise ValueError("The 1D.1 thresholds are not preregistered.")
    duration = float(payload["duration_seconds"])
    step = float(payload["output_step_seconds"])
    if duration <= 0.0 or step <= 0.0 or abs(duration / step - round(duration / step)) > 1e-12:
        raise ValueError("Duration must be a positive integer multiple of output step.")
    if float(payload["time_grid_tolerance_seconds"]) <= 0.0:
        raise ValueError("Time-grid tolerance must be positive.")
    return payload


def _gmat_epoch(value: str) -> str:
    epoch = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)
    return epoch.astimezone(timezone.utc).strftime("%d %b %Y %H:%M:%S.%f")[:-3]


def _gmat_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _resource_block(
    model: dict[str, Any],
    baseline: dict[str, Any],
    config: dict[str, Any],
    *,
    gravity_file: Path,
    output_ephemeris: Path,
) -> tuple[str, str]:
    state = initial_state_from_config(baseline)
    model_id = str(model["model_id"])
    degree = int(model["degree"])
    order = int(model["order"])
    integrator = config["integrator"]
    step = float(config["output_step_seconds"])
    max_step = min(float(integrator["gmat_maximum_step_seconds"]), step)
    resource = f"""
% {model_id}: degree {degree}, order {order}
Create Spacecraft {model_id}Sat;
{model_id}Sat.DateFormat = UTCGregorian;
{model_id}Sat.Epoch = '{_gmat_epoch(state.epoch_utc)}';
{model_id}Sat.CoordinateSystem = EarthMJ2000Eq;
{model_id}Sat.DisplayStateType = Cartesian;
{model_id}Sat.X = {state.position_km[0]:.15f};
{model_id}Sat.Y = {state.position_km[1]:.15f};
{model_id}Sat.Z = {state.position_km[2]:.15f};
{model_id}Sat.VX = {state.velocity_km_s[0]:.15f};
{model_id}Sat.VY = {state.velocity_km_s[1]:.15f};
{model_id}Sat.VZ = {state.velocity_km_s[2]:.15f};
{model_id}Sat.DryMass = 500;
{model_id}Sat.Cd = 2.2;
{model_id}Sat.Cr = 1.0;
{model_id}Sat.DragArea = 4;
{model_id}Sat.SRPArea = 4;

Create ForceModel {model_id}FM;
{model_id}FM.CentralBody = Earth;
{model_id}FM.PrimaryBodies = {{Earth}};
{model_id}FM.Drag = None;
{model_id}FM.SRP = Off;
{model_id}FM.RelativisticCorrection = Off;
{model_id}FM.ErrorControl = RSSStep;
{model_id}FM.GravityField.Earth.Degree = {degree};
{model_id}FM.GravityField.Earth.Order = {order};
{model_id}FM.GravityField.Earth.PotentialFile = '{_gmat_path(gravity_file)}';
{model_id}FM.GravityField.Earth.TideModel = 'None';

Create Propagator {model_id}Prop;
{model_id}Prop.FM = {model_id}FM;
{model_id}Prop.Type = {integrator['gmat_method']};
{model_id}Prop.InitialStepSize = {step:.15g};
{model_id}Prop.Accuracy = {float(integrator['gmat_accuracy']):.15g};
{model_id}Prop.MinStep = 1e-6;
{model_id}Prop.MaxStep = {max_step:.15g};
{model_id}Prop.MaxStepAttempts = 50;
{model_id}Prop.StopIfAccuracyIsViolated = true;

Create EphemerisFile {model_id}Eph;
{model_id}Eph.Spacecraft = {model_id}Sat;
{model_id}Eph.Filename = '{_gmat_path(output_ephemeris)}';
{model_id}Eph.FileFormat = STK-TimePosVel;
{model_id}Eph.EpochFormat = UTCGregorian;
{model_id}Eph.InitialEpoch = InitialSpacecraftEpoch;
{model_id}Eph.FinalEpoch = FinalSpacecraftEpoch;
{model_id}Eph.StepSize = {step:.15g};
{model_id}Eph.Interpolator = Lagrange;
{model_id}Eph.InterpolationOrder = 7;
{model_id}Eph.CoordinateSystem = EarthMJ2000Eq;
{model_id}Eph.WriteEphemeris = true;
"""
    mission = (
        f"Propagate {model_id}Prop({model_id}Sat) "
        f"{{{model_id}Sat.ElapsedSecs = {float(config['duration_seconds']):.15g}}};"
    )
    return resource, mission


def build_gravity_short_arc_master_script(
    config: dict[str, Any],
    baseline: dict[str, Any],
    *,
    gravity_file: Path,
    output_directory: Path,
) -> str:
    resources: list[str] = []
    missions: list[str] = []
    for model in config["models"]:
        model_id = str(model["model_id"])
        resource, mission = _resource_block(
            model,
            baseline,
            config,
            gravity_file=gravity_file,
            output_ephemeris=output_directory / f"{model_id}_SHORT_ARC.e",
        )
        resources.append(resource)
        missions.append(mission)
    return (
        "%\n% Research Core 1D.1 higher-degree/order short-arc matrix\n"
        "% Target GMAT R2026a; frozen JGM2; 30 minutes; 10-second outputs.\n"
        "% Run once. Thresholds were preregistered before these outputs.\n%\n"
        + "".join(resources)
        + "\nBeginMissionSequence;\n"
        + "\n".join(missions)
        + "\n"
    )


def prepare_gravity_short_arc(
    config_path: str | Path,
    *,
    project_root: str | Path,
) -> PreparedGravityShortArc:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_gravity_short_arc_config(config_file)
    closure_path = _project_path(root, config["prerequisite_closure"], "prerequisite_closure")
    closure = verify_gravity_ladder_closure(closure_path, project_root=root)
    baseline_path = _project_path(root, config["baseline_configuration"], "baseline_configuration")
    gravity_file = _project_path(root, config["gravity_file"], "gravity_file")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    field = CofGravityField.from_file(gravity_file)
    reference = _project_path(root, config["reference_root"], "reference_root")
    scripts = reference / "scripts"
    outputs = reference / "output"
    for directory in (scripts, outputs):
        directory.mkdir(parents=True, exist_ok=True)
    expected_names = {f"{item['model_id']}_SHORT_ARC.e" for item in config["models"]}
    existing = [path for path in outputs.glob("*.e") if path.name in expected_names]
    archived: list[Path] = []
    if existing:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")
        archive = reference / "archive" / stamp
        archive.mkdir(parents=True, exist_ok=False)
        for source in existing:
            destination = archive / source.name
            shutil.move(str(source), destination)
            archived.append(destination)
    master = scripts / "RUN_GRAVITY_SHORT_ARCS_1D1.script"
    master.write_text(
        build_gravity_short_arc_master_script(
            config, baseline, gravity_file=gravity_file, output_directory=outputs
        ),
        encoding="utf-8",
        newline="\n",
    )
    individual_records = []
    for model in config["models"]:
        model_id = str(model["model_id"])
        output = outputs / f"{model_id}_SHORT_ARC.e"
        resource, mission = _resource_block(
            model, baseline, config, gravity_file=gravity_file, output_ephemeris=output
        )
        script = scripts / f"{model_id}_SHORT_ARC.script"
        script.write_text(
            "% Research Core 1D.1 fallback individual short arc\n"
            + resource
            + "\nBeginMissionSequence;\n"
            + mission
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        individual_records.append(
            {
                **model,
                "script": _relative(script, root),
                "script_sha256": _sha256(script),
                "output": _relative(output, root),
            }
        )
    run_order = reference / "RUN_ORDER_1D1.txt"
    run_order.write_text(
        "RESEARCH CORE 1D.1 GMAT RUN ORDER\n\n"
        "Preferred: run scripts/RUN_GRAVITY_SHORT_ARCS_1D1.script once.\n"
        "Fallback only if the master does not interpret: run G20, G44, G88, then G2020.\n"
        "Expected untouched ephemerides: 4. Do not rerun after successful creation.\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = reference / "GMAT_GRAVITY_SHORT_ARC_1D1_MANIFEST.json"
    _write_json(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": config["experiment_id"],
            "status": "scripts_prepared_gmat_execution_pending",
            "prerequisite_closure_id": closure.closure_id,
            "configuration": _relative(config_file, root),
            "configuration_sha256": _sha256(config_file),
            "gravity_file": _relative(gravity_file, root),
            "gravity_file_sha256": field.source_sha256,
            "master_script": _relative(master, root),
            "master_script_sha256": _sha256(master),
            "duration_seconds": config["duration_seconds"],
            "output_step_seconds": config["output_step_seconds"],
            "model_count": len(config["models"]),
            "expected_output_count": len(config["models"]),
            "archived_previous_output_count": len(archived),
            "models": individual_records,
        },
        manifest,
    )
    return PreparedGravityShortArc(
        experiment_id=str(config["experiment_id"]),
        model_count=len(config["models"]),
        expected_output_count=len(config["models"]),
        master_script=master,
        run_order=run_order,
        manifest=manifest,
        archived_outputs=tuple(archived),
    )


def _initial_difference(initial_state: Any, history: Any) -> tuple[float, float]:
    position = float(np.linalg.norm(history.positions_km[0] - initial_state.position_km) * 1000.0)
    velocity = float(np.linalg.norm(history.velocities_km_s[0] - initial_state.velocity_km_s) * 1.0e6)
    return position, velocity


def _report_html(status: str, decision: str, records: list[dict[str, Any]]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['model_id'])}</td><td>{item['degree']}×{item['order']}</td>"
        f"<td>{item['maximum_position_difference_m']:.9e}</td>"
        f"<td>{item['maximum_velocity_difference_mm_s']:.9e}</td>"
        f"<td>{html.escape(item['status'])}</td></tr>"
        for item in records
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Research Core 1D.1 GMAT Short-Arc Validation</title><style>
body{{font-family:Arial,sans-serif;margin:2rem;color:#172033}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccd3df;padding:.5rem}}th{{background:#edf2f8}}</style></head><body>
<h1>Research Core 1D.1 GMAT Short-Arc Validation</h1><p><strong>Status:</strong> {status}</p>
<p><strong>Decision:</strong> {decision}</p><table><thead><tr><th>Model</th><th>Degree/order</th>
<th>Maximum position difference (m)</th><th>Maximum velocity difference (mm/s)</th>
<th>Status</th></tr></thead><tbody>{rows}</tbody></table>
<p>All arcs are 30 minutes with 10-second output spacing. This is independent software-model
agreement, not measured-orbit truth or flight qualification.</p></body></html>"""


def run_gravity_short_arc_validation(
    config_path: str | Path,
    *,
    project_root: str | Path,
) -> GravityShortArcResult:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_gravity_short_arc_config(config_file)
    closure_path = _project_path(root, config["prerequisite_closure"], "prerequisite_closure")
    closure = verify_gravity_ladder_closure(closure_path, project_root=root)
    baseline = json.loads(
        _project_path(root, config["baseline_configuration"], "baseline_configuration").read_text(
            encoding="utf-8"
        )
    )
    initial_state = initial_state_from_config(baseline)
    field = CofGravityField.from_file(_project_path(root, config["gravity_file"], "gravity_file"))
    eop = GmatEopDataset.from_file(
        _project_path(root, config["eop_file"], "eop_file"),
        expected_sha256=GMAT_R2026A_EOP_SHA256,
    )
    reference = _project_path(root, config["reference_root"], "reference_root")
    output_directory = reference / "output"
    missing = [
        output_directory / f"{item['model_id']}_SHORT_ARC.e"
        for item in config["models"]
        if not (output_directory / f"{item['model_id']}_SHORT_ARC.e").is_file()
    ]
    if missing:
        shown = "\n".join(f"  - {_relative(path, root)}" for path in missing)
        raise FileNotFoundError(
            f"The 1D.1 GMAT matrix is incomplete; {len(missing)} ephemerides are missing:\n{shown}"
        )
    duration = float(config["duration_seconds"])
    step = float(config["output_step_seconds"])
    times = np.arange(int(round(duration / step)) + 1, dtype=float) * step
    integrator = config["integrator"]
    thresholds = config["thresholds"]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")
    result_directory = root / "results" / str(config["experiment_id"]) / stamp
    result_directory.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    all_checks: list[dict[str, Any]] = []
    for model in config["models"]:
        model_id = str(model["model_id"])
        source_path = output_directory / f"{model_id}_SHORT_ARC.e"
        imported = parse_stk_time_pos_vel(source_path, model_name=f"gmat_{model_id.lower()}")
        imported, grid = _canonicalize_nominal_output_grid(
            imported,
            expected_step_seconds=step,
            expected_duration_seconds=duration,
            tolerance_seconds=float(config["time_grid_tolerance_seconds"]),
        )
        python_history = propagate_spherical_harmonic_gravity(
            initial_state,
            field,
            eop,
            times,
            degree=int(model["degree"]),
            order=int(model["order"]),
            method=str(integrator["python_method"]),
            relative_tolerance=float(integrator["python_relative_tolerance"]),
            absolute_tolerance=float(integrator["python_absolute_tolerance"]),
            maximum_step_seconds=float(integrator["python_maximum_step_seconds"]),
        )
        comparison = compare_state_histories(imported, python_history)
        summary = create_error_summary(comparison)
        initial_position, initial_velocity = _initial_difference(initial_state, imported)
        maximum_position = float(summary["position_difference_m"]["maximum_absolute"])
        maximum_velocity = float(summary["velocity_difference_mm_s"]["maximum_absolute"])
        checks = [
            ("initial_position", initial_position, float(thresholds["initial_position_difference_m"]), "m"),
            ("initial_velocity", initial_velocity, float(thresholds["initial_velocity_difference_mm_s"]), "mm/s"),
            ("maximum_position", maximum_position, float(thresholds["maximum_position_difference_m"]), "m"),
            ("maximum_velocity", maximum_velocity, float(thresholds["maximum_velocity_difference_mm_s"]), "mm/s"),
        ]
        check_records = [
            {
                "check_id": f"1D1-{model_id}-{name}",
                "measured_value": value,
                "limit": limit,
                "unit": unit,
                "status": "passed" if value <= limit else "failed",
            }
            for name, value, limit, unit in checks
        ]
        all_checks.extend(check_records)
        model_status = "passed" if all(item["status"] == "passed" for item in check_records) else "failed"
        comparison_path = result_directory / f"{model_id}_python_vs_gmat.csv"
        python_path = result_directory / f"{model_id}_python_states.csv"
        write_comparison_csv(comparison_path, comparison)
        write_state_history_csv(python_path, python_history)
        records.append(
            {
                **model,
                "status": model_status,
                "source_ephemeris": _relative(source_path, root),
                "source_ephemeris_sha256": _sha256(source_path),
                "sample_count": int(imported.elapsed_seconds.size),
                "grid_diagnostics": grid,
                "initial_position_difference_m": initial_position,
                "initial_velocity_difference_mm_s": initial_velocity,
                "maximum_position_difference_m": maximum_position,
                "maximum_velocity_difference_mm_s": maximum_velocity,
                "final_position_difference_m": float(summary["position_difference_m"]["final"]),
                "final_velocity_difference_mm_s": float(summary["velocity_difference_mm_s"]["final"]),
                "python_runtime_seconds": python_history.runtime_seconds,
                "python_function_evaluations": python_history.function_evaluations,
                "checks": check_records,
            }
        )
    passed = sum(item["status"] == "passed" for item in records)
    status = "passed_with_warnings" if passed == len(records) else "failed_validation"
    decision = (
        "advance_to_1d2_multicase_full_arc_validation"
        if passed == len(records)
        else "stop_and_investigate_failed_short_arc_model"
    )
    summary_path = result_directory / "gravity_short_arc_summary.json"
    _write_json(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": config["experiment_id"],
            "status": status,
            "decision": decision,
            "prerequisite_closure_id": closure.closure_id,
            "model_count": len(records),
            "passed_model_count": passed,
            "failed_model_count": len(records) - passed,
            "check_count": len(all_checks),
            "passed_check_count": sum(item["status"] == "passed" for item in all_checks),
            "duration_seconds": duration,
            "output_step_seconds": step,
            "thresholds": thresholds,
            "models": records,
            "scientific_scope": config["scientific_cautions"],
        },
        summary_path,
    )
    csv_path = result_directory / "gravity_short_arc_model_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "model_id", "degree", "order", "status", "sample_count",
            "initial_position_difference_m", "initial_velocity_difference_mm_s",
            "maximum_position_difference_m", "maximum_velocity_difference_mm_s",
            "final_position_difference_m", "final_velocity_difference_mm_s",
            "python_runtime_seconds", "python_function_evaluations",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: item[key] for key in fieldnames} for item in records)
    report = result_directory / "GMAT_GRAVITY_SHORT_ARC_1D1_REPORT.html"
    report.write_text(_report_html(status, decision, records), encoding="utf-8", newline="\n")
    manifest = result_directory / "RUN_MANIFEST.json"
    files = [path for path in result_directory.iterdir() if path.is_file() and path != manifest]
    _write_json(
        {
            "files": [
                {"path": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in sorted(files)
            ]
        },
        manifest,
    )
    return GravityShortArcResult(
        experiment_id=str(config["experiment_id"]),
        status=status,
        decision=decision,
        model_count=len(records),
        passed_model_count=passed,
        check_count=len(all_checks),
        maximum_position_difference_m=max(item["maximum_position_difference_m"] for item in records),
        maximum_velocity_difference_mm_s=max(item["maximum_velocity_difference_mm_s"] for item in records),
        result_directory=result_directory,
        report_path=report,
    )


def package_gravity_short_arc_results(
    config_path: str | Path,
    *,
    project_root: str | Path,
    output_path: str | Path,
) -> Path:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_gravity_short_arc_config(config_file)
    reference = _project_path(root, config["reference_root"], "reference_root")
    expected = [reference / "output" / f"{item['model_id']}_SHORT_ARC.e" for item in config["models"]]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot package 1D.1: {len(missing)} GMAT ephemerides are missing.")
    result_root = root / "results" / str(config["experiment_id"])
    completed = sorted(
        path for path in result_root.glob("*")
        if path.is_dir() and (path / "gravity_short_arc_summary.json").is_file()
    )
    if not completed:
        raise FileNotFoundError("Run the Python 1D.1 validation before packaging.")
    members: set[Path] = {config_file}
    members.update(path for path in reference.rglob("*") if path.is_file() and "archive" not in path.parts)
    members.update(path for path in completed[-1].rglob("*") if path.is_file())
    members.update(
        {
            _project_path(root, config["prerequisite_closure"], "prerequisite_closure"),
            _project_path(root, config["gravity_file"], "gravity_file"),
            root / "data/reference/gmat_r2026a/JGM2_PROVENANCE_1D0.json",
            root / "data/reference/gmat_1d0/output/GMAT_GRAVITY_LADDER_1D0.csv",
            root / "results/EXP-GMAT-GRAVITY-1D0-001/2026-07-19_202029_369249Z/gravity_ladder_summary.json",
            root / "results/EXP-GMAT-GRAVITY-1D0-001/2026-07-19_202029_369249Z/RUN_MANIFEST.json",
        }
    )
    archive = Path(output_path).resolve()
    if archive.exists():
        raise FileExistsError(f"Move or rename the existing result ZIP first: {archive}")
    temporary = archive.with_name(f".{archive.name}.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
        for member in sorted(members):
            stream.write(member, _relative(member, root))
    os.replace(temporary, archive)
    with zipfile.ZipFile(archive) as stream:
        bad = stream.testzip()
        if bad:
            raise RuntimeError(f"Created 1D.1 result ZIP failed at {bad}.")
    return archive
