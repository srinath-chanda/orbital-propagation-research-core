"""Research Core 1D.2 multi-case full-arc GMAT gravity validation."""

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
from typing import Any, Callable

import numpy as np

from .analysis.comparison import compare_state_histories, create_error_summary
from .external_validation import (
    _canonicalize_nominal_output_grid,
    initial_state_from_config,
    parse_stk_time_pos_vel,
)
from .gmat_eop import GMAT_R2026A_EOP_SHA256, GmatEopDataset
from .gmat_gravity_short_arc import EXPECTED_MODELS
from .gmat_gravity_short_arc_closure import verify_gravity_short_arc_closure
from .gravity_harmonics import CofGravityField
from .outputs import write_comparison_csv, write_state_history_csv
from .propagators.numerical_gravity import propagate_spherical_harmonic_gravity


SCHEMA_VERSION = "1D.2"
EXPECTED_CASES = (
    ("D01_LOW_350KM_I1_FEB_6H", 6.0),
    ("D02_MID_500KM_I28P5_MAR_12H", 12.0),
    ("D03_CRITICAL_700KM_I63P4_MAY_24H", 24.0),
    ("D04_SSO_900KM_I97P6_JUN_36H", 36.0),
    ("D05_RETRO_1200KM_I120_AUG_48H", 48.0),
    ("D06_LONG_550KM_I45_SEP_72H", 72.0),
)


@dataclass(frozen=True)
class PreparedGravityMulticase:
    experiment_id: str
    case_count: int
    model_count: int
    expected_output_count: int
    expected_check_count: int
    master_script: Path
    run_order: Path
    manifest: Path
    archived_outputs: tuple[Path, ...]


@dataclass(frozen=True)
class GravityMulticaseResult:
    experiment_id: str
    status: str
    decision: str
    case_count: int
    passed_case_count: int
    model_run_count: int
    passed_model_run_count: int
    check_count: int
    passed_check_count: int
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
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must remain inside the project.") from exc
    return resolved


def _duration_thresholds(config: dict[str, Any], duration_hours: float) -> tuple[float, float]:
    for tier in config["threshold_policy"]["duration_tiers"]:
        if duration_hours <= float(tier["maximum_duration_hours"]):
            return (
                float(tier["maximum_position_difference_m"]),
                float(tier["maximum_velocity_difference_mm_s"]),
            )
    raise ValueError(f"No preregistered threshold covers {duration_hours:g} hours.")


def load_gravity_multicase_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Gravity multi-case schema must be {SCHEMA_VERSION!r}.")
    models = tuple(
        (str(item["model_id"]), int(item["degree"]), int(item["order"]))
        for item in payload.get("models", [])
    )
    if models != EXPECTED_MODELS:
        raise ValueError(f"The preregistered gravity models must be {EXPECTED_MODELS!r}.")
    cases = payload.get("cases", [])
    observed_cases = tuple(
        (str(item.get("case_id")), float(item.get("duration_hours", -1.0))) for item in cases
    )
    if observed_cases != EXPECTED_CASES:
        raise ValueError(f"The preregistered 1D.2 cases must be {EXPECTED_CASES!r}.")
    if payload.get("preregistration_status") != "frozen_before_any_1d2_gmat_output":
        raise ValueError("The 1D.2 matrix is not frozen before GMAT execution.")
    if payload.get("threshold_policy", {}).get("status") != "preregistered_before_first_1d2_gmat_run":
        raise ValueError("The 1D.2 thresholds are not preregistered.")
    step = float(payload["output_step_seconds"])
    if step <= 0.0 or float(payload["time_grid_tolerance_seconds"]) <= 0.0:
        raise ValueError("The output step and time-grid tolerance must be positive.")
    for case in cases:
        case_id = str(case["case_id"])
        duration_seconds = float(case["duration_hours"]) * 3600.0
        if abs(duration_seconds / step - round(duration_seconds / step)) > 1e-12:
            raise ValueError(f"{case_id} duration is not an integer output-step multiple.")
        if float(case["altitude_km"]) <= 100.0:
            raise ValueError(f"{case_id} altitude must exceed 100 km.")
        if not 0.0 <= float(case["inclination_deg"]) <= 180.0:
            raise ValueError(f"{case_id} inclination must be in [0, 180] degrees.")
        if not 0.0 <= float(case["eccentricity"]) < 1.0:
            raise ValueError(f"{case_id} eccentricity must be in [0, 1).")
        datetime.fromisoformat(str(case["epoch_utc"]).replace("Z", "+00:00"))
        _duration_thresholds(payload, float(case["duration_hours"]))
    return payload


def _case_configuration(
    baseline: dict[str, Any], case: dict[str, Any], experiment_id: str
) -> dict[str, Any]:
    result = deepcopy(baseline)
    radius = float(result["earth_model"]["equatorial_radius_km"])
    result["experiment"].update(
        {
            "experiment_id": f"{experiment_id}-{case['case_id']}",
            "case_id": str(case["case_id"]),
            "title": f"Research Core 1D.2 gravity validation: {case['case_id']}",
        }
    )
    result["initial_state"].update(
        {
            "epoch_utc": str(case["epoch_utc"]),
            "semi_major_axis_km": radius + float(case["altitude_km"]),
            "eccentricity": float(case["eccentricity"]),
            "inclination_deg": float(case["inclination_deg"]),
            "raan_deg": float(case["raan_deg"]),
            "argument_of_perigee_deg": float(case["argument_of_perigee_deg"]),
            "true_anomaly_deg": float(case["true_anomaly_deg"]),
            "notes": (
                "Preregistered Research Core 1D.2 full-arc case; altitude defines "
                "semi-major axis relative to the matched JGM2 radius."
            ),
        }
    )
    result["propagation"]["default_duration_hours"] = float(case["duration_hours"])
    return result


def _gmat_epoch(value: str) -> str:
    epoch = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)
    return epoch.astimezone(timezone.utc).strftime("%d %b %Y %H:%M:%S.%f")[:-3]


def _gmat_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _resource_block(
    case_number: int,
    model: dict[str, Any],
    case_config: dict[str, Any],
    config: dict[str, Any],
    *,
    gravity_file: Path,
    output_ephemeris: Path,
) -> tuple[str, str]:
    state = initial_state_from_config(case_config)
    model_id = str(model["model_id"])
    prefix = f"D{case_number:02d}{model_id}"
    degree = int(model["degree"])
    order = int(model["order"])
    integrator = config["integrator"]
    step = float(config["output_step_seconds"])
    duration = float(case_config["propagation"]["default_duration_hours"]) * 3600.0
    maximum_step = min(float(integrator["gmat_maximum_step_seconds"]), step)
    resource = f"""
% {case_config['experiment']['case_id']} / {model_id} degree {degree}, order {order}
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
{prefix}FM.GravityField.Earth.Order = {order};
{prefix}FM.GravityField.Earth.PotentialFile = '{_gmat_path(gravity_file)}';
{prefix}FM.GravityField.Earth.TideModel = 'None';

Create Propagator {prefix}Prop;
{prefix}Prop.FM = {prefix}FM;
{prefix}Prop.Type = {integrator['gmat_method']};
{prefix}Prop.InitialStepSize = {step:.15g};
{prefix}Prop.Accuracy = {float(integrator['gmat_accuracy']):.15g};
{prefix}Prop.MinStep = 1e-6;
{prefix}Prop.MaxStep = {maximum_step:.15g};
{prefix}Prop.MaxStepAttempts = 50;
{prefix}Prop.StopIfAccuracyIsViolated = true;

Create EphemerisFile {prefix}Eph;
{prefix}Eph.Spacecraft = {prefix}Sat;
{prefix}Eph.Filename = '{_gmat_path(output_ephemeris)}';
{prefix}Eph.FileFormat = STK-TimePosVel;
{prefix}Eph.EpochFormat = UTCGregorian;
{prefix}Eph.InitialEpoch = InitialSpacecraftEpoch;
{prefix}Eph.FinalEpoch = FinalSpacecraftEpoch;
{prefix}Eph.StepSize = {step:.15g};
{prefix}Eph.Interpolator = Lagrange;
{prefix}Eph.InterpolationOrder = 7;
{prefix}Eph.CoordinateSystem = EarthMJ2000Eq;
{prefix}Eph.WriteEphemeris = true;
"""
    mission = f"Propagate {prefix}Prop({prefix}Sat) {{{prefix}Sat.ElapsedSecs = {duration:.15g}}};"
    return resource, mission


def build_gravity_multicase_master_script(
    config: dict[str, Any],
    baseline: dict[str, Any],
    *,
    gravity_file: Path,
    output_directory: Path,
) -> str:
    resources: list[str] = []
    missions: list[str] = []
    for case_number, case in enumerate(config["cases"], start=1):
        case_config = _case_configuration(baseline, case, str(config["experiment_id"]))
        for model in config["models"]:
            output = output_directory / f"{case['case_id']}_{model['model_id']}.e"
            resource, mission = _resource_block(
                case_number,
                model,
                case_config,
                config,
                gravity_file=gravity_file,
                output_ephemeris=output,
            )
            resources.append(resource)
            missions.append(mission)
    return (
        "%\n% Research Core 1D.2 multi-case full-arc normalized JGM2 matrix\n"
        "% Target GMAT R2026a; 6 cases; 4 gravity models; 60-second outputs.\n"
        "% Run once. All cases and thresholds were frozen before GMAT output.\n%\n"
        + "".join(resources)
        + "\nBeginMissionSequence;\n"
        + "\n".join(missions)
        + "\n"
    )


def prepare_gravity_multicase(
    config_path: str | Path,
    *,
    project_root: str | Path,
) -> PreparedGravityMulticase:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_gravity_multicase_config(config_file)
    closure = verify_gravity_short_arc_closure(
        _project_path(root, config["prerequisite_closure"], "prerequisite_closure"),
        project_root=root,
    )
    baseline = json.loads(
        _project_path(root, config["baseline_configuration"], "baseline_configuration").read_text(
            encoding="utf-8"
        )
    )
    gravity_file = _project_path(root, config["gravity_file"], "gravity_file")
    field = CofGravityField.from_file(gravity_file)
    if (field.maximum_degree, field.maximum_order) != (70, 70):
        raise ValueError("The 1D.2 gravity file must contain normalized JGM2 through 70x70.")
    eop_file = _project_path(root, config["eop_file"], "eop_file")
    if _sha256(eop_file) != GMAT_R2026A_EOP_SHA256:
        raise ValueError("The 1D.2 EOP file does not match the adopted R2026a evidence.")
    reference = _project_path(root, config["reference_root"], "reference_root")
    scripts = reference / "scripts"
    outputs = reference / "output"
    cases_directory = reference / "cases"
    for directory in (scripts, outputs, cases_directory):
        directory.mkdir(parents=True, exist_ok=True)

    expected_names = {
        f"{case['case_id']}_{model['model_id']}.e"
        for case in config["cases"]
        for model in config["models"]
    }
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

    master = scripts / "RUN_GRAVITY_MULTICASE_1D2.script"
    master.write_text(
        build_gravity_multicase_master_script(
            config, baseline, gravity_file=gravity_file, output_directory=outputs
        ),
        encoding="utf-8",
        newline="\n",
    )
    records: list[dict[str, Any]] = []
    for case_number, case in enumerate(config["cases"], start=1):
        case_config = _case_configuration(baseline, case, str(config["experiment_id"]))
        case_path = cases_directory / f"{case['case_id']}.json"
        _write_json(case_config, case_path)
        for model in config["models"]:
            output = outputs / f"{case['case_id']}_{model['model_id']}.e"
            resource, mission = _resource_block(
                case_number,
                model,
                case_config,
                config,
                gravity_file=gravity_file,
                output_ephemeris=output,
            )
            script = scripts / f"{case['case_id']}_{model['model_id']}.script"
            script.write_text(
                "% Research Core 1D.2 fallback individual full-arc case/model\n"
                + resource
                + "\nBeginMissionSequence;\n"
                + mission
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            records.append(
                {
                    "case_id": case["case_id"],
                    "model_id": model["model_id"],
                    "degree": model["degree"],
                    "order": model["order"],
                    "duration_hours": case["duration_hours"],
                    "script": _relative(script, root),
                    "script_sha256": _sha256(script),
                    "output": _relative(output, root),
                }
            )
    run_order = reference / "RUN_ORDER_1D2.txt"
    run_order.write_text(
        "RESEARCH CORE 1D.2 GMAT RUN ORDER\n\n"
        "Preferred: run scripts/RUN_GRAVITY_MULTICASE_1D2.script once.\n"
        "The mission propagates 6 cases x 4 gravity models in registered order.\n"
        "Fallback only if the master does not interpret: run the 24 individual scripts.\n"
        "Expected untouched ephemerides: 24. Do not rerun after successful creation.\n",
        encoding="utf-8",
        newline="\n",
    )
    readme = reference / "README.md"
    readme.write_text(
        "# Research Core 1D.2 GMAT matrix\n\n"
        "Run `scripts/RUN_GRAVITY_MULTICASE_1D2.script` once in GMAT R2026a. "
        "It creates 24 STK-TimePosVel ephemerides in `output`. Do not alter the "
        "frozen configuration or thresholds after inspecting an output.\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = reference / "GMAT_GRAVITY_MULTICASE_1D2_MANIFEST.json"
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
            "eop_file": _relative(eop_file, root),
            "eop_file_sha256": _sha256(eop_file),
            "master_script": _relative(master, root),
            "master_script_sha256": _sha256(master),
            "output_step_seconds": config["output_step_seconds"],
            "case_count": len(config["cases"]),
            "model_count": len(config["models"]),
            "expected_output_count": len(records),
            "expected_check_count": len(records) * 4,
            "archived_previous_output_count": len(archived),
            "runs": records,
        },
        manifest,
    )
    return PreparedGravityMulticase(
        experiment_id=str(config["experiment_id"]),
        case_count=len(config["cases"]),
        model_count=len(config["models"]),
        expected_output_count=len(records),
        expected_check_count=len(records) * 4,
        master_script=master,
        run_order=run_order,
        manifest=manifest,
        archived_outputs=tuple(archived),
    )


def _initial_difference(initial_state: Any, history: Any) -> tuple[float, float]:
    position = float(np.linalg.norm(history.positions_km[0] - initial_state.position_km) * 1000.0)
    velocity = float(np.linalg.norm(history.velocities_km_s[0] - initial_state.velocity_km_s) * 1.0e6)
    return position, velocity


def _report_html(status: str, decision: str, cases: list[dict[str, Any]]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(case['case_id'])}</td>"
        f"<td>{case['duration_hours']:g}</td>"
        f"<td>{case['passed_model_count']}/{case['model_count']}</td>"
        f"<td>{case['maximum_position_difference_m']:.9e}</td>"
        f"<td>{case['maximum_velocity_difference_mm_s']:.9e}</td>"
        f"<td>{html.escape(case['status'])}</td></tr>"
        for case in cases
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Research Core 1D.2 GMAT Gravity Matrix</title><style>
body{{font-family:Arial,sans-serif;margin:2rem;color:#172033}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccd3df;padding:.5rem}}th{{background:#edf2f8}}</style></head><body>
<h1>Research Core 1D.2 GMAT Gravity Matrix</h1><p><strong>Status:</strong> {status}</p>
<p><strong>Decision:</strong> {decision}</p><table><thead><tr><th>Case</th><th>Hours</th>
<th>Models passed</th><th>Maximum position difference (m)</th>
<th>Maximum velocity difference (mm/s)</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table>
<p>Six preregistered cases span 6–72 hours. Each uses 2×0, 4×4, 8×8, and 20×20
normalized JGM2 gravity. This is independent software-model agreement, not measured-orbit
truth or flight qualification.</p></body></html>"""


def run_gravity_multicase_validation(
    config_path: str | Path,
    *,
    project_root: str | Path,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
) -> GravityMulticaseResult:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_gravity_multicase_config(config_file)
    closure = verify_gravity_short_arc_closure(
        _project_path(root, config["prerequisite_closure"], "prerequisite_closure"),
        project_root=root,
    )
    baseline = json.loads(
        _project_path(root, config["baseline_configuration"], "baseline_configuration").read_text(
            encoding="utf-8"
        )
    )
    field = CofGravityField.from_file(_project_path(root, config["gravity_file"], "gravity_file"))
    eop = GmatEopDataset.from_file(
        _project_path(root, config["eop_file"], "eop_file"),
        expected_sha256=GMAT_R2026A_EOP_SHA256,
    )
    reference = _project_path(root, config["reference_root"], "reference_root")
    output_directory = reference / "output"
    expected = [
        output_directory / f"{case['case_id']}_{model['model_id']}.e"
        for case in config["cases"]
        for model in config["models"]
    ]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        shown = "\n".join(f"  - {_relative(path, root)}" for path in missing)
        raise FileNotFoundError(
            f"The 1D.2 GMAT matrix is incomplete; {len(missing)} ephemerides are missing:\n{shown}"
        )

    step = float(config["output_step_seconds"])
    integrator = config["integrator"]
    policy = config["threshold_policy"]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")
    result_directory = root / "results" / str(config["experiment_id"]) / stamp
    result_directory.mkdir(parents=True, exist_ok=False)
    case_records: list[dict[str, Any]] = []
    flat_records: list[dict[str, Any]] = []
    all_checks: list[dict[str, Any]] = []
    completed_runs = 0
    total_runs = len(config["cases"]) * len(config["models"])

    for case_index, case in enumerate(config["cases"], start=1):
        case_id = str(case["case_id"])
        duration_hours = float(case["duration_hours"])
        duration = duration_hours * 3600.0
        times = np.arange(int(round(duration / step)) + 1, dtype=float) * step
        case_config = _case_configuration(baseline, case, str(config["experiment_id"]))
        initial_state = initial_state_from_config(case_config)
        maximum_position_limit, maximum_velocity_limit = _duration_thresholds(
            config, duration_hours
        )
        model_records: list[dict[str, Any]] = []
        case_directory = result_directory / case_id
        case_directory.mkdir(parents=True, exist_ok=False)
        for model_index, model in enumerate(config["models"], start=1):
            model_id = str(model["model_id"])
            source_path = output_directory / f"{case_id}_{model_id}.e"
            imported = parse_stk_time_pos_vel(
                source_path, model_name=f"gmat_{case_id.lower()}_{model_id.lower()}"
            )
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
            error = create_error_summary(comparison)
            initial_position, initial_velocity = _initial_difference(initial_state, imported)
            maximum_position = float(error["position_difference_m"]["maximum_absolute"])
            maximum_velocity = float(error["velocity_difference_mm_s"]["maximum_absolute"])
            checks = (
                (
                    "initial_position",
                    initial_position,
                    float(policy["initial_position_difference_m"]),
                    "m",
                ),
                (
                    "initial_velocity",
                    initial_velocity,
                    float(policy["initial_velocity_difference_mm_s"]),
                    "mm/s",
                ),
                ("maximum_position", maximum_position, maximum_position_limit, "m"),
                ("maximum_velocity", maximum_velocity, maximum_velocity_limit, "mm/s"),
            )
            check_records = [
                {
                    "check_id": f"1D2-{case_index:02d}-{model_index:02d}-{name}",
                    "measured_value": value,
                    "limit": limit,
                    "unit": unit,
                    "status": "passed" if value <= limit else "failed",
                }
                for name, value, limit, unit in checks
            ]
            all_checks.extend(check_records)
            model_status = (
                "passed" if all(item["status"] == "passed" for item in check_records) else "failed"
            )
            comparison_path = case_directory / f"{model_id}_python_vs_gmat.csv"
            python_path = case_directory / f"{model_id}_python_states.csv"
            write_comparison_csv(comparison_path, comparison)
            write_state_history_csv(python_path, python_history)
            record = {
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
                "final_position_difference_m": float(error["position_difference_m"]["final"]),
                "final_velocity_difference_mm_s": float(error["velocity_difference_mm_s"]["final"]),
                "python_runtime_seconds": python_history.runtime_seconds,
                "python_function_evaluations": python_history.function_evaluations,
                "checks": check_records,
            }
            model_records.append(record)
            flat_records.append({"case_id": case_id, "duration_hours": duration_hours, **record})
            completed_runs += 1
            if progress_callback is not None:
                progress_callback(completed_runs, total_runs, case_id, model_id)
        passed_models = sum(item["status"] == "passed" for item in model_records)
        case_records.append(
            {
                **case,
                "status": "passed" if passed_models == len(model_records) else "failed",
                "model_count": len(model_records),
                "passed_model_count": passed_models,
                "failed_model_count": len(model_records) - passed_models,
                "check_count": sum(len(item["checks"]) for item in model_records),
                "passed_check_count": sum(
                    check["status"] == "passed"
                    for item in model_records
                    for check in item["checks"]
                ),
                "maximum_position_limit_m": maximum_position_limit,
                "maximum_velocity_limit_mm_s": maximum_velocity_limit,
                "maximum_position_difference_m": max(
                    item["maximum_position_difference_m"] for item in model_records
                ),
                "maximum_velocity_difference_mm_s": max(
                    item["maximum_velocity_difference_mm_s"] for item in model_records
                ),
                "models": model_records,
            }
        )
    passed_cases = sum(item["status"] == "passed" for item in case_records)
    passed_models = sum(item["status"] == "passed" for item in flat_records)
    passed_checks = sum(item["status"] == "passed" for item in all_checks)
    status = "passed_with_warnings" if passed_cases == len(case_records) else "failed_validation"
    decision = (
        "close_higher_order_gravity_validation_and_advance_to_drag"
        if status == "passed_with_warnings"
        else "stop_and_investigate_failed_case_or_model"
    )
    summary_path = result_directory / "gravity_multicase_summary.json"
    _write_json(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": config["experiment_id"],
            "status": status,
            "decision": decision,
            "prerequisite_closure_id": closure.closure_id,
            "case_count": len(case_records),
            "passed_case_count": passed_cases,
            "failed_case_count": len(case_records) - passed_cases,
            "model_run_count": len(flat_records),
            "passed_model_run_count": passed_models,
            "failed_model_run_count": len(flat_records) - passed_models,
            "check_count": len(all_checks),
            "passed_check_count": passed_checks,
            "failed_check_count": len(all_checks) - passed_checks,
            "output_step_seconds": step,
            "threshold_policy": policy,
            "cases": case_records,
            "scientific_scope": config["scientific_cautions"],
        },
        summary_path,
    )
    csv_path = result_directory / "gravity_multicase_model_summary.csv"
    fieldnames = [
        "case_id",
        "duration_hours",
        "model_id",
        "degree",
        "order",
        "status",
        "sample_count",
        "initial_position_difference_m",
        "initial_velocity_difference_mm_s",
        "maximum_position_difference_m",
        "maximum_velocity_difference_mm_s",
        "final_position_difference_m",
        "final_velocity_difference_mm_s",
        "python_runtime_seconds",
        "python_function_evaluations",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: item[key] for key in fieldnames} for item in flat_records)
    report = result_directory / "GMAT_GRAVITY_MULTICASE_1D2_REPORT.html"
    report.write_text(_report_html(status, decision, case_records), encoding="utf-8", newline="\n")
    manifest = result_directory / "RUN_MANIFEST.json"
    files = [path for path in result_directory.rglob("*") if path.is_file() and path != manifest]
    _write_json(
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
        manifest,
    )
    return GravityMulticaseResult(
        experiment_id=str(config["experiment_id"]),
        status=status,
        decision=decision,
        case_count=len(case_records),
        passed_case_count=passed_cases,
        model_run_count=len(flat_records),
        passed_model_run_count=passed_models,
        check_count=len(all_checks),
        passed_check_count=passed_checks,
        maximum_position_difference_m=max(
            item["maximum_position_difference_m"] for item in flat_records
        ),
        maximum_velocity_difference_mm_s=max(
            item["maximum_velocity_difference_mm_s"] for item in flat_records
        ),
        result_directory=result_directory,
        report_path=report,
    )


def package_gravity_multicase_results(
    config_path: str | Path,
    *,
    project_root: str | Path,
    output_path: str | Path,
) -> Path:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_gravity_multicase_config(config_file)
    reference = _project_path(root, config["reference_root"], "reference_root")
    expected = [
        reference / "output" / f"{case['case_id']}_{model['model_id']}.e"
        for case in config["cases"]
        for model in config["models"]
    ]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot package 1D.2: {len(missing)} GMAT ephemerides are missing.")
    result_root = root / "results" / str(config["experiment_id"])
    completed = sorted(
        path
        for path in result_root.glob("*")
        if path.is_dir() and (path / "gravity_multicase_summary.json").is_file()
    )
    if not completed:
        raise FileNotFoundError("Run the Python 1D.2 validation before packaging.")

    closure_path = _project_path(root, config["prerequisite_closure"], "prerequisite_closure")
    closure_record = json.loads(closure_path.read_text(encoding="utf-8"))
    prerequisite_result = _project_path(
        root, closure_record["official_result_directory"], "official_result_directory"
    )
    members: set[Path] = {config_file, closure_path}
    members.update(path for path in reference.rglob("*") if path.is_file() and "archive" not in path.parts)
    members.update(path for path in completed[-1].rglob("*") if path.is_file())
    members.update(
        {
            _project_path(root, closure_record["configuration"], "1d1_configuration"),
            _project_path(root, config["gravity_file"], "gravity_file"),
            _project_path(root, config["eop_file"], "eop_file"),
            root / "data/reference/gmat_r2026a/JGM2_PROVENANCE_1D0.json",
            prerequisite_result / "gravity_short_arc_summary.json",
            prerequisite_result / "RUN_MANIFEST.json",
        }
    )
    members.update(
        _project_path(root, item["path"], f"{item['model_id']}_1d1_ephemeris")
        for item in closure_record["ephemerides"]
    )
    archive = Path(output_path).resolve()
    if archive.exists():
        raise FileExistsError(f"Move or rename the existing result ZIP first: {archive}")
    temporary = archive.with_name(f".{archive.name}.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
        for member in sorted(members):
            if member.is_file():
                stream.write(member, _relative(member, root))
    os.replace(temporary, archive)
    with zipfile.ZipFile(archive) as stream:
        bad = stream.testzip()
        if bad:
            raise RuntimeError(f"Created 1D.2 result ZIP failed at {bad}.")
    return archive
