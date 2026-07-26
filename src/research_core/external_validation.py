"""GMAT script preparation, STK ephemeris import, and external validation."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis.comparison import compare_state_histories, create_error_summary
from .analysis.j2 import compare_in_reference_rtn
from .data_models import CartesianState, StateHistory
from .gmat_j2_diagnostics import build_gmat_acceleration_diagnostic_script
from .orbital_elements import elements_from_config, elements_to_cartesian
from .outputs import (
    write_comparison_csv,
    write_json,
    write_rtn_comparison_csv,
    write_state_history_csv,
)
from .propagators.numerical_j2 import (
    propagate_numerical_j2,
    propagate_numerical_j2_gmat_matched,
)
from .propagators.numerical_two_body import propagate_numerical_two_body
from .time_utils import timestamps_from_epoch


@dataclass(frozen=True)
class PreparedGmatFiles:
    two_body_script: Path
    j2_script: Path
    j2_short_arc_script: Path
    acceleration_diagnostic_script: Path
    two_body_ephemeris: Path
    j2_ephemeris: Path
    j2_short_arc_ephemeris: Path
    acceleration_diagnostic_report: Path
    metadata_file: Path
    initial_state: CartesianState


@dataclass(frozen=True)
class ExternalValidationResult:
    experiment_id: str
    result_directory: Path
    validation_status: str
    two_body_maximum_position_difference_m: float
    two_body_maximum_velocity_difference_mm_s: float
    j2_maximum_position_difference_m: float
    j2_maximum_velocity_difference_mm_s: float
    j2_gmat_matched_maximum_position_difference_m: float
    j2_gmat_matched_maximum_velocity_difference_mm_s: float
    two_body_final_position_difference_m: float
    j2_final_position_difference_m: float
    j2_gmat_matched_final_position_difference_m: float
    warnings: tuple[str, ...]
    created_files: tuple[Path, ...]
    report_path: Path


@dataclass(frozen=True)
class J2ArcValidationResult:
    experiment_id: str
    result_directory: Path
    validation_status: str
    fixed_axis_maximum_position_difference_m: float
    fixed_axis_maximum_velocity_difference_mm_s: float
    gmat_matched_maximum_position_difference_m: float
    gmat_matched_maximum_velocity_difference_mm_s: float
    warnings: tuple[str, ...]
    created_files: tuple[Path, ...]
    report_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_from_config(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _gmat_epoch(value: str) -> str:
    epoch = _utc_from_config(value)
    return epoch.strftime("%d %b %Y %H:%M:%S.%f")[:-3]


def _portable_gmat_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _recorded_source_path(path: str | Path, project_root: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:
        return str(resolved)


def initial_state_from_config(config: dict[str, Any]) -> CartesianState:
    elements = elements_from_config(config["initial_state"])
    mu = float(config["earth_model"]["gravitational_parameter_km3_s2"])
    position, velocity = elements_to_cartesian(elements, mu)
    return CartesianState(
        epoch_utc=str(config["initial_state"]["epoch_utc"]),
        frame=str(config["external_validation"]["frame"]),
        position_km=position,
        velocity_km_s=velocity,
    )


def build_gmat_script(
    config: dict[str, Any],
    *,
    model: str,
    output_ephemeris: Path,
    duration_seconds: float | None = None,
    output_step_seconds: float | None = None,
    stage_label: str | None = None,
    generated_release: str = "1C.0",
) -> str:
    """Build a GMAT R2026a script for point-mass or zonal-J2 validation."""
    if model not in {"two_body", "j2"}:
        raise ValueError("model must be 'two_body' or 'j2'.")

    state = initial_state_from_config(config)
    ext = config["external_validation"]
    degree = 0 if model == "two_body" else 2
    label = "TwoBody" if model == "two_body" else "J2"
    duration = float(
        ext["duration_seconds"]
        if duration_seconds is None
        else duration_seconds
    )
    output_step = float(
        ext["output_step_seconds"]
        if output_step_seconds is None
        else output_step_seconds
    )
    accuracy = float(ext["gmat_accuracy"])
    initial_step = min(float(ext["gmat_initial_step_seconds"]), output_step)
    max_step = min(float(ext["gmat_maximum_step_seconds"]), output_step)
    gravity_file = str(ext["gravity_file"])
    epoch = _gmat_epoch(state.epoch_utc)
    ephemeris_path = _portable_gmat_path(output_ephemeris)

    stage = stage_label or "full_arc"
    case_id = str(config["experiment"]["case_id"])
    return f"""%
% {case_id} {label} {stage} external-validation script
% Generated by Orbital Propagation Research Core {generated_release}
% Target GMAT release: {ext['tool_version']}
%
% This is a controlled model-matching case, not measured-orbit truth.
%

Create Spacecraft ValidationSat;
ValidationSat.DateFormat = UTCGregorian;
ValidationSat.Epoch = '{epoch}';
ValidationSat.CoordinateSystem = EarthMJ2000Eq;
ValidationSat.DisplayStateType = Cartesian;
ValidationSat.X = {state.position_km[0]:.15f};
ValidationSat.Y = {state.position_km[1]:.15f};
ValidationSat.Z = {state.position_km[2]:.15f};
ValidationSat.VX = {state.velocity_km_s[0]:.15f};
ValidationSat.VY = {state.velocity_km_s[1]:.15f};
ValidationSat.VZ = {state.velocity_km_s[2]:.15f};
ValidationSat.DryMass = 500;
ValidationSat.Cd = 2.2;
ValidationSat.Cr = 1.0;
ValidationSat.DragArea = 4;
ValidationSat.SRPArea = 4;

Create ForceModel ValidationFM;
ValidationFM.CentralBody = Earth;
ValidationFM.PrimaryBodies = {{Earth}};
ValidationFM.Drag = None;
ValidationFM.SRP = Off;
ValidationFM.RelativisticCorrection = Off;
ValidationFM.ErrorControl = RSSStep;
ValidationFM.GravityField.Earth.Degree = {degree};
ValidationFM.GravityField.Earth.Order = 0;
ValidationFM.GravityField.Earth.PotentialFile = '{gravity_file}';
ValidationFM.GravityField.Earth.TideModel = 'None';

Create Propagator ValidationProp;
ValidationProp.FM = ValidationFM;
ValidationProp.Type = {ext['gmat_integrator']};
ValidationProp.InitialStepSize = {initial_step:.15g};
ValidationProp.Accuracy = {accuracy:.15g};
ValidationProp.MinStep = 1e-6;
ValidationProp.MaxStep = {max_step:.15g};
ValidationProp.MaxStepAttempts = 50;
ValidationProp.StopIfAccuracyIsViolated = true;

Create EphemerisFile ValidationEphemeris;
ValidationEphemeris.Spacecraft = ValidationSat;
ValidationEphemeris.Filename = '{ephemeris_path}';
ValidationEphemeris.FileFormat = STK-TimePosVel;
ValidationEphemeris.EpochFormat = UTCGregorian;
ValidationEphemeris.InitialEpoch = InitialSpacecraftEpoch;
ValidationEphemeris.FinalEpoch = FinalSpacecraftEpoch;
ValidationEphemeris.StepSize = {output_step:.15g};
ValidationEphemeris.Interpolator = Lagrange;
ValidationEphemeris.InterpolationOrder = 7;
ValidationEphemeris.CoordinateSystem = EarthMJ2000Eq;
ValidationEphemeris.WriteEphemeris = true;

BeginMissionSequence;
Propagate ValidationProp(ValidationSat) {{ValidationSat.ElapsedSecs = {duration:.15g}}};
"""


def prepare_gmat_files(
    config_path: str | Path,
    *,
    project_root: str | Path,
) -> PreparedGmatFiles:
    config_file = Path(config_path).resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    root = Path(project_root).resolve()
    reference_root = root / "data" / "reference" / "gmat"
    scripts_dir = reference_root / "scripts"
    output_dir = reference_root / "output"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    two_body_ephemeris = output_dir / "CASE_LEO400_GMAT_TWO_BODY.e"
    j2_ephemeris = output_dir / "CASE_LEO400_GMAT_J2.e"
    j2_short_arc_ephemeris = output_dir / "CASE_LEO400_GMAT_J2_SHORT_ARC.e"
    acceleration_diagnostic_report = (
        output_dir / "CASE_LEO400_GMAT_ACCELERATION_DIAGNOSTIC.csv"
    )
    two_body_script = scripts_dir / "CASE_LEO400_GMAT_TWO_BODY.script"
    j2_script = scripts_dir / "CASE_LEO400_GMAT_J2.script"
    j2_short_arc_script = scripts_dir / "CASE_LEO400_GMAT_J2_SHORT_ARC.script"
    acceleration_diagnostic_script = (
        scripts_dir / "CASE_LEO400_GMAT_ACCELERATION_DIAGNOSTIC.script"
    )

    two_body_script.write_text(
        build_gmat_script(config, model="two_body", output_ephemeris=two_body_ephemeris),
        encoding="utf-8",
        newline="\n",
    )
    j2_script.write_text(
        build_gmat_script(config, model="j2", output_ephemeris=j2_ephemeris),
        encoding="utf-8",
        newline="\n",
    )
    short_arc = config["external_validation"]["short_arc"]
    j2_short_arc_script.write_text(
        build_gmat_script(
            config,
            model="j2",
            output_ephemeris=j2_short_arc_ephemeris,
            duration_seconds=float(short_arc["duration_seconds"]),
            output_step_seconds=float(short_arc["output_step_seconds"]),
            stage_label="short_arc",
        ),
        encoding="utf-8",
        newline="\n",
    )
    acceleration_diagnostic_script.write_text(
        build_gmat_acceleration_diagnostic_script(
            config,
            output_report=acceleration_diagnostic_report,
        ),
        encoding="utf-8",
        newline="\n",
    )

    state = initial_state_from_config(config)
    metadata = {
        "research_core_version": "1C.0",
        "target_tool": config["external_validation"]["tool"],
        "target_tool_version": config["external_validation"]["tool_version"],
        "configuration_file": str(config_file),
        "configuration_sha256": _sha256(config_file),
        "frame": state.frame,
        "epoch_utc": state.epoch_utc,
        "initial_position_km": state.position_km.tolist(),
        "initial_velocity_km_s": state.velocity_km_s.tolist(),
        "earth_model": config["earth_model"],
        "external_validation": config["external_validation"],
        "generated_files": {
            "two_body_script": str(two_body_script),
            "j2_script": str(j2_script),
            "j2_short_arc_script": str(j2_short_arc_script),
            "acceleration_diagnostic_script": str(
                acceleration_diagnostic_script
            ),
            "two_body_ephemeris": str(two_body_ephemeris),
            "j2_ephemeris": str(j2_ephemeris),
            "j2_short_arc_ephemeris": str(j2_short_arc_ephemeris),
            "acceleration_diagnostic_report": str(
                acceleration_diagnostic_report
            ),
        },
        "script_sha256": {
            "two_body": _sha256(two_body_script),
            "j2": _sha256(j2_script),
            "j2_short_arc": _sha256(j2_short_arc_script),
            "acceleration_diagnostic": _sha256(
                acceleration_diagnostic_script
            ),
        },
        "status": "scripts_prepared_gmat_execution_pending",
    }
    metadata_file = reference_root / "GMAT_PREPARATION_METADATA.json"
    write_json(metadata, metadata_file)

    return PreparedGmatFiles(
        two_body_script=two_body_script,
        j2_script=j2_script,
        j2_short_arc_script=j2_short_arc_script,
        acceleration_diagnostic_script=acceleration_diagnostic_script,
        two_body_ephemeris=two_body_ephemeris,
        j2_ephemeris=j2_ephemeris,
        j2_short_arc_ephemeris=j2_short_arc_ephemeris,
        acceleration_diagnostic_report=acceleration_diagnostic_report,
        metadata_file=metadata_file,
        initial_state=state,
    )


_GMAT_EPOCH_FORMATS = (
    "%d %b %Y %H:%M:%S.%f",
    "%d %b %Y %H:%M:%S",
)


def _parse_scenario_epoch(text: str) -> datetime:
    value = text.strip().strip("'").strip('"')
    for fmt in _GMAT_EPOCH_FORMATS:
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Unsupported STK ScenarioEpoch format: {text!r}")


def parse_stk_time_pos_vel(
    path: str | Path,
    *,
    model_name: str,
    frame: str = "EarthMJ2000Eq",
) -> StateHistory:
    """Parse a GMAT STK-TimePosVel ASCII ephemeris into a StateHistory."""
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"GMAT ephemeris not found: {source}")
    lines = source.read_text(encoding="utf-8-sig", errors="strict").splitlines()

    scenario_epoch: datetime | None = None
    data_start: int | None = None
    for index, raw in enumerate(lines):
        line = raw.strip()
        if line.lower().startswith("scenarioepoch"):
            scenario_epoch = _parse_scenario_epoch(line[len("ScenarioEpoch"):].strip())
        if line.lower() == "ephemeristimeposvel":
            data_start = index + 1
            break

    if scenario_epoch is None:
        raise ValueError("STK ephemeris is missing ScenarioEpoch.")
    if data_start is None:
        raise ValueError("STK ephemeris is missing EphemerisTimePosVel.")

    rows: list[list[float]] = []
    for raw in lines[data_start:]:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("end ephemeris"):
            break
        parts = re.split(r"[\s,]+", line)
        if len(parts) < 7:
            continue
        try:
            values = [float(value) for value in parts[:7]]
        except ValueError:
            continue
        rows.append(values)

    if len(rows) < 2:
        raise ValueError("STK ephemeris contains fewer than two numeric states.")

    array = np.asarray(rows, dtype=float)
    elapsed = array[:, 0]
    if not np.all(np.isfinite(array)):
        raise ValueError("STK ephemeris contains non-finite values.")
    if abs(float(elapsed[0])) > 1e-6:
        raise ValueError("STK ephemeris must begin at elapsed time zero.")
    if np.any(np.diff(elapsed) <= 0.0):
        raise ValueError("STK ephemeris times must be strictly increasing.")

    epoch_utc = scenario_epoch.isoformat().replace("+00:00", "Z")
    return StateHistory(
        model_name=model_name,
        frame=frame,
        epoch_utc=epoch_utc,
        elapsed_seconds=elapsed,
        timestamps_utc=timestamps_from_epoch(epoch_utc, elapsed),
        positions_km=array[:, 1:4],
        velocities_km_s=array[:, 4:7],
        runtime_seconds=0.0,
        solver_status=f"Imported from {source.name}",
        function_evaluations=None,
    )


def _canonicalize_nominal_output_grid(
    history: StateHistory,
    *,
    expected_step_seconds: float,
    expected_duration_seconds: float,
    tolerance_seconds: float = 1.0e-6,
) -> tuple[StateHistory, dict[str, float]]:
    """Snap sub-microsecond GMAT epoch noise to the requested fixed output grid.

    GMAT STK-TimePosVel files can represent nominally identical output epochs
    with tiny floating-point differences between independently generated files.
    States are accepted only when every timestamp is already within the stated
    tolerance of the configured grid. The state vectors are not interpolated or
    otherwise modified.
    """
    step = float(expected_step_seconds)
    duration = float(expected_duration_seconds)
    tolerance = float(tolerance_seconds)
    if step <= 0.0 or duration <= 0.0 or tolerance <= 0.0:
        raise ValueError("Grid step, duration and tolerance must be positive.")

    times = np.asarray(history.elapsed_seconds, dtype=float)
    expected_count_float = duration / step
    expected_count_rounded = round(expected_count_float)
    if abs(expected_count_float - expected_count_rounded) > 1.0e-12:
        raise ValueError(
            "Configured duration must be an integer multiple of the output step."
        )
    expected_count = int(expected_count_rounded) + 1
    if times.size != expected_count:
        raise ValueError(
            f"{history.model_name} contains {times.size} states; "
            f"expected {expected_count} for duration {duration} s and step {step} s."
        )

    canonical = np.arange(expected_count, dtype=float) * step
    residual = times - canonical
    maximum_absolute_residual = float(np.max(np.abs(residual)))
    final_time_residual = float(times[-1] - duration)
    if maximum_absolute_residual > tolerance:
        raise ValueError(
            f"{history.model_name} output epochs differ from the configured "
            f"{step:g}-second grid by as much as "
            f"{maximum_absolute_residual:.12g} s, exceeding the "
            f"{tolerance:.12g} s synchronization tolerance."
        )

    synchronized = StateHistory(
        model_name=history.model_name,
        frame=history.frame,
        epoch_utc=history.epoch_utc,
        elapsed_seconds=canonical,
        timestamps_utc=timestamps_from_epoch(history.epoch_utc, canonical),
        positions_km=history.positions_km,
        velocities_km_s=history.velocities_km_s,
        runtime_seconds=history.runtime_seconds,
        solver_status=(
            f"{history.solver_status}; nominal output grid canonicalized "
            f"(max raw residual {maximum_absolute_residual:.12g} s)"
        ),
        function_evaluations=history.function_evaluations,
    )
    diagnostics = {
        "maximum_absolute_raw_time_residual_seconds": maximum_absolute_residual,
        "final_raw_time_residual_seconds": final_time_residual,
        "synchronization_tolerance_seconds": tolerance,
        "configured_step_seconds": step,
        "configured_duration_seconds": duration,
    }
    return synchronized, diagnostics


def _maximum_step_deviation_seconds(times: np.ndarray, expected_step: float) -> float:
    differences = np.diff(times)
    return float(np.max(np.abs(differences - expected_step)))


def _initial_difference(
    expected: CartesianState,
    reference: StateHistory,
) -> tuple[float, float]:
    position_m = float(np.linalg.norm(reference.positions_km[0] - expected.position_km) * 1000.0)
    velocity_mm_s = float(
        np.linalg.norm(reference.velocities_km_s[0] - expected.velocity_km_s) * 1e6
    )
    return position_m, velocity_mm_s


def _status_check(
    check_id: str,
    name: str,
    value: float | bool,
    criterion: str,
    passed: bool,
) -> dict[str, Any]:
    return {
        "validation_id": check_id,
        "name": name,
        "measured_value": value,
        "criterion": criterion,
        "status": "passed" if passed else "failed",
    }


def _write_summary_csv(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    fields = [
        "model",
        "final_position_difference_m",
        "maximum_position_difference_m",
        "rms_position_difference_m",
        "final_velocity_difference_mm_s",
        "maximum_velocity_difference_mm_s",
        "rms_velocity_difference_mm_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model, summary in summaries.items():
            writer.writerow(
                {
                    "model": model,
                    "final_position_difference_m": summary["position_difference_m"]["final"],
                    "maximum_position_difference_m": summary["position_difference_m"]["maximum_absolute"],
                    "rms_position_difference_m": summary["position_difference_m"]["rms"],
                    "final_velocity_difference_mm_s": summary["velocity_difference_mm_s"]["final"],
                    "maximum_velocity_difference_mm_s": summary["velocity_difference_mm_s"]["maximum_absolute"],
                    "rms_velocity_difference_mm_s": summary["velocity_difference_mm_s"]["rms"],
                }
            )


def _save_figures(
    directory: Path,
    comparisons: dict[str, dict[str, Any]],
    rtn: dict[str, dict[str, Any]],
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    fig, axis = plt.subplots(figsize=(10, 5.5))
    for model, data in comparisons.items():
        axis.plot(
            np.asarray(data["elapsed_seconds"]) / 3600.0,
            np.asarray(data["position_difference_m"]),
            label=model,
        )
    axis.set_xlabel("Elapsed time (hours)")
    axis.set_ylabel("Position difference (m)")
    axis.set_title("Python versus GMAT position difference")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = directory / f"gmat_position_difference.{suffix}"
        fig.savefig(path, dpi=180)
        created.append(path)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5.5))
    for model, data in comparisons.items():
        axis.plot(
            np.asarray(data["elapsed_seconds"]) / 3600.0,
            np.asarray(data["velocity_difference_mm_s"]),
            label=model,
        )
    axis.set_xlabel("Elapsed time (hours)")
    axis.set_ylabel("Velocity difference (mm/s)")
    axis.set_title("Python versus GMAT velocity difference")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = directory / f"gmat_velocity_difference.{suffix}"
        fig.savefig(path, dpi=180)
        created.append(path)
    plt.close(fig)

    for model, data in rtn.items():
        fig, axis = plt.subplots(figsize=(10, 5.5))
        hours = np.asarray(data["elapsed_seconds"]) / 3600.0
        axis.plot(hours, np.asarray(data["radial_position_difference_m"]), label="Radial")
        axis.plot(hours, np.asarray(data["along_track_position_difference_m"]), label="Along-track")
        axis.plot(hours, np.asarray(data["cross_track_position_difference_m"]), label="Cross-track")
        axis.set_xlabel("Elapsed time (hours)")
        axis.set_ylabel("RTN position difference (m)")
        axis.set_title(f"Python versus GMAT RTN difference — {model}")
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()
        safe = model.replace(" ", "_")
        for suffix in ("png", "pdf"):
            path = directory / f"gmat_{safe}_rtn_difference.{suffix}"
            fig.savefig(path, dpi=180)
            created.append(path)
        plt.close(fig)

    return created


def _report_html(
    *,
    experiment_id: str,
    config: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
    validation: dict[str, Any],
    warnings: list[str],
) -> str:
    rows = []
    for model, summary in summaries.items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(model)}</td>"
            f"<td>{summary['position_difference_m']['maximum_absolute']:.9g}</td>"
            f"<td>{summary['position_difference_m']['final']:.9g}</td>"
            f"<td>{summary['velocity_difference_mm_s']['maximum_absolute']:.9g}</td>"
            f"<td>{summary['velocity_difference_mm_s']['final']:.9g}</td>"
            "</tr>"
        )
    checks = "".join(
        "<tr>"
        f"<td>{html.escape(str(item['validation_id']))}</td>"
        f"<td>{html.escape(str(item['name']))}</td>"
        f"<td>{html.escape(str(item['measured_value']))}</td>"
        f"<td>{html.escape(str(item['criterion']))}</td>"
        f"<td>{html.escape(str(item['status']))}</td>"
        "</tr>"
        for item in validation["checks"]
    )
    warning_items = "".join(f"<li>{html.escape(item)}</li>" for item in warnings)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GMAT External Validation — {html.escape(experiment_id)}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 2rem auto; max-width: 1100px; line-height: 1.5; color: #1f2937; }}
h1, h2 {{ color: #111827; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
th, td {{ border: 1px solid #cbd5e1; padding: .55rem; text-align: left; }}
th {{ background: #f1f5f9; }}
.badge {{ display: inline-block; padding: .25rem .6rem; border-radius: 999px; background: #e2e8f0; }}
.warning {{ background: #fff7ed; border-left: 4px solid #f97316; padding: 1rem; }}
img {{ max-width: 100%; height: auto; margin: .7rem 0 1.5rem; }}
code {{ background: #f1f5f9; padding: .1rem .3rem; }}
</style>
</head>
<body>
<h1>GMAT External Validation Report</h1>
<p><strong>Experiment:</strong> {html.escape(experiment_id)}</p>
<p><strong>Status:</strong> <span class="badge">{html.escape(validation['overall_status'])}</span></p>
<p>This report compares the Research Core numerical point-mass, fixed-axis J2, and pole-aware J2 models against independently generated GMAT R2026a ephemerides. The fixed-axis result is diagnostic-only. The pole-aware result tests the GMAT-matched claim. This is a controlled implementation comparison, not measured-orbit truth.</p>

<h2>Matched setup</h2>
<ul>
<li>Frame: {html.escape(str(config['external_validation']['frame']))}</li>
<li>Gravity file: {html.escape(str(config['external_validation']['gravity_file']))}</li>
<li>Duration: {config['external_validation']['duration_seconds']} s</li>
<li>Output step: {config['external_validation']['output_step_seconds']} s</li>
<li>Python integrator: {html.escape(str(config['integrator']['method']))}</li>
<li>GMAT integrator: {html.escape(str(config['external_validation']['gmat_integrator']))}</li>
</ul>

<h2>Comparison summary</h2>
<table>
<thead><tr><th>Model</th><th>Max position (m)</th><th>Final position (m)</th><th>Max velocity (mm/s)</th><th>Final velocity (mm/s)</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>

<h2>Figures</h2>
<img src="figures/gmat_position_difference.png" alt="Python versus GMAT position difference">
<img src="figures/gmat_velocity_difference.png" alt="Python versus GMAT velocity difference">
<img src="figures/gmat_two_body_rtn_difference.png" alt="Two-body RTN differences">
<img src="figures/gmat_j2_fixed_axis_rtn_difference.png" alt="Fixed-axis J2 RTN differences">
<img src="figures/gmat_j2_gmat_matched_rtn_difference.png" alt="Pole-aware J2 RTN differences">

<h2>Validation checks</h2>
<table>
<thead><tr><th>ID</th><th>Check</th><th>Measured</th><th>Criterion</th><th>Status</th></tr></thead>
<tbody>{checks}</tbody>
</table>

<div class="warning">
<h2>Warnings and interpretation</h2>
<ul>{warning_items}</ul>
</div>

<h2>Scientific meaning</h2>
<p>A small pole-aware difference supports the independent Python implementation under the matched assumptions. The fixed-axis difference measures the effect of the simplified textbook symmetry-axis assumption. Remaining differences must be investigated through acceleration, frame, Earth-orientation, time-system, Earth-constant, integrator, and output-interpolation checks. Thresholds must not be relaxed solely to produce a passing result.</p>
</body>
</html>
"""


def _create_manifest(directory: Path) -> dict[str, Any]:
    files = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == "RUN_MANIFEST.json":
            continue
        files.append(
            {
                "relative_path": path.relative_to(directory).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {"file_count": len(files), "files": files}


def run_gmat_external_validation(
    config_path: str | Path,
    two_body_ephemeris: str | Path,
    j2_ephemeris: str | Path,
    *,
    project_root: str | Path,
) -> ExternalValidationResult:
    config_file = Path(config_path).resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    experiment_id = str(config["experiment"]["experiment_id"])
    root = Path(project_root).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")
    result_directory = root / "results" / experiment_id / stamp
    result_directory.mkdir(parents=True, exist_ok=False)
    created: list[Path] = []

    frame = str(config["external_validation"]["frame"])
    gmat_two_body = parse_stk_time_pos_vel(
        two_body_ephemeris, model_name="gmat_two_body", frame=frame
    )
    gmat_j2 = parse_stk_time_pos_vel(
        j2_ephemeris, model_name="gmat_j2", frame=frame
    )

    if gmat_two_body.epoch_utc != gmat_j2.epoch_utc:
        raise ValueError("GMAT two-body and J2 files use different scenario epochs.")

    expected_step = float(config["external_validation"]["output_step_seconds"])
    duration = float(config["external_validation"]["duration_seconds"])
    synchronization_tolerance = 1.0e-6

    gmat_two_body, two_body_grid_diagnostics = _canonicalize_nominal_output_grid(
        gmat_two_body,
        expected_step_seconds=expected_step,
        expected_duration_seconds=duration,
        tolerance_seconds=synchronization_tolerance,
    )
    gmat_j2, j2_grid_diagnostics = _canonicalize_nominal_output_grid(
        gmat_j2,
        expected_step_seconds=expected_step,
        expected_duration_seconds=duration,
        tolerance_seconds=synchronization_tolerance,
    )

    if not np.array_equal(
        gmat_two_body.elapsed_seconds,
        gmat_j2.elapsed_seconds,
    ):
        raise RuntimeError(
            "Internal error: canonical GMAT output grids are not identical."
        )

    expected_state = initial_state_from_config(config)
    if gmat_two_body.epoch_utc != expected_state.epoch_utc:
        raise ValueError(
            f"GMAT ScenarioEpoch {gmat_two_body.epoch_utc} does not match "
            f"configuration epoch {expected_state.epoch_utc}."
        )

    times = gmat_two_body.elapsed_seconds
    mu = float(config["earth_model"]["gravitational_parameter_km3_s2"])
    radius = float(config["earth_model"]["equatorial_radius_km"])
    j2 = float(config["earth_model"]["j2"])
    integ = config["integrator"]

    python_two_body = propagate_numerical_two_body(
        expected_state,
        mu,
        times,
        method=str(integ["method"]),
        relative_tolerance=float(integ["relative_tolerance"]),
        absolute_tolerance=float(integ["absolute_tolerance"]),
        maximum_step_seconds=float(integ["maximum_step_seconds"]),
    )
    python_j2 = propagate_numerical_j2(
        expected_state,
        mu,
        radius,
        j2,
        times,
        method=str(integ["method"]),
        relative_tolerance=float(integ["relative_tolerance"]),
        absolute_tolerance=float(integ["absolute_tolerance"]),
        maximum_step_seconds=float(integ["maximum_step_seconds"]),
    )
    python_j2_gmat_matched = propagate_numerical_j2_gmat_matched(
        expected_state,
        mu,
        radius,
        j2,
        times,
        method=str(integ["method"]),
        relative_tolerance=float(integ["relative_tolerance"]),
        absolute_tolerance=float(integ["absolute_tolerance"]),
        maximum_step_seconds=float(integ["maximum_step_seconds"]),
    )

    comparisons = {
        "two_body": compare_state_histories(gmat_two_body, python_two_body),
        "j2_fixed_axis": compare_state_histories(gmat_j2, python_j2),
        "j2_gmat_matched": compare_state_histories(
            gmat_j2,
            python_j2_gmat_matched,
        ),
    }
    rtn = {
        "two_body": compare_in_reference_rtn(gmat_two_body, python_two_body),
        "j2_fixed_axis": compare_in_reference_rtn(gmat_j2, python_j2),
        "j2_gmat_matched": compare_in_reference_rtn(
            gmat_j2,
            python_j2_gmat_matched,
        ),
    }
    summaries = {name: create_error_summary(data) for name, data in comparisons.items()}

    histories = {
        "gmat_two_body_states.csv": gmat_two_body,
        "python_two_body_states.csv": python_two_body,
        "gmat_j2_states.csv": gmat_j2,
        "python_j2_fixed_axis_states.csv": python_j2,
        "python_j2_gmat_matched_states.csv": python_j2_gmat_matched,
    }
    for filename, history in histories.items():
        path = result_directory / filename
        write_state_history_csv(path, history)
        created.append(path)

    for name, data in comparisons.items():
        path = result_directory / f"python_vs_gmat_{name}_cartesian.csv"
        write_comparison_csv(path, data)
        created.append(path)
    for name, data in rtn.items():
        path = result_directory / f"python_vs_gmat_{name}_rtn.csv"
        write_rtn_comparison_csv(path, data)
        created.append(path)

    # Preserve the 1A.8.1 filenames as explicit fixed-axis diagnostic aliases.
    legacy_cartesian_path = result_directory / "python_vs_gmat_j2_cartesian.csv"
    write_comparison_csv(legacy_cartesian_path, comparisons["j2_fixed_axis"])
    created.append(legacy_cartesian_path)
    legacy_rtn_path = result_directory / "python_vs_gmat_j2_rtn.csv"
    write_rtn_comparison_csv(legacy_rtn_path, rtn["j2_fixed_axis"])
    created.append(legacy_rtn_path)

    summary_csv = result_directory / "external_validation_summary.csv"
    _write_summary_csv(summary_csv, summaries)
    created.append(summary_csv)

    tb_initial_pos, tb_initial_vel = _initial_difference(expected_state, gmat_two_body)
    j2_initial_pos, j2_initial_vel = _initial_difference(expected_state, gmat_j2)
    thresholds = config["external_validation"]["thresholds"]

    checks = [
        _status_check(
            "VAL-GMAT-001",
            "GMAT two-body initial position matches generated Cartesian state",
            tb_initial_pos,
            f"<= {thresholds['initial_position_difference_m']} m",
            tb_initial_pos <= float(thresholds["initial_position_difference_m"]),
        ),
        _status_check(
            "VAL-GMAT-002",
            "GMAT two-body initial velocity matches generated Cartesian state",
            tb_initial_vel,
            f"<= {thresholds['initial_velocity_difference_mm_s']} mm/s",
            tb_initial_vel <= float(thresholds["initial_velocity_difference_mm_s"]),
        ),
        _status_check(
            "VAL-GMAT-003",
            "GMAT J2 initial position matches generated Cartesian state",
            j2_initial_pos,
            f"<= {thresholds['initial_position_difference_m']} m",
            j2_initial_pos <= float(thresholds["initial_position_difference_m"]),
        ),
        _status_check(
            "VAL-GMAT-004",
            "GMAT J2 initial velocity matches generated Cartesian state",
            j2_initial_vel,
            f"<= {thresholds['initial_velocity_difference_mm_s']} mm/s",
            j2_initial_vel <= float(thresholds["initial_velocity_difference_mm_s"]),
        ),
        _status_check(
            "VAL-GMAT-005",
            "GMAT output duration matches requested duration",
            float(times[-1]),
            f"abs(final-{duration}) <= 1e-6 s",
            abs(float(times[-1]) - duration) <= 1e-6,
        ),
        _status_check(
            "VAL-GMAT-006",
            "GMAT output grid matches requested step",
            _maximum_step_deviation_seconds(times, expected_step),
            "maximum step deviation <= 1e-6 s",
            _maximum_step_deviation_seconds(times, expected_step) <= 1e-6,
        ),
        _status_check(
            "VAL-GMAT-007",
            "GMAT two-body raw epochs lie on the configured nominal grid",
            two_body_grid_diagnostics[
                "maximum_absolute_raw_time_residual_seconds"
            ],
            f"<= {synchronization_tolerance} s",
            two_body_grid_diagnostics[
                "maximum_absolute_raw_time_residual_seconds"
            ]
            <= synchronization_tolerance,
        ),
        _status_check(
            "VAL-GMAT-008",
            "GMAT J2 raw epochs lie on the configured nominal grid",
            j2_grid_diagnostics[
                "maximum_absolute_raw_time_residual_seconds"
            ],
            f"<= {synchronization_tolerance} s",
            j2_grid_diagnostics[
                "maximum_absolute_raw_time_residual_seconds"
            ]
            <= synchronization_tolerance,
        ),
        _status_check(
            "VAL-GMAT-TB-P",
            "Python point-mass position agrees with GMAT",
            float(summaries["two_body"]["position_difference_m"]["maximum_absolute"]),
            f"<= {thresholds['two_body_maximum_position_difference_m']} m",
            float(summaries["two_body"]["position_difference_m"]["maximum_absolute"])
            <= float(thresholds["two_body_maximum_position_difference_m"]),
        ),
        _status_check(
            "VAL-GMAT-TB-V",
            "Python point-mass velocity agrees with GMAT",
            float(summaries["two_body"]["velocity_difference_mm_s"]["maximum_absolute"]),
            f"<= {thresholds['two_body_maximum_velocity_difference_mm_s']} mm/s",
            float(summaries["two_body"]["velocity_difference_mm_s"]["maximum_absolute"])
            <= float(thresholds["two_body_maximum_velocity_difference_mm_s"]),
        ),
        _status_check(
            "VAL-GMAT-J2-MATCHED-P",
            "Python pole-aware J2 position agrees with GMAT",
            float(summaries["j2_gmat_matched"]["position_difference_m"]["maximum_absolute"]),
            f"<= {thresholds['j2_maximum_position_difference_m']} m",
            float(summaries["j2_gmat_matched"]["position_difference_m"]["maximum_absolute"])
            <= float(thresholds["j2_maximum_position_difference_m"]),
        ),
        _status_check(
            "VAL-GMAT-J2-MATCHED-V",
            "Python pole-aware J2 velocity agrees with GMAT",
            float(summaries["j2_gmat_matched"]["velocity_difference_mm_s"]["maximum_absolute"]),
            f"<= {thresholds['j2_maximum_velocity_difference_mm_s']} mm/s",
            float(summaries["j2_gmat_matched"]["velocity_difference_mm_s"]["maximum_absolute"])
            <= float(thresholds["j2_maximum_velocity_difference_mm_s"]),
        ),
    ]
    failed = [item for item in checks if item["status"] == "failed"]
    overall = "passed_with_warnings" if not failed else "failed_validation"
    warnings = [
        "The fixed-axis J2 comparison is diagnostic-only and is not used to pass the GMAT-matched validation claim.",
        "The pole-aware model uses ERFA IAU-1976 precession and IAU-1980 nutation evaluated in TT.",
        "Polar motion and the exact GMAT EOP realization remain outside the current pole-aware model.",
        "The acceptance thresholds remain provisional diagnostic limits.",
        "Sub-microsecond textual epoch residuals are snapped to the configured nominal grid only after a strict tolerance check; state vectors are not interpolated by Research Core.",
        "GMAT STK-TimePosVel output is interpolated to fixed 60-second epochs.",
        "This is an independent implementation comparison, not measured-orbit truth.",
        "A failed check must be investigated; thresholds must not be loosened merely to obtain a pass.",
    ]
    validation = {
        "overall_status": overall,
        "threshold_status": config["external_validation"]["threshold_status"],
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "warnings": warnings,
    }
    validation_path = result_directory / "external_validation_status.json"
    write_json(validation, validation_path)
    created.append(validation_path)

    summary_payload = {
        "experiment_id": experiment_id,
        "tool": "GMAT",
        "tool_version": config["external_validation"]["tool_version"],
        "frame": frame,
        "epoch_utc": expected_state.epoch_utc,
        "duration_seconds": float(times[-1]),
        "state_count": int(times.size),
        "earth_model": config["earth_model"],
        "models": summaries,
        "model_roles": {
            "two_body": "external_validation",
            "j2_fixed_axis": "diagnostic_only_textbook_assumption",
            "j2_gmat_matched": "external_validation",
        },
        "initial_state_checks": {
            "two_body_position_difference_m": tb_initial_pos,
            "two_body_velocity_difference_mm_s": tb_initial_vel,
            "j2_position_difference_m": j2_initial_pos,
            "j2_velocity_difference_mm_s": j2_initial_vel,
        },
        "time_grid_synchronization": {
            "canonical_grid_used_for_comparison": True,
            "two_body_raw_grid": two_body_grid_diagnostics,
            "j2_raw_grid": j2_grid_diagnostics,
        },
        "source_files": {
            "configuration": _recorded_source_path(config_file, project_root),
            "gmat_two_body": _recorded_source_path(
                two_body_ephemeris,
                project_root,
            ),
            "gmat_j2": _recorded_source_path(j2_ephemeris, project_root),
            "configuration_sha256": _sha256(config_file),
            "gmat_two_body_sha256": _sha256(Path(two_body_ephemeris).resolve()),
            "gmat_j2_sha256": _sha256(Path(j2_ephemeris).resolve()),
        },
        "validation_status": overall,
    }
    summary_json = result_directory / "external_validation_summary.json"
    write_json(summary_payload, summary_json)
    created.append(summary_json)

    figure_files = _save_figures(result_directory / "figures", comparisons, rtn)
    created.extend(figure_files)

    report_path = result_directory / "GMAT_VALIDATION_REPORT.html"
    report_path.write_text(
        _report_html(
            experiment_id=experiment_id,
            config=config,
            summaries=summaries,
            validation=validation,
            warnings=warnings,
        ),
        encoding="utf-8",
        newline="\n",
    )
    created.append(report_path)

    run_log = result_directory / "run_log.txt"
    run_log.write_text(
        "\n".join(
            [
                "Research Core 1C.0 GMAT External Validation",
                f"Experiment ID: {experiment_id}",
                f"Status: {overall}",
                f"GMAT two-body file: {_recorded_source_path(two_body_ephemeris, project_root)}",
                f"GMAT J2 file: {_recorded_source_path(j2_ephemeris, project_root)}",
                f"Two-body maximum position difference (m): {summaries['two_body']['position_difference_m']['maximum_absolute']:.12g}",
                f"Fixed-axis J2 maximum position difference (m): {summaries['j2_fixed_axis']['position_difference_m']['maximum_absolute']:.12g}",
                f"Pole-aware J2 maximum position difference (m): {summaries['j2_gmat_matched']['position_difference_m']['maximum_absolute']:.12g}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    created.append(run_log)

    manifest_path = result_directory / "RUN_MANIFEST.json"
    write_json(_create_manifest(result_directory), manifest_path)
    created.append(manifest_path)

    return ExternalValidationResult(
        experiment_id=experiment_id,
        result_directory=result_directory,
        validation_status=overall,
        two_body_maximum_position_difference_m=float(
            summaries["two_body"]["position_difference_m"]["maximum_absolute"]
        ),
        two_body_maximum_velocity_difference_mm_s=float(
            summaries["two_body"]["velocity_difference_mm_s"]["maximum_absolute"]
        ),
        j2_maximum_position_difference_m=float(
            summaries["j2_fixed_axis"]["position_difference_m"]["maximum_absolute"]
        ),
        j2_maximum_velocity_difference_mm_s=float(
            summaries["j2_fixed_axis"]["velocity_difference_mm_s"]["maximum_absolute"]
        ),
        j2_gmat_matched_maximum_position_difference_m=float(
            summaries["j2_gmat_matched"]["position_difference_m"]["maximum_absolute"]
        ),
        j2_gmat_matched_maximum_velocity_difference_mm_s=float(
            summaries["j2_gmat_matched"]["velocity_difference_mm_s"]["maximum_absolute"]
        ),
        two_body_final_position_difference_m=float(
            summaries["two_body"]["position_difference_m"]["final"]
        ),
        j2_final_position_difference_m=float(
            summaries["j2_fixed_axis"]["position_difference_m"]["final"]
        ),
        j2_gmat_matched_final_position_difference_m=float(
            summaries["j2_gmat_matched"]["position_difference_m"]["final"]
        ),
        warnings=tuple(warnings),
        created_files=tuple(created),
        report_path=report_path,
    )


def _j2_arc_report_html(
    *,
    experiment_id: str,
    status: str,
    duration_seconds: float,
    output_step_seconds: float,
    summaries: dict[str, dict[str, Any]],
    checks: list[dict[str, Any]],
    warnings: list[str],
) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{summary['position_difference_m']['maximum_absolute']:.12g}</td>"
        f"<td>{summary['velocity_difference_mm_s']['maximum_absolute']:.12g}</td>"
        "</tr>"
        for name, summary in summaries.items()
    )
    check_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item['validation_id']))}</td>"
        f"<td>{html.escape(str(item['name']))}</td>"
        f"<td>{html.escape(str(item['measured_value']))}</td>"
        f"<td>{html.escape(str(item['criterion']))}</td>"
        f"<td>{html.escape(str(item['status']))}</td>"
        "</tr>"
        for item in checks
    )
    warning_items = "".join(f"<li>{html.escape(item)}</li>" for item in warnings)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>GMAT J2 Short-Arc Validation</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 2rem auto; max-width: 1050px; line-height: 1.5; color: #1f2937; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
th, td {{ border: 1px solid #cbd5e1; padding: .55rem; text-align: left; }}
th {{ background: #f1f5f9; }} img {{ max-width: 100%; height: auto; }}
.warning {{ background: #fff7ed; border-left: 4px solid #f97316; padding: 1rem; }}
</style></head><body>
<h1>GMAT J2 Short-Arc Validation</h1>
<p><strong>Experiment:</strong> {html.escape(experiment_id)}</p>
<p><strong>Status:</strong> {html.escape(status)}</p>
<p>Duration: {duration_seconds:g} s. Output step: {output_step_seconds:g} s.</p>
<table><thead><tr><th>Python model</th><th>Maximum position difference (m)</th><th>Maximum velocity difference (mm/s)</th></tr></thead><tbody>{rows}</tbody></table>
<img src="figures/gmat_j2_fixed_axis_rtn_difference.png" alt="Fixed-axis short-arc RTN difference">
<img src="figures/gmat_j2_gmat_matched_rtn_difference.png" alt="Pole-aware short-arc RTN difference">
<h2>Validation checks</h2>
<table><thead><tr><th>ID</th><th>Check</th><th>Measured</th><th>Criterion</th><th>Status</th></tr></thead><tbody>{check_rows}</tbody></table>
<div class="warning"><h2>Assumptions and limitations</h2><ul>{warning_items}</ul></div>
</body></html>
"""


def run_gmat_j2_short_arc_validation(
    config_path: str | Path,
    j2_ephemeris: str | Path,
    *,
    project_root: str | Path,
) -> J2ArcValidationResult:
    """Run the 10-minute, one-second J2 validation stage."""
    config_file = Path(config_path).resolve()
    source_file = Path(j2_ephemeris).resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    short = config["external_validation"]["short_arc"]
    duration = float(short["duration_seconds"])
    output_step = float(short["output_step_seconds"])
    frame = str(config["external_validation"]["frame"])
    gmat_j2 = parse_stk_time_pos_vel(
        source_file,
        model_name="gmat_j2_short_arc",
        frame=frame,
    )
    gmat_j2, grid_diagnostics = _canonicalize_nominal_output_grid(
        gmat_j2,
        expected_step_seconds=output_step,
        expected_duration_seconds=duration,
        tolerance_seconds=1.0e-6,
    )
    expected_state = initial_state_from_config(config)
    if gmat_j2.epoch_utc != expected_state.epoch_utc:
        raise ValueError(
            f"GMAT ScenarioEpoch {gmat_j2.epoch_utc} does not match "
            f"configuration epoch {expected_state.epoch_utc}."
        )
    times = gmat_j2.elapsed_seconds
    earth = config["earth_model"]
    integ = config["integrator"]
    mu = float(earth["gravitational_parameter_km3_s2"])
    radius = float(earth["equatorial_radius_km"])
    j2 = float(earth["j2"])
    fixed = propagate_numerical_j2(
        expected_state,
        mu,
        radius,
        j2,
        times,
        method=str(integ["method"]),
        relative_tolerance=float(integ["relative_tolerance"]),
        absolute_tolerance=float(integ["absolute_tolerance"]),
        maximum_step_seconds=min(
            float(integ["maximum_step_seconds"]),
            output_step,
        ),
    )
    matched = propagate_numerical_j2_gmat_matched(
        expected_state,
        mu,
        radius,
        j2,
        times,
        method=str(integ["method"]),
        relative_tolerance=float(integ["relative_tolerance"]),
        absolute_tolerance=float(integ["absolute_tolerance"]),
        maximum_step_seconds=min(
            float(integ["maximum_step_seconds"]),
            output_step,
        ),
    )
    comparisons = {
        "j2_fixed_axis": compare_state_histories(gmat_j2, fixed),
        "j2_gmat_matched": compare_state_histories(gmat_j2, matched),
    }
    rtn = {
        "j2_fixed_axis": compare_in_reference_rtn(gmat_j2, fixed),
        "j2_gmat_matched": compare_in_reference_rtn(gmat_j2, matched),
    }
    summaries = {
        name: create_error_summary(data) for name, data in comparisons.items()
    }
    thresholds = short["thresholds"]
    initial_position, initial_velocity = _initial_difference(
        expected_state,
        gmat_j2,
    )
    checks = [
        _status_check(
            "VAL-GMAT-SHORT-001",
            "GMAT short arc starts from the generated Cartesian position",
            initial_position,
            "<= 0.001 m",
            initial_position <= 0.001,
        ),
        _status_check(
            "VAL-GMAT-SHORT-002",
            "GMAT short arc starts from the generated Cartesian velocity",
            initial_velocity,
            "<= 0.001 mm/s",
            initial_velocity <= 0.001,
        ),
        _status_check(
            "VAL-GMAT-SHORT-003",
            "Raw GMAT epochs lie on the one-second grid",
            grid_diagnostics["maximum_absolute_raw_time_residual_seconds"],
            "<= 1e-6 s",
            grid_diagnostics["maximum_absolute_raw_time_residual_seconds"]
            <= 1.0e-6,
        ),
        _status_check(
            "VAL-GMAT-SHORT-J2-P",
            "Pole-aware J2 short-arc position agrees with GMAT",
            summaries["j2_gmat_matched"]["position_difference_m"][
                "maximum_absolute"
            ],
            f"<= {thresholds['gmat_matched_maximum_position_difference_m']} m",
            summaries["j2_gmat_matched"]["position_difference_m"][
                "maximum_absolute"
            ]
            <= float(thresholds["gmat_matched_maximum_position_difference_m"]),
        ),
        _status_check(
            "VAL-GMAT-SHORT-J2-V",
            "Pole-aware J2 short-arc velocity agrees with GMAT",
            summaries["j2_gmat_matched"]["velocity_difference_mm_s"][
                "maximum_absolute"
            ],
            f"<= {thresholds['gmat_matched_maximum_velocity_difference_mm_s']} mm/s",
            summaries["j2_gmat_matched"]["velocity_difference_mm_s"][
                "maximum_absolute"
            ]
            <= float(
                thresholds["gmat_matched_maximum_velocity_difference_mm_s"]
            ),
        ),
    ]
    failed = [item for item in checks if item["status"] == "failed"]
    overall = "passed_with_warnings" if not failed else "failed_validation"
    warnings = [
        "The fixed-axis result is diagnostic-only and does not control the short-arc validation status.",
        "The pole-aware model uses ERFA IAU-1976 precession and IAU-1980 nutation in TT.",
        "Polar motion and the exact GMAT EOP realization remain unresolved.",
        "The short-arc thresholds remain provisional until the multi-case validation matrix is complete.",
        "This is an implementation comparison, not measured orbit truth.",
    ]
    experiment_id = str(config["experiment"]["experiment_id"])
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")
    directory = (
        Path(project_root).resolve()
        / "results"
        / experiment_id
        / f"gmat_j2_short_arc_{stamp}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    created: list[Path] = []
    for filename, history in {
        "gmat_j2_short_arc_states.csv": gmat_j2,
        "python_j2_fixed_axis_short_arc_states.csv": fixed,
        "python_j2_gmat_matched_short_arc_states.csv": matched,
    }.items():
        path = directory / filename
        write_state_history_csv(path, history)
        created.append(path)
    for name, data in comparisons.items():
        path = directory / f"python_vs_gmat_{name}_short_arc_cartesian.csv"
        write_comparison_csv(path, data)
        created.append(path)
    for name, data in rtn.items():
        path = directory / f"python_vs_gmat_{name}_short_arc_rtn.csv"
        write_rtn_comparison_csv(path, data)
        created.append(path)
    summary_path = directory / "short_arc_validation_summary.json"
    write_json(
        {
            "research_core_version": "1C.0",
            "experiment_id": experiment_id,
            "status": overall,
            "duration_seconds": duration,
            "output_step_seconds": output_step,
            "models": summaries,
            "checks": checks,
            "time_grid": grid_diagnostics,
            "source_files": {
                "configuration": _recorded_source_path(
                    config_file,
                    project_root,
                ),
                "configuration_sha256": _sha256(config_file),
                "gmat_j2_short_arc": _recorded_source_path(
                    source_file,
                    project_root,
                ),
                "gmat_j2_short_arc_sha256": _sha256(source_file),
            },
            "assumptions_and_limitations": warnings,
        },
        summary_path,
    )
    created.append(summary_path)
    created.extend(_save_figures(directory / "figures", comparisons, rtn))
    report_path = directory / "GMAT_J2_SHORT_ARC_REPORT.html"
    report_path.write_text(
        _j2_arc_report_html(
            experiment_id=experiment_id,
            status=overall,
            duration_seconds=duration,
            output_step_seconds=output_step,
            summaries=summaries,
            checks=checks,
            warnings=warnings,
        ),
        encoding="utf-8",
        newline="\n",
    )
    created.append(report_path)
    manifest_path = directory / "RUN_MANIFEST.json"
    write_json(_create_manifest(directory), manifest_path)
    created.append(manifest_path)
    return J2ArcValidationResult(
        experiment_id=experiment_id,
        result_directory=directory,
        validation_status=overall,
        fixed_axis_maximum_position_difference_m=float(
            summaries["j2_fixed_axis"]["position_difference_m"][
                "maximum_absolute"
            ]
        ),
        fixed_axis_maximum_velocity_difference_mm_s=float(
            summaries["j2_fixed_axis"]["velocity_difference_mm_s"][
                "maximum_absolute"
            ]
        ),
        gmat_matched_maximum_position_difference_m=float(
            summaries["j2_gmat_matched"]["position_difference_m"][
                "maximum_absolute"
            ]
        ),
        gmat_matched_maximum_velocity_difference_mm_s=float(
            summaries["j2_gmat_matched"]["velocity_difference_mm_s"][
                "maximum_absolute"
            ]
        ),
        warnings=tuple(warnings),
        created_files=tuple(created),
        report_path=report_path,
    )
