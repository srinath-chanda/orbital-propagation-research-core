"""GMAT acceleration diagnostics for fixed-axis and pole-aware J2 models."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data_models import CartesianState
from .orbital_elements import elements_from_config, elements_to_cartesian
from .outputs import write_json
from .propagators.numerical_j2 import (
    central_gravity_acceleration,
    j2_perturbing_acceleration,
    j2_perturbing_acceleration_gmat_matched,
)


@dataclass(frozen=True)
class GmatAccelerationSamples:
    elapsed_seconds: np.ndarray
    positions_km: np.ndarray
    point_mass_accelerations_km_s2: np.ndarray
    degree2_accelerations_km_s2: np.ndarray


@dataclass(frozen=True)
class AccelerationValidationResult:
    result_directory: Path
    validation_status: str
    sample_count: int
    fixed_axis_maximum_difference_km_s2: float
    gmat_matched_maximum_difference_km_s2: float
    gmat_matched_maximum_relative_difference: float
    report_path: Path
    created_files: tuple[Path, ...]
    warnings: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_gmat_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _recorded_source_path(path: Path, project_root: str | Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _gmat_epoch(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%d %b %Y %H:%M:%S.%f")[:-3]


def _initial_state_from_config(config: dict[str, Any]) -> CartesianState:
    elements = elements_from_config(config["initial_state"])
    mu = float(config["earth_model"]["gravitational_parameter_km3_s2"])
    position, velocity = elements_to_cartesian(elements, mu)
    return CartesianState(
        epoch_utc=str(config["initial_state"]["epoch_utc"]),
        frame=str(config["external_validation"]["frame"]),
        position_km=position,
        velocity_km_s=velocity,
    )


def acceleration_sample_times(
    duration_seconds: float,
    sample_count: int,
) -> np.ndarray:
    """Return an inclusive, uniform diagnostic grid containing 20–50 states."""
    duration = float(duration_seconds)
    count = int(sample_count)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("Acceleration diagnostic duration must be positive.")
    if count < 20 or count > 50:
        raise ValueError("Acceleration diagnostic sample_count must be between 20 and 50.")
    return np.linspace(0.0, duration, count, dtype=float)


def build_gmat_acceleration_diagnostic_script(
    config: dict[str, Any],
    *,
    output_report: Path,
) -> str:
    """Build a GMAT script that evaluates degree 0 and degree 2 at shared states."""
    state = _initial_state_from_config(config)
    ext = config["external_validation"]
    diagnostic = ext["acceleration_diagnostic"]
    sample_times = acceleration_sample_times(
        diagnostic.get("duration_seconds", ext["duration_seconds"]),
        int(diagnostic["sample_count"]),
    )
    sample_step = float(sample_times[1] - sample_times[0])
    gravity_file = str(ext["gravity_file"])
    accuracy = float(ext["gmat_accuracy"])
    initial_step = min(float(ext["gmat_initial_step_seconds"]), sample_step)
    maximum_step = min(float(ext["gmat_maximum_step_seconds"]), sample_step)
    fields = " ".join(
        [
            "DiagnosticSat.ElapsedSecs",
            "DiagnosticSat.EarthMJ2000Eq.X",
            "DiagnosticSat.EarthMJ2000Eq.Y",
            "DiagnosticSat.EarthMJ2000Eq.Z",
            "DiagnosticSat.EarthMJ2000Eq.VX",
            "DiagnosticSat.EarthMJ2000Eq.VY",
            "DiagnosticSat.EarthMJ2000Eq.VZ",
            "DiagnosticSat.PointMassFM.AccelerationX",
            "DiagnosticSat.PointMassFM.AccelerationY",
            "DiagnosticSat.PointMassFM.AccelerationZ",
            "DiagnosticSat.Degree2FM.AccelerationX",
            "DiagnosticSat.Degree2FM.AccelerationY",
            "DiagnosticSat.Degree2FM.AccelerationZ",
        ]
    )
    mission_lines = ["BeginMissionSequence;", f"Report AccelerationReport {fields};"]
    for _ in sample_times[1:]:
        mission_lines.append(
            "Propagate Degree2Prop(DiagnosticSat) "
            f"{{DiagnosticSat.ElapsedSecs = {sample_step:.15g}}};"
        )
        mission_lines.append(f"Report AccelerationReport {fields};")

    return f"""%
% CASE-LEO400 shared-state acceleration diagnostic
% Generated by Orbital Propagation Research Core 1C.0
% Target GMAT release: {ext['tool_version']}
%
% Both force models are evaluated at the same DiagnosticSat state.  The
% isolated GMAT J2 term is Degree2FM acceleration minus PointMassFM
% acceleration.  Degree2FM also propagates the state between samples.
%

Create Spacecraft DiagnosticSat;
DiagnosticSat.DateFormat = UTCGregorian;
DiagnosticSat.Epoch = '{_gmat_epoch(state.epoch_utc)}';
DiagnosticSat.CoordinateSystem = EarthMJ2000Eq;
DiagnosticSat.DisplayStateType = Cartesian;
DiagnosticSat.X = {state.position_km[0]:.15f};
DiagnosticSat.Y = {state.position_km[1]:.15f};
DiagnosticSat.Z = {state.position_km[2]:.15f};
DiagnosticSat.VX = {state.velocity_km_s[0]:.15f};
DiagnosticSat.VY = {state.velocity_km_s[1]:.15f};
DiagnosticSat.VZ = {state.velocity_km_s[2]:.15f};
DiagnosticSat.DryMass = 500;
DiagnosticSat.Cd = 2.2;
DiagnosticSat.Cr = 1.0;
DiagnosticSat.DragArea = 4;
DiagnosticSat.SRPArea = 4;

Create ForceModel PointMassFM;
PointMassFM.CentralBody = Earth;
PointMassFM.PrimaryBodies = {{Earth}};
PointMassFM.Drag = None;
PointMassFM.SRP = Off;
PointMassFM.RelativisticCorrection = Off;
PointMassFM.ErrorControl = RSSStep;
PointMassFM.GravityField.Earth.Degree = 0;
PointMassFM.GravityField.Earth.Order = 0;
PointMassFM.GravityField.Earth.PotentialFile = '{gravity_file}';
PointMassFM.GravityField.Earth.TideModel = 'None';

Create ForceModel Degree2FM;
Degree2FM.CentralBody = Earth;
Degree2FM.PrimaryBodies = {{Earth}};
Degree2FM.Drag = None;
Degree2FM.SRP = Off;
Degree2FM.RelativisticCorrection = Off;
Degree2FM.ErrorControl = RSSStep;
Degree2FM.GravityField.Earth.Degree = 2;
Degree2FM.GravityField.Earth.Order = 0;
Degree2FM.GravityField.Earth.PotentialFile = '{gravity_file}';
Degree2FM.GravityField.Earth.TideModel = 'None';

Create Propagator PointMassProp;
PointMassProp.FM = PointMassFM;
PointMassProp.Type = {ext['gmat_integrator']};
PointMassProp.InitialStepSize = {initial_step:.15g};
PointMassProp.Accuracy = {accuracy:.15g};
PointMassProp.MinStep = 1e-6;
PointMassProp.MaxStep = {maximum_step:.15g};
PointMassProp.MaxStepAttempts = 50;
PointMassProp.StopIfAccuracyIsViolated = true;

Create Propagator Degree2Prop;
Degree2Prop.FM = Degree2FM;
Degree2Prop.Type = {ext['gmat_integrator']};
Degree2Prop.InitialStepSize = {initial_step:.15g};
Degree2Prop.Accuracy = {accuracy:.15g};
Degree2Prop.MinStep = 1e-6;
Degree2Prop.MaxStep = {maximum_step:.15g};
Degree2Prop.MaxStepAttempts = 50;
Degree2Prop.StopIfAccuracyIsViolated = true;

Create ReportFile AccelerationReport;
AccelerationReport.Filename = '{_portable_gmat_path(output_report)}';
AccelerationReport.Precision = 16;
AccelerationReport.ColumnWidth = 24;
AccelerationReport.WriteHeaders = true;
AccelerationReport.LeftJustify = Off;
AccelerationReport.ZeroFill = On;
AccelerationReport.FixedWidth = Off;
AccelerationReport.Delimiter = Comma;

{chr(10).join(mission_lines)}
"""


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _column_index(headers: list[str], required_suffix: str) -> int:
    suffix = _normalize_header(required_suffix)
    matches = [
        index
        for index, header in enumerate(headers)
        if _normalize_header(header).endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one GMAT acceleration-report column ending in "
            f"{required_suffix!r}; found {len(matches)}."
        )
    return matches[0]


def _split_report_line(value: str) -> list[str]:
    """Split supported GMAT ReportFile layouts without altering source bytes."""
    stripped = value.strip()
    if not stripped:
        return []
    if "," in stripped:
        return [cell.strip() for cell in next(csv.reader([stripped]))]
    whitespace_cells = re.split(r"\s+", stripped)
    if len(whitespace_cells) > 1:
        return whitespace_cells
    # GMAT R2026a on Windows was observed writing the first character of the
    # `Comma` enumeration between numeric fields.  Accept that raw layout so
    # the independently generated report remains untouched evidence.
    character_cells = stripped.split("C")
    if len(character_cells) > 1:
        return [cell.strip() for cell in character_cells]
    return [stripped]


def parse_gmat_acceleration_report(path: str | Path) -> GmatAccelerationSamples:
    """Parse normalized or raw GMAT R2026a acceleration reports."""
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"GMAT acceleration report not found: {source}")
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    rows = [
        (line_number, cells)
        for line_number, line in enumerate(lines, start=1)
        if (cells := _split_report_line(line))
    ]
    if len(rows) < 2:
        raise ValueError("GMAT acceleration report contains no numeric samples.")
    headers = rows[0][1]
    suffixes = [
        "ElapsedSecs",
        "EarthMJ2000Eq.X",
        "EarthMJ2000Eq.Y",
        "EarthMJ2000Eq.Z",
        "PointMassFM.AccelerationX",
        "PointMassFM.AccelerationY",
        "PointMassFM.AccelerationZ",
        "Degree2FM.AccelerationX",
        "Degree2FM.AccelerationY",
        "Degree2FM.AccelerationZ",
    ]
    indices = [_column_index(headers, suffix) for suffix in suffixes]
    numeric_rows: list[list[float]] = []
    for line_number, row in rows[1:]:
        if _normalize_header(row[0]).endswith("elapsedsecs"):
            # The GMAT Report command writes the header again at each explicit
            # sample.  Repeated headers contain no new scientific data.
            continue
        try:
            numeric_rows.append([float(row[index].strip()) for index in indices])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"Invalid numeric row in GMAT acceleration report at line {line_number}."
            ) from exc
    if not numeric_rows:
        raise ValueError("GMAT acceleration report contains no numeric samples.")
    values = np.asarray(numeric_rows, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("GMAT acceleration report contains non-finite values.")
    elapsed = values[:, 0]
    if abs(float(elapsed[0])) > 1.0e-6 or np.any(np.diff(elapsed) <= 0.0):
        raise ValueError(
            "GMAT acceleration samples must begin at zero and increase strictly."
        )
    return GmatAccelerationSamples(
        elapsed_seconds=elapsed,
        positions_km=values[:, 1:4],
        point_mass_accelerations_km_s2=values[:, 4:7],
        degree2_accelerations_km_s2=values[:, 7:10],
    )


def compare_acceleration_samples(
    samples: GmatAccelerationSamples,
    *,
    epoch_utc: str,
    gravitational_parameter_km3_s2: float,
    earth_equatorial_radius_km: float,
    j2: float,
) -> dict[str, np.ndarray]:
    """Compare isolated GMAT J2 with both Python J2 formulations."""
    gmat_j2 = (
        samples.degree2_accelerations_km_s2
        - samples.point_mass_accelerations_km_s2
    )
    python_point_mass = []
    python_fixed = []
    python_matched = []
    for elapsed, position in zip(samples.elapsed_seconds, samples.positions_km):
        python_point_mass.append(
            central_gravity_acceleration(position, gravitational_parameter_km3_s2)
        )
        python_fixed.append(
            j2_perturbing_acceleration(
                position,
                gravitational_parameter_km3_s2,
                earth_equatorial_radius_km,
                j2,
            )
        )
        python_matched.append(
            j2_perturbing_acceleration_gmat_matched(
                position,
                epoch_utc,
                float(elapsed),
                gravitational_parameter_km3_s2,
                earth_equatorial_radius_km,
                j2,
            )
        )
    point_mass = np.asarray(python_point_mass)
    fixed = np.asarray(python_fixed)
    matched = np.asarray(python_matched)
    fixed_difference = fixed - gmat_j2
    matched_difference = matched - gmat_j2
    point_mass_difference = point_mass - samples.point_mass_accelerations_km_s2
    gmat_magnitude = np.linalg.norm(gmat_j2, axis=1)
    denominator = np.maximum(gmat_magnitude, np.finfo(float).tiny)
    return {
        "elapsed_seconds": samples.elapsed_seconds,
        "positions_km": samples.positions_km,
        "gmat_point_mass_km_s2": samples.point_mass_accelerations_km_s2,
        "python_point_mass_km_s2": point_mass,
        "point_mass_difference_km_s2": point_mass_difference,
        "gmat_isolated_j2_km_s2": gmat_j2,
        "python_fixed_axis_j2_km_s2": fixed,
        "python_gmat_matched_j2_km_s2": matched,
        "fixed_axis_difference_km_s2": fixed_difference,
        "gmat_matched_difference_km_s2": matched_difference,
        "fixed_axis_vector_difference_km_s2": np.linalg.norm(
            fixed_difference, axis=1
        ),
        "gmat_matched_vector_difference_km_s2": np.linalg.norm(
            matched_difference, axis=1
        ),
        "point_mass_vector_difference_km_s2": np.linalg.norm(
            point_mass_difference, axis=1
        ),
        "fixed_axis_relative_difference": np.linalg.norm(
            fixed_difference, axis=1
        )
        / denominator,
        "gmat_matched_relative_difference": np.linalg.norm(
            matched_difference, axis=1
        )
        / denominator,
    }


def _summary(values: dict[str, np.ndarray], prefix: str) -> dict[str, float]:
    vector = np.asarray(values[f"{prefix}_vector_difference_km_s2"])
    component = np.asarray(values[f"{prefix}_difference_km_s2"])
    result = {
        "maximum_vector_difference_km_s2": float(np.max(vector)),
        "rms_vector_difference_km_s2": float(np.sqrt(np.mean(vector**2))),
        "maximum_absolute_component_difference_km_s2": float(
            np.max(np.abs(component))
        ),
    }
    relative_key = f"{prefix}_relative_difference"
    if relative_key in values:
        result["maximum_relative_difference"] = float(
            np.max(values[relative_key])
        )
    return result


def _write_comparison_csv(path: Path, values: dict[str, np.ndarray]) -> None:
    fields = [
        "elapsed_seconds",
        "x_km",
        "y_km",
        "z_km",
        "gmat_j2_ax_km_s2",
        "gmat_j2_ay_km_s2",
        "gmat_j2_az_km_s2",
        "python_fixed_ax_km_s2",
        "python_fixed_ay_km_s2",
        "python_fixed_az_km_s2",
        "python_matched_ax_km_s2",
        "python_matched_ay_km_s2",
        "python_matched_az_km_s2",
        "fixed_difference_ax_km_s2",
        "fixed_difference_ay_km_s2",
        "fixed_difference_az_km_s2",
        "matched_difference_ax_km_s2",
        "matched_difference_ay_km_s2",
        "matched_difference_az_km_s2",
        "fixed_vector_difference_km_s2",
        "matched_vector_difference_km_s2",
        "fixed_relative_difference",
        "matched_relative_difference",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        count = int(values["elapsed_seconds"].size)
        for index in range(count):
            writer.writerow(
                [
                    values["elapsed_seconds"][index],
                    *values["positions_km"][index],
                    *values["gmat_isolated_j2_km_s2"][index],
                    *values["python_fixed_axis_j2_km_s2"][index],
                    *values["python_gmat_matched_j2_km_s2"][index],
                    *values["fixed_axis_difference_km_s2"][index],
                    *values["gmat_matched_difference_km_s2"][index],
                    values["fixed_axis_vector_difference_km_s2"][index],
                    values["gmat_matched_vector_difference_km_s2"][index],
                    values["fixed_axis_relative_difference"][index],
                    values["gmat_matched_relative_difference"][index],
                ]
            )


def _save_figure(directory: Path, values: dict[str, np.ndarray]) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    hours = values["elapsed_seconds"] / 3600.0
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.semilogy(
        hours,
        np.maximum(values["fixed_axis_vector_difference_km_s2"], 1.0e-20),
        label="fixed-axis J2",
    )
    axis.semilogy(
        hours,
        np.maximum(values["gmat_matched_vector_difference_km_s2"], 1.0e-20),
        label="IAU-1976/1980 pole-aware J2",
    )
    axis.set_xlabel("Elapsed time (hours)")
    axis.set_ylabel("J2 acceleration vector difference (km/s²)")
    axis.set_title("Python versus isolated GMAT J2 acceleration")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    created: list[Path] = []
    for suffix in ("png", "pdf"):
        path = directory / f"gmat_j2_acceleration_difference.{suffix}"
        figure.savefig(path, dpi=180)
        created.append(path)
    plt.close(figure)
    return created


def _report_html(
    *,
    experiment_id: str,
    status: str,
    sample_count: int,
    point_mass: dict[str, float],
    fixed: dict[str, float],
    matched: dict[str, float],
    thresholds: dict[str, Any],
    warnings: list[str],
) -> str:
    warning_items = "".join(f"<li>{html.escape(item)}</li>" for item in warnings)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>GMAT J2 Acceleration Diagnostic</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 2rem auto; max-width: 1050px; line-height: 1.5; color: #1f2937; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
th, td {{ border: 1px solid #cbd5e1; padding: .55rem; text-align: left; }}
th {{ background: #f1f5f9; }} img {{ max-width: 100%; height: auto; }}
.warning {{ background: #fff7ed; border-left: 4px solid #f97316; padding: 1rem; }}
</style></head><body>
<h1>GMAT J2 Acceleration Diagnostic</h1>
<p><strong>Experiment:</strong> {html.escape(experiment_id)}</p>
<p><strong>Status:</strong> {html.escape(status)}</p>
<p><strong>Shared-state samples:</strong> {sample_count}</p>
<p>GMAT J2 is isolated as degree-2/order-0 acceleration minus degree-0/order-0 acceleration at one shared spacecraft state.</p>
<table><thead><tr><th>Model</th><th>Maximum vector difference (km/s²)</th><th>Maximum relative difference</th></tr></thead>
<tbody>
<tr><td>Point mass</td><td>{point_mass['maximum_vector_difference_km_s2']:.12g}</td><td>Not applicable</td></tr>
<tr><td>Fixed-axis J2</td><td>{fixed['maximum_vector_difference_km_s2']:.12g}</td><td>{fixed['maximum_relative_difference']:.12g}</td></tr>
<tr><td>IAU-1976/1980 pole-aware J2</td><td>{matched['maximum_vector_difference_km_s2']:.12g}</td><td>{matched['maximum_relative_difference']:.12g}</td></tr>
</tbody></table>
<p>Acceptance limits for the pole-aware model: {thresholds['maximum_vector_difference_km_s2']} km/s² and {thresholds['maximum_relative_difference']} relative difference.</p>
<img src="figures/gmat_j2_acceleration_difference.png" alt="J2 acceleration difference">
<div class="warning"><h2>Assumptions and limitations</h2><ul>{warning_items}</ul></div>
</body></html>
"""


def run_gmat_acceleration_validation(
    config_path: str | Path,
    acceleration_report: str | Path,
    *,
    project_root: str | Path,
) -> AccelerationValidationResult:
    """Import a GMAT acceleration report and validate both Python J2 models."""
    config_file = Path(config_path).resolve()
    source_file = Path(acceleration_report).resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    samples = parse_gmat_acceleration_report(source_file)
    diagnostic = config["external_validation"]["acceleration_diagnostic"]
    requested_count = int(diagnostic["sample_count"])
    if samples.elapsed_seconds.size != requested_count:
        raise ValueError(
            f"GMAT acceleration report contains {samples.elapsed_seconds.size} samples; "
            f"expected {requested_count}."
        )
    expected_duration = float(
        diagnostic.get(
            "duration_seconds",
            config["external_validation"]["duration_seconds"],
        )
    )
    if abs(float(samples.elapsed_seconds[-1]) - expected_duration) > 1.0e-5:
        raise ValueError("GMAT acceleration report duration does not match configuration.")

    earth = config["earth_model"]
    epoch_utc = str(config["initial_state"]["epoch_utc"])
    values = compare_acceleration_samples(
        samples,
        epoch_utc=epoch_utc,
        gravitational_parameter_km3_s2=float(
            earth["gravitational_parameter_km3_s2"]
        ),
        earth_equatorial_radius_km=float(earth["equatorial_radius_km"]),
        j2=float(earth["j2"]),
    )
    point_mass = _summary(values, "point_mass")
    fixed = _summary(values, "fixed_axis")
    matched = _summary(values, "gmat_matched")
    thresholds = diagnostic["thresholds"]
    absolute_pass = matched["maximum_vector_difference_km_s2"] <= float(
        thresholds["maximum_vector_difference_km_s2"]
    )
    relative_pass = matched["maximum_relative_difference"] <= float(
        thresholds["maximum_relative_difference"]
    )
    point_mass_pass = point_mass["maximum_vector_difference_km_s2"] <= float(
        thresholds["point_mass_maximum_vector_difference_km_s2"]
    )
    overall = (
        "passed_with_warnings"
        if absolute_pass and relative_pass and point_mass_pass
        else "failed_validation"
    )

    experiment_id = str(config["experiment"]["experiment_id"])
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")
    directory = (
        Path(project_root).resolve()
        / "results"
        / experiment_id
        / f"gmat_acceleration_{stamp}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    created: list[Path] = []

    csv_path = directory / "python_vs_gmat_j2_acceleration.csv"
    _write_comparison_csv(csv_path, values)
    created.append(csv_path)
    warnings = [
        "The fixed-axis model is retained as a textbook model and is diagnostic-only in this GMAT comparison.",
        "The pole-aware model uses ERFA IAU-1976 precession and IAU-1980 nutation in TT.",
        "Polar motion and sub-daily EOP corrections are not included without the exact GMAT EOP data set.",
        "This comparison checks implementation agreement, not measured orbit truth.",
        "The raw GMAT R2026a report is imported directly; repeated headers and the observed single-character C delimiter are normalized in memory only.",
        "Thresholds remain provisional until the multi-case validation matrix is complete.",
    ]
    summary_payload = {
        "research_core_version": "1C.0",
        "experiment_id": experiment_id,
        "status": overall,
        "sample_count": int(samples.elapsed_seconds.size),
        "epoch_utc": epoch_utc,
        "frame": config["external_validation"]["frame"],
        "point_mass": point_mass,
        "fixed_axis_j2": fixed,
        "gmat_matched_j2": matched,
        "thresholds": thresholds,
        "checks": {
            "point_mass_passed": point_mass_pass,
            "gmat_matched_absolute_passed": absolute_pass,
            "gmat_matched_relative_passed": relative_pass,
        },
        "source_files": {
            "configuration": _recorded_source_path(config_file, project_root),
            "configuration_sha256": _sha256(config_file),
            "gmat_acceleration_report": _recorded_source_path(
                source_file,
                project_root,
            ),
            "gmat_acceleration_report_sha256": _sha256(source_file),
        },
        "assumptions_and_limitations": warnings,
    }
    summary_path = directory / "acceleration_validation_summary.json"
    write_json(summary_payload, summary_path)
    created.append(summary_path)
    created.extend(_save_figure(directory / "figures", values))

    report_path = directory / "GMAT_J2_ACCELERATION_REPORT.html"
    report_path.write_text(
        _report_html(
            experiment_id=experiment_id,
            status=overall,
            sample_count=int(samples.elapsed_seconds.size),
            point_mass=point_mass,
            fixed=fixed,
            matched=matched,
            thresholds=thresholds,
            warnings=warnings,
        ),
        encoding="utf-8",
        newline="\n",
    )
    created.append(report_path)
    manifest_path = directory / "RUN_MANIFEST.json"
    manifest_files = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == "RUN_MANIFEST.json":
            continue
        manifest_files.append(
            {
                "relative_path": path.relative_to(directory).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    write_json(
        {"file_count": len(manifest_files), "files": manifest_files},
        manifest_path,
    )
    created.append(manifest_path)
    return AccelerationValidationResult(
        result_directory=directory,
        validation_status=overall,
        sample_count=int(samples.elapsed_seconds.size),
        fixed_axis_maximum_difference_km_s2=fixed[
            "maximum_vector_difference_km_s2"
        ],
        gmat_matched_maximum_difference_km_s2=matched[
            "maximum_vector_difference_km_s2"
        ],
        gmat_matched_maximum_relative_difference=matched[
            "maximum_relative_difference"
        ],
        report_path=report_path,
        created_files=tuple(created),
        warnings=tuple(warnings),
    )
