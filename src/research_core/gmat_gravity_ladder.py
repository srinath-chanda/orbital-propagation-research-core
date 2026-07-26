"""Research Core 1D.0 higher-degree/order GMAT acceleration ladder."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from astropy.time import Time, TimeDelta

from .external_validation import initial_state_from_config
from .gmat_eop import (
    GMAT_R2026A_EOP_SHA256,
    GmatEopDataset,
    gmat_r2026a_inertial_to_fixed_matrix,
)
from .gravity_harmonics import CofGravityField, gravity_acceleration_inertial_km_s2


SCHEMA_VERSION = "1D.0"
EXPECTED_LADDER = ((0, 0), (2, 0), (4, 0), (4, 4), (8, 8), (20, 20))


@dataclass(frozen=True)
class ImportedGravityFile:
    destination: Path
    provenance: Path
    sha256: str
    maximum_degree: int
    maximum_order: int


@dataclass(frozen=True)
class PreparedGravityLadder:
    experiment_id: str
    master_script: Path
    output_report: Path
    manifest: Path
    sample_count: int
    model_count: int
    archived_outputs: tuple[Path, ...]


@dataclass(frozen=True)
class GravityLadderResult:
    experiment_id: str
    status: str
    decision: str
    sample_count: int
    model_count: int
    maximum_difference_km_s2: float
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


def load_gravity_ladder_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Gravity ladder schema must be {SCHEMA_VERSION!r}.")
    ladder = tuple((int(item["degree"]), int(item["order"])) for item in payload["ladder"])
    if ladder != EXPECTED_LADDER:
        raise ValueError(f"The preregistered ladder must be {EXPECTED_LADDER!r}.")
    if payload.get("threshold_status") != "preregistered_before_first_1d0_gmat_run":
        raise ValueError("The 1D.0 threshold status is not preregistered.")
    sample_count = int(payload["sample_count"])
    duration = float(payload["duration_seconds"])
    if not 20 <= sample_count <= 50 or duration <= 0.0:
        raise ValueError("1D.0 requires 20–50 samples over a positive duration.")
    return payload


def _find_jgm2(gmat_root: Path) -> Path:
    candidates = (
        gmat_root / "data" / "gravity" / "earth" / "JGM2.cof",
        gmat_root / "application" / "data" / "gravity" / "earth" / "JGM2.cof",
    )
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        listed = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(f"Expected one GMAT JGM2.cof at:\n{listed}")
    return matches[0]


def import_gmat_jgm2(
    gmat_root: str | Path,
    *,
    project_root: str | Path,
) -> ImportedGravityFile:
    """Copy the installed JGM2 file into frozen project evidence."""
    root = Path(project_root).resolve()
    source = _find_jgm2(Path(gmat_root).resolve())
    field = CofGravityField.from_file(source)
    if (field.maximum_degree, field.maximum_order) != (70, 70):
        raise ValueError("GMAT JGM2.cof must declare degree/order 70/70.")
    if abs(field.gravitational_parameter_km3_s2 - 398600.4415) > 1.0e-9:
        raise ValueError("GMAT JGM2 gravitational parameter is unexpected.")
    if abs(field.reference_radius_km - 6378.1363) > 1.0e-9:
        raise ValueError("GMAT JGM2 reference radius is unexpected.")
    if abs(field.j2 - 0.001082626724392697) > 2.0e-10:
        raise ValueError("GMAT JGM2 C20/J2 value is unexpected.")
    destination = root / "data" / "reference" / "gmat_r2026a" / "JGM2.cof"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and _sha256(destination) != field.source_sha256:
        raise ValueError(
            "A different frozen JGM2.cof already exists. Move it aside and investigate; "
            "it will not be overwritten."
        )
    if not destination.exists():
        shutil.copy2(source, destination)
    provenance = destination.with_name("JGM2_PROVENANCE_1D0.json")
    _write_json(
        {
            "schema_version": SCHEMA_VERSION,
            "source": str(source),
            "destination": _relative(destination, root),
            "sha256": field.source_sha256,
            "size_bytes": destination.stat().st_size,
            "maximum_degree": field.maximum_degree,
            "maximum_order": field.maximum_order,
            "normalized": field.normalized,
            "gravitational_parameter_km3_s2": field.gravitational_parameter_km3_s2,
            "reference_radius_km": field.reference_radius_km,
            "derived_j2": field.j2,
            "imported_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_policy": "read_only_copy_from_user_verified_GMAT_R2026a_install",
        },
        provenance,
    )
    return ImportedGravityFile(
        destination=destination,
        provenance=provenance,
        sha256=field.source_sha256,
        maximum_degree=field.maximum_degree,
        maximum_order=field.maximum_order,
    )


def _gmat_epoch(value: str) -> str:
    epoch = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)
    return epoch.astimezone(timezone.utc).strftime("%d %b %Y %H:%M:%S.%f")[:-3]


def _gmat_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def build_gravity_ladder_script(
    config: dict[str, Any],
    baseline: dict[str, Any],
    *,
    gravity_file: Path,
    output_report: Path,
) -> str:
    state = initial_state_from_config(baseline)
    samples = int(config["sample_count"])
    step = float(config["duration_seconds"]) / (samples - 1)
    resource_blocks: list[str] = []
    field_names = [
        "LadderSat.ElapsedSecs",
        "LadderSat.EarthMJ2000Eq.X",
        "LadderSat.EarthMJ2000Eq.Y",
        "LadderSat.EarthMJ2000Eq.Z",
    ]
    for item in config["ladder"]:
        alias = str(item["alias"])
        degree = int(item["degree"])
        order = int(item["order"])
        resource_blocks.append(
            f"""
Create ForceModel {alias}FM;
{alias}FM.CentralBody = Earth;
{alias}FM.PrimaryBodies = {{Earth}};
{alias}FM.Drag = None;
{alias}FM.SRP = Off;
{alias}FM.RelativisticCorrection = Off;
{alias}FM.ErrorControl = RSSStep;
{alias}FM.GravityField.Earth.Degree = {degree};
{alias}FM.GravityField.Earth.Order = {order};
{alias}FM.GravityField.Earth.PotentialFile = '{_gmat_path(gravity_file)}';
{alias}FM.GravityField.Earth.TideModel = 'None';
"""
        )
        field_names.extend(
            [
                f"LadderSat.{alias}FM.AccelerationX",
                f"LadderSat.{alias}FM.AccelerationY",
                f"LadderSat.{alias}FM.AccelerationZ",
            ]
        )
    propagate = str(config["ladder"][-1]["alias"])
    resource_blocks.append(
        f"""
Create Propagator LadderProp;
LadderProp.FM = {propagate}FM;
LadderProp.Type = PrinceDormand78;
LadderProp.InitialStepSize = {step:.15g};
LadderProp.Accuracy = 1e-13;
LadderProp.MinStep = 1e-6;
LadderProp.MaxStep = {step:.15g};
LadderProp.MaxStepAttempts = 50;
LadderProp.StopIfAccuracyIsViolated = true;

Create ReportFile GravityLadderReport;
GravityLadderReport.Filename = '{_gmat_path(output_report)}';
GravityLadderReport.Precision = 16;
GravityLadderReport.ColumnWidth = 24;
GravityLadderReport.WriteHeaders = true;
GravityLadderReport.LeftJustify = Off;
GravityLadderReport.ZeroFill = On;
GravityLadderReport.FixedWidth = Off;
GravityLadderReport.Delimiter = Comma;
"""
    )
    fields = " ".join(field_names)
    mission = ["BeginMissionSequence;", f"Report GravityLadderReport {fields};"]
    for _ in range(samples - 1):
        mission.append(
            f"Propagate LadderProp(LadderSat) {{LadderSat.ElapsedSecs = {step:.15g}}};"
        )
        mission.append(f"Report GravityLadderReport {fields};")
    return f"""%
% Research Core 1D.0 higher-degree/order shared-state gravity ladder
% Target: GMAT R2026a; JGM2; no tides, drag, SRP, third bodies, or relativity.
%

Create Spacecraft LadderSat;
LadderSat.DateFormat = UTCGregorian;
LadderSat.Epoch = '{_gmat_epoch(state.epoch_utc)}';
LadderSat.CoordinateSystem = EarthMJ2000Eq;
LadderSat.DisplayStateType = Cartesian;
LadderSat.X = {state.position_km[0]:.15f};
LadderSat.Y = {state.position_km[1]:.15f};
LadderSat.Z = {state.position_km[2]:.15f};
LadderSat.VX = {state.velocity_km_s[0]:.15f};
LadderSat.VY = {state.velocity_km_s[1]:.15f};
LadderSat.VZ = {state.velocity_km_s[2]:.15f};
LadderSat.DryMass = 500;
LadderSat.Cd = 2.2;
LadderSat.Cr = 1.0;
LadderSat.DragArea = 4;
LadderSat.SRPArea = 4;
{''.join(resource_blocks)}
{chr(10).join(mission)}
"""


def prepare_gravity_ladder(
    config_path: str | Path,
    *,
    project_root: str | Path,
) -> PreparedGravityLadder:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_gravity_ladder_config(config_file)
    baseline_path = root / str(config["baseline_configuration"])
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    gravity_file = root / str(config["gravity_file"])
    field = CofGravityField.from_file(gravity_file)
    reference = root / str(config["reference_root"])
    scripts = reference / "scripts"
    outputs = reference / "output"
    scripts.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    output_report = outputs / "GMAT_GRAVITY_LADDER_1D0.csv"
    archived: list[Path] = []
    if output_report.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")
        destination = reference / "archive" / stamp / output_report.name
        destination.parent.mkdir(parents=True, exist_ok=False)
        shutil.move(str(output_report), destination)
        archived.append(destination)
    master = scripts / "RUN_GRAVITY_LADDER_1D0.script"
    master.write_text(
        build_gravity_ladder_script(
            config, baseline, gravity_file=gravity_file, output_report=output_report
        ),
        encoding="utf-8",
        newline="\n",
    )
    run_order = reference / "RUN_ORDER_1D0.txt"
    run_order.write_text(
        "RESEARCH CORE 1D.0 GMAT RUN ORDER\n\n"
        "1. Verify EOP and import JGM2 using START_HERE_1D_0.txt.\n"
        "2. Open scripts/RUN_GRAVITY_LADDER_1D0.script in GMAT R2026a.\n"
        "3. Run once; one CSV with 25 shared states is expected.\n"
        "4. Return to PowerShell and run the Python validation command.\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = reference / "GMAT_GRAVITY_LADDER_1D0_MANIFEST.json"
    _write_json(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": config["experiment_id"],
            "status": "scripts_prepared_gmat_execution_pending",
            "configuration": _relative(config_file, root),
            "configuration_sha256": _sha256(config_file),
            "baseline_configuration": _relative(baseline_path, root),
            "gravity_file": _relative(gravity_file, root),
            "gravity_file_sha256": field.source_sha256,
            "master_script": _relative(master, root),
            "master_script_sha256": _sha256(master),
            "output_report": _relative(output_report, root),
            "sample_count": int(config["sample_count"]),
            "model_count": len(config["ladder"]),
            "ladder": config["ladder"],
            "archived_previous_output_count": len(archived),
        },
        manifest,
    )
    return PreparedGravityLadder(
        experiment_id=str(config["experiment_id"]),
        master_script=master,
        output_report=output_report,
        manifest=manifest,
        sample_count=int(config["sample_count"]),
        model_count=len(config["ladder"]),
        archived_outputs=tuple(archived),
    )


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _split(value: str) -> list[str]:
    stripped = value.strip()
    if not stripped:
        return []
    if "," in stripped:
        return [cell.strip() for cell in next(csv.reader([stripped]))]
    whitespace = re.split(r"\s+", stripped)
    if len(whitespace) > 1:
        return whitespace
    cells = stripped.split("C")
    return [cell.strip() for cell in cells]


def _index(headers: list[str], suffix: str) -> int:
    target = _normalize(suffix)
    matches = [i for i, header in enumerate(headers) if _normalize(header).endswith(target)]
    if len(matches) != 1:
        raise ValueError(f"Expected one GMAT column ending in {suffix!r}; found {len(matches)}.")
    return matches[0]


def parse_gravity_ladder_report(
    path: str | Path, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"GMAT 1D.0 report not found: {source}")
    rows = [cells for line in source.read_text(encoding="utf-8-sig").splitlines() if (cells := _split(line))]
    if len(rows) < 2:
        raise ValueError("GMAT 1D.0 report has no numeric samples.")
    headers = rows[0]
    suffixes = ["ElapsedSecs", "EarthMJ2000Eq.X", "EarthMJ2000Eq.Y", "EarthMJ2000Eq.Z"]
    aliases = [str(item["alias"]) for item in config["ladder"]]
    for alias in aliases:
        suffixes.extend([f"{alias}FM.AccelerationX", f"{alias}FM.AccelerationY", f"{alias}FM.AccelerationZ"])
    indices = [_index(headers, suffix) for suffix in suffixes]
    numeric: list[list[float]] = []
    for row in rows[1:]:
        if _normalize(row[0]).endswith("elapsedsecs"):
            continue
        try:
            numeric.append([float(row[index]) for index in indices])
        except (IndexError, ValueError) as exc:
            raise ValueError("Invalid numeric row in GMAT 1D.0 report.") from exc
    values = np.asarray(numeric, dtype=float)
    if values.shape[0] != int(config["sample_count"]) or not np.all(np.isfinite(values)):
        raise ValueError("GMAT 1D.0 report sample count or numeric content is invalid.")
    elapsed = values[:, 0]
    if abs(float(elapsed[0])) > 1.0e-6 or np.any(np.diff(elapsed) <= 0.0):
        raise ValueError("GMAT 1D.0 elapsed times must start at zero and increase.")
    accelerations = {
        alias: values[:, 4 + 3 * index : 7 + 3 * index]
        for index, alias in enumerate(aliases)
    }
    return elapsed, values[:, 1:4], accelerations


def _report_html(status: str, decision: str, rows: list[dict[str, Any]]) -> str:
    body = "".join(
        "<tr>"
        f"<td>{html.escape(row['label'])}</td><td>{row['degree']}</td><td>{row['order']}</td>"
        f"<td>{row['maximum_difference_km_s2']:.9e}</td>"
        f"<td>{row['maximum_relative_difference']:.9e}</td>"
        f"<td>{'passed' if row['passed'] else 'failed'}</td></tr>"
        for row in rows
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Research Core 1D.0 Gravity Ladder</title><style>
body{{font-family:Arial,sans-serif;margin:2rem;color:#172033}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccd3df;padding:.5rem}}th{{background:#edf2f8}}</style></head><body>
<h1>Research Core 1D.0 GMAT Gravity Ladder</h1><p><strong>Status:</strong> {status}</p>
<p><strong>Decision:</strong> {decision}</p><table><thead><tr><th>Model</th><th>Degree</th>
<th>Order</th><th>Maximum vector difference (km/s²)</th><th>Maximum relative difference</th>
<th>Check</th></tr></thead><tbody>{body}</tbody></table>
<p>This shared-state diagnostic checks software-model agreement. It is not a full-arc validation,
measured-orbit truth, or flight qualification.</p></body></html>"""


def run_gravity_ladder_validation(
    config_path: str | Path,
    *,
    project_root: str | Path,
) -> GravityLadderResult:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_gravity_ladder_config(config_file)
    gravity_path = root / str(config["gravity_file"])
    eop_path = root / str(config["eop_file"])
    output = root / str(config["reference_root"]) / "output" / "GMAT_GRAVITY_LADDER_1D0.csv"
    field = CofGravityField.from_file(gravity_path)
    eop = GmatEopDataset.from_file(eop_path, expected_sha256=GMAT_R2026A_EOP_SHA256)
    elapsed, positions, gmat = parse_gravity_ladder_report(output, config)
    epoch = str(config["epoch_utc"])
    thresholds = config["thresholds"]
    records: list[dict[str, Any]] = []
    comparison_columns: dict[str, np.ndarray] = {}
    for item in config["ladder"]:
        alias = str(item["alias"])
        degree = int(item["degree"])
        order = int(item["order"])
        python_rows = []
        for seconds, position in zip(elapsed, positions):
            evaluation = Time(epoch, scale="utc") + TimeDelta(float(seconds), format="sec")
            sample = eop.sample(evaluation.utc)
            rotation = gmat_r2026a_inertial_to_fixed_matrix(evaluation.utc, sample)
            python_rows.append(
                gravity_acceleration_inertial_km_s2(
                    position, rotation, field, degree=degree, order=order
                )
            )
        python_values = np.asarray(python_rows)
        differences = python_values - gmat[alias]
        vector = np.linalg.norm(differences, axis=1)
        relative = vector / np.maximum(np.linalg.norm(gmat[alias], axis=1), np.finfo(float).tiny)
        max_absolute = float(np.max(vector))
        max_relative = float(np.max(relative))
        passed = (
            max_absolute <= float(thresholds["maximum_vector_difference_km_s2"])
            and max_relative <= float(thresholds["maximum_relative_difference"])
        )
        records.append(
            {
                **item,
                "maximum_difference_km_s2": max_absolute,
                "rms_difference_km_s2": float(np.sqrt(np.mean(vector**2))),
                "maximum_relative_difference": max_relative,
                "passed": passed,
            }
        )
        comparison_columns[f"{alias}_difference_km_s2"] = vector
    all_passed = all(item["passed"] for item in records)
    status = "passed_with_warnings" if all_passed else "failed_validation"
    decision = "advance_to_1d1_short_arc_validation" if all_passed else "stop_and_investigate_failed_gravity_level"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")
    result_dir = root / "results" / str(config["experiment_id"]) / stamp
    result_dir.mkdir(parents=True, exist_ok=False)
    summary = result_dir / "gravity_ladder_summary.json"
    _write_json(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": config["experiment_id"],
            "status": status,
            "decision": decision,
            "sample_count": int(elapsed.size),
            "model_count": len(records),
            "gravity_file_sha256": field.source_sha256,
            "eop_file_sha256": eop.source_sha256,
            "thresholds": thresholds,
            "models": records,
            "scientific_scope": config["scientific_cautions"],
        },
        summary,
    )
    csv_path = result_dir / "gravity_ladder_differences.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["elapsed_seconds", *comparison_columns])
        for index, seconds in enumerate(elapsed):
            writer.writerow([seconds, *[values[index] for values in comparison_columns.values()]])
    report = result_dir / "GMAT_GRAVITY_LADDER_1D0_REPORT.html"
    report.write_text(_report_html(status, decision, records), encoding="utf-8", newline="\n")
    manifest = result_dir / "RUN_MANIFEST.json"
    _write_json(
        {
            "files": [
                {"path": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in (summary, csv_path, report)
            ]
        },
        manifest,
    )
    return GravityLadderResult(
        experiment_id=str(config["experiment_id"]),
        status=status,
        decision=decision,
        sample_count=int(elapsed.size),
        model_count=len(records),
        maximum_difference_km_s2=max(item["maximum_difference_km_s2"] for item in records),
        result_directory=result_dir,
        report_path=report,
    )


def package_gravity_ladder_results(
    config_path: str | Path,
    *,
    project_root: str | Path,
    output_path: str | Path,
) -> Path:
    root = Path(project_root).resolve()
    config = load_gravity_ladder_config(config_path)
    reference = root / str(config["reference_root"])
    raw = reference / "output" / "GMAT_GRAVITY_LADDER_1D0.csv"
    if not raw.is_file():
        raise FileNotFoundError("Run the 1D.0 GMAT script before packaging results.")
    result_root = root / "results" / str(config["experiment_id"])
    completed = sorted(
        path for path in result_root.glob("*")
        if path.is_dir() and (path / "gravity_ladder_summary.json").is_file()
    )
    if not completed:
        raise FileNotFoundError("Run the Python 1D.0 validation before packaging results.")
    members = [Path(config_path).resolve()]
    members.extend(path for path in reference.rglob("*") if path.is_file() and "archive" not in path.parts)
    members.extend(path for path in completed[-1].rglob("*") if path.is_file())
    members.extend(
        [
            root / str(config["gravity_file"]),
            root / "data/reference/gmat_r2026a/JGM2_PROVENANCE_1D0.json",
        ]
    )
    archive = Path(output_path).resolve()
    if archive.exists():
        raise FileExistsError(f"Move or rename the existing results ZIP first: {archive}")
    temporary = archive.with_name(f".{archive.name}.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
        for member in sorted(set(members)):
            stream.write(member, _relative(member, root))
    os.replace(temporary, archive)
    with zipfile.ZipFile(archive) as stream:
        bad = stream.testzip()
        if bad:
            raise RuntimeError(f"Created 1D.0 results ZIP failed at {bad}.")
    return archive
