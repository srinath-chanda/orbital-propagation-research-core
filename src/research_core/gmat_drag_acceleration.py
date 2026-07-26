"""Research Core 1E.0 deterministic GMAT exponential-drag acceleration gate."""

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
from .gmat_gravity_multicase_closure import verify_gravity_multicase_closure
from .gravity_harmonics import CofGravityField


SCHEMA_VERSION = "1E.0"
TIME_GRID_TOLERANCE_SECONDS = 5.0e-6
EXPECTED_SCENARIOS = (
    ("NOMINAL", "DGN", 500.0, 4.0, 2.2, 1.0),
    ("AREA_X2", "DGA2", 500.0, 8.0, 2.2, 2.0),
    ("MASS_HALF", "DGM2", 250.0, 4.0, 2.2, 2.0),
    ("CD_HALF", "DGC5", 500.0, 4.0, 1.1, 0.5),
)


@dataclass(frozen=True)
class ExponentialAtmosphereTable:
    """Frozen three-column GMAT exponential atmosphere table."""

    source_path: Path
    source_sha256: str
    reference_height_km: np.ndarray
    reference_density_kg_m3: np.ndarray
    scale_height_km: np.ndarray

    @classmethod
    def from_file(
        cls, path: str | Path, *, expected_sha256: str | None = None
    ) -> "ExponentialAtmosphereTable":
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"GMAT exponential atmosphere file not found: {source}")
        actual = _sha256(source)
        if expected_sha256 is not None and actual != expected_sha256:
            raise ValueError(
                f"Atmosphere checksum mismatch: expected {expected_sha256}, found {actual}."
            )
        rows: list[tuple[float, float, float]] = []
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 3:
                raise ValueError(f"Invalid atmosphere row at line {line_number}.")
            try:
                rows.append(tuple(float(field) for field in fields))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid atmosphere number at line {line_number}."
                ) from exc
        values = np.asarray(rows, dtype=float)
        if values.shape != (28, 3) or not np.all(np.isfinite(values)):
            raise ValueError("The GMAT Earth exponential table must contain 28 finite rows.")
        if np.any(np.diff(values[:, 0]) <= 0.0):
            raise ValueError("Atmosphere reference heights must increase strictly.")
        if np.any(values[:, 1:] <= 0.0):
            raise ValueError("Atmosphere densities and scale heights must be positive.")
        return cls(source, actual, values[:, 0], values[:, 1], values[:, 2])

    def density_kg_m3(self, geodetic_height_km: float) -> float:
        """Evaluate GMAT R2026a's unsmoothed piecewise exponential rule."""
        height = float(geodetic_height_km)
        if not np.isfinite(height) or height < 0.0:
            raise ValueError("Geodetic height must be finite and non-negative.")
        index = int(np.searchsorted(self.reference_height_km, height, side="right") - 1)
        index = min(max(index, 0), self.reference_height_km.size - 1)
        return float(
            self.reference_density_kg_m3[index]
            * np.exp(
                -(height - self.reference_height_km[index])
                / self.scale_height_km[index]
            )
        )


@dataclass(frozen=True)
class PreparedDragAcceleration:
    experiment_id: str
    scenario_count: int
    sample_count: int
    expected_output_count: int
    master_script: Path
    run_order: Path
    manifest: Path
    archived_outputs: tuple[Path, ...]


@dataclass(frozen=True)
class DragAccelerationResult:
    experiment_id: str
    status: str
    decision: str
    scenario_count: int
    passed_scenario_count: int
    check_count: int
    passed_check_count: int
    maximum_drag_difference_km_s2: float
    maximum_drag_relative_difference: float
    maximum_density_relative_difference: float
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


def load_drag_acceleration_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Drag acceleration schema must be {SCHEMA_VERSION!r}.")
    observed = tuple(
        (
            str(item["scenario_id"]),
            str(item["alias"]),
            float(item["mass_kg"]),
            float(item["drag_area_m2"]),
            float(item["drag_coefficient"]),
            float(item["expected_acceleration_scale"]),
        )
        for item in payload.get("scenarios", [])
    )
    if observed != EXPECTED_SCENARIOS:
        raise ValueError(f"The preregistered drag scenarios must be {EXPECTED_SCENARIOS!r}.")
    if payload.get("threshold_status") != "preregistered_before_first_1e0_gmat_output":
        raise ValueError("The 1E.0 thresholds are not preregistered.")
    sample_count = int(payload["sample_count"])
    duration = float(payload["duration_seconds"])
    if sample_count != 25 or duration <= 0.0:
        raise ValueError("1E.0 requires exactly 25 samples over a positive duration.")
    time_grid_tolerance = float(payload["time_grid_tolerance_seconds"])
    if abs(time_grid_tolerance - TIME_GRID_TOLERANCE_SECONDS) > 1.0e-15:
        raise ValueError(
            "The 1E.0 time-grid tolerance must reuse the closed 1D.2 limit of 5 microseconds."
        )
    if int(payload["gravity_degree"]) != 20 or int(payload["gravity_order"]) != 20:
        raise ValueError("1E.0 must reuse the closed 20x20 gravity baseline.")
    if abs(float(payload["earth_equatorial_radius_km"]) - 6378.1363) > 1.0e-12:
        raise ValueError("The GMAT Earth equatorial radius is incorrect.")
    if abs(float(payload["earth_flattening"]) - 0.00335270) > 1.0e-12:
        raise ValueError("The GMAT Earth flattening is incorrect.")
    thresholds = payload["thresholds"]
    for key, value in thresholds.items():
        if key == "maximum_direction_cosine":
            if not -1.0 < float(value) < 0.0:
                raise ValueError("The direction-cosine limit must lie between -1 and 0.")
        elif float(value) <= 0.0:
            raise ValueError(f"Threshold {key} must be positive.")
    return payload


def gmat_geodetic_height_km(
    body_fixed_position_km: np.ndarray,
    *,
    equatorial_radius_km: float,
    flattening: float,
) -> float:
    """Reproduce GMAT AtmosphereModel::CalculateGeodetics (Vallado algorithm 12)."""
    position = np.asarray(body_fixed_position_km, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("Body-fixed position must be a finite three-vector.")
    radius = float(equatorial_radius_km)
    flat = float(flattening)
    if radius <= 0.0 or not 0.0 <= flat < 1.0:
        raise ValueError("Ellipsoid radius or flattening is invalid.")
    rxy = float(np.hypot(position[0], position[1]))
    latitude = float(np.arctan2(position[2], rxy))
    eccentricity_squared = flat * (2.0 - flat)
    delta = 1.0
    while delta > 1.0e-7:
        old = latitude
        sine = float(np.sin(old))
        factor = radius / np.sqrt(1.0 - eccentricity_squared * sine * sine)
        latitude = float(np.arctan2(position[2] + factor * eccentricity_squared * sine, rxy))
        delta = abs(latitude - old)
    sine = float(np.sin(latitude))
    factor = radius / np.sqrt(1.0 - eccentricity_squared * sine * sine)
    cosine = float(np.cos(latitude))
    if abs(cosine) < 1.0e-12:
        polar_radius = radius * (1.0 - flat)
        return float(abs(position[2]) - polar_radius)
    return float(rxy / cosine - factor)


def gmat_earth_angular_velocity_inertial_rad_s(
    evaluation_time_utc: Time,
    dataset: GmatEopDataset,
    *,
    difference_step_seconds: float = 0.5,
) -> np.ndarray:
    """Derive the GMAT body-fixed angular-velocity vector from its validated rotation."""
    step = float(difference_step_seconds)
    if step <= 0.0 or not np.isfinite(step):
        raise ValueError("Rotation derivative step must be positive and finite.")
    time = evaluation_time_utc.utc

    def rotation(at: Time) -> np.ndarray:
        sample = dataset.sample(at.utc)
        return gmat_r2026a_inertial_to_fixed_matrix(at.utc, sample)

    offset = TimeDelta(step, format="sec")
    central = rotation(time)
    derivative = (rotation(time + offset) - rotation(time - offset)) / (2.0 * step)
    skew = -(central.T @ derivative)
    skew = 0.5 * (skew - skew.T)
    angular_velocity = np.asarray(
        [skew[2, 1], skew[0, 2], skew[1, 0]], dtype=float
    )
    if not np.all(np.isfinite(angular_velocity)):
        raise RuntimeError("Earth angular-velocity calculation became non-finite.")
    return angular_velocity


def gmat_exponential_drag_acceleration_km_s2(
    position_km: np.ndarray,
    velocity_km_s: np.ndarray,
    evaluation_time_utc: Time,
    *,
    eop: GmatEopDataset,
    atmosphere: ExponentialAtmosphereTable,
    equatorial_radius_km: float,
    flattening: float,
    mass_kg: float,
    drag_area_m2: float,
    drag_coefficient: float,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Return matched GMAT drag, density, height, and relative velocity."""
    position = np.asarray(position_km, dtype=float)
    velocity = np.asarray(velocity_km_s, dtype=float)
    if position.shape != (3,) or velocity.shape != (3,):
        raise ValueError("Position and velocity must be three-vectors.")
    sample = eop.sample(evaluation_time_utc.utc)
    rotation = gmat_r2026a_inertial_to_fixed_matrix(evaluation_time_utc.utc, sample)
    height = gmat_geodetic_height_km(
        rotation @ position,
        equatorial_radius_km=equatorial_radius_km,
        flattening=flattening,
    )
    density = atmosphere.density_kg_m3(height)
    angular_velocity = gmat_earth_angular_velocity_inertial_rad_s(
        evaluation_time_utc.utc, eop
    )
    relative_velocity = velocity - np.cross(angular_velocity, position)
    speed = float(np.linalg.norm(relative_velocity))
    coefficient = -500.0 * float(drag_coefficient) * float(drag_area_m2) / float(mass_kg)
    acceleration = coefficient * density * speed * relative_velocity
    return acceleration, density, height, relative_velocity


def _gmat_epoch(value: str) -> str:
    epoch = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)
    return epoch.astimezone(timezone.utc).strftime("%d %b %Y %H:%M:%S.%f")[:-3]


def _gmat_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _scenario_script_parts(
    scenario: dict[str, Any],
    config: dict[str, Any],
    baseline: dict[str, Any],
    *,
    gravity_file: Path,
    atmosphere_file: Path,
    output_report: Path,
) -> tuple[str, list[str]]:
    state = initial_state_from_config(baseline)
    alias = str(scenario["alias"])
    satellite = f"{alias}Sat"
    gravity = f"{alias}GravityFM"
    drag = f"{alias}DragFM"
    propagator = f"{alias}Prop"
    report = f"{alias}Report"
    step = float(config["duration_seconds"]) / (int(config["sample_count"]) - 1)
    resource = f"""
% {scenario['scenario_id']}: isolated exponential drag at shared gravity-only states
Create Spacecraft {satellite};
{satellite}.DateFormat = UTCGregorian;
{satellite}.Epoch = '{_gmat_epoch(state.epoch_utc)}';
{satellite}.CoordinateSystem = EarthMJ2000Eq;
{satellite}.DisplayStateType = Cartesian;
{satellite}.X = {state.position_km[0]:.15f};
{satellite}.Y = {state.position_km[1]:.15f};
{satellite}.Z = {state.position_km[2]:.15f};
{satellite}.VX = {state.velocity_km_s[0]:.15f};
{satellite}.VY = {state.velocity_km_s[1]:.15f};
{satellite}.VZ = {state.velocity_km_s[2]:.15f};
{satellite}.DryMass = {float(scenario['mass_kg']):.15g};
{satellite}.Cd = {float(scenario['drag_coefficient']):.15g};
{satellite}.Cr = 1.0;
{satellite}.DragArea = {float(scenario['drag_area_m2']):.15g};
{satellite}.SRPArea = 4.0;
{satellite}.AtmosDensityScaleFactor = 1.0;

Create ForceModel {gravity};
{gravity}.CentralBody = Earth;
{gravity}.PrimaryBodies = {{Earth}};
{gravity}.Drag = None;
{gravity}.SRP = Off;
{gravity}.RelativisticCorrection = Off;
{gravity}.ErrorControl = RSSStep;
{gravity}.GravityField.Earth.Degree = {int(config['gravity_degree'])};
{gravity}.GravityField.Earth.Order = {int(config['gravity_order'])};
{gravity}.GravityField.Earth.PotentialFile = '{_gmat_path(gravity_file)}';
{gravity}.GravityField.Earth.TideModel = 'None';

Create ForceModel {drag};
{drag}.CentralBody = Earth;
{drag}.PrimaryBodies = {{Earth}};
{drag}.Drag.AtmosphereModel = Exponential;
{drag}.Drag.DragModel = 'Spherical';
{drag}.Drag.InputFile = '{_gmat_path(atmosphere_file)}';
{drag}.SRP = Off;
{drag}.RelativisticCorrection = Off;
{drag}.ErrorControl = RSSStep;
{drag}.GravityField.Earth.Degree = {int(config['gravity_degree'])};
{drag}.GravityField.Earth.Order = {int(config['gravity_order'])};
{drag}.GravityField.Earth.PotentialFile = '{_gmat_path(gravity_file)}';
{drag}.GravityField.Earth.TideModel = 'None';

Create Propagator {propagator};
{propagator}.FM = {gravity};
{propagator}.Type = {config['integrator']['gmat_method']};
{propagator}.InitialStepSize = {step:.15g};
{propagator}.Accuracy = {float(config['integrator']['gmat_accuracy']):.15g};
{propagator}.MinStep = {float(config['integrator']['gmat_minimum_step_seconds']):.15g};
{propagator}.MaxStep = {min(step, float(config['integrator']['gmat_maximum_step_seconds'])):.15g};
{propagator}.MaxStepAttempts = 50;
{propagator}.StopIfAccuracyIsViolated = true;

Create ReportFile {report};
{report}.Filename = '{_gmat_path(output_report)}';
{report}.Precision = 16;
{report}.ColumnWidth = 24;
{report}.WriteHeaders = true;
{report}.LeftJustify = Off;
{report}.ZeroFill = On;
{report}.FixedWidth = Off;
{report}.Delimiter = Comma;
"""
    fields = " ".join(
        [
            f"{satellite}.ElapsedSecs",
            f"{satellite}.EarthMJ2000Eq.X",
            f"{satellite}.EarthMJ2000Eq.Y",
            f"{satellite}.EarthMJ2000Eq.Z",
            f"{satellite}.EarthMJ2000Eq.VX",
            f"{satellite}.EarthMJ2000Eq.VY",
            f"{satellite}.EarthMJ2000Eq.VZ",
            f"{satellite}.{gravity}.AccelerationX",
            f"{satellite}.{gravity}.AccelerationY",
            f"{satellite}.{gravity}.AccelerationZ",
            f"{satellite}.{drag}.AccelerationX",
            f"{satellite}.{drag}.AccelerationY",
            f"{satellite}.{drag}.AccelerationZ",
        ]
    )
    mission = [f"Report {report} {fields};"]
    for _ in range(int(config["sample_count"]) - 1):
        mission.append(
            f"Propagate {propagator}({satellite}) "
            f"{{{satellite}.ElapsedSecs = {step:.15g}}};"
        )
        mission.append(f"Report {report} {fields};")
    return resource, mission


def build_drag_acceleration_master_script(
    config: dict[str, Any],
    baseline: dict[str, Any],
    *,
    gravity_file: Path,
    atmosphere_file: Path,
    output_directory: Path,
) -> str:
    resources: list[str] = []
    missions: list[str] = []
    for scenario in config["scenarios"]:
        resource, mission = _scenario_script_parts(
            scenario,
            config,
            baseline,
            gravity_file=gravity_file,
            atmosphere_file=atmosphere_file,
            output_report=output_directory / f"{scenario['scenario_id']}_DRAG_ACCELERATION_1E0.csv",
        )
        resources.append(resource)
        missions.extend(mission)
    return (
        "%\n% Research Core 1E.0 GMAT exponential-drag acceleration gate\n"
        "% Four ballistic scenarios; 25 gravity-only shared states each.\n"
        "% Run once in GMAT R2026a. Thresholds were frozen before output.\n%\n"
        + "".join(resources)
        + "\nBeginMissionSequence;\n"
        + "\n".join(missions)
        + "\n"
    )


def prepare_drag_acceleration(
    config_path: str | Path,
    *,
    project_root: str | Path,
) -> PreparedDragAcceleration:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_drag_acceleration_config(config_file)
    closure = verify_gravity_multicase_closure(
        _project_path(root, config["prerequisite_closure"], "prerequisite_closure"),
        project_root=root,
    )
    baseline_path = _project_path(root, config["baseline_configuration"], "baseline_configuration")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if str(baseline["initial_state"]["epoch_utc"]) != str(config["epoch_utc"]):
        raise ValueError("The baseline epoch does not match the 1E.0 epoch.")
    gravity_file = _project_path(root, config["gravity_file"], "gravity_file")
    field = CofGravityField.from_file(gravity_file)
    if (field.maximum_degree, field.maximum_order) != (70, 70):
        raise ValueError("The 1E.0 gravity file must contain JGM2 through 70x70.")
    eop_file = _project_path(root, config["eop_file"], "eop_file")
    if _sha256(eop_file) != GMAT_R2026A_EOP_SHA256:
        raise ValueError("The 1E.0 EOP file is not the validated R2026a file.")
    atmosphere_file = _project_path(root, config["atmosphere_file"], "atmosphere_file")
    atmosphere = ExponentialAtmosphereTable.from_file(
        atmosphere_file, expected_sha256=str(config["atmosphere_file_sha256"])
    )
    reference = _project_path(root, config["reference_root"], "reference_root")
    scripts = reference / "scripts"
    outputs = reference / "output"
    scripts.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    expected_names = {
        f"{item['scenario_id']}_DRAG_ACCELERATION_1E0.csv"
        for item in config["scenarios"]
    }
    existing = [path for path in outputs.glob("*.csv") if path.name in expected_names]
    archived: list[Path] = []
    if existing:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")
        archive = reference / "archive" / stamp
        archive.mkdir(parents=True, exist_ok=False)
        for source in existing:
            destination = archive / source.name
            shutil.move(str(source), destination)
            archived.append(destination)
    master = scripts / "RUN_DRAG_ACCELERATION_1E0.script"
    master.write_text(
        build_drag_acceleration_master_script(
            config,
            baseline,
            gravity_file=gravity_file,
            atmosphere_file=atmosphere_file,
            output_directory=outputs,
        ),
        encoding="utf-8",
        newline="\n",
    )
    runs: list[dict[str, Any]] = []
    for scenario in config["scenarios"]:
        output = outputs / f"{scenario['scenario_id']}_DRAG_ACCELERATION_1E0.csv"
        resource, mission = _scenario_script_parts(
            scenario,
            config,
            baseline,
            gravity_file=gravity_file,
            atmosphere_file=atmosphere_file,
            output_report=output,
        )
        script = scripts / f"{scenario['scenario_id']}_DRAG_ACCELERATION_1E0.script"
        script.write_text(
            "% Research Core 1E.0 fallback single ballistic scenario\n"
            + resource
            + "\nBeginMissionSequence;\n"
            + "\n".join(mission)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        runs.append(
            {
                "scenario_id": scenario["scenario_id"],
                "script": _relative(script, root),
                "script_sha256": _sha256(script),
                "output": _relative(output, root),
            }
        )
    run_order = reference / "RUN_ORDER_1E0.txt"
    run_order.write_text(
        "RESEARCH CORE 1E.0 GMAT RUN ORDER\n\n"
        "1. Run scripts/RUN_DRAG_ACCELERATION_1E0.script once in GMAT R2026a.\n"
        "2. Expect four CSV files in output, one for each ballistic scenario.\n"
        "3. Do not rerun after successful creation; return to PowerShell for validation.\n"
        "4. Use the four individual scripts only if the master script will not interpret.\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = reference / "GMAT_DRAG_ACCELERATION_1E0_MANIFEST.json"
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
            "atmosphere_file": _relative(atmosphere.source_path, root),
            "atmosphere_file_sha256": atmosphere.source_sha256,
            "master_script": _relative(master, root),
            "master_script_sha256": _sha256(master),
            "scenario_count": len(config["scenarios"]),
            "sample_count_per_scenario": int(config["sample_count"]),
            "expected_output_count": len(runs),
            "archived_previous_output_count": len(archived),
            "runs": runs,
        },
        manifest,
    )
    return PreparedDragAcceleration(
        experiment_id=str(config["experiment_id"]),
        scenario_count=len(config["scenarios"]),
        sample_count=int(config["sample_count"]),
        expected_output_count=len(runs),
        master_script=master,
        run_order=run_order,
        manifest=manifest,
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
    return [cell.strip() for cell in stripped.split("C")]


def _index(headers: list[str], suffix: str) -> int:
    target = _normalize(suffix)
    matches = [index for index, header in enumerate(headers) if _normalize(header).endswith(target)]
    if len(matches) != 1:
        raise ValueError(f"Expected one GMAT column ending in {suffix!r}; found {len(matches)}.")
    return matches[0]


def parse_drag_acceleration_report(
    path: str | Path,
    scenario: dict[str, Any],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"GMAT 1E.0 report not found: {source}")
    rows = [
        cells
        for line in source.read_text(encoding="utf-8-sig").splitlines()
        if (cells := _split(line))
    ]
    if len(rows) < 2:
        raise ValueError("GMAT 1E.0 report has no numeric samples.")
    alias = str(scenario["alias"])
    suffixes = [
        "ElapsedSecs",
        "EarthMJ2000Eq.X",
        "EarthMJ2000Eq.Y",
        "EarthMJ2000Eq.Z",
        "EarthMJ2000Eq.VX",
        "EarthMJ2000Eq.VY",
        "EarthMJ2000Eq.VZ",
        f"{alias}GravityFM.AccelerationX",
        f"{alias}GravityFM.AccelerationY",
        f"{alias}GravityFM.AccelerationZ",
        f"{alias}DragFM.AccelerationX",
        f"{alias}DragFM.AccelerationY",
        f"{alias}DragFM.AccelerationZ",
    ]
    headers = rows[0]
    indices = [_index(headers, suffix) for suffix in suffixes]
    numeric: list[list[float]] = []
    for row in rows[1:]:
        if _normalize(row[0]).endswith("elapsedsecs"):
            continue
        try:
            numeric.append([float(row[index]) for index in indices])
        except (IndexError, ValueError) as exc:
            raise ValueError("Invalid numeric row in GMAT 1E.0 report.") from exc
    values = np.asarray(numeric, dtype=float)
    if values.shape != (int(config["sample_count"]), 13) or not np.all(np.isfinite(values)):
        raise ValueError("GMAT 1E.0 sample count or numeric content is invalid.")
    raw_elapsed = values[:, 0]
    expected = np.linspace(0.0, float(config["duration_seconds"]), int(config["sample_count"]))
    residual = raw_elapsed - expected
    maximum_residual = float(np.max(np.abs(residual)))
    final_residual = float(residual[-1])
    tolerance = float(config["time_grid_tolerance_seconds"])
    if maximum_residual > tolerance:
        raise ValueError(
            "GMAT 1E.0 elapsed times differ from the registered grid by "
            f"{maximum_residual:.12g} s, exceeding the {tolerance:.12g} s limit."
        )
    grid_diagnostics = {
        "maximum_absolute_raw_time_residual_seconds": maximum_residual,
        "final_raw_time_residual_seconds": final_residual,
        "synchronization_tolerance_seconds": tolerance,
    }
    # Preserve the raw state vectors and snap only the accepted textual epochs to
    # the preregistered nominal grid; no state interpolation is performed.
    return expected, values[:, 1:7], values[:, 7:10], values[:, 10:13], grid_diagnostics


def _check(check_id: str, value: float, limit: float, unit: str, *, mode: str = "maximum") -> dict[str, Any]:
    if mode == "maximum":
        passed = value <= limit
    else:
        raise ValueError(f"Unsupported check mode: {mode}")
    return {
        "check_id": check_id,
        "measured_value": float(value),
        "limit": float(limit),
        "unit": unit,
        "status": "passed" if passed else "failed",
    }


def _write_comparison_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _report_html(status: str, decision: str, scenarios: list[dict[str, Any]], checks: list[dict[str, Any]]) -> str:
    scenario_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['scenario_id'])}</td>"
        f"<td>{item['maximum_drag_difference_km_s2']:.9e}</td>"
        f"<td>{item['maximum_drag_relative_difference']:.9e}</td>"
        f"<td>{item['maximum_density_relative_difference']:.9e}</td>"
        f"<td>{item['worst_direction_cosine']:.12f}</td>"
        f"<td>{html.escape(item['status'])}</td></tr>"
        for item in scenarios
    )
    passed = sum(item["status"] == "passed" for item in checks)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Research Core 1E.0 GMAT Drag Acceleration Gate</title><style>
body{{font-family:Arial,sans-serif;margin:2rem;color:#172033}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccd3df;padding:.5rem}}th{{background:#edf2f8}}</style></head><body>
<h1>Research Core 1E.0 GMAT Exponential-Drag Acceleration Gate</h1>
<p><strong>Status:</strong> {html.escape(status)}</p><p><strong>Decision:</strong> {html.escape(decision)}</p>
<p><strong>Checks passed:</strong> {passed}/{len(checks)}</p><table><thead><tr><th>Scenario</th>
<th>Max vector difference (km/s²)</th><th>Max relative difference</th>
<th>Max inferred-density relative difference</th><th>Worst direction cosine</th><th>Status</th>
</tr></thead><tbody>{scenario_rows}</tbody></table>
<p>The GMAT and Python drag accelerations use the same frozen R2026a 28-band exponential table,
geodetic altitude, rigid co-rotation, spherical area, mass, and drag coefficient. This deterministic
software comparison is not measured-orbit truth, a real-date density forecast, or flight qualification.</p>
</body></html>"""


def run_drag_acceleration_validation(
    config_path: str | Path,
    *,
    project_root: str | Path,
) -> DragAccelerationResult:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_drag_acceleration_config(config_file)
    closure = verify_gravity_multicase_closure(
        _project_path(root, config["prerequisite_closure"], "prerequisite_closure"),
        project_root=root,
    )
    eop = GmatEopDataset.from_file(
        _project_path(root, config["eop_file"], "eop_file"),
        expected_sha256=GMAT_R2026A_EOP_SHA256,
    )
    atmosphere = ExponentialAtmosphereTable.from_file(
        _project_path(root, config["atmosphere_file"], "atmosphere_file"),
        expected_sha256=str(config["atmosphere_file_sha256"]),
    )
    output_directory = _project_path(root, config["reference_root"], "reference_root") / "output"
    expected = {
        str(item["scenario_id"]): output_directory
        / f"{item['scenario_id']}_DRAG_ACCELERATION_1E0.csv"
        for item in config["scenarios"]
    }
    missing = [path for path in expected.values() if not path.is_file()]
    if missing:
        shown = "\n".join(f"  - {_relative(path, root)}" for path in missing)
        raise FileNotFoundError(f"The 1E.0 GMAT run is incomplete; missing:\n{shown}")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")
    result_directory = root / "results" / str(config["experiment_id"]) / stamp
    result_directory.mkdir(parents=True, exist_ok=False)
    thresholds = config["thresholds"]
    epoch = Time(str(config["epoch_utc"]), scale="utc")
    scenario_records: list[dict[str, Any]] = []
    all_checks: list[dict[str, Any]] = []
    state_histories: dict[str, np.ndarray] = {}
    gmat_drag_histories: dict[str, np.ndarray] = {}

    for scenario_index, scenario in enumerate(config["scenarios"], start=1):
        scenario_id = str(scenario["scenario_id"])
        source = expected[scenario_id]
        (
            elapsed,
            states,
            gravity_acceleration,
            total_acceleration,
            grid_diagnostics,
        ) = parse_drag_acceleration_report(source, scenario, config)
        gmat_drag = total_acceleration - gravity_acceleration
        python_rows: list[np.ndarray] = []
        densities: list[float] = []
        heights: list[float] = []
        relative_velocities: list[np.ndarray] = []
        for seconds, state in zip(elapsed, states):
            evaluation = epoch + TimeDelta(float(seconds), format="sec")
            acceleration, density, height, relative_velocity = (
                gmat_exponential_drag_acceleration_km_s2(
                    state[:3],
                    state[3:],
                    evaluation,
                    eop=eop,
                    atmosphere=atmosphere,
                    equatorial_radius_km=float(config["earth_equatorial_radius_km"]),
                    flattening=float(config["earth_flattening"]),
                    mass_kg=float(scenario["mass_kg"]),
                    drag_area_m2=float(scenario["drag_area_m2"]),
                    drag_coefficient=float(scenario["drag_coefficient"]),
                )
            )
            python_rows.append(acceleration)
            densities.append(density)
            heights.append(height)
            relative_velocities.append(relative_velocity)
        python_drag = np.asarray(python_rows)
        density_values = np.asarray(densities)
        height_values = np.asarray(heights)
        relative = np.asarray(relative_velocities)
        differences = np.linalg.norm(gmat_drag - python_drag, axis=1)
        python_magnitude = np.linalg.norm(python_drag, axis=1)
        relative_difference = differences / python_magnitude
        gmat_magnitude = np.linalg.norm(gmat_drag, axis=1)
        relative_speed = np.linalg.norm(relative, axis=1)
        ballistic_prefactor = (
            500.0
            * float(scenario["drag_coefficient"])
            * float(scenario["drag_area_m2"])
            / float(scenario["mass_kg"])
        )
        inferred_density = gmat_magnitude / (ballistic_prefactor * relative_speed**2)
        density_relative = np.abs(inferred_density - density_values) / density_values
        direction_cosine = np.sum(gmat_drag * relative, axis=1) / (
            gmat_magnitude * relative_speed
        )
        maximum_difference = float(np.max(differences))
        maximum_relative = float(np.max(relative_difference))
        maximum_density_relative = float(np.max(density_relative))
        worst_direction = float(np.max(direction_cosine))
        checks = [
            _check(
                f"1E0-{scenario_index:02d}-drag-vector",
                maximum_difference,
                float(thresholds["maximum_drag_vector_difference_km_s2"]),
                "km/s^2",
            ),
            _check(
                f"1E0-{scenario_index:02d}-drag-relative",
                maximum_relative,
                float(thresholds["maximum_drag_relative_difference"]),
                "relative",
            ),
            _check(
                f"1E0-{scenario_index:02d}-density",
                maximum_density_relative,
                float(thresholds["maximum_inferred_density_relative_difference"]),
                "relative",
            ),
            _check(
                f"1E0-{scenario_index:02d}-direction",
                worst_direction,
                float(thresholds["maximum_direction_cosine"]),
                "cosine",
            ),
        ]
        all_checks.extend(checks)
        comparison_rows: list[dict[str, float]] = []
        for index in range(elapsed.size):
            comparison_rows.append(
                {
                    "elapsed_seconds": float(elapsed[index]),
                    "geodetic_height_km": float(height_values[index]),
                    "python_density_kg_m3": float(density_values[index]),
                    "gmat_inferred_density_kg_m3": float(inferred_density[index]),
                    "relative_speed_km_s": float(relative_speed[index]),
                    "gmat_drag_x_km_s2": float(gmat_drag[index, 0]),
                    "gmat_drag_y_km_s2": float(gmat_drag[index, 1]),
                    "gmat_drag_z_km_s2": float(gmat_drag[index, 2]),
                    "python_drag_x_km_s2": float(python_drag[index, 0]),
                    "python_drag_y_km_s2": float(python_drag[index, 1]),
                    "python_drag_z_km_s2": float(python_drag[index, 2]),
                    "vector_difference_km_s2": float(differences[index]),
                    "relative_difference": float(relative_difference[index]),
                    "density_relative_difference": float(density_relative[index]),
                    "direction_cosine": float(direction_cosine[index]),
                }
            )
        comparison_path = result_directory / f"{scenario_id}_python_vs_gmat_drag.csv"
        _write_comparison_csv(comparison_path, comparison_rows)
        status = "passed" if all(item["status"] == "passed" for item in checks) else "failed"
        scenario_records.append(
            {
                **scenario,
                "status": status,
                "source_report": _relative(source, root),
                "source_report_sha256": _sha256(source),
                "sample_count": int(elapsed.size),
                "time_grid": grid_diagnostics,
                "minimum_geodetic_height_km": float(np.min(height_values)),
                "maximum_geodetic_height_km": float(np.max(height_values)),
                "minimum_density_kg_m3": float(np.min(density_values)),
                "maximum_density_kg_m3": float(np.max(density_values)),
                "maximum_drag_difference_km_s2": maximum_difference,
                "maximum_drag_relative_difference": maximum_relative,
                "maximum_density_relative_difference": maximum_density_relative,
                "worst_direction_cosine": worst_direction,
                "checks": checks,
            }
        )
        state_histories[scenario_id] = states
        gmat_drag_histories[scenario_id] = gmat_drag

    nominal_states = state_histories["NOMINAL"]
    nominal_drag = gmat_drag_histories["NOMINAL"]
    shared_records: list[dict[str, Any]] = []
    scaling_records: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(config["scenarios"][1:], start=2):
        scenario_id = str(scenario["scenario_id"])
        state_difference = state_histories[scenario_id] - nominal_states
        maximum_position = float(np.max(np.linalg.norm(state_difference[:, :3], axis=1)))
        maximum_velocity = float(np.max(np.linalg.norm(state_difference[:, 3:], axis=1)))
        position_check = _check(
            f"1E0-{scenario_index:02d}-shared-position",
            maximum_position,
            float(thresholds["maximum_shared_position_difference_km"]),
            "km",
        )
        velocity_check = _check(
            f"1E0-{scenario_index:02d}-shared-velocity",
            maximum_velocity,
            float(thresholds["maximum_shared_velocity_difference_km_s"]),
            "km/s",
        )
        all_checks.extend((position_check, velocity_check))
        shared_records.append(
            {
                "scenario_id": scenario_id,
                "maximum_position_difference_km": maximum_position,
                "maximum_velocity_difference_km_s": maximum_velocity,
                "checks": [position_check, velocity_check],
            }
        )
        scale = float(scenario["expected_acceleration_scale"])
        expected_drag = scale * nominal_drag
        scaling_relative = np.linalg.norm(
            gmat_drag_histories[scenario_id] - expected_drag, axis=1
        ) / np.linalg.norm(expected_drag, axis=1)
        maximum_scaling = float(np.max(scaling_relative))
        scaling_check = _check(
            f"1E0-{scenario_index:02d}-ballistic-scaling",
            maximum_scaling,
            float(thresholds["maximum_ballistic_scaling_relative_difference"]),
            "relative",
        )
        all_checks.append(scaling_check)
        scaling_records.append(
            {
                "scenario_id": scenario_id,
                "expected_scale": scale,
                "maximum_scaling_relative_difference": maximum_scaling,
                "check": scaling_check,
            }
        )

    passed_scenarios = sum(item["status"] == "passed" for item in scenario_records)
    passed_checks = sum(item["status"] == "passed" for item in all_checks)
    status = (
        "passed_with_warnings"
        if passed_scenarios == len(scenario_records) and passed_checks == len(all_checks)
        else "failed_validation"
    )
    decision = (
        "advance_to_1e1_drag_short_arc_validation"
        if status == "passed_with_warnings"
        else "stop_and_investigate_drag_density_rotation_or_ballistic_scaling"
    )
    summary_path = result_directory / "drag_acceleration_summary.json"
    _write_json(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": config["experiment_id"],
            "status": status,
            "decision": decision,
            "prerequisite_closure_id": closure.closure_id,
            "scenario_count": len(scenario_records),
            "passed_scenario_count": passed_scenarios,
            "failed_scenario_count": len(scenario_records) - passed_scenarios,
            "check_count": len(all_checks),
            "passed_check_count": passed_checks,
            "failed_check_count": len(all_checks) - passed_checks,
            "atmosphere_file_sha256": atmosphere.source_sha256,
            "thresholds": thresholds,
            "scenarios": scenario_records,
            "shared_state_checks": shared_records,
            "ballistic_scaling_checks": scaling_records,
            "checks": all_checks,
            "scientific_scope": config["scientific_cautions"],
        },
        summary_path,
    )
    report = result_directory / "GMAT_DRAG_ACCELERATION_1E0_REPORT.html"
    report.write_text(
        _report_html(status, decision, scenario_records, all_checks),
        encoding="utf-8",
        newline="\n",
    )
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
    return DragAccelerationResult(
        experiment_id=str(config["experiment_id"]),
        status=status,
        decision=decision,
        scenario_count=len(scenario_records),
        passed_scenario_count=passed_scenarios,
        check_count=len(all_checks),
        passed_check_count=passed_checks,
        maximum_drag_difference_km_s2=max(
            float(item["maximum_drag_difference_km_s2"]) for item in scenario_records
        ),
        maximum_drag_relative_difference=max(
            float(item["maximum_drag_relative_difference"]) for item in scenario_records
        ),
        maximum_density_relative_difference=max(
            float(item["maximum_density_relative_difference"]) for item in scenario_records
        ),
        result_directory=result_directory,
        report_path=report,
    )


def package_drag_acceleration_results(
    config_path: str | Path,
    *,
    project_root: str | Path,
    output_path: str | Path,
) -> Path:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_drag_acceleration_config(config_file)
    reference = _project_path(root, config["reference_root"], "reference_root")
    expected = [
        reference / "output" / f"{item['scenario_id']}_DRAG_ACCELERATION_1E0.csv"
        for item in config["scenarios"]
    ]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot package 1E.0: {len(missing)} GMAT reports are missing.")
    result_root = root / "results" / str(config["experiment_id"])
    completed = sorted(
        path
        for path in result_root.glob("*")
        if path.is_dir() and (path / "drag_acceleration_summary.json").is_file()
    )
    if not completed:
        raise FileNotFoundError("Run the Python 1E.0 validation before packaging.")
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
            _project_path(root, config["gravity_file"], "gravity_file"),
            _project_path(root, config["eop_file"], "eop_file"),
            _project_path(root, config["atmosphere_file"], "atmosphere_file"),
            root / "data/reference/gmat_r2026a/EARTH_EXPONENTIAL_PROVENANCE_1E0.json",
            prerequisite_result / "gravity_multicase_summary.json",
            prerequisite_result / "RUN_MANIFEST.json",
        }
    )
    members.update(
        _project_path(root, item["path"], f"{item['case_id']}_{item['model_id']}_ephemeris")
        for item in closure_record["ephemerides"]
    )
    members.update(path for path in prerequisite_result.rglob("*") if path.is_file())
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
            raise RuntimeError(f"Created 1E.0 result ZIP failed at {bad}.")
    return archive
