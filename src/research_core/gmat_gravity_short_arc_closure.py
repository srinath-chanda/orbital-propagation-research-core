"""Checksum-gated closure of the successful 1D.1 short-arc validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .external_validation import parse_stk_time_pos_vel
from .gmat_eop import GMAT_R2026A_EOP_SHA256
from .gmat_gravity_short_arc import EXPECTED_MODELS, load_gravity_short_arc_config
from .gravity_harmonics import CofGravityField


CLOSURE_SCHEMA = "1D.1-closure"


@dataclass(frozen=True)
class GravityShortArcClosure:
    closure_id: str
    experiment_id: str
    model_count: int
    check_count: int
    sample_count_per_model: int
    maximum_position_difference_m: float
    maximum_velocity_difference_mm_s: float
    official_result_directory: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def verify_gravity_short_arc_closure(
    closure_path: str | Path,
    *,
    project_root: str | Path,
) -> GravityShortArcClosure:
    """Verify the official result, raw ephemerides, model pass, and manifest."""
    root = Path(project_root).resolve()
    record = json.loads(Path(closure_path).read_text(encoding="utf-8"))
    if record.get("schema_version") != CLOSURE_SCHEMA:
        raise ValueError(f"Closure schema must be {CLOSURE_SCHEMA!r}.")

    config_path = _project_path(root, record["configuration"], "configuration")
    gravity_path = _project_path(root, record["gravity_file"], "gravity_file")
    eop_path = _project_path(root, record["eop_file"], "eop_file")
    result_directory = _project_path(
        root, record["official_result_directory"], "official_result_directory"
    )
    summary_path = result_directory / "gravity_short_arc_summary.json"
    manifest_path = result_directory / "RUN_MANIFEST.json"
    for path in (config_path, gravity_path, eop_path, summary_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required 1D.1 evidence is missing: {path}")

    expected_hashes = (
        (config_path, record["configuration_sha256"]),
        (gravity_path, record["gravity_file_sha256"]),
        (eop_path, record["eop_file_sha256"]),
        (summary_path, record["official_summary_sha256"]),
    )
    for path, expected in expected_hashes:
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"1D.1 evidence checksum mismatch for {path.name}: "
                f"expected {expected}, found {actual}."
            )
    if record["eop_file_sha256"] != GMAT_R2026A_EOP_SHA256:
        raise ValueError("The 1D.1 closure does not reference the adopted R2026a EOP file.")
    field = CofGravityField.from_file(gravity_path)
    if (field.maximum_degree, field.maximum_order) != (70, 70):
        raise ValueError("The frozen JGM2 evidence is not degree/order 70/70.")
    load_gravity_short_arc_config(config_path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    requirements = record["requirements"]
    scalar_requirements = {
        "status": requirements["status"],
        "decision": requirements["decision"],
        "model_count": int(requirements["model_count"]),
        "passed_model_count": int(requirements["passed_model_count"]),
        "check_count": int(requirements["check_count"]),
        "passed_check_count": int(requirements["passed_check_count"]),
    }
    for key, expected in scalar_requirements.items():
        if summary.get(key) != expected:
            raise ValueError(f"The official 1D.1 {key} does not satisfy closure.")

    expected_models = {item[0]: item[1:] for item in EXPECTED_MODELS}
    models = summary.get("models", [])
    if len(models) != len(expected_models):
        raise ValueError("The official 1D.1 model set is incomplete.")
    if requirements["all_models_passed"] and not all(
        item.get("status") == "passed" for item in models
    ):
        raise ValueError("Not every 1D.1 model passed.")
    for item in models:
        model_id = str(item.get("model_id"))
        if model_id not in expected_models:
            raise ValueError(f"Unexpected 1D.1 model: {model_id}")
        degree, order = expected_models[model_id]
        if (int(item.get("degree", -1)), int(item.get("order", -1))) != (degree, order):
            raise ValueError(f"The 1D.1 {model_id} degree/order is incorrect.")
        if int(item.get("sample_count", -1)) != int(requirements["sample_count_per_model"]):
            raise ValueError(f"The 1D.1 {model_id} sample count is incorrect.")
        checks = item.get("checks", [])
        if len(checks) != 4 or not all(check.get("status") == "passed" for check in checks):
            raise ValueError(f"The 1D.1 {model_id} checks are incomplete or failed.")
        if not all(float(check["measured_value"]) <= float(check["limit"]) for check in checks):
            raise ValueError(f"The 1D.1 {model_id} reported a false pass.")

    ephemeris_records = record.get("ephemerides", [])
    if {item.get("model_id") for item in ephemeris_records} != set(expected_models):
        raise ValueError("The 1D.1 raw ephemeris evidence set is incomplete.")
    raw_states: dict[str, np.ndarray] = {}
    for item in ephemeris_records:
        model_id = str(item["model_id"])
        path = _project_path(root, item["path"], f"{model_id}_ephemeris")
        if not path.is_file() or _sha256(path) != item["sha256"]:
            raise ValueError(f"The 1D.1 {model_id} ephemeris checksum is invalid.")
        matching = next(model for model in models if model["model_id"] == model_id)
        if matching["source_ephemeris_sha256"] != item["sha256"]:
            raise ValueError(f"The 1D.1 {model_id} summary does not match its raw ephemeris.")
        history = parse_stk_time_pos_vel(path, model_name=f"closure_{model_id.lower()}")
        if history.elapsed_seconds.size != int(requirements["sample_count_per_model"]):
            raise ValueError(f"The 1D.1 {model_id} raw ephemeris sample count is incorrect.")
        raw_states[model_id] = np.column_stack((history.positions_km, history.velocities_km_s))
    if requirements["all_raw_ephemerides_distinct"]:
        identifiers = sorted(raw_states)
        for index, left in enumerate(identifiers):
            for right in identifiers[index + 1 :]:
                if np.array_equal(raw_states[left], raw_states[right]):
                    raise ValueError(f"The 1D.1 {left} and {right} ephemerides are duplicates.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest.get("files", []):
        path = result_directory / str(item["path"])
        if not path.is_file() or path.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"1D.1 result manifest size mismatch: {path.name}")
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"1D.1 result manifest hash mismatch: {path.name}")

    return GravityShortArcClosure(
        closure_id=str(record["closure_id"]),
        experiment_id=str(record["experiment_id"]),
        model_count=len(models),
        check_count=sum(len(item["checks"]) for item in models),
        sample_count_per_model=int(requirements["sample_count_per_model"]),
        maximum_position_difference_m=max(
            float(item["maximum_position_difference_m"]) for item in models
        ),
        maximum_velocity_difference_mm_s=max(
            float(item["maximum_velocity_difference_mm_s"]) for item in models
        ),
        official_result_directory=result_directory,
    )
