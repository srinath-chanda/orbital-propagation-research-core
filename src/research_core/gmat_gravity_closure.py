"""Evidence gate closing 1D.0 before 1D.1 short-arc work begins."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .gmat_gravity_ladder import (
    load_gravity_ladder_config,
    parse_gravity_ladder_report,
)
from .gravity_harmonics import CofGravityField


CLOSURE_SCHEMA = "1D.0-closure"


@dataclass(frozen=True)
class GravityLadderClosure:
    closure_id: str
    experiment_id: str
    sample_count: int
    model_count: int
    largest_model_difference_km_s2: float
    smallest_adjacent_physical_increment_km_s2: float
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


def verify_gravity_ladder_closure(
    closure_path: str | Path,
    *,
    project_root: str | Path,
) -> GravityLadderClosure:
    """Verify hashes, pass decision, raw samples, and distinct gravity levels."""
    root = Path(project_root).resolve()
    record = json.loads(Path(closure_path).read_text(encoding="utf-8"))
    if record.get("schema_version") != CLOSURE_SCHEMA:
        raise ValueError(f"Closure schema must be {CLOSURE_SCHEMA!r}.")
    configuration = _project_path(root, record["configuration"], "configuration")
    raw_report = _project_path(root, record["raw_report"], "raw_report")
    gravity_file = _project_path(root, record["gravity_file"], "gravity_file")
    result_directory = _project_path(
        root, record["official_result_directory"], "official_result_directory"
    )
    summary_path = result_directory / "gravity_ladder_summary.json"
    manifest_path = result_directory / "RUN_MANIFEST.json"
    for path in (configuration, raw_report, gravity_file, summary_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required 1D.0 evidence is missing: {path}")
    expected_hashes = (
        (raw_report, record["raw_report_sha256"]),
        (gravity_file, record["gravity_file_sha256"]),
        (summary_path, record["official_summary_sha256"]),
    )
    for path, expected in expected_hashes:
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"1D.0 evidence checksum mismatch for {path.name}: "
                f"expected {expected}, found {actual}."
            )

    field = CofGravityField.from_file(gravity_file)
    if (field.maximum_degree, field.maximum_order) != (70, 70):
        raise ValueError("The frozen JGM2 evidence is not degree/order 70/70.")
    config = load_gravity_ladder_config(configuration)
    elapsed, _positions, accelerations = parse_gravity_ladder_report(raw_report, config)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    requirements = record["requirements"]
    if summary.get("status") != requirements["status"]:
        raise ValueError("The official 1D.0 status does not satisfy closure.")
    if summary.get("decision") != requirements["decision"]:
        raise ValueError("The official 1D.0 decision does not authorize 1D.1.")
    if int(summary.get("sample_count", -1)) != int(requirements["sample_count"]):
        raise ValueError("The official 1D.0 sample count is incorrect.")
    models = summary.get("models", [])
    if len(models) != int(requirements["model_count"]):
        raise ValueError("The official 1D.0 model count is incorrect.")
    if requirements["all_models_passed"] and not all(item.get("passed") for item in models):
        raise ValueError("Not every 1D.0 gravity level passed.")

    aliases = [str(item["alias"]) for item in config["ladder"]]
    adjacent = [
        float(np.max(np.linalg.norm(accelerations[right] - accelerations[left], axis=1)))
        for left, right in zip(aliases[:-1], aliases[1:])
    ]
    if requirements["all_adjacent_gravity_levels_distinct"] and not all(
        value > 1.0e-12 for value in adjacent
    ):
        raise ValueError("The raw GMAT report contains a duplicate gravity level.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest.get("files", []):
        path = result_directory / str(item["path"])
        if not path.is_file() or path.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"1D.0 result manifest size mismatch: {path.name}")
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"1D.0 result manifest hash mismatch: {path.name}")

    largest = max(float(item["maximum_difference_km_s2"]) for item in models)
    return GravityLadderClosure(
        closure_id=str(record["closure_id"]),
        experiment_id=str(record["experiment_id"]),
        sample_count=int(elapsed.size),
        model_count=len(models),
        largest_model_difference_km_s2=largest,
        smallest_adjacent_physical_increment_km_s2=min(adjacent),
        official_result_directory=result_directory,
    )
