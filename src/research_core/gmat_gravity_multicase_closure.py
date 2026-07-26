"""Evidence gate closing the 1D.2 gravity matrix before drag validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CLOSURE_SCHEMA = "1D.2-closure"


@dataclass(frozen=True)
class GravityMulticaseClosure:
    """Verified summary of the closed higher-order gravity campaign."""

    closure_id: str
    experiment_id: str
    case_count: int
    model_run_count: int
    check_count: int
    maximum_position_difference_m: float
    maximum_velocity_difference_mm_s: float
    maximum_time_residual_seconds: float
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


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(
            f"The official 1D.2 {label} is {actual!r}; expected {expected!r}."
        )


def verify_gravity_multicase_closure(
    closure_path: str | Path,
    *,
    project_root: str | Path,
) -> GravityMulticaseClosure:
    """Verify all frozen 1D.2 outputs, results, counts, and the pass decision."""
    root = Path(project_root).resolve()
    record = json.loads(Path(closure_path).read_text(encoding="utf-8"))
    if record.get("schema_version") != CLOSURE_SCHEMA:
        raise ValueError(f"Closure schema must be {CLOSURE_SCHEMA!r}.")

    configuration = _project_path(root, record["configuration"], "configuration")
    gravity_file = _project_path(root, record["gravity_file"], "gravity_file")
    eop_file = _project_path(root, record["eop_file"], "eop_file")
    result_directory = _project_path(
        root, record["official_result_directory"], "official_result_directory"
    )
    summary_path = result_directory / "gravity_multicase_summary.json"
    manifest_path = result_directory / "RUN_MANIFEST.json"
    required = (configuration, gravity_file, eop_file, summary_path, manifest_path)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required 1D.2 evidence is missing: {path}")

    expected_hashes = (
        (configuration, record["configuration_sha256"]),
        (gravity_file, record["gravity_file_sha256"]),
        (eop_file, record["eop_file_sha256"]),
        (summary_path, record["official_summary_sha256"]),
        (manifest_path, record["official_manifest_sha256"]),
    )
    for path, expected in expected_hashes:
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"1D.2 evidence checksum mismatch for {path.name}: "
                f"expected {expected}, found {actual}."
            )

    ephemerides = record.get("ephemerides", [])
    requirements = record["requirements"]
    if len(ephemerides) != int(requirements["model_run_count"]):
        raise ValueError("The 1D.2 closure does not list every raw ephemeris.")
    hashes_by_case: dict[str, set[str]] = {}
    for item in ephemerides:
        path = _project_path(root, item["path"], "1D.2 ephemeris")
        if not path.is_file():
            raise FileNotFoundError(f"Required 1D.2 ephemeris is missing: {path}")
        actual = _sha256(path)
        if actual != item["sha256"]:
            raise ValueError(f"1D.2 ephemeris checksum mismatch: {path.name}")
        hashes_by_case.setdefault(str(item["case_id"]), set()).add(actual)
    if requirements["all_four_models_distinct_per_case"] and not all(
        len(values) == 4 for values in hashes_by_case.values()
    ):
        raise ValueError("A 1D.2 case contains duplicate raw model ephemerides.")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in (
        "status",
        "decision",
        "case_count",
        "passed_case_count",
        "model_run_count",
        "passed_model_run_count",
        "check_count",
        "passed_check_count",
    ):
        _require_equal(summary.get(key), requirements[key], key)
    if int(summary.get("failed_case_count", -1)) != 0:
        raise ValueError("The official 1D.2 result contains a failed case.")
    if int(summary.get("failed_model_run_count", -1)) != 0:
        raise ValueError("The official 1D.2 result contains a failed model run.")
    if int(summary.get("failed_check_count", -1)) != 0:
        raise ValueError("The official 1D.2 result contains a failed check.")

    models = [model for case in summary["cases"] for model in case["models"]]
    maximum_position = max(
        float(model["maximum_position_difference_m"]) for model in models
    )
    maximum_velocity = max(
        float(model["maximum_velocity_difference_mm_s"]) for model in models
    )
    maximum_time = max(
        float(model["grid_diagnostics"]["maximum_absolute_raw_time_residual_seconds"])
        for model in models
    )
    if maximum_time > float(requirements["maximum_time_residual_seconds"]):
        raise ValueError("A 1D.2 time-grid residual exceeds the closure limit.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest.get("files", []):
        path = result_directory / str(item["path"])
        if not path.is_file() or path.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"1D.2 result manifest size mismatch: {path}")
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"1D.2 result manifest hash mismatch: {path}")

    return GravityMulticaseClosure(
        closure_id=str(record["closure_id"]),
        experiment_id=str(record["experiment_id"]),
        case_count=int(summary["case_count"]),
        model_run_count=int(summary["model_run_count"]),
        check_count=int(summary["check_count"]),
        maximum_position_difference_m=maximum_position,
        maximum_velocity_difference_mm_s=maximum_velocity,
        maximum_time_residual_seconds=maximum_time,
        official_result_directory=result_directory,
    )
