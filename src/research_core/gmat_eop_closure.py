"""Research Core 1C.3 evidence-gated adoption of the validated GMAT EOP model."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gmat_eop import GMAT_R2026A_EOP_SHA256, GMAT_VALIDATED_EOP_MODEL


SCHEMA_VERSION = "1C.3"
ADOPTION_DECISION = "adopt_gmat_r2026a_eop_full_as_validated_baseline"


@dataclass(frozen=True)
class GmatEopClosureResult:
    closure_id: str
    matrix_id: str
    validated_model: str
    adoption_decision: str
    case_count: int
    check_count: int
    raw_ephemeris_count: int
    manifest_record_count: int
    maximum_position_difference_m: float
    maximum_velocity_difference_mm_s: float
    official_summary: Path
    official_report: Path


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_project_path(value: str, root: Path, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"{label} must be project-relative.")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must remain inside the project root.") from exc
    return resolved


def load_adoption_record(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Adoption schema must be {SCHEMA_VERSION!r}.")
    if payload.get("status") != "validated_baseline_adopted":
        raise ValueError("Adoption record status is not closed.")
    if payload.get("validated_model") != GMAT_VALIDATED_EOP_MODEL:
        raise ValueError("Adoption record does not select the validated EOP model.")
    if payload.get("adoption_decision") != ADOPTION_DECISION:
        raise ValueError("Adoption decision is not the accepted 1C.2 decision.")
    requirements = payload.get("requirements")
    if not isinstance(requirements, dict) or not requirements:
        raise ValueError("Adoption requirements must be a non-empty object.")
    if not all(value is True for value in requirements.values()):
        raise ValueError("Every adoption safeguard must be true.")
    return payload


def _verify_manifest(manifest_path: Path, root: Path) -> int:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Manifest contains no file inventory: {manifest_path}")
    base = manifest_path.parent.resolve()
    checked = 0
    seen: set[str] = set()
    for record in records:
        relative = str(record.get("path", ""))
        if not relative or relative in seen:
            raise ValueError(f"Manifest has an invalid or duplicate path: {relative!r}")
        seen.add(relative)
        target = (base / relative).resolve()
        try:
            target.relative_to(base)
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Manifest path escapes the evidence root: {relative}") from exc
        if not target.is_file():
            raise FileNotFoundError(f"Manifest evidence is missing: {target}")
        if target.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"Manifest size mismatch: {target}")
        if _sha256(target) != str(record["sha256"]):
            raise ValueError(f"Manifest checksum mismatch: {target}")
        checked += 1
    return checked


def verify_gmat_eop_adoption(
    adoption_path: str | Path, *, project_root: str | Path
) -> GmatEopClosureResult:
    root = Path(project_root).resolve()
    adoption = load_adoption_record(adoption_path)
    summary_path = _resolve_project_path(
        str(adoption["official_summary"]), root, "official_summary"
    )
    run_manifest_path = _resolve_project_path(
        str(adoption["official_run_manifest"]), root, "official_run_manifest"
    )
    result_dir = _resolve_project_path(
        str(adoption["official_result_directory"]), root, "official_result_directory"
    )
    eop_path = _resolve_project_path(
        str(adoption["eop_source"]), root, "eop_source"
    )
    if _sha256(summary_path) != str(adoption["official_summary_sha256"]):
        raise ValueError("Official 1C.2 summary checksum mismatch.")
    if _sha256(run_manifest_path) != str(adoption["official_run_manifest_sha256"]):
        raise ValueError("Official 1C.2 run-manifest checksum mismatch.")
    if _sha256(eop_path) != GMAT_R2026A_EOP_SHA256:
        raise ValueError("Frozen validated EOP checksum mismatch.")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "matrix_id": adoption["matrix_id"],
        "validation_status": "passed_with_warnings",
        "adoption_decision": ADOPTION_DECISION,
        "candidate_model": GMAT_VALIDATED_EOP_MODEL,
        "case_count": int(adoption["required_case_count"]),
        "passed_case_count": int(adoption["required_case_count"]),
        "failed_case_count": 0,
        "incomplete_case_count": 0,
        "thresholds_preregistered": True,
        "thresholds_relaxed_after_results": False,
        "eop_source_sha256": GMAT_R2026A_EOP_SHA256,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(
                f"Official 1C.2 summary field {key!r} is invalid: "
                f"expected {value!r}, found {summary.get(key)!r}."
            )
    cases = summary.get("cases")
    if not isinstance(cases, list) or len(cases) != expected["case_count"]:
        raise ValueError("Official 1C.2 case inventory is incomplete.")

    case_ids: set[str] = set()
    raw_sources: set[Path] = set()
    total_checks = 0
    maximum_position = 0.0
    maximum_velocity = 0.0
    manifest_records = _verify_manifest(run_manifest_path, root)
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in case_ids:
            raise ValueError(f"Invalid or duplicate official case ID: {case_id!r}")
        case_ids.add(case_id)
        if case.get("status") != "passed_with_warnings":
            raise ValueError(f"Official case is not passed: {case_id}")
        if int(case.get("failed_check_count", -1)) != 0:
            raise ValueError(f"Official case contains a failed check: {case_id}")
        case_summary_path = result_dir / "cases" / case_id / "case_validation_summary.json"
        case_manifest_path = result_dir / "cases" / case_id / "RUN_MANIFEST.json"
        case_summary = json.loads(case_summary_path.read_text(encoding="utf-8"))
        if case_summary.get("status") != "passed_with_warnings":
            raise ValueError(f"Saved case summary is not passed: {case_id}")
        checks = case_summary.get("checks")
        if not isinstance(checks, list) or any(
            item.get("status") != "passed" for item in checks
        ):
            raise ValueError(f"Saved case checks are incomplete or failed: {case_id}")
        if len(checks) != int(case.get("check_count", -1)):
            raise ValueError(f"Saved case check count differs from aggregate: {case_id}")
        total_checks += len(checks)
        manifest_records += _verify_manifest(case_manifest_path, root)
        sources = case_summary.get("source_files", {})
        for key in ("configuration", "gmat_two_body", "gmat_j2"):
            source_path = _resolve_project_path(str(sources[key]), root, key)
            if _sha256(source_path) != str(sources[f"{key}_sha256"]):
                raise ValueError(f"Official source checksum mismatch: {source_path}")
            if key.startswith("gmat_"):
                raw_sources.add(source_path)
        maximum_position = max(
            maximum_position,
            float(case["candidate_maximum_position_difference_m"]),
        )
        maximum_velocity = max(
            maximum_velocity,
            float(case["candidate_maximum_velocity_difference_mm_s"]),
        )

    if total_checks != int(adoption["required_check_count"]):
        raise ValueError("Official 1C.2 validation-check inventory is incomplete.")
    if len(raw_sources) != int(adoption["required_raw_ephemeris_count"]):
        raise ValueError("Official raw GMAT ephemeris inventory is incomplete.")
    if maximum_position != float(
        adoption["observed_maximum_candidate_position_difference_m"]
    ):
        raise ValueError("Official maximum candidate position residual changed.")
    if maximum_velocity != float(
        adoption["observed_maximum_candidate_velocity_difference_mm_s"]
    ):
        raise ValueError("Official maximum candidate velocity residual changed.")
    report_path = result_dir / "GMAT_EOP_1C2_INDEPENDENT_REPORT.html"
    if not report_path.is_file():
        raise FileNotFoundError("Official 1C.2 HTML report is missing.")

    return GmatEopClosureResult(
        closure_id=str(adoption["closure_id"]),
        matrix_id=str(adoption["matrix_id"]),
        validated_model=GMAT_VALIDATED_EOP_MODEL,
        adoption_decision=ADOPTION_DECISION,
        case_count=len(cases),
        check_count=total_checks,
        raw_ephemeris_count=len(raw_sources),
        manifest_record_count=manifest_records,
        maximum_position_difference_m=maximum_position,
        maximum_velocity_difference_mm_s=maximum_velocity,
        official_summary=summary_path,
        official_report=report_path,
    )
