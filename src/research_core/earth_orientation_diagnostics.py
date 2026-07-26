"""Research Core 1C Earth-orientation residual attribution."""

from __future__ import annotations

import csv
import hashlib
import html
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis.comparison import compare_state_histories, create_error_summary
from .analysis.j2 import compare_in_reference_rtn
from .earth_orientation import (
    POLE_MODEL_DESCRIPTIONS,
    SUPPORTED_POLE_MODELS,
    earth_pole_unit_vector,
    pole_angular_separation_arcsec,
)
from .external_validation import (
    _canonicalize_nominal_output_grid,
    initial_state_from_config,
    parse_stk_time_pos_vel,
)
from .gmat_multicase import load_gmat_matrix_spec
from .propagators.numerical_j2 import propagate_numerical_j2_orientation_model


DIAGNOSTIC_SCHEMA_VERSION = "1C.0"


@dataclass(frozen=True)
class EarthOrientationDiagnosticResult:
    result_directory: Path
    diagnostic_status: str
    decision: str
    baseline_model: str
    recommended_model: str
    case_count: int
    model_count: int
    summary_json: Path
    report_path: Path
    created_files: tuple[Path, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(payload: Any, path: Path) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _resolve_project_path(value: str, root: Path, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"{label} must be project-relative.")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the project root.") from exc
    return resolved


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def load_earth_orientation_diagnostic_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != DIAGNOSTIC_SCHEMA_VERSION:
        raise ValueError(
            f"Earth-orientation diagnostic schema must be {DIAGNOSTIC_SCHEMA_VERSION!r}."
        )
    models = payload.get("models")
    if not isinstance(models, list) or len(models) < 2:
        raise ValueError("The diagnostic requires at least two orientation models.")
    identifiers = [str(model.get("model_id")) for model in models]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Orientation model identifiers must be unique.")
    unsupported = sorted(set(identifiers) - set(SUPPORTED_POLE_MODELS))
    if unsupported:
        raise ValueError(f"Unsupported orientation models: {', '.join(unsupported)}.")
    baseline = str(payload.get("baseline_model"))
    if baseline not in identifiers:
        raise ValueError("baseline_model must appear in models.")
    rule = payload.get("decision_rule", {})
    for name in (
        "maximum_median_position_ratio",
        "maximum_worst_position_ratio",
        "maximum_median_velocity_ratio",
        "maximum_worst_velocity_ratio",
    ):
        value = float(rule.get(name, -1.0))
        if value <= 0.0:
            raise ValueError(f"decision_rule.{name} must be positive.")
    return payload


def _maximum_absolute(values: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(values, dtype=float))))


def _ratio_to_baseline(value: float, baseline: float) -> float:
    """Return a finite diagnostic ratio, including an exact-zero baseline."""
    if baseline > 0.0:
        return value / baseline
    if value == 0.0:
        return 1.0
    return float(np.finfo(float).max)


def _case_model_row(
    *,
    case: dict[str, Any],
    model: dict[str, Any],
    comparison: dict[str, Any],
    rtn: dict[str, Any],
    thresholds: dict[str, Any],
    pole_start_arcsec: float,
    pole_end_arcsec: float,
    runtime_seconds: float,
) -> dict[str, Any]:
    summary = create_error_summary(comparison)
    position_max = float(
        summary["position_difference_m"]["maximum_absolute"]
    )
    velocity_max = float(
        summary["velocity_difference_mm_s"]["maximum_absolute"]
    )
    position_gate = float(thresholds["j2_maximum_position_difference_m"])
    velocity_gate = float(thresholds["j2_maximum_velocity_difference_mm_s"])
    return {
        "case_id": case["case_id"],
        "factor": case["factor"],
        "epoch_utc": case["epoch_utc"],
        "altitude_km": float(case["altitude_km"]),
        "inclination_deg": float(case["inclination_deg"]),
        "duration_hours": float(case["duration_hours"]),
        "model_id": model["model_id"],
        "role": model["role"],
        "eligible_candidate": bool(model["eligible_candidate"]),
        "maximum_position_difference_m": position_max,
        "maximum_velocity_difference_mm_s": velocity_max,
        "final_position_difference_m": float(
            summary["position_difference_m"]["final"]
        ),
        "final_velocity_difference_mm_s": float(
            summary["velocity_difference_mm_s"]["final"]
        ),
        "maximum_radial_position_difference_m": _maximum_absolute(
            rtn["radial_position_difference_m"]
        ),
        "maximum_along_track_position_difference_m": _maximum_absolute(
            rtn["along_track_position_difference_m"]
        ),
        "maximum_cross_track_position_difference_m": _maximum_absolute(
            rtn["cross_track_position_difference_m"]
        ),
        "maximum_radial_velocity_difference_mm_s": _maximum_absolute(
            rtn["radial_velocity_difference_mm_s"]
        ),
        "maximum_along_track_velocity_difference_mm_s": _maximum_absolute(
            rtn["along_track_velocity_difference_mm_s"]
        ),
        "maximum_cross_track_velocity_difference_mm_s": _maximum_absolute(
            rtn["cross_track_velocity_difference_mm_s"]
        ),
        "position_gate_m": position_gate,
        "velocity_gate_mm_s": velocity_gate,
        "position_gate_passed": position_max <= position_gate,
        "velocity_gate_passed": velocity_max <= velocity_gate,
        "pole_separation_from_baseline_start_arcsec": pole_start_arcsec,
        "pole_separation_from_baseline_end_arcsec": pole_end_arcsec,
        "runtime_seconds": runtime_seconds,
    }


def _aggregate_models(
    rows: list[dict[str, Any]],
    model_specs: list[dict[str, Any]],
    baseline_model: str,
    descriptions: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    description_lookup = descriptions or POLE_MODEL_DESCRIPTIONS
    aggregates: list[dict[str, Any]] = []
    for model in model_specs:
        model_id = str(model["model_id"])
        selected = [row for row in rows if row["model_id"] == model_id]
        position = np.asarray(
            [row["maximum_position_difference_m"] for row in selected], dtype=float
        )
        velocity = np.asarray(
            [row["maximum_velocity_difference_mm_s"] for row in selected], dtype=float
        )
        worst_index = int(np.argmax(position))
        aggregates.append(
            {
                "model_id": model_id,
                "description": description_lookup[model_id],
                "role": model["role"],
                "eligible_candidate": bool(model["eligible_candidate"]),
                "case_count": len(selected),
                "all_existing_gates_passed": all(
                    row["position_gate_passed"] and row["velocity_gate_passed"]
                    for row in selected
                ),
                "median_case_maximum_position_difference_m": float(
                    np.median(position)
                ),
                "worst_case_maximum_position_difference_m": float(np.max(position)),
                "median_case_maximum_velocity_difference_mm_s": float(
                    np.median(velocity)
                ),
                "worst_case_maximum_velocity_difference_mm_s": float(
                    np.max(velocity)
                ),
                "worst_position_case_id": selected[worst_index]["case_id"],
                "maximum_pole_separation_from_baseline_arcsec": float(
                    max(
                        max(
                            row["pole_separation_from_baseline_start_arcsec"],
                            row["pole_separation_from_baseline_end_arcsec"],
                        )
                        for row in selected
                    )
                ),
                "total_runtime_seconds": float(
                    sum(row["runtime_seconds"] for row in selected)
                ),
            }
        )
    baseline = next(item for item in aggregates if item["model_id"] == baseline_model)
    for item in aggregates:
        item["median_position_ratio_to_baseline"] = _ratio_to_baseline(
            item["median_case_maximum_position_difference_m"],
            baseline["median_case_maximum_position_difference_m"],
        )
        item["worst_position_ratio_to_baseline"] = _ratio_to_baseline(
            item["worst_case_maximum_position_difference_m"],
            baseline["worst_case_maximum_position_difference_m"],
        )
        item["median_velocity_ratio_to_baseline"] = _ratio_to_baseline(
            item["median_case_maximum_velocity_difference_mm_s"],
            baseline["median_case_maximum_velocity_difference_mm_s"],
        )
        item["worst_velocity_ratio_to_baseline"] = _ratio_to_baseline(
            item["worst_case_maximum_velocity_difference_mm_s"],
            baseline["worst_case_maximum_velocity_difference_mm_s"],
        )
    return aggregates


def _apply_decision_rule(
    aggregates: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    baseline = str(config["baseline_model"])
    rule = config["decision_rule"]
    qualifying = []
    for item in aggregates:
        if not item["eligible_candidate"] or item["model_id"] == baseline:
            continue
        passed = (
            item["all_existing_gates_passed"]
            and item["median_position_ratio_to_baseline"]
            <= float(rule["maximum_median_position_ratio"])
            and item["worst_position_ratio_to_baseline"]
            <= float(rule["maximum_worst_position_ratio"])
            and item["median_velocity_ratio_to_baseline"]
            <= float(rule["maximum_median_velocity_ratio"])
            and item["worst_velocity_ratio_to_baseline"]
            <= float(rule["maximum_worst_velocity_ratio"])
        )
        item["decision_rule_passed"] = passed
        if passed:
            qualifying.append(item)
    for item in aggregates:
        item.setdefault("decision_rule_passed", False)
    if not qualifying:
        return {
            "decision": "baseline_retained_no_candidate_met_preregistered_rule",
            "recommended_model": baseline,
            "candidate_requires_independent_validation": False,
            "qualifying_candidates": [],
        }
    winner = min(
        qualifying,
        key=lambda item: (
            item["median_position_ratio_to_baseline"],
            item["median_velocity_ratio_to_baseline"],
        ),
    )
    return {
        "decision": "candidate_identified_requires_independent_validation",
        "recommended_model": winner["model_id"],
        "candidate_requires_independent_validation": True,
        "qualifying_candidates": [item["model_id"] for item in qualifying],
    }


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_figures(
    figures_dir: Path,
    rows: list[dict[str, Any]],
    model_specs: list[dict[str, Any]],
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    case_ids = list(dict.fromkeys(row["case_id"] for row in rows))
    short_ids = [case_id.split("_")[0] for case_id in case_ids]
    created: list[Path] = []

    figure, axis = plt.subplots(figsize=(12, 6))
    for model in model_specs:
        model_id = model["model_id"]
        selected = [row for row in rows if row["model_id"] == model_id]
        axis.plot(
            range(len(case_ids)),
            [row["maximum_position_difference_m"] for row in selected],
            marker="o",
            linewidth=1.5,
            label=model_id,
        )
    axis.set_yscale("log")
    axis.set_xticks(range(len(case_ids)), short_ids)
    axis.set_xlabel("Matrix case")
    axis.set_ylabel("Maximum position difference (m, log scale)")
    axis.set_title("GMAT residual by Earth-orientation realization")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        path = figures_dir / f"orientation_model_position_residual.{suffix}"
        figure.savefig(path, dpi=180 if suffix == "png" else None)
        created.append(path)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    model_ids = [model["model_id"] for model in model_specs]
    values = []
    for model_id in model_ids:
        selected = [row for row in rows if row["model_id"] == model_id]
        values.append(
            max(
                max(
                    row["pole_separation_from_baseline_start_arcsec"],
                    row["pole_separation_from_baseline_end_arcsec"],
                )
                for row in selected
            )
        )
    axis.bar(range(len(model_ids)), values, color="#4267b2")
    axis.set_xticks(range(len(model_ids)), model_ids, rotation=20, ha="right")
    axis.set_ylabel("Maximum pole separation from baseline (arcsec)")
    axis.set_title("Orientation-model pole separation")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        path = figures_dir / f"orientation_model_pole_separation.{suffix}"
        figure.savefig(path, dpi=180 if suffix == "png" else None)
        created.append(path)
    plt.close(figure)
    return created


def _report_html(
    *,
    config: dict[str, Any],
    aggregates: list[dict[str, Any]],
    decision: dict[str, Any],
) -> str:
    rows = []
    for item in aggregates:
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['model_id'])}</td>"
            f"<td>{html.escape(item['role'])}</td>"
            f"<td>{item['median_case_maximum_position_difference_m']:.6g}</td>"
            f"<td>{item['worst_case_maximum_position_difference_m']:.6g}</td>"
            f"<td>{item['median_case_maximum_velocity_difference_mm_s']:.6g}</td>"
            f"<td>{item['worst_case_maximum_velocity_difference_mm_s']:.6g}</td>"
            f"<td>{item['median_position_ratio_to_baseline']:.4f}</td>"
            f"<td>{'yes' if item['decision_rule_passed'] else 'no'}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Research Core 1C Earth-Orientation Diagnostic</title>
<style>body{{font-family:Arial,sans-serif;margin:2rem;color:#172033}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd3df;
padding:.45rem;text-align:right}}th:first-child,td:first-child,
th:nth-child(2),td:nth-child(2){{text-align:left}}th{{background:#edf2f8}}</style>
</head><body><h1>Research Core 1C Earth-Orientation Diagnostic</h1>
<p><strong>Experiment:</strong> {html.escape(str(config['experiment_id']))}</p>
<p><strong>Status:</strong> diagnostic completed</p>
<p><strong>Decision:</strong> {html.escape(decision['decision'])}</p>
<p><strong>Recommended model:</strong> {html.escape(decision['recommended_model'])}</p>
<table><thead><tr><th>Model</th><th>Role</th><th>Median position m</th>
<th>Worst position m</th><th>Median velocity mm/s</th><th>Worst velocity mm/s</th>
<th>Median position ratio</th><th>Rule passed</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Interpretation</h2><p>Alternative models are evaluated on the same evidence
used for diagnosis. A selected candidate would require a new independent
validation run before replacing the closed 1B baseline.</p>
<p>Fixed and precession-only models are ablations, not replacement candidates.
Polar motion and observed EOP corrections are not included in this diagnostic.</p>
<img src="figures/orientation_model_position_residual.png" style="max-width:100%" alt="Residual comparison">
<img src="figures/orientation_model_pole_separation.png" style="max-width:100%" alt="Pole separation">
</body></html>"""


def run_earth_orientation_diagnostics(
    config_path: str | Path,
    *,
    project_root: str | Path,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
) -> EarthOrientationDiagnosticResult:
    root = Path(project_root).resolve()
    diagnostic_path = Path(config_path).resolve()
    config = load_earth_orientation_diagnostic_config(diagnostic_path)
    matrix_path = _resolve_project_path(
        str(config["matrix_specification"]), root, "matrix_specification"
    )
    matrix = load_gmat_matrix_spec(matrix_path)
    reference_root = _resolve_project_path(
        str(matrix["reference_root"]), root, "matrix.reference_root"
    )
    models = config["models"]
    baseline_model = str(config["baseline_model"])
    rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    total_comparisons = len(matrix["cases"]) * len(models)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")
    result_dir = root / "results" / str(config["experiment_id"]) / stamp
    result_dir.mkdir(parents=True, exist_ok=False)
    created: list[Path] = []

    for case in matrix["cases"]:
        case_id = str(case["case_id"])
        case_config_path = reference_root / "cases" / f"{case_id}.json"
        gmat_path = reference_root / "output" / f"{case_id}_J2.e"
        case_config = json.loads(case_config_path.read_text(encoding="utf-8"))
        frame = str(case_config["external_validation"]["frame"])
        expected_step = float(
            case_config["external_validation"]["output_step_seconds"]
        )
        duration = float(case_config["external_validation"]["duration_seconds"])
        gmat = parse_stk_time_pos_vel(gmat_path, model_name="gmat_j2", frame=frame)
        gmat, _ = _canonicalize_nominal_output_grid(
            gmat,
            expected_step_seconds=expected_step,
            expected_duration_seconds=duration,
            tolerance_seconds=1.0e-6,
        )
        initial_state = initial_state_from_config(case_config)
        earth = case_config["earth_model"]
        integrator = case_config["integrator"]
        baseline_start = earth_pole_unit_vector(
            initial_state.epoch_utc, 0.0, baseline_model
        )
        baseline_end = earth_pole_unit_vector(
            initial_state.epoch_utc, duration, baseline_model
        )
        evidence.append(
            {
                "case_id": case_id,
                "configuration": _relative(case_config_path, root),
                "configuration_sha256": _sha256(case_config_path),
                "gmat_j2_ephemeris": _relative(gmat_path, root),
                "gmat_j2_ephemeris_sha256": _sha256(gmat_path),
            }
        )
        for model in models:
            model_id = str(model["model_id"])
            history = propagate_numerical_j2_orientation_model(
                initial_state,
                float(earth["gravitational_parameter_km3_s2"]),
                float(earth["equatorial_radius_km"]),
                float(earth["j2"]),
                gmat.elapsed_seconds,
                orientation_model=model_id,
                method=str(integrator["method"]),
                relative_tolerance=float(integrator["relative_tolerance"]),
                absolute_tolerance=float(integrator["absolute_tolerance"]),
                maximum_step_seconds=float(integrator["maximum_step_seconds"]),
            )
            comparison = compare_state_histories(gmat, history)
            rtn = compare_in_reference_rtn(gmat, history)
            model_start = earth_pole_unit_vector(
                initial_state.epoch_utc, 0.0, model_id
            )
            model_end = earth_pole_unit_vector(
                initial_state.epoch_utc, duration, model_id
            )
            rows.append(
                _case_model_row(
                    case=case,
                    model=model,
                    comparison=comparison,
                    rtn=rtn,
                    thresholds=case_config["external_validation"]["thresholds"],
                    pole_start_arcsec=pole_angular_separation_arcsec(
                        baseline_start, model_start
                    ),
                    pole_end_arcsec=pole_angular_separation_arcsec(
                        baseline_end, model_end
                    ),
                    runtime_seconds=history.runtime_seconds,
                )
            )
            if progress_callback is not None:
                progress_callback(
                    len(rows), total_comparisons, case_id, model_id
                )

    aggregates = _aggregate_models(rows, models, baseline_model)
    decision = _apply_decision_rule(aggregates, config)
    summary_csv = result_dir / "earth_orientation_case_model_summary.csv"
    _write_rows_csv(summary_csv, rows)
    created.append(summary_csv)
    summary = {
        "research_core_version": "1C.0",
        "experiment_id": config["experiment_id"],
        "diagnostic_status": "diagnostic_completed_with_warnings",
        "case_count": len(matrix["cases"]),
        "model_count": len(models),
        "comparison_count": len(rows),
        "baseline_model": baseline_model,
        "decision_rule_preregistered": True,
        "decision_rule": config["decision_rule"],
        **decision,
        "model_aggregates": aggregates,
        "case_model_results": rows,
        "source_files": {
            "diagnostic_configuration": _relative(diagnostic_path, root),
            "diagnostic_configuration_sha256": _sha256(diagnostic_path),
            "matrix_specification": _relative(matrix_path, root),
            "matrix_specification_sha256": _sha256(matrix_path),
            "cases": evidence,
        },
        "warnings": [
            "This is residual attribution on existing validation evidence, not independent candidate validation.",
            "The closed 1B IAU-1976/1980 baseline remains authoritative unless a candidate passes a new validation gate.",
            "Polar motion, observed EOP corrections, and subdaily terms are not evaluated here.",
            "Model agreement is not measured-orbit truth or flight qualification.",
        ],
    }
    summary_json = result_dir / "earth_orientation_diagnostic_summary.json"
    _write_json(summary, summary_json)
    created.append(summary_json)
    created.extend(_save_figures(result_dir / "figures", rows, models))
    report = result_dir / "EARTH_ORIENTATION_DIAGNOSTIC_REPORT.html"
    report.write_text(
        _report_html(config=config, aggregates=aggregates, decision=decision),
        encoding="utf-8",
        newline="\n",
    )
    created.append(report)
    manifest_path = result_dir / "RUN_MANIFEST.json"
    _write_json(
        {
            "files": [
                {
                    "path": path.relative_to(result_dir).as_posix(),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(created)
            ]
        },
        manifest_path,
    )
    created.append(manifest_path)
    return EarthOrientationDiagnosticResult(
        result_directory=result_dir,
        diagnostic_status="diagnostic_completed_with_warnings",
        decision=str(decision["decision"]),
        baseline_model=baseline_model,
        recommended_model=str(decision["recommended_model"]),
        case_count=len(matrix["cases"]),
        model_count=len(models),
        summary_json=summary_json,
        report_path=report,
        created_files=tuple(created),
    )
