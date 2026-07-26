"""Research Core 1C.1 GMAT R2026a EOP/polar-motion attribution."""

from __future__ import annotations

import csv
import html
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time, TimeDelta

from .analysis.comparison import compare_state_histories
from .analysis.j2 import compare_in_reference_rtn
from .earth_orientation import earth_pole_unit_vector, pole_angular_separation_arcsec
from .earth_orientation_diagnostics import (
    _aggregate_models,
    _apply_decision_rule,
    _case_model_row,
    _relative,
    _resolve_project_path,
    _sha256,
    _write_json,
)
from .external_validation import (
    _canonicalize_nominal_output_grid,
    initial_state_from_config,
    parse_stk_time_pos_vel,
)
from .gmat_eop import (
    GMAT_EOP_MODEL_DESCRIPTIONS,
    GMAT_EOP_POLE_MODELS,
    GmatEopDataset,
    gmat_r2026a_eop_pole_unit_vector,
)
from .gmat_multicase import load_gmat_matrix_spec
from .propagators.numerical_j2 import propagate_numerical_j2_pole_provider


DIAGNOSTIC_SCHEMA_VERSION = "1C.1"
BASELINE_MODEL = "iau1976_1980"
MODEL_DESCRIPTIONS = {
    BASELINE_MODEL: "Closed 1B IAU-1976/IAU-1980 baseline without polar motion",
    **GMAT_EOP_MODEL_DESCRIPTIONS,
}


@dataclass(frozen=True)
class GmatEopDiagnosticResult:
    result_directory: Path
    diagnostic_status: str
    decision: str
    baseline_model: str
    recommended_model: str
    case_count: int
    model_count: int
    comparison_count: int
    summary_json: Path
    report_path: Path
    created_files: tuple[Path, ...]


def load_gmat_eop_diagnostic_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != DIAGNOSTIC_SCHEMA_VERSION:
        raise ValueError(
            f"GMAT EOP diagnostic schema must be {DIAGNOSTIC_SCHEMA_VERSION!r}."
        )
    models = payload.get("models")
    if not isinstance(models, list) or len(models) < 2:
        raise ValueError("The GMAT EOP diagnostic requires at least two models.")
    model_ids = [str(item.get("model_id")) for item in models]
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("GMAT EOP model identifiers must be unique.")
    supported = {BASELINE_MODEL, *GMAT_EOP_POLE_MODELS}
    unsupported = sorted(set(model_ids) - supported)
    if unsupported:
        raise ValueError(f"Unsupported GMAT EOP models: {', '.join(unsupported)}.")
    baseline = str(payload.get("baseline_model"))
    if baseline != BASELINE_MODEL or baseline not in model_ids:
        raise ValueError(f"baseline_model must be {BASELINE_MODEL!r}.")
    expected_hash = str(payload.get("eop_expected_sha256", ""))
    if len(expected_hash) != 64 or any(
        character not in "0123456789abcdef" for character in expected_hash
    ):
        raise ValueError("eop_expected_sha256 must be a lowercase SHA-256 digest.")
    rule = payload.get("decision_rule", {})
    for field in (
        "maximum_median_position_ratio",
        "maximum_worst_position_ratio",
        "maximum_median_velocity_ratio",
        "maximum_worst_velocity_ratio",
    ):
        if float(rule.get(field, -1.0)) <= 0.0:
            raise ValueError(f"decision_rule.{field} must be positive.")
    return payload


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _pole_provider(
    model_id: str,
    dataset: GmatEopDataset,
) -> Callable[[str, float], np.ndarray]:
    if model_id == BASELINE_MODEL:
        return partial(earth_pole_unit_vector, model=BASELINE_MODEL)
    return partial(gmat_r2026a_eop_pole_unit_vector, dataset=dataset, model=model_id)


def _save_figures(
    figures_dir: Path,
    rows: list[dict[str, Any]],
    model_specs: list[dict[str, Any]],
    eop_cases: list[dict[str, Any]],
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    case_ids = list(dict.fromkeys(row["case_id"] for row in rows))
    short_ids = [case_id.split("_")[0] for case_id in case_ids]

    figure, axis = plt.subplots(figsize=(12, 6))
    for model in model_specs:
        model_id = str(model["model_id"])
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
    axis.set_xlabel("Saved GMAT matrix case")
    axis.set_ylabel("Maximum position difference (m, log scale)")
    axis.set_title("GMAT residual after exact tagged EOP polar motion")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        path = figures_dir / f"gmat_eop_position_residual.{suffix}"
        figure.savefig(path, dpi=180 if suffix == "png" else None)
        created.append(path)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 5))
    x_values = [item["start_sample"]["x_arcsec"] for item in eop_cases]
    y_values = [item["start_sample"]["y_arcsec"] for item in eop_cases]
    positions = np.arange(len(eop_cases), dtype=float)
    width = 0.38
    axis.bar(positions - width / 2.0, x_values, width, label="EOP x")
    axis.bar(positions + width / 2.0, y_values, width, label="EOP y")
    axis.set_xticks(positions, short_ids)
    axis.set_ylabel("Polar motion at case start (arcsec)")
    axis.set_title("Frozen GMAT R2026a EOP values used by the diagnostic")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        path = figures_dir / f"gmat_eop_case_values.{suffix}"
        figure.savefig(path, dpi=180 if suffix == "png" else None)
        created.append(path)
    plt.close(figure)
    return created


def _report_html(
    *,
    config: dict[str, Any],
    dataset: GmatEopDataset,
    aggregates: list[dict[str, Any]],
    decision: dict[str, Any],
    eop_cases: list[dict[str, Any]],
) -> str:
    aggregate_rows = []
    for item in aggregates:
        aggregate_rows.append(
            "<tr>"
            f"<td>{html.escape(item['model_id'])}</td>"
            f"<td>{item['median_case_maximum_position_difference_m']:.9g}</td>"
            f"<td>{item['worst_case_maximum_position_difference_m']:.9g}</td>"
            f"<td>{item['median_case_maximum_velocity_difference_mm_s']:.9g}</td>"
            f"<td>{item['worst_case_maximum_velocity_difference_mm_s']:.9g}</td>"
            f"<td>{item['median_position_ratio_to_baseline']:.6f}</td>"
            f"<td>{'yes' if item['decision_rule_passed'] else 'no'}</td>"
            "</tr>"
        )
    coverage_rows = []
    for item in eop_cases:
        start = item["start_sample"]
        end = item["end_sample"]
        coverage_rows.append(
            "<tr>"
            f"<td>{html.escape(item['case_id'])}</td>"
            f"<td>{start['x_arcsec']:.6f}</td>"
            f"<td>{start['y_arcsec']:.6f}</td>"
            f"<td>{html.escape(start['coverage_status'])}</td>"
            f"<td>{html.escape(end['coverage_status'])}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Research Core 1C.1 GMAT EOP Diagnostic</title>
<style>body{{font-family:Arial,sans-serif;margin:2rem;color:#172033;line-height:1.45}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #ccd3df;
padding:.45rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}
th{{background:#edf2f8}}code{{background:#f1f3f7;padding:.1rem .25rem}}</style>
</head><body><h1>Research Core 1C.1 GMAT R2026a EOP/Polar-Motion Diagnostic</h1>
<p><strong>Experiment:</strong> {html.escape(str(config['experiment_id']))}<br>
<strong>Status:</strong> diagnostic completed with warnings<br>
<strong>Decision:</strong> {html.escape(decision['decision'])}<br>
<strong>Recommended candidate:</strong> {html.escape(decision['recommended_model'])}</p>
<h2>Model comparison</h2><table><thead><tr><th>Model</th>
<th>Median position m</th><th>Worst position m</th>
<th>Median velocity mm/s</th><th>Worst velocity mm/s</th>
<th>Median position ratio</th><th>Rule passed</th></tr></thead>
<tbody>{''.join(aggregate_rows)}</tbody></table>
<h2>Frozen EOP evidence</h2>
<p>Source rows: {dataset.row_count}; MJD {dataset.first_mjd_utc:.0f} to
{dataset.last_mjd_utc:.0f}; SHA-256 <code>{dataset.source_sha256}</code>.</p>
<table><thead><tr><th>Case</th><th>Start x arcsec</th><th>Start y arcsec</th>
<th>Start coverage</th><th>End coverage</th></tr></thead>
<tbody>{''.join(coverage_rows)}</tbody></table>
<h2>Interpretation</h2>
<p>The full x/y EOP polar-motion realization is compared with the closed 1B
baseline and x-only/y-only ablations. The exact file and boundary behavior are
frozen to reproduce GMAT R2026a. The October case is beyond the tagged file and
therefore uses GMAT's documented endpoint-clamping behavior.</p>
<p>A candidate passing this diagnostic still requires a new independent GMAT
matrix before it can replace the closed 1B baseline. This is software-model
agreement, not measured-orbit truth or flight qualification.</p>
<img src="figures/gmat_eop_position_residual.png" style="max-width:100%" alt="Residual comparison">
<img src="figures/gmat_eop_case_values.png" style="max-width:100%" alt="EOP case values">
</body></html>"""


def run_gmat_eop_diagnostics(
    config_path: str | Path,
    *,
    project_root: str | Path,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
) -> GmatEopDiagnosticResult:
    root = Path(project_root).resolve()
    diagnostic_path = Path(config_path).resolve()
    config = load_gmat_eop_diagnostic_config(diagnostic_path)
    matrix_path = _resolve_project_path(
        str(config["matrix_specification"]), root, "matrix_specification"
    )
    matrix = load_gmat_matrix_spec(matrix_path)
    reference_root = _resolve_project_path(
        str(matrix["reference_root"]), root, "matrix.reference_root"
    )
    eop_path = _resolve_project_path(str(config["eop_file"]), root, "eop_file")
    provenance_path = _resolve_project_path(
        str(config["eop_provenance"]), root, "eop_provenance"
    )
    dataset = GmatEopDataset.from_file(
        eop_path, expected_sha256=str(config["eop_expected_sha256"])
    )
    models = config["models"]
    baseline_model = str(config["baseline_model"])
    total_comparisons = len(matrix["cases"]) * len(models)
    rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    eop_cases: list[dict[str, Any]] = []

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
        start_time = Time(initial_state.epoch_utc, scale="utc")
        end_time = start_time + TimeDelta(duration, format="sec")
        start_sample = dataset.sample(start_time)
        end_sample = dataset.sample(end_time)
        eop_cases.append(
            {
                "case_id": case_id,
                "epoch_utc": initial_state.epoch_utc,
                "duration_seconds": duration,
                "start_sample": asdict(start_sample),
                "end_sample": asdict(end_sample),
            }
        )
        baseline_provider = _pole_provider(baseline_model, dataset)
        baseline_start = baseline_provider(initial_state.epoch_utc, 0.0)
        baseline_end = baseline_provider(initial_state.epoch_utc, duration)
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
            provider = _pole_provider(model_id, dataset)
            history = propagate_numerical_j2_pole_provider(
                initial_state,
                float(earth["gravitational_parameter_km3_s2"]),
                float(earth["equatorial_radius_km"]),
                float(earth["j2"]),
                gmat.elapsed_seconds,
                pole_provider=provider,
                model_name=f"numerical_j2_{model_id}",
                method=str(integrator["method"]),
                relative_tolerance=float(integrator["relative_tolerance"]),
                absolute_tolerance=float(integrator["absolute_tolerance"]),
                maximum_step_seconds=float(integrator["maximum_step_seconds"]),
            )
            comparison = compare_state_histories(gmat, history)
            rtn = compare_in_reference_rtn(gmat, history)
            model_start = provider(initial_state.epoch_utc, 0.0)
            model_end = provider(initial_state.epoch_utc, duration)
            row = _case_model_row(
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
            row["eop_start_coverage_status"] = start_sample.coverage_status
            row["eop_end_coverage_status"] = end_sample.coverage_status
            rows.append(row)
            if progress_callback is not None:
                progress_callback(len(rows), total_comparisons, case_id, model_id)

    aggregates = _aggregate_models(
        rows, models, baseline_model, descriptions=MODEL_DESCRIPTIONS
    )
    decision = _apply_decision_rule(aggregates, config)
    summary_csv = result_dir / "gmat_eop_case_model_summary.csv"
    _write_rows_csv(summary_csv, rows)
    created.append(summary_csv)
    clamped_cases = [
        item["case_id"]
        for item in eop_cases
        if "clamped" in item["start_sample"]["coverage_status"]
        or "clamped" in item["end_sample"]["coverage_status"]
    ]
    placeholder_cases = [
        item["case_id"]
        for item in eop_cases
        if "placeholder" in item["start_sample"]["uncertainty_status"]
        or "placeholder" in item["end_sample"]["uncertainty_status"]
    ]
    summary = {
        "research_core_version": "1C.1",
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
        "eop_case_samples": eop_cases,
        "eop_coverage": {
            "clamped_case_ids": clamped_cases,
            "placeholder_uncertainty_case_ids": placeholder_cases,
        },
        "source_files": {
            "diagnostic_configuration": _relative(diagnostic_path, root),
            "diagnostic_configuration_sha256": _sha256(diagnostic_path),
            "matrix_specification": _relative(matrix_path, root),
            "matrix_specification_sha256": _sha256(matrix_path),
            "gmat_r2026a_eop_file": _relative(eop_path, root),
            "gmat_r2026a_eop_file_sha256": dataset.source_sha256,
            "gmat_r2026a_eop_file_size_bytes": eop_path.stat().st_size,
            "gmat_r2026a_eop_first_mjd": dataset.first_mjd_utc,
            "gmat_r2026a_eop_last_mjd": dataset.last_mjd_utc,
            "gmat_r2026a_eop_row_count": dataset.row_count,
            "eop_provenance": _relative(provenance_path, root),
            "eop_provenance_sha256": _sha256(provenance_path),
            "cases": evidence,
        },
        "warnings": [
            "This is residual attribution on existing validation evidence, not independent validation of the selected candidate.",
            "The exact GMAT R2026a tagged EOP file is frozen; replacing it with newer EOP data would not reproduce the saved GMAT runs.",
            "The October case is after the tagged EOP coverage and therefore reproduces GMAT's last-row clamping behavior.",
            "Some tagged-file rows use placeholder uncertainty fields; they are retained because they are the values consumed by GMAT R2026a.",
            "The closed 1B baseline remains authoritative until a new independent candidate-validation matrix passes.",
            "Agreement is not measured-orbit truth, operational readiness, or flight qualification.",
        ],
    }
    summary_json = result_dir / "gmat_eop_diagnostic_summary.json"
    _write_json(summary, summary_json)
    created.append(summary_json)
    created.extend(_save_figures(result_dir / "figures", rows, models, eop_cases))
    report = result_dir / "GMAT_EOP_DIAGNOSTIC_REPORT.html"
    report.write_text(
        _report_html(
            config=config,
            dataset=dataset,
            aggregates=aggregates,
            decision=decision,
            eop_cases=eop_cases,
        ),
        encoding="utf-8",
        newline="\n",
    )
    created.append(report)
    manifest = result_dir / "RUN_MANIFEST.json"
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
        manifest,
    )
    created.append(manifest)
    return GmatEopDiagnosticResult(
        result_directory=result_dir,
        diagnostic_status="diagnostic_completed_with_warnings",
        decision=str(decision["decision"]),
        baseline_model=baseline_model,
        recommended_model=str(decision["recommended_model"]),
        case_count=len(matrix["cases"]),
        model_count=len(models),
        comparison_count=len(rows),
        summary_json=summary_json,
        report_path=report,
        created_files=tuple(created),
    )
