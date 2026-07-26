"""Paper 1 results consolidation for the frozen production matrix.

This module consumes the aggregate output from the validated Paper 1
production runner.  It does not propagate an orbit, change force models, or
reinterpret validation thresholds.  Its only job is to turn registered
production evidence into publication-oriented tables, figures, and prose.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


BUILD_MARKER = "RESEARCH-CORE-2A1-PAPER1-RESULTS-CONSOLIDATED"
MATRIX_ID = "EXP-PAPER1-PRODUCTION-001"
CONSOLIDATION_ID = "EXP-PAPER1-RESULTS-CONSOLIDATION-001"
EXPECTED_MATRIX = {
    "CASE-LEO400": (6, 24, 72, 168),
    "CASE-SSO700": (6, 24, 72, 168),
    "CASE-ISS-TLE": (6, 24, 72),
}


@dataclass(frozen=True)
class ConsolidationResult:
    result_directory: Path
    report_path: Path
    manifest_path: Path
    status: str
    figure_count: int
    table_count: int


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def find_latest_production_result(project_root: Path) -> Path:
    base = project_root / "results" / MATRIX_ID
    candidates = sorted(
        path.parent
        for path in base.glob("*/paper1_production_summary.json")
        if path.is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            "No Paper 1 production result was found. Expected "
            f"{base}\\<timestamp>\\paper1_production_summary.json"
        )
    return candidates[-1]


def validate_production_summary(summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if summary.get("matrix_id") != MATRIX_ID:
        errors.append(f"matrix_id must be {MATRIX_ID}")
    if summary.get("status") not in {"passed", "passed_with_warnings"}:
        errors.append("production status is not passing")
    if summary.get("completed_experiment_count") != 11:
        errors.append("completed_experiment_count must equal 11")
    if summary.get("expected_experiment_count") != 11:
        errors.append("expected_experiment_count must equal 11")
    if summary.get("failed_experiment_count") != 0:
        errors.append("failed_experiment_count must equal 0")
    if summary.get("primary_model_run_count") != 43:
        errors.append("primary_model_run_count must equal 43")
    if summary.get("executed_model_run_count") != 47:
        errors.append("executed_model_run_count must equal 47")
    if summary.get("failures"):
        errors.append("failure records are present")
    if summary.get("scope_decision") != "paper1_models_frozen_no_additional_force_models":
        errors.append("Paper 1 scope-freeze decision is missing")

    convergence = summary.get("convergence") or {}
    if convergence.get("validation_status") not in {"passed", "passed_with_warnings"}:
        errors.append("convergence validation did not pass")
    if convergence.get("evaluated_setting_count") != 36:
        errors.append("convergence evaluated_setting_count must equal 36")
    if convergence.get("passing_candidate_count") != 36:
        errors.append("convergence passing_candidate_count must equal 36")

    observed: dict[str, list[int]] = {}
    runs = summary.get("runs")
    if not isinstance(runs, list):
        errors.append("runs must be a list")
        runs = []
    for run in runs:
        case_id = str(run.get("case_id"))
        try:
            hours = int(run.get("duration_hours"))
        except (TypeError, ValueError):
            errors.append(f"invalid duration for {case_id}")
            continue
        observed.setdefault(case_id, []).append(hours)
        if run.get("validation_status") not in {"passed", "passed_with_warnings"}:
            errors.append(f"run did not pass: {case_id} {hours} h")
    normalized = {key: tuple(sorted(value)) for key, value in observed.items()}
    if normalized != EXPECTED_MATRIX:
        errors.append(f"run matrix differs from the frozen matrix: {normalized}")
    return errors


def _controlled_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in summary["runs"]:
        if run["case_id"] not in {"CASE-LEO400", "CASE-SSO700"}:
            continue
        rows.append(
            {
                "case_id": run["case_id"],
                "duration_hours": run["duration_hours"],
                "maximum_two_body_position_difference_m": run[
                    "maximum_two_body_position_difference_m"
                ],
                "maximum_two_body_velocity_difference_mm_s": run[
                    "maximum_two_body_velocity_difference_mm_s"
                ],
                "maximum_j2_two_body_position_difference_km": run[
                    "maximum_j2_two_body_position_difference_km"
                ],
                "analytical_raan_rate_deg_day": run["analytical_raan_rate_deg_day"],
                "fitted_raan_rate_deg_day": run["fitted_raan_rate_deg_day"],
                "raan_rate_relative_difference": run["raan_rate_relative_difference"],
                "maximum_drag_j2_position_difference_km": run[
                    "maximum_drag_j2_position_difference_km"
                ],
                "final_drag_semi_major_axis_difference_vs_j2_m": run[
                    "final_drag_semi_major_axis_difference_vs_j2_m"
                ],
                "drag_total_specific_energy_loss_km2_s2": run[
                    "drag_total_specific_energy_loss_km2_s2"
                ],
            }
        )
    return sorted(rows, key=lambda row: (row["case_id"], row["duration_hours"]))


def _sgp4_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in summary["runs"]:
        if run["case_id"] != "CASE-ISS-TLE":
            continue
        hours = run["duration_hours"]
        for model, maximum in run["maximum_separation_km_by_model"].items():
            rows.append(
                {
                    "case_id": run["case_id"],
                    "duration_hours": hours,
                    "model": model,
                    "maximum_separation_from_sgp4_km": maximum,
                    "final_separation_from_sgp4_km": run[
                        "final_separation_km_by_model"
                    ][model],
                    "model_pass_count": run["pass_count_by_model"].get(model),
                    "matched_pass_count": run["matched_pass_count_by_model"].get(model),
                    "maximum_absolute_aos_difference_seconds": run[
                        "maximum_absolute_aos_difference_seconds_by_model"
                    ].get(model),
                    "maximum_absolute_los_difference_seconds": run[
                        "maximum_absolute_los_difference_seconds_by_model"
                    ].get(model),
                }
            )
    return sorted(rows, key=lambda row: (row["duration_hours"], row["model"]))


def _runtime_rows(runtime_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in _read_csv(runtime_path):
        rows.append(
            {
                "case_id": raw["case_id"],
                "duration_hours": int(raw["duration_hours"]),
                "model": raw["model"],
                "runtime_seconds": float(raw["runtime_seconds"]),
                "function_evaluations": (
                    int(raw["function_evaluations"])
                    if raw.get("function_evaluations")
                    else None
                ),
            }
        )
    return sorted(rows, key=lambda row: (row["case_id"], row["model"], row["duration_hours"]))


def _select(rows: list[dict[str, Any]], **values: Any) -> dict[str, Any]:
    for row in rows:
        if all(row.get(key) == value for key, value in values.items()):
            return row
    raise KeyError(f"No row matches {values}")


def _save_figure(fig: Any, output_dir: Path, stem: str) -> list[Path]:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    return [png, pdf]


def _publication_style() -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "figure.dpi": 120,
            "savefig.dpi": 220,
        }
    )
    return plt, ScalarFormatter


def _figure_j2(controlled: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    plt, _ = _publication_style()
    colors = {"CASE-LEO400": "#0072B2", "CASE-SSO700": "#D55E00"}
    labels = {"CASE-LEO400": "400 km, 51.6°", "CASE-SSO700": "700 km, 98°"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    for axis, case_id in zip(axes, ("CASE-LEO400", "CASE-SSO700")):
        subset = [row for row in controlled if row["case_id"] == case_id]
        x = [row["duration_hours"] for row in subset]
        y = [row["maximum_j2_two_body_position_difference_km"] for row in subset]
        axis.plot(x, y, marker="o", color=colors[case_id])
        axis.set_title(labels[case_id])
        axis.set_xlabel("Propagation duration (h)")
        axis.set_ylabel("Maximum J2–two-body separation (km)")
        axis.set_xticks(x)
        axis.annotate(f"{y[-1]:,.0f} km", (x[-1], y[-1]), xytext=(-6, -18),
                      textcoords="offset points", ha="right", color=colors[case_id])
    fig.suptitle("Accumulated effect of J2 relative to two-body propagation", fontsize=11)
    paths = _save_figure(fig, output_dir, "figure_01_j2_effect")
    plt.close(fig)
    return paths


def _figure_drag(controlled: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    plt, _ = _publication_style()
    case_info = (
        ("CASE-LEO400", "400 km, 51.6°", "#0072B2"),
        ("CASE-SSO700", "700 km, 98°", "#D55E00"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    for row_index, (case_id, title, color) in enumerate(case_info):
        subset = [row for row in controlled if row["case_id"] == case_id]
        x = [row["duration_hours"] for row in subset]
        separation = [row["maximum_drag_j2_position_difference_km"] for row in subset]
        sma_loss = [-row["final_drag_semi_major_axis_difference_vs_j2_m"] for row in subset]
        left, right = axes[row_index]
        left.plot(x, separation, marker="o", color=color)
        right.plot(x, sma_loss, marker="s", color=color)
        left.set_ylabel(f"{title}\nMaximum separation (km)")
        right.set_ylabel(f"{title}\nFinal semi-major-axis reduction (m)")
        for axis in (left, right):
            axis.set_xticks(x)
            axis.set_xlabel("Propagation duration (h)")
    axes[0, 0].set_title("J2+drag separation from J2")
    axes[0, 1].set_title("Cumulative semi-major-axis reduction")
    fig.suptitle("Altitude dependence of the simplified-drag sensitivity", fontsize=11)
    paths = _save_figure(fig, output_dir, "figure_02_drag_effect")
    plt.close(fig)
    return paths


def _figure_sgp4(sgp4_rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    plt, _ = _publication_style()
    models = (
        ("analytical_two_body", "Analytical two-body", "#7F7F7F", "o"),
        ("numerical_two_body", "Numerical two-body", "#000000", "x"),
        ("numerical_j2", "Numerical J2", "#D55E00", "s"),
        ("numerical_j2_drag", "Numerical J2 + drag", "#009E73", "^"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    for axis, field, title in (
        (axes[0], "maximum_separation_from_sgp4_km", "Maximum separation"),
        (axes[1], "final_separation_from_sgp4_km", "Final-epoch separation"),
    ):
        for model, label, color, marker in models:
            subset = [row for row in sgp4_rows if row["model"] == model]
            axis.plot(
                [row["duration_hours"] for row in subset],
                [row[field] for row in subset],
                marker=marker,
                color=color,
                label=label,
            )
        axis.set_yscale("log")
        axis.set_xticks((6, 24, 72))
        axis.set_xlabel("Duration from fixed TLE epoch (h)")
        axis.set_ylabel("Separation from SGP4 (km, log scale)")
        axis.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Model separation from the fixed-TLE SGP4 reference", fontsize=11, y=0.97)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.80, bottom=0.28, wspace=0.26)
    paths = _save_figure(fig, output_dir, "figure_03_sgp4_comparison")
    plt.close(fig)
    return paths


def _figure_runtime(runtime_rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    plt, _ = _publication_style()
    colors = {
        "sgp4": "#CC79A7",
        "analytical_two_body": "#56B4E9",
        "numerical_two_body": "#7F7F7F",
        "numerical_j2": "#D55E00",
        "numerical_j2_drag": "#009E73",
    }
    labels = {
        "sgp4": "SGP4",
        "analytical_two_body": "Analytical two-body",
        "numerical_two_body": "Numerical two-body",
        "numerical_j2": "Numerical J2",
        "numerical_j2_drag": "Numerical J2 + drag",
    }
    case_titles = {
        "CASE-LEO400": "LEO400",
        "CASE-SSO700": "SSO700",
        "CASE-ISS-TLE": "ISS-TLE",
    }
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.9))
    legend_handles: dict[str, Any] = {}
    for axis, case_id in zip(axes, ("CASE-LEO400", "CASE-SSO700", "CASE-ISS-TLE")):
        case_rows = [row for row in runtime_rows if row["case_id"] == case_id]
        for model in sorted({row["model"] for row in case_rows}):
            subset = [row for row in case_rows if row["model"] == model]
            line = axis.plot(
                [row["duration_hours"] for row in subset],
                [row["runtime_seconds"] for row in subset],
                marker="o",
                color=colors.get(model, "#333333"),
                label=labels.get(model, model),
            )[0]
            legend_handles.setdefault(model, line)
        axis.set_yscale("log")
        axis.set_title(case_titles[case_id])
        axis.set_xlabel("Propagation duration (h)")
        axis.set_ylabel("Wall-clock runtime (s, log scale)")
        axis.set_xticks(sorted({row["duration_hours"] for row in case_rows}))
    order = [key for key in colors if key in legend_handles]
    fig.legend(
        [legend_handles[key] for key in order],
        [labels[key] for key in order],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle("Observed runtime scaling on the production computer", fontsize=11, y=0.97)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.27, wspace=0.32)
    paths = _save_figure(fig, output_dir, "figure_04_runtime_scaling")
    plt.close(fig)
    return paths


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "not available"
    return f"{float(value):,.{digits}f}"


def _key_results(
    summary: dict[str, Any],
    controlled: list[dict[str, Any]],
    sgp4_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    leo168 = _select(controlled, case_id="CASE-LEO400", duration_hours=168)
    sso168 = _select(controlled, case_id="CASE-SSO700", duration_hours=168)
    j2_72 = _select(sgp4_rows, duration_hours=72, model="numerical_j2")
    drag_72 = _select(sgp4_rows, duration_hours=72, model="numerical_j2_drag")
    tb_72 = _select(sgp4_rows, duration_hours=72, model="numerical_two_body")
    improvement = 100.0 * (
        1.0
        - drag_72["maximum_separation_from_sgp4_km"]
        / j2_72["maximum_separation_from_sgp4_km"]
    )
    return {
        "schema_version": "P1R.0",
        "build_marker": BUILD_MARKER,
        "source_matrix_id": summary["matrix_id"],
        "source_status": summary["status"],
        "production_experiments": summary["completed_experiment_count"],
        "failed_experiments": summary["failed_experiment_count"],
        "convergence_candidates_passing": summary["convergence"]["passing_candidate_count"],
        "convergence_candidates_evaluated": summary["convergence"]["evaluated_setting_count"],
        "balanced_integrator_case": summary["convergence"]["balanced_case_id"],
        "leo400_168h": {
            "j2_vs_two_body_maximum_position_separation_km": leo168[
                "maximum_j2_two_body_position_difference_km"
            ],
            "drag_vs_j2_maximum_position_separation_km": leo168[
                "maximum_drag_j2_position_difference_km"
            ],
            "drag_final_semi_major_axis_difference_vs_j2_m": leo168[
                "final_drag_semi_major_axis_difference_vs_j2_m"
            ],
            "raan_rate_relative_difference": leo168["raan_rate_relative_difference"],
        },
        "sso700_168h": {
            "j2_vs_two_body_maximum_position_separation_km": sso168[
                "maximum_j2_two_body_position_difference_km"
            ],
            "drag_vs_j2_maximum_position_separation_km": sso168[
                "maximum_drag_j2_position_difference_km"
            ],
            "drag_final_semi_major_axis_difference_vs_j2_m": sso168[
                "final_drag_semi_major_axis_difference_vs_j2_m"
            ],
            "raan_rate_relative_difference": sso168["raan_rate_relative_difference"],
        },
        "iss_tle_72h": {
            "numerical_two_body_maximum_separation_from_sgp4_km": tb_72[
                "maximum_separation_from_sgp4_km"
            ],
            "numerical_j2_maximum_separation_from_sgp4_km": j2_72[
                "maximum_separation_from_sgp4_km"
            ],
            "numerical_j2_drag_maximum_separation_from_sgp4_km": drag_72[
                "maximum_separation_from_sgp4_km"
            ],
            "j2_drag_reduction_relative_to_j2_percent": improvement,
            "numerical_j2_drag_maximum_aos_difference_seconds": drag_72[
                "maximum_absolute_aos_difference_seconds"
            ],
            "numerical_j2_drag_maximum_los_difference_seconds": drag_72[
                "maximum_absolute_los_difference_seconds"
            ],
        },
        "interpretation_limits": [
            "Simplified drag is an illustrative sensitivity model, not a high-fidelity atmosphere.",
            "Fixed-TLE SGP4 is a comparison reference, not measured-orbit truth.",
            "Ground-station pass comparisons are geometric and exclude link-budget effects.",
            "Runtime values describe the production computer and are not universal benchmarks.",
        ],
    }


def _narrative(key: dict[str, Any]) -> str:
    leo = key["leo400_168h"]
    sso = key["sso700_168h"]
    iss = key["iss_tle_72h"]
    return f"""# Paper 1 quantitative results narrative

## Production and numerical controls

The frozen production matrix completed all {key['production_experiments']} experiments with no failed experiments. The convergence study evaluated {key['convergence_candidates_evaluated']} solver settings, and all {key['convergence_candidates_passing']} satisfied the registered acceptance criteria. The balanced production setting was `{key['balanced_integrator_case']}`.

## Controlled two-body and J2 comparison

Over 168 hours, the maximum J2-to-two-body position separation reached {_format_number(leo['j2_vs_two_body_maximum_position_separation_km'])} km for the 400 km, 51.6° case and {_format_number(sso['j2_vs_two_body_maximum_position_separation_km'])} km for the 700 km, 98° case. The fitted secular RAAN rates remained within {_format_number(100.0 * leo['raan_rate_relative_difference'])}% and {_format_number(100.0 * sso['raan_rate_relative_difference'])}% of the corresponding first-order analytical rates. These results demonstrate the rapid accumulation of J2-driven trajectory separation while retaining agreement with the expected secular nodal trend.

## Simplified-drag sensitivity

At 168 hours, the maximum separation between J2+drag and J2 was {_format_number(leo['drag_vs_j2_maximum_position_separation_km'])} km at 400 km, compared with {_format_number(sso['drag_vs_j2_maximum_position_separation_km'])} km at 700 km. The final semi-major-axis differences relative to J2 were {_format_number(leo['drag_final_semi_major_axis_difference_vs_j2_m'])} m and {_format_number(sso['drag_final_semi_major_axis_difference_vs_j2_m'])} m, respectively. This strong altitude dependence is a sensitivity result for the registered exponential-density model and must not be presented as a high-fidelity atmospheric prediction.

## Fixed-TLE SGP4 comparison

At 72 hours from the fixed TLE epoch, the maximum position separation from SGP4 was {_format_number(iss['numerical_two_body_maximum_separation_from_sgp4_km'])} km for numerical two-body, {_format_number(iss['numerical_j2_maximum_separation_from_sgp4_km'])} km for numerical J2, and {_format_number(iss['numerical_j2_drag_maximum_separation_from_sgp4_km'])} km for numerical J2+drag. Adding the registered simplified-drag term reduced the maximum separation relative to J2 alone by {_format_number(iss['j2_drag_reduction_relative_to_j2_percent'], 1)}%. The J2+drag geometric pass comparison had maximum absolute AOS and LOS differences of {_format_number(iss['numerical_j2_drag_maximum_aos_difference_seconds'])} s and {_format_number(iss['numerical_j2_drag_maximum_los_difference_seconds'])} s. SGP4 remains a fixed-TLE model reference rather than measured-orbit truth, so these quantities describe model-to-model separation, not orbit-determination accuracy.

## Scope statement

No additional force models were introduced during consolidation. The figures and tables report the frozen Paper 1 production evidence without changing the underlying states, thresholds, or validation decisions.
"""


def _captions() -> str:
    return """# Paper 1 figure captions

**Figure 1.** Maximum Cartesian position separation between numerical J2 and two-body propagation as a function of propagation duration for the controlled 400 km, 51.6° and 700 km, 98° cases. Separate panels prevent the larger LEO400 separation from compressing the SSO700 trend.

**Figure 2.** Simplified-drag sensitivity relative to numerical J2. The left column shows maximum position separation; the right column shows final semi-major-axis reduction. Rows use independent vertical scales for the 400 km and 700 km cases. Values describe the registered exponential-density sensitivity model, not a high-fidelity atmospheric prediction.

**Figure 3.** Maximum and final-epoch position separation of analytical two-body, numerical two-body, numerical J2, and numerical J2+drag from the fixed-TLE SGP4 reference. A logarithmic vertical scale exposes both the small J2/J2+drag differences and the much larger two-body separation. SGP4 is a comparison reference, not measured-orbit truth.

**Figure 4.** Observed wall-clock runtime against propagation duration on the production computer. Panels separate the three cases, and logarithmic vertical scaling accommodates analytical, SGP4, and numerical runtimes. These timings demonstrate scaling within this implementation and hardware environment; they are not universal performance benchmarks.
"""


def _html_report(
    key: dict[str, Any],
    controlled: list[dict[str, Any]],
    sgp4_rows: list[dict[str, Any]],
) -> str:
    leo = key["leo400_168h"]
    sso = key["sso700_168h"]
    iss = key["iss_tle_72h"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Paper 1 Results Consolidation</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1050px;margin:2rem auto;line-height:1.5;color:#222}}
h1,h2{{color:#17365d}} .pass{{color:#176b2c;font-weight:bold}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #bbb;padding:.4rem;text-align:left}}
img{{max-width:100%;height:auto;border:1px solid #ddd;margin:.5rem 0 1.5rem}}
.note{{background:#f5f7fa;border-left:4px solid #597ea6;padding:.7rem 1rem}}
</style></head><body>
<h1>Paper 1 Results Consolidation</h1>
<p>Build marker: <code>{html.escape(BUILD_MARKER)}</code></p>
<p class="pass">Status: ready_for_manuscript_drafting_with_scope_warnings</p>
<table><tr><th>Production experiments</th><th>Failures</th><th>Convergence</th><th>Model scope</th></tr>
<tr><td>{key['production_experiments']}/11</td><td>{key['failed_experiments']}</td>
<td>{key['convergence_candidates_passing']}/{key['convergence_candidates_evaluated']} passed</td>
<td>Frozen; no additional force models</td></tr></table>
<h2>Key quantitative findings</h2>
<ul>
<li>168 h J2–two-body maximum separation: {_format_number(leo['j2_vs_two_body_maximum_position_separation_km'])} km (LEO400) and {_format_number(sso['j2_vs_two_body_maximum_position_separation_km'])} km (SSO700).</li>
<li>168 h drag–J2 maximum separation: {_format_number(leo['drag_vs_j2_maximum_position_separation_km'])} km (LEO400) and {_format_number(sso['drag_vs_j2_maximum_position_separation_km'])} km (SSO700).</li>
<li>72 h maximum separation from fixed-TLE SGP4: {_format_number(iss['numerical_two_body_maximum_separation_from_sgp4_km'])} km (two-body), {_format_number(iss['numerical_j2_maximum_separation_from_sgp4_km'])} km (J2), and {_format_number(iss['numerical_j2_drag_maximum_separation_from_sgp4_km'])} km (J2+drag).</li>
</ul>
<div class="note"><strong>Interpretation:</strong> the drag model is a registered sensitivity model, and SGP4 is a fixed-TLE comparison reference. Neither is measured-orbit truth.</div>
<h2>Publication figures</h2>
<img src="figure_01_j2_effect.png" alt="J2 effect">
<img src="figure_02_drag_effect.png" alt="Drag sensitivity">
<img src="figure_03_sgp4_comparison.png" alt="SGP4 comparison">
<img src="figure_04_runtime_scaling.png" alt="Runtime scaling">
<h2>Included tables and text</h2>
<ul><li>table_01_controlled_model_effects.csv</li><li>table_02_sgp4_comparison.csv</li>
<li>table_03_runtime_scaling.csv</li><li>table_04_validation_summary.csv</li>
<li>PAPER1_RESULTS_NARRATIVE.md</li><li>PAPER1_FIGURE_CAPTIONS.md</li></ul>
</body></html>"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_manifest(result_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(result_dir.iterdir()):
        if not path.is_file() or path.name == "RUN_MANIFEST.json":
            continue
        files.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema_version": "P1R.0",
        "build_marker": BUILD_MARKER,
        "consolidation_id": CONSOLIDATION_ID,
        "status": "ready_for_manuscript_drafting_with_scope_warnings",
        "file_count_excluding_manifest": len(files),
        "files": files,
    }


def run_consolidation(
    project_root: Path,
    production_dir: Path | None = None,
    output_root: Path | None = None,
) -> ConsolidationResult:
    project_root = project_root.resolve()
    production_dir = (
        production_dir.resolve()
        if production_dir is not None
        else find_latest_production_result(project_root)
    )
    summary_path = production_dir / "paper1_production_summary.json"
    runtime_path = production_dir / "paper1_runtime_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing production summary: {summary_path}")
    if not runtime_path.is_file():
        raise FileNotFoundError(f"Missing production runtime table: {runtime_path}")

    summary = _load_json(summary_path)
    errors = validate_production_summary(summary)
    if errors:
        joined = "\n  - ".join(errors)
        raise ValueError(f"Paper 1 production evidence failed consolidation gates:\n  - {joined}")

    controlled = _controlled_rows(summary)
    sgp4_rows = _sgp4_rows(summary)
    runtime_rows = _runtime_rows(runtime_path)
    if len(controlled) != 8:
        raise ValueError(f"Expected 8 controlled rows, found {len(controlled)}")
    if len(sgp4_rows) != 12:
        raise ValueError(f"Expected 12 SGP4 comparison rows, found {len(sgp4_rows)}")
    if len(runtime_rows) != 47:
        raise ValueError(f"Expected 47 runtime rows, found {len(runtime_rows)}")

    if output_root is None:
        output_root = project_root / "results" / CONSOLIDATION_ID
    result_dir = output_root.resolve() / _utc_stamp()
    result_dir.mkdir(parents=True, exist_ok=False)

    controlled_fields = list(controlled[0].keys())
    sgp4_fields = list(sgp4_rows[0].keys())
    runtime_fields = list(runtime_rows[0].keys())
    _write_csv(result_dir / "table_01_controlled_model_effects.csv", controlled_fields, controlled)
    _write_csv(result_dir / "table_02_sgp4_comparison.csv", sgp4_fields, sgp4_rows)
    _write_csv(result_dir / "table_03_runtime_scaling.csv", runtime_fields, runtime_rows)
    validation_rows = [
        {"metric": "production_status", "value": summary["status"]},
        {"metric": "completed_experiments", "value": summary["completed_experiment_count"]},
        {"metric": "expected_experiments", "value": summary["expected_experiment_count"]},
        {"metric": "failed_experiments", "value": summary["failed_experiment_count"]},
        {"metric": "primary_model_runs", "value": summary["primary_model_run_count"]},
        {"metric": "executed_model_runs", "value": summary["executed_model_run_count"]},
        {"metric": "convergence_candidates_evaluated", "value": summary["convergence"]["evaluated_setting_count"]},
        {"metric": "convergence_candidates_passing", "value": summary["convergence"]["passing_candidate_count"]},
        {"metric": "balanced_integrator_case", "value": summary["convergence"]["balanced_case_id"]},
        {"metric": "paper1_scope", "value": summary["scope_decision"]},
    ]
    _write_csv(result_dir / "table_04_validation_summary.csv", ["metric", "value"], validation_rows)

    figure_paths: list[Path] = []
    figure_paths.extend(_figure_j2(controlled, result_dir))
    figure_paths.extend(_figure_drag(controlled, result_dir))
    figure_paths.extend(_figure_sgp4(sgp4_rows, result_dir))
    figure_paths.extend(_figure_runtime(runtime_rows, result_dir))

    key = _key_results(summary, controlled, sgp4_rows)
    key["created_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    key["source_production_directory"] = str(production_dir)
    _write_json(result_dir / "paper1_key_results.json", key)
    (result_dir / "PAPER1_RESULTS_NARRATIVE.md").write_text(_narrative(key), encoding="utf-8")
    (result_dir / "PAPER1_FIGURE_CAPTIONS.md").write_text(_captions(), encoding="utf-8")
    readiness = {
        "schema_version": "P1R.0",
        "build_marker": BUILD_MARKER,
        "status": "ready_for_manuscript_drafting_with_scope_warnings",
        "source_matrix_id": summary["matrix_id"],
        "source_status": summary["status"],
        "gates": {
            "exact_frozen_matrix": True,
            "all_experiments_completed": True,
            "zero_failed_experiments": True,
            "convergence_gate_passed": True,
            "force_model_scope_unchanged": True,
        },
        "next_stage": "paper1_manuscript_and_github_release",
    }
    _write_json(result_dir / "publication_readiness.json", readiness)
    report_path = result_dir / "PAPER1_RESULTS_CONSOLIDATION_REPORT.html"
    report_path.write_text(_html_report(key, controlled, sgp4_rows), encoding="utf-8")
    manifest_path = result_dir / "RUN_MANIFEST.json"
    _write_json(manifest_path, create_manifest(result_dir))

    return ConsolidationResult(
        result_directory=result_dir,
        report_path=report_path,
        manifest_path=manifest_path,
        status=readiness["status"],
        figure_count=4,
        table_count=4,
    )


def verify_result_directory(result_dir: Path) -> list[str]:
    manifest_path = result_dir / "RUN_MANIFEST.json"
    if not manifest_path.is_file():
        return ["RUN_MANIFEST.json is missing"]
    manifest = _load_json(manifest_path)
    errors: list[str] = []
    if manifest.get("build_marker") != BUILD_MARKER:
        errors.append("build marker does not match")
    if manifest.get("status") != "ready_for_manuscript_drafting_with_scope_warnings":
        errors.append("readiness status does not match")
    for record in manifest.get("files", []):
        path = result_dir / record["path"]
        if not path.is_file():
            errors.append(f"missing file: {record['path']}")
            continue
        if path.stat().st_size != record["size_bytes"]:
            errors.append(f"size mismatch: {record['path']}")
        if _sha256(path) != record["sha256"]:
            errors.append(f"hash mismatch: {record['path']}")
    if len(manifest.get("files", [])) != manifest.get("file_count_excluding_manifest"):
        errors.append("manifest file count does not match")
    return errors
