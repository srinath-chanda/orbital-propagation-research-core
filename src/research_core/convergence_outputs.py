"""CSV, figure, and report outputs for convergence studies."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONVERGENCE_COLUMNS = (
    "row_role",
    "case_id",
    "method",
    "relative_tolerance",
    "absolute_tolerance",
    "maximum_step_seconds",
    "runtime_repetitions",
    "minimum_runtime_seconds",
    "median_runtime_seconds",
    "maximum_runtime_seconds",
    "function_evaluations",
    "maximum_position_difference_vs_analytical_m",
    "final_position_difference_vs_analytical_m",
    "rms_position_difference_vs_analytical_m",
    "maximum_velocity_difference_vs_analytical_mm_s",
    "maximum_position_difference_vs_reference_m",
    "maximum_velocity_difference_vs_reference_mm_s",
    "maximum_absolute_relative_energy_drift",
    "maximum_absolute_relative_angular_momentum_drift",
    "passes_provisional_thresholds",
    "is_current_configuration",
)


def write_convergence_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CONVERGENCE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in CONVERGENCE_COLUMNS})


def _save_figure(
    figure: plt.Figure,
    base_path: Path,
    *,
    save_png: bool,
    save_pdf: bool,
) -> list[Path]:
    created: list[Path] = []
    base_path.parent.mkdir(parents=True, exist_ok=True)
    if save_png:
        png = base_path.with_suffix(".png")
        figure.savefig(png, dpi=180, bbox_inches="tight")
        created.append(png)
    if save_pdf:
        pdf = base_path.with_suffix(".pdf")
        figure.savefig(pdf, bbox_inches="tight")
        created.append(pdf)
    plt.close(figure)
    return created


def create_convergence_figures(
    figure_directory: str | Path,
    rows: list[dict[str, Any]],
    selection: dict[str, Any],
    *,
    save_png: bool,
    save_pdf: bool,
) -> list[Path]:
    directory = Path(figure_directory)
    created: list[Path] = []

    runtimes = np.array([row["median_runtime_seconds"] for row in rows], dtype=float)
    position_errors = np.array(
        [row["maximum_position_difference_vs_analytical_m"] for row in rows],
        dtype=float,
    )
    energy_drifts = np.array(
        [row["maximum_absolute_relative_energy_drift"] for row in rows],
        dtype=float,
    )
    evaluations = np.array([row["function_evaluations"] for row in rows], dtype=float)

    figure, axis = plt.subplots(figsize=(8.0, 5.4))
    axis.scatter(runtimes, position_errors)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Median runtime (s)")
    axis.set_ylabel("Maximum position difference vs analytical (m)")
    axis.set_title("Two-body convergence: error versus runtime")
    axis.grid(True, which="both", alpha=0.3)
    balanced = selection.get("balanced_recommendation")
    if balanced:
        axis.annotate(
            "balanced recommendation",
            (
                balanced["median_runtime_seconds"],
                balanced["maximum_position_difference_vs_analytical_m"],
            ),
            xytext=(8, 8),
            textcoords="offset points",
        )
    created.extend(
        _save_figure(
            figure,
            directory / "convergence_error_vs_runtime",
            save_png=save_png,
            save_pdf=save_pdf,
        )
    )

    figure, axis = plt.subplots(figsize=(9.0, 5.8))
    groups: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["relative_tolerance"], row["absolute_tolerance"])
        groups.setdefault(key, []).append(row)
    for (rtol, atol), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda item: item["maximum_step_seconds"])
        axis.plot(
            [item["maximum_step_seconds"] for item in ordered],
            [item["maximum_position_difference_vs_analytical_m"] for item in ordered],
            marker="o",
            label=f"rtol={rtol:.0e}, atol={atol:.0e}",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Maximum integration step (s)")
    axis.set_ylabel("Maximum position difference vs analytical (m)")
    axis.set_title("Position convergence by tolerance pair")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=7, ncol=2)
    created.extend(
        _save_figure(
            figure,
            directory / "convergence_position_error",
            save_png=save_png,
            save_pdf=save_pdf,
        )
    )

    figure, axis = plt.subplots(figsize=(8.0, 5.4))
    axis.scatter(runtimes, energy_drifts)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Median runtime (s)")
    axis.set_ylabel("Maximum absolute relative energy drift")
    axis.set_title("Energy conservation versus runtime")
    axis.grid(True, which="both", alpha=0.3)
    created.extend(
        _save_figure(
            figure,
            directory / "convergence_energy_drift",
            save_png=save_png,
            save_pdf=save_pdf,
        )
    )

    figure, axis = plt.subplots(figsize=(8.0, 5.4))
    axis.scatter(evaluations, position_errors)
    axis.set_yscale("log")
    axis.set_xlabel("Force-function evaluations")
    axis.set_ylabel("Maximum position difference vs analytical (m)")
    axis.set_title("Function evaluations versus position error")
    axis.grid(True, which="both", alpha=0.3)
    created.extend(
        _save_figure(
            figure,
            directory / "convergence_function_evaluations",
            save_png=save_png,
            save_pdf=save_pdf,
        )
    )

    return created


def convergence_summary_markdown(
    *,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    reference_summary: dict[str, Any],
    selection: dict[str, Any],
    validation_status: dict[str, Any],
) -> str:
    balanced = selection.get("balanced_recommendation")
    fastest = selection.get("fastest_passing")
    accurate = selection.get("most_accurate_passing")

    def setting_text(setting: dict[str, Any] | None) -> str:
        if not setting:
            return "No qualifying setting was found."
        return (
            f"`{setting['case_id']}`: rtol={setting['relative_tolerance']:.0e}, "
            f"atol={setting['absolute_tolerance']:.0e}, "
            f"max step={setting['maximum_step_seconds']:g} s, "
            f"median runtime={setting['median_runtime_seconds']:.6f} s, "
            f"maximum position difference="
            f"{setting['maximum_position_difference_vs_analytical_m']:.6e} m."
        )

    return f"""# Numerical Convergence Summary

## Experiment

- Experiment ID: `{config['experiment']['experiment_id']}`
- Case ID: `{config['experiment']['case_id']}`
- Duration: {config['convergence']['duration_hours']} hours
- Output interval: {config['convergence']['output_step_seconds']} seconds
- Integrator: {config['convergence']['method']}
- Evaluated settings: {len(rows)}
- Matrix candidates: {len([row for row in rows if row.get('row_role') == 'matrix_candidate'])}
- Current-configuration baselines: {len([row for row in rows if row.get('row_role') == 'current_configuration'])}
- Runtime repetitions per candidate: {config['convergence']['runtime_repetitions']}

## High-accuracy numerical reference

- Relative tolerance: {reference_summary['relative_tolerance']:.0e}
- Absolute tolerance: {reference_summary['absolute_tolerance']:.0e}
- Maximum step: {reference_summary['maximum_step_seconds']:g} s
- Runtime: {reference_summary['runtime_seconds']:.6f} s
- Maximum position difference versus analytical: {reference_summary['maximum_position_difference_vs_analytical_m']:.6e} m
- Maximum relative energy drift: {reference_summary['maximum_absolute_relative_energy_drift']:.6e}

This is a numerical convergence reference, not an independent physical truth model.

## Recommendations

### Fastest candidate passing provisional thresholds

{setting_text(fastest)}

### Most accurate candidate passing provisional thresholds

{setting_text(accurate)}

### Balanced provisional recommendation

{setting_text(balanced)}

Selection method:

> {selection['selection_method']}

## Validation status

**{validation_status['overall_status']}**

Passing candidate cases: {selection['passing_case_count']} of {len(rows)}.

## Scientific interpretation

All runs use the same point-mass two-body force model and common Cartesian
initial state. The study measures numerical integration behaviour and runtime.
It does not measure real-satellite prediction accuracy.

## Next decision

The balanced setting is a provisional engineering recommendation. Review the
figures and CSV before updating the production integrator configuration. The
next physical-model release remains J2 propagation and RAAN validation.
"""
