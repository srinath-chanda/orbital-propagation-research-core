"""CSV, figure and Markdown outputs for Research Core 1A.6 pass analysis."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .ground_station import GroundStationPass, VisibilityHistory


def _safe_name(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


def write_visibility_history_csv(path: str | Path, visibility: VisibilityHistory) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "station_id",
        "station_name",
        "minimum_elevation_deg",
        "elapsed_seconds",
        "timestamp_utc",
        "azimuth_deg",
        "elevation_deg",
        "range_km",
        "range_rate_km_s",
        "above_elevation_mask",
    ]
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, elapsed in enumerate(visibility.elapsed_seconds):
            writer.writerow(
                {
                    "model": visibility.model_name,
                    "station_id": visibility.station.station_id,
                    "station_name": visibility.station.name,
                    "minimum_elevation_deg": visibility.station.minimum_elevation_deg,
                    "elapsed_seconds": float(elapsed),
                    "timestamp_utc": visibility.timestamps_utc[index],
                    "azimuth_deg": visibility.azimuth_deg[index],
                    "elevation_deg": visibility.elevation_deg[index],
                    "range_km": visibility.range_km[index],
                    "range_rate_km_s": visibility.range_rate_km_s[index],
                    "above_elevation_mask": bool(
                        visibility.elevation_deg[index]
                        >= visibility.station.minimum_elevation_deg
                    ),
                }
            )


def write_passes_csv(path: str | Path, passes: list[GroundStationPass]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = list(GroundStationPass.__dataclass_fields__)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in passes:
            writer.writerow(item.as_dict())


def write_pass_comparison_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "match_status",
        "reference_model",
        "comparison_model",
        "reference_pass_id",
        "comparison_pass_id",
        "reference_aos_utc",
        "comparison_aos_utc",
        "aos_difference_seconds",
        "maximum_time_difference_seconds",
        "los_difference_seconds",
        "duration_difference_seconds",
        "maximum_elevation_difference_deg",
        "closest_range_difference_km",
    ]
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_pass_summary_csv(path: str | Path, summary: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "pass_count",
        "total_visible_time_seconds",
        "maximum_elevation_deg",
        "minimum_closest_range_km",
        "partial_pass_count",
        "matched_pass_count_against_sgp4",
        "reference_unmatched_count",
        "comparison_unmatched_count",
        "maximum_absolute_aos_difference_seconds",
        "maximum_absolute_maximum_time_difference_seconds",
        "maximum_absolute_los_difference_seconds",
        "rms_aos_difference_seconds",
        "rms_los_difference_seconds",
    ]
    comparisons = summary["comparisons_against_sgp4"]
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model, values in summary["models"].items():
            comparison = comparisons.get(model, {})
            writer.writerow(
                {
                    "model": model,
                    **values,
                    "matched_pass_count_against_sgp4": comparison.get("matched_pass_count"),
                    "reference_unmatched_count": comparison.get("reference_unmatched_count"),
                    "comparison_unmatched_count": comparison.get("comparison_unmatched_count"),
                    "maximum_absolute_aos_difference_seconds": comparison.get("maximum_absolute_aos_difference_seconds"),
                    "maximum_absolute_maximum_time_difference_seconds": comparison.get("maximum_absolute_maximum_time_difference_seconds"),
                    "maximum_absolute_los_difference_seconds": comparison.get("maximum_absolute_los_difference_seconds"),
                    "rms_aos_difference_seconds": comparison.get("rms_aos_difference_seconds"),
                    "rms_los_difference_seconds": comparison.get("rms_los_difference_seconds"),
                }
            )


def _save(
    figure: plt.Figure,
    directory: Path,
    stem: str,
    save_png: bool,
    save_pdf: bool,
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    if save_png:
        path = directory / f"{stem}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        created.append(path)
    if save_pdf:
        path = directory / f"{stem}.pdf"
        figure.savefig(path, bbox_inches="tight")
        created.append(path)
    plt.close(figure)
    return created


def _masked_visible(values: np.ndarray, elevation: np.ndarray, threshold: float) -> np.ndarray:
    output = np.asarray(values, dtype=float).copy()
    output[elevation < threshold] = np.nan
    return output


def create_pass_figures(
    directory: str | Path,
    visibility_by_model: dict[str, VisibilityHistory],
    passes_by_model: dict[str, list[GroundStationPass]],
    comparisons_by_model: dict[str, list[dict[str, Any]]],
    *,
    save_png: bool,
    save_pdf: bool,
) -> list[Path]:
    output_directory = Path(directory)
    created: list[Path] = []
    reference = visibility_by_model["sgp4"]
    station_stem = _safe_name(reference.station.name)
    hours = reference.elapsed_seconds / 3600.0
    threshold = reference.station.minimum_elevation_deg

    figure, ax = plt.subplots(figsize=(10.2, 5.6))
    for model, visibility in visibility_by_model.items():
        ax.plot(hours, visibility.elevation_deg, label=model, linewidth=1.0)
    ax.axhline(threshold, linestyle="--", linewidth=1.0, label="elevation mask")
    ax.set_xlabel("Elapsed time (hours)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_title(f"{reference.station.name} ground-station elevation history")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    created += _save(
        figure,
        output_directory,
        f"{station_stem}_elevation_history",
        save_png,
        save_pdf,
    )

    figure, ax = plt.subplots(figsize=(10.2, 5.6))
    for model, visibility in visibility_by_model.items():
        visible_range = _masked_visible(
            visibility.range_km,
            visibility.elevation_deg,
            threshold,
        )
        ax.plot(hours, visible_range, label=model, linewidth=1.1)
    ax.set_xlabel("Elapsed time (hours)")
    ax.set_ylabel("Topocentric range while visible (km)")
    ax.set_title(f"{reference.station.name} visible-range history")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    created += _save(
        figure,
        output_directory,
        f"{station_stem}_visible_range_history",
        save_png,
        save_pdf,
    )

    sgp4_passes = passes_by_model.get("sgp4", [])
    figure = plt.figure(figsize=(8.2, 7.0))
    ax = figure.add_subplot(111, projection="polar")
    visible_mask = reference.elevation_deg >= threshold
    segments = np.where(np.diff(np.concatenate(([False], visible_mask, [False])).astype(int)) != 0)[0]
    for pass_number, (start, end) in enumerate(segments.reshape(-1, 2), start=1):
        azimuth = np.radians(reference.azimuth_deg[start:end])
        radius = 90.0 - reference.elevation_deg[start:end]
        ax.plot(azimuth, radius, label=f"Pass {pass_number}")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(90.0 - max(90.0, threshold), 90.0 - threshold)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_yticklabels(["90°", "70°", "50°", "30°", "10°"])
    ax.set_title(f"SGP4 sky paths above {threshold:.1f}° at {reference.station.name}")
    if sgp4_passes:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
    created += _save(
        figure,
        output_directory,
        f"{station_stem}_sgp4_sky_paths",
        save_png,
        save_pdf,
    )

    models = [model for model in comparisons_by_model if model != "sgp4"]
    figure, axes = plt.subplots(3, 1, figsize=(10.2, 9.0), sharex=True)
    for model in models:
        matched = [
            row
            for row in comparisons_by_model[model]
            if row["match_status"] == "matched"
        ]
        if not matched:
            continue
        indices = np.arange(1, len(matched) + 1)
        axes[0].plot(
            indices,
            [row["aos_difference_seconds"] for row in matched],
            marker="o",
            label=model,
        )
        axes[1].plot(
            indices,
            [row["maximum_time_difference_seconds"] for row in matched],
            marker="o",
            label=model,
        )
        axes[2].plot(
            indices,
            [row["los_difference_seconds"] for row in matched],
            marker="o",
            label=model,
        )
    labels = ["AOS difference (s)", "Maximum-time difference (s)", "LOS difference (s)"]
    for ax, label in zip(axes, labels):
        ax.axhline(0.0, linewidth=0.8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Matched SGP4 pass sequence")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(ncol=2)
    figure.suptitle(f"{reference.station.name} pass-timing differences from SGP4")
    created += _save(
        figure,
        output_directory,
        f"{station_stem}_pass_timing_differences",
        save_png,
        save_pdf,
    )

    figure, ax = plt.subplots(figsize=(10.2, 5.6))
    for model, passes in passes_by_model.items():
        if not passes:
            continue
        ax.plot(
            np.arange(1, len(passes) + 1),
            [item.maximum_elevation_deg for item in passes],
            marker="o",
            label=model,
        )
    ax.axhline(threshold, linestyle="--", linewidth=1.0, label="elevation mask")
    ax.set_xlabel("Pass sequence within model")
    ax.set_ylabel("Maximum elevation (deg)")
    ax.set_title(f"{reference.station.name} maximum-elevation comparison")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    created += _save(
        figure,
        output_directory,
        f"{station_stem}_maximum_elevation_comparison",
        save_png,
        save_pdf,
    )

    return created


def write_pass_technical_summary(
    path: str | Path,
    *,
    summary: dict[str, Any],
    validation_status: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    station = summary["station"]
    lines = [
        "# Ground-Station Pass Technical Summary",
        "",
        "## Station",
        "",
        f"- ID: `{station['station_id']}`",
        f"- Name: {station['name']}",
        f"- Latitude: {station['latitude_deg']:.6f}°",
        f"- Longitude: {station['longitude_deg']:.6f}°",
        f"- Altitude: {station['altitude_m']:.3f} m",
        f"- Minimum elevation: {station['minimum_elevation_deg']:.3f}°",
        "",
        "## Method",
        "",
        f"- Coarse access grid: {summary['coarse_step_seconds']:.3f} s",
        f"- Boundary refinement tolerance: {summary['refinement_tolerance_seconds']:.3f} s",
        "- AOS and LOS are refined using a shape-preserving elevation interpolator and Brent root finding.",
        "- Maximum elevation and closest range are refined inside each pass using bounded scalar optimisation.",
        "- Visibility is purely geometric above the configured elevation mask.",
        "",
        "## Pass counts",
        "",
        "| Model | Passes | Visible time (s) | Highest elevation (deg) | Minimum range (km) |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, values in summary["models"].items():
        maximum = "—" if values["maximum_elevation_deg"] is None else f"{values['maximum_elevation_deg']:.6f}"
        minimum = "—" if values["minimum_closest_range_km"] is None else f"{values['minimum_closest_range_km']:.6f}"
        lines.append(
            f"| {model} | {values['pass_count']} | {values['total_visible_time_seconds']:.3f} | {maximum} | {minimum} |"
        )

    lines += [
        "",
        "## Timing comparison against SGP4",
        "",
        "| Model | Matched | Reference unmatched | Model unmatched | Max |AOS Δ| (s) | Max |LOS Δ| (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, values in summary["comparisons_against_sgp4"].items():
        aos = "—" if values["maximum_absolute_aos_difference_seconds"] is None else f"{values['maximum_absolute_aos_difference_seconds']:.6f}"
        los = "—" if values["maximum_absolute_los_difference_seconds"] is None else f"{values['maximum_absolute_los_difference_seconds']:.6f}"
        lines.append(
            f"| {model} | {values['matched_pass_count']} | {values['reference_unmatched_count']} | {values['comparison_unmatched_count']} | {aos} | {los} |"
        )

    lines += [
        "",
        "## Validation",
        "",
        f"Overall status: **{validation_status['overall_status']}**",
        "",
        f"Failed checks: {validation_status['failed_check_count']}",
        "",
        "## Interpretation",
        "",
        "Pass-time differences are consequences of propagation-model differences under one common initial state. They are not direct errors against measured station tracking data.",
        "",
        "## Assumptions",
        "",
        "- WGS-84 fixed station coordinates",
        "- ITRS satellite states",
        "- geometric line of sight",
        "- no terrain or local horizon mask",
        "- no atmospheric refraction",
        "- no antenna slew, gain or link-budget constraints",
        "",
        "## Known limitations",
        "",
        "A predicted pass does not guarantee a usable communications contact. Local obstructions, refraction, radio-frequency effects and operational constraints are excluded.",
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
