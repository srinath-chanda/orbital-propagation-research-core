"""Outputs for the fixed-TLE and SGP4 common-state experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data_models import StateHistory


def write_json(data: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")


def write_tle_age_csv(path: str | Path, report: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["timestamp_utc", "elapsed_seconds", "tle_age_seconds", "tle_age_hours"],
        )
        writer.writeheader()
        for i, timestamp in enumerate(report["timestamps_utc"]):
            writer.writerow({
                "timestamp_utc": timestamp,
                "elapsed_seconds": report["elapsed_seconds"][i],
                "tle_age_seconds": report["tle_age_seconds"][i],
                "tle_age_hours": report["tle_age_hours"][i],
            })


def write_sgp4_model_summary_csv(path: str | Path, summary: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "reference_model", "comparison_model", "frame", "duration_hours",
        "final_position_difference_km", "maximum_position_difference_km",
        "rms_position_difference_km", "final_velocity_difference_mm_s",
        "maximum_velocity_difference_mm_s", "runtime_seconds", "function_evaluations",
    ]
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model_name, model in summary["models"].items():
            writer.writerow({
                "reference_model": "sgp4",
                "comparison_model": model_name,
                "frame": summary["frame"],
                "duration_hours": summary["duration_hours"],
                "final_position_difference_km": model["final_position_difference_km"],
                "maximum_position_difference_km": model["maximum_position_difference_km"],
                "rms_position_difference_km": model["rms_position_difference_km"],
                "final_velocity_difference_mm_s": model["velocity_difference_mm_s"]["final"],
                "maximum_velocity_difference_mm_s": model["velocity_difference_mm_s"]["maximum_absolute"],
                "runtime_seconds": model["runtime_seconds"],
                "function_evaluations": model["function_evaluations"],
            })


def _save(fig: plt.Figure, directory: Path, stem: str, save_png: bool, save_pdf: bool) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    if save_png:
        path = directory / f"{stem}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        created.append(path)
    if save_pdf:
        path = directory / f"{stem}.pdf"
        fig.savefig(path, bbox_inches="tight")
        created.append(path)
    plt.close(fig)
    return created


def create_sgp4_figures(
    figures_directory: str | Path,
    sgp4_history: StateHistory,
    comparisons: list[dict[str, Any]],
    rtn_comparisons: list[dict[str, Any]],
    age_report: dict[str, Any],
    *,
    save_png: bool,
    save_pdf: bool,
) -> list[Path]:
    directory = Path(figures_directory)
    created: list[Path] = []
    hours = sgp4_history.elapsed_seconds / 3600.0

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for comparison in comparisons:
        ax.plot(hours, np.asarray(comparison["position_difference_m"]) / 1000.0, label=comparison["comparison_model"])
    ax.set_xlabel("Elapsed time (hours)")
    ax.set_ylabel("Position separation from SGP4 (km)")
    ax.set_title("Common-state model separation from SGP4")
    ax.grid(True, alpha=0.3)
    ax.legend()
    created += _save(fig, directory, "sgp4_model_position_separation", save_png, save_pdf)

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for comparison in comparisons:
        ax.plot(hours, np.asarray(comparison["velocity_difference_mm_s"]) / 1000.0, label=comparison["comparison_model"])
    ax.set_xlabel("Elapsed time (hours)")
    ax.set_ylabel("Velocity separation from SGP4 (m/s)")
    ax.set_title("Velocity-model separation from SGP4")
    ax.grid(True, alpha=0.3)
    ax.legend()
    created += _save(fig, directory, "sgp4_model_velocity_separation", save_png, save_pdf)

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 9.0), sharex=True)
    components = [
        ("radial_position_difference_m", "Radial (km)"),
        ("along_track_position_difference_m", "Along-track (km)"),
        ("cross_track_position_difference_m", "Cross-track (km)"),
    ]
    for ax, (key, label) in zip(axes, components):
        for comparison in rtn_comparisons:
            ax.plot(hours, np.asarray(comparison[key]) / 1000.0, label=comparison["comparison_model"])
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("Elapsed time (hours)")
    fig.suptitle("RTN position differences relative to SGP4")
    created += _save(fig, directory, "sgp4_rtn_position_differences", save_png, save_pdf)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    radius = np.linalg.norm(sgp4_history.positions_km, axis=1)
    ax.plot(hours, radius)
    ax.set_xlabel("Elapsed time (hours)")
    ax.set_ylabel("GCRS geocentric radius (km)")
    ax.set_title("SGP4 transformed GCRS radius history")
    ax.grid(True, alpha=0.3)
    created += _save(fig, directory, "sgp4_gcrs_radius_history", save_png, save_pdf)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(hours, age_report["tle_age_hours"])
    ax.set_xlabel("Elapsed time (hours)")
    ax.set_ylabel("Age relative to TLE epoch (hours)")
    ax.set_title("Frozen TLE age during experiment")
    ax.grid(True, alpha=0.3)
    created += _save(fig, directory, "tle_age_history", save_png, save_pdf)

    return created


def write_technical_summary(
    path: str | Path,
    *,
    provenance: dict[str, Any],
    summary: dict[str, Any],
    initial_differences: dict[str, Any],
    frame_roundtrip: dict[str, Any],
    age_report: dict[str, Any],
    validation_status: dict[str, Any],
) -> None:
    lines = [
        "# Research Core 1A.5 — Fixed TLE, SGP4 and Ground-Track Foundation",
        "",
        "## Purpose",
        "",
        "This run propagates one frozen ISS TLE with SGP4, transforms every TEME state to GCRS,",
        "and starts the analytical and numerical comparison models from the same GCRS Cartesian state.",
        "",
        "## Frozen input",
        "",
        f"- Object: {provenance['object_name']}",
        f"- NORAD catalog number: {provenance['norad_catalog_number']}",
        f"- TLE epoch: {provenance['tle_epoch_utc']}",
        f"- Retrieved: {provenance.get('retrieved_utc')}",
        f"- Source: {provenance.get('source_organisation')}",
        f"- File SHA-256: `{provenance['tle_file_sha256']}`",
        "",
        "## Frame handling",
        "",
        "SGP4 natively produces TEME states. This run uses Astropy to transform every SGP4 state",
        "to GCRS before model-to-model subtraction. The numerical models use the transformed",
        "GCRS state at the TLE epoch as their common initial condition.",
        "",
        f"TEME→GCRS→TEME round-trip position residual: {frame_roundtrip['position_roundtrip_error_m']:.6e} m.",
        f"Round-trip velocity residual: {frame_roundtrip['velocity_roundtrip_error_mm_s']:.6e} mm/s.",
        "",
        "## TLE age",
        "",
        f"The run begins at {age_report['start_age_hours']:.6f} hours from the TLE epoch and ends at {age_report['end_age_hours']:.6f} hours.",
        "",
        "## Common-state comparisons",
        "",
        "| Model | Final separation (km) | Maximum separation (km) | RMS separation (km) |",
        "|---|---:|---:|---:|",
    ]
    for model_name, model in summary["models"].items():
        lines.append(
            f"| {model_name} | {model['final_position_difference_km']:.6f} | "
            f"{model['maximum_position_difference_km']:.6f} | {model['rms_position_difference_km']:.6f} |"
        )
    lines += [
        "",
        "## Initial-state equality",
        "",
    ]
    for model_name, values in initial_differences.items():
        lines.append(
            f"- {model_name}: {values['position_difference_m']:.6e} m position, "
            f"{values['velocity_difference_mm_s']:.6e} mm/s velocity."
        )
    lines += [
        "",
        "## Validation status",
        "",
        f"Overall status: **{validation_status['overall_status']}**.",
        "",
        "## Interpretation",
        "",
        "The separation curves are differences between propagation models that share an initial state.",
        "They are not direct errors against measured ISS truth. SGP4 is also a model tied to the selected TLE.",
        "",
        "## Known limitations",
        "",
        "- The TLE is a frozen snapshot and becomes older as propagation proceeds.",
        "- TLE elements are SGP4-specific mean elements, not ordinary osculating Keplerian elements.",
        "- The numerical J2 model uses a simplified fixed-axis representation in GCRS.",
        "- The drag atmosphere remains illustrative and is not fitted to the TLE B* term.",
        "- No measured ephemeris, covariance, orbit determination, terrain mask, or pass validation is included yet.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
