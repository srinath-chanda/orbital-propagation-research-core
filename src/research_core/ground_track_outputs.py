"""CSV, figure and Markdown outputs for Research Core 1A.5 ground tracks."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .ground_track import GroundTrackHistory, split_at_antimeridian


def write_ground_track_csv(path: str | Path, track: GroundTrackHistory) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "source_frame",
        "earth_fixed_frame",
        "ellipsoid",
        "epoch_utc",
        "elapsed_seconds",
        "timestamp_utc",
        "itrs_x_km",
        "itrs_y_km",
        "itrs_z_km",
        "itrs_vx_km_s",
        "itrs_vy_km_s",
        "itrs_vz_km_s",
        "latitude_deg",
        "longitude_deg",
        "altitude_km",
    ]
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, elapsed in enumerate(track.elapsed_seconds):
            writer.writerow(
                {
                    "model": track.model_name,
                    "source_frame": track.source_frame,
                    "earth_fixed_frame": track.earth_fixed_frame,
                    "ellipsoid": track.ellipsoid,
                    "epoch_utc": track.epoch_utc,
                    "elapsed_seconds": float(elapsed),
                    "timestamp_utc": track.timestamps_utc[index],
                    "itrs_x_km": track.positions_itrs_km[index, 0],
                    "itrs_y_km": track.positions_itrs_km[index, 1],
                    "itrs_z_km": track.positions_itrs_km[index, 2],
                    "itrs_vx_km_s": track.velocities_itrs_km_s[index, 0],
                    "itrs_vy_km_s": track.velocities_itrs_km_s[index, 1],
                    "itrs_vz_km_s": track.velocities_itrs_km_s[index, 2],
                    "latitude_deg": track.latitude_deg[index],
                    "longitude_deg": track.longitude_deg[index],
                    "altitude_km": track.altitude_km[index],
                }
            )


def write_ground_track_comparison_csv(path: str | Path, comparison: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "reference_model",
        "comparison_model",
        "frame",
        "ellipsoid",
        "elapsed_seconds",
        "timestamp_utc",
        "latitude_difference_deg",
        "longitude_difference_deg",
        "altitude_difference_km",
        "surface_separation_km",
        "itrs_position_separation_km",
    ]
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        n = len(comparison["timestamps_utc"])
        for index in range(n):
            writer.writerow(
                {
                    "reference_model": comparison["reference_model"],
                    "comparison_model": comparison["comparison_model"],
                    "frame": comparison["frame"],
                    "ellipsoid": comparison["ellipsoid"],
                    "elapsed_seconds": comparison["elapsed_seconds"][index],
                    "timestamp_utc": comparison["timestamps_utc"][index],
                    "latitude_difference_deg": comparison[
                        "latitude_difference_deg"
                    ][index],
                    "longitude_difference_deg": comparison[
                        "longitude_difference_deg"
                    ][index],
                    "altitude_difference_km": comparison[
                        "altitude_difference_km"
                    ][index],
                    "surface_separation_km": comparison[
                        "surface_separation_km"
                    ][index],
                    "itrs_position_separation_km": comparison[
                        "itrs_position_separation_km"
                    ][index],
                }
            )


def write_ground_track_summary_csv(path: str | Path, summary: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "reference_model",
        "comparison_model",
        "frame",
        "ellipsoid",
        "duration_hours",
        "final_surface_separation_km",
        "maximum_surface_separation_km",
        "rms_surface_separation_km",
        "final_altitude_difference_km",
        "maximum_absolute_altitude_difference_km",
    ]
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model_name, values in summary["models"].items():
            writer.writerow(
                {
                    "reference_model": summary["reference_model"],
                    "comparison_model": model_name,
                    "frame": summary["frame"],
                    "ellipsoid": summary["ellipsoid"],
                    "duration_hours": summary["duration_hours"],
                    **values,
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


def _draw_map_background(ax: plt.Axes, background_file: Path | None) -> None:
    if background_file is not None and background_file.is_file():
        image = plt.imread(background_file)
        ax.imshow(image, extent=[-180, 180, -90, 90], aspect="auto", alpha=0.72)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xticks(np.arange(-180.0, 181.0, 30.0))
    ax.set_yticks(np.arange(-90.0, 91.0, 15.0))
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Geodetic latitude (deg)")


def _plot_track(
    ax: plt.Axes,
    track: GroundTrackHistory,
    *,
    label: str,
    linewidth: float,
    alpha: float,
) -> None:
    first = True
    for longitude, latitude in split_at_antimeridian(
        track.longitude_deg, track.latitude_deg
    ):
        ax.plot(
            longitude,
            latitude,
            linewidth=linewidth,
            alpha=alpha,
            label=label if first else None,
        )
        first = False


def create_ground_track_figures(
    directory: str | Path,
    tracks: list[GroundTrackHistory],
    comparisons: list[dict[str, Any]],
    *,
    background_file: str | Path | None,
    save_png: bool,
    save_pdf: bool,
) -> list[Path]:
    output_directory = Path(directory)
    background = Path(background_file) if background_file else None
    created: list[Path] = []
    reference = tracks[0]
    hours = reference.elapsed_seconds / 3600.0

    figure, ax = plt.subplots(figsize=(12.0, 6.2))
    _draw_map_background(ax, background)
    for index, track in enumerate(tracks):
        _plot_track(
            ax,
            track,
            label=track.model_name,
            linewidth=2.0 if index == 0 else 1.1,
            alpha=1.0 if index == 0 else 0.82,
        )
    ax.scatter(
        [reference.longitude_deg[0]],
        [reference.latitude_deg[0]],
        marker="o",
        s=28,
        label="start",
    )
    ax.set_title("24-hour Earth-fixed ground-track comparison")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3)
    created += _save(
        figure,
        output_directory,
        "ground_track_comparison",
        save_png,
        save_pdf,
    )

    figure, ax = plt.subplots(figsize=(12.0, 6.2))
    _draw_map_background(ax, background)
    _plot_track(ax, reference, label=reference.model_name, linewidth=1.5, alpha=1.0)
    ax.scatter(
        reference.longitude_deg[:: max(1, reference.longitude_deg.size // 24)],
        reference.latitude_deg[:: max(1, reference.latitude_deg.size // 24)],
        s=10,
        label="approximately hourly samples",
    )
    ax.set_title("SGP4 Earth-fixed ground track")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2)
    created += _save(
        figure,
        output_directory,
        "sgp4_ground_track",
        save_png,
        save_pdf,
    )

    figure, ax = plt.subplots(figsize=(9.2, 5.4))
    for comparison in comparisons:
        ax.plot(
            hours,
            comparison["surface_separation_km"],
            label=comparison["comparison_model"],
        )
    ax.set_xlabel("Elapsed time (hours)")
    ax.set_ylabel("Sub-satellite-point separation (km)")
    ax.set_title("Ground-track surface separation from SGP4")
    ax.grid(True, alpha=0.3)
    ax.legend()
    created += _save(
        figure,
        output_directory,
        "ground_track_surface_separation",
        save_png,
        save_pdf,
    )

    figure, ax = plt.subplots(figsize=(9.2, 5.4))
    for track in tracks:
        ax.plot(hours, track.altitude_km, label=track.model_name)
    ax.set_xlabel("Elapsed time (hours)")
    ax.set_ylabel("WGS-84 geodetic altitude (km)")
    ax.set_title("Geodetic altitude histories")
    ax.grid(True, alpha=0.3)
    ax.legend()
    created += _save(
        figure,
        output_directory,
        "ground_track_altitude_history",
        save_png,
        save_pdf,
    )

    figure, axes = plt.subplots(2, 1, figsize=(9.2, 7.4), sharex=True)
    for comparison in comparisons:
        axes[0].plot(
            hours,
            comparison["latitude_difference_deg"],
            label=comparison["comparison_model"],
        )
        axes[1].plot(
            hours,
            comparison["longitude_difference_deg"],
            label=comparison["comparison_model"],
        )
    axes[0].set_ylabel("Latitude difference (deg)")
    axes[1].set_ylabel("Wrapped longitude difference (deg)")
    axes[1].set_xlabel("Elapsed time (hours)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    figure.suptitle("Ground-track angular differences from SGP4")
    created += _save(
        figure,
        output_directory,
        "ground_track_angular_differences",
        save_png,
        save_pdf,
    )
    return created


def write_ground_track_technical_summary(
    path: str | Path,
    *,
    summary: dict[str, Any],
    roundtrip: dict[str, Any],
    validation_status: dict[str, Any],
) -> None:
    lines = [
        "# Research Core 1A.5 — Earth-Fixed Coordinates and Ground Tracks",
        "",
        "## Purpose",
        "",
        "This analysis transforms synchronized GCRS trajectories to ITRS and then",
        "to WGS-84 geodetic latitude, longitude and altitude.",
        "",
        "## Coordinate workflow",
        "",
        "```text",
        "GCRS Cartesian state",
        "→ Astropy GCRS-to-ITRS transformation",
        "→ EarthLocation WGS-84 geodetic conversion",
        "→ latitude, longitude and altitude",
        "```",
        "",
        f"Earth-fixed frame: `{summary['frame']}`.",
        f"Reference ellipsoid: `{summary['ellipsoid']}`.",
        f"Samples: {summary['sample_count']} over {summary['duration_hours']:.6f} hours.",
        "",
        "## Geodetic reconstruction validation",
        "",
        f"Maximum geodetic→ITRS reconstruction residual: {roundtrip['maximum_position_residual_m']:.6e} m.",
        f"RMS reconstruction residual: {roundtrip['rms_position_residual_m']:.6e} m.",
        "",
        "## SGP4 reference ground track",
        "",
        f"Latitude range: {summary['reference_latitude_range_deg'][0]:.6f}° to {summary['reference_latitude_range_deg'][1]:.6f}°.",
        f"Altitude range: {summary['reference_minimum_altitude_km']:.6f} km to {summary['reference_maximum_altitude_km']:.6f} km.",
        "",
        "## Model ground-track separations",
        "",
        "| Model | Final surface separation (km) | Maximum surface separation (km) | RMS surface separation (km) | Final altitude difference (km) |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_name, model in summary["models"].items():
        lines.append(
            f"| {model_name} | {model['final_surface_separation_km']:.6f} | "
            f"{model['maximum_surface_separation_km']:.6f} | "
            f"{model['rms_surface_separation_km']:.6f} | "
            f"{model['final_altitude_difference_km']:.6f} |"
        )
    lines += [
        "",
        "## Validation status",
        "",
        f"Overall status: **{validation_status['overall_status']}**.",
        "",
        "## Interpretation",
        "",
        "Ground-track separation is the distance between model sub-satellite points.",
        "It is calculated from geodetic latitude and longitude using a spherical",
        "central-angle approximation with the configured Earth radius. It is not a",
        "full ellipsoidal geodesic and is not measured orbit error.",
        "",
        "## Known limitations",
        "",
        "- Earth-orientation accuracy depends on the installed Astropy IERS data.",
        "- The map background is presentation context and is not used in calculations.",
        "- Ground-track lines are split at the anti-meridian to avoid false map-spanning segments.",
        "- Terrain, topography, refraction and uncertainty corridors are not included.",
        "- Ground-station pass prediction is not included until Research Core 1A.6.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
