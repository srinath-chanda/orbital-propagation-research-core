"""Research data exports and technical figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data_models import CartesianState, ClassicalElements, StateHistory


def write_json(data: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")


def write_initial_conditions_csv(
    output_path: str | Path,
    elements: ClassicalElements,
    initial_state: CartesianState,
    reconstructed_elements: dict[str, Any],
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = elements.as_degrees_dict()
    row = {
        "epoch_utc": initial_state.epoch_utc,
        "frame": initial_state.frame,
        **values,
        "position_x_km": initial_state.position_km[0],
        "position_y_km": initial_state.position_km[1],
        "position_z_km": initial_state.position_km[2],
        "velocity_x_km_s": initial_state.velocity_km_s[0],
        "velocity_y_km_s": initial_state.velocity_km_s[1],
        "velocity_z_km_s": initial_state.velocity_km_s[2],
        "reconstructed_semi_major_axis_km": reconstructed_elements[
            "semi_major_axis_km"
        ],
        "reconstructed_eccentricity": reconstructed_elements["eccentricity"],
        "reconstructed_inclination_deg": reconstructed_elements["inclination_deg"],
        "reconstructed_raan_deg": reconstructed_elements["raan_deg"],
        "reconstructed_argument_of_perigee_deg": reconstructed_elements[
            "argument_of_perigee_deg"
        ],
        "reconstructed_true_anomaly_deg": reconstructed_elements[
            "true_anomaly_deg"
        ],
    }
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_state_history_csv(output_path: str | Path, history: StateHistory) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "epoch_utc",
        "frame",
        "elapsed_seconds",
        "timestamp_utc",
        "position_x_km",
        "position_y_km",
        "position_z_km",
        "velocity_x_km_s",
        "velocity_y_km_s",
        "velocity_z_km_s",
        "position_magnitude_km",
        "velocity_magnitude_km_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, elapsed in enumerate(history.elapsed_seconds):
            position = history.positions_km[index]
            velocity = history.velocities_km_s[index]
            writer.writerow(
                {
                    "model": history.model_name,
                    "epoch_utc": history.epoch_utc,
                    "frame": history.frame,
                    "elapsed_seconds": elapsed,
                    "timestamp_utc": history.timestamps_utc[index],
                    "position_x_km": position[0],
                    "position_y_km": position[1],
                    "position_z_km": position[2],
                    "velocity_x_km_s": velocity[0],
                    "velocity_y_km_s": velocity[1],
                    "velocity_z_km_s": velocity[2],
                    "position_magnitude_km": np.linalg.norm(position),
                    "velocity_magnitude_km_s": np.linalg.norm(velocity),
                }
            )


def write_comparison_csv(output_path: str | Path, data: dict[str, Any]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array_fields = [
        "position_delta_x_m",
        "position_delta_y_m",
        "position_delta_z_m",
        "position_difference_m",
        "velocity_delta_x_mm_s",
        "velocity_delta_y_mm_s",
        "velocity_delta_z_mm_s",
        "velocity_difference_mm_s",
    ]
    fields = [
        "reference_model",
        "comparison_model",
        "frame",
        "elapsed_seconds",
        "timestamp_utc",
        *array_fields,
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, elapsed in enumerate(data["elapsed_seconds"]):
            row = {
                "reference_model": data["reference_model"],
                "comparison_model": data["comparison_model"],
                "frame": data["frame"],
                "elapsed_seconds": elapsed,
                "timestamp_utc": data["timestamps_utc"][index],
            }
            for name in array_fields:
                row[name] = data[name][index]
            writer.writerow(row)


def write_conservation_csv(
    output_path: str | Path,
    diagnostics: Iterable[dict[str, Any]],
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "elapsed_seconds",
        "timestamp_utc",
        "specific_energy_km2_s2",
        "relative_energy_drift",
        "angular_momentum_magnitude_km2_s",
        "relative_angular_momentum_drift",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for diagnostic in diagnostics:
            for index, elapsed in enumerate(diagnostic["elapsed_seconds"]):
                writer.writerow(
                    {
                        "model": diagnostic["model_name"],
                        "elapsed_seconds": elapsed,
                        "timestamp_utc": diagnostic["timestamps_utc"][index],
                        "specific_energy_km2_s2": diagnostic[
                            "specific_energy_km2_s2"
                        ][index],
                        "relative_energy_drift": diagnostic[
                            "relative_energy_drift"
                        ][index],
                        "angular_momentum_magnitude_km2_s": diagnostic[
                            "angular_momentum_magnitude_km2_s"
                        ][index],
                        "relative_angular_momentum_drift": diagnostic[
                            "relative_angular_momentum_drift"
                        ][index],
                    }
                )


def write_model_error_summary_csv(
    output_path: str | Path,
    error_summary: dict[str, Any],
    histories: Iterable[StateHistory],
    diagnostics: Iterable[dict[str, Any]],
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history_map = {history.model_name: history for history in histories}
    diagnostic_map = {
        diagnostic["model_name"]: diagnostic for diagnostic in diagnostics
    }
    row = {
        "reference_model": error_summary["reference_model"],
        "comparison_model": error_summary["comparison_model"],
        "frame": error_summary["frame"],
        "final_position_difference_m": error_summary["position_difference_m"][
            "final"
        ],
        "maximum_position_difference_m": error_summary[
            "position_difference_m"
        ]["maximum_absolute"],
        "rms_position_difference_m": error_summary["position_difference_m"]["rms"],
        "time_of_maximum_position_difference_seconds": error_summary[
            "position_difference_m"
        ]["time_of_maximum_seconds"],
        "final_velocity_difference_mm_s": error_summary[
            "velocity_difference_mm_s"
        ]["final"],
        "maximum_velocity_difference_mm_s": error_summary[
            "velocity_difference_mm_s"
        ]["maximum_absolute"],
        "rms_velocity_difference_mm_s": error_summary[
            "velocity_difference_mm_s"
        ]["rms"],
        "analytical_runtime_seconds": history_map[
            "analytical_two_body"
        ].runtime_seconds,
        "numerical_runtime_seconds": history_map[
            "numerical_two_body"
        ].runtime_seconds,
        "numerical_function_evaluations": history_map[
            "numerical_two_body"
        ].function_evaluations,
        "analytical_maximum_relative_energy_drift": diagnostic_map[
            "analytical_two_body"
        ]["maximum_absolute_relative_energy_drift"],
        "numerical_maximum_relative_energy_drift": diagnostic_map[
            "numerical_two_body"
        ]["maximum_absolute_relative_energy_drift"],
        "analytical_maximum_relative_angular_momentum_drift": diagnostic_map[
            "analytical_two_body"
        ]["maximum_absolute_relative_angular_momentum_drift"],
        "numerical_maximum_relative_angular_momentum_drift": diagnostic_map[
            "numerical_two_body"
        ]["maximum_absolute_relative_angular_momentum_drift"],
    }
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _save_figure(
    figure: plt.Figure,
    figures_directory: Path,
    stem: str,
    *,
    save_png: bool,
    save_pdf: bool,
) -> list[Path]:
    figures_directory.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    if save_png:
        png_path = figures_directory / f"{stem}.png"
        figure.savefig(png_path, dpi=180, bbox_inches="tight")
        created.append(png_path)
    if save_pdf:
        pdf_path = figures_directory / f"{stem}.pdf"
        figure.savefig(pdf_path, bbox_inches="tight")
        created.append(pdf_path)
    plt.close(figure)
    return created


def create_two_body_figures(
    figures_directory: str | Path,
    comparison_data: dict[str, Any],
    diagnostics: Iterable[dict[str, Any]],
    *,
    save_png: bool,
    save_pdf: bool,
) -> list[Path]:
    directory = Path(figures_directory)
    elapsed_hours = np.asarray(comparison_data["elapsed_seconds"]) / 3600.0
    created: list[Path] = []

    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.plot(elapsed_hours, comparison_data["position_difference_m"])
    axis.set_xlabel("Elapsed time (hours)")
    axis.set_ylabel("Position difference (m)")
    axis.set_title("Numerical versus analytical two-body position difference")
    axis.grid(True)
    created.extend(
        _save_figure(
            figure,
            directory,
            "position_difference_vs_time",
            save_png=save_png,
            save_pdf=save_pdf,
        )
    )

    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.plot(elapsed_hours, comparison_data["velocity_difference_mm_s"])
    axis.set_xlabel("Elapsed time (hours)")
    axis.set_ylabel("Velocity difference (mm/s)")
    axis.set_title("Numerical versus analytical two-body velocity difference")
    axis.grid(True)
    created.extend(
        _save_figure(
            figure,
            directory,
            "velocity_difference_vs_time",
            save_png=save_png,
            save_pdf=save_pdf,
        )
    )

    figure = plt.figure()
    axis = figure.add_subplot(111)
    for diagnostic in diagnostics:
        axis.plot(
            np.asarray(diagnostic["elapsed_seconds"]) / 3600.0,
            diagnostic["relative_energy_drift"],
            label=diagnostic["model_name"],
        )
    axis.set_xlabel("Elapsed time (hours)")
    axis.set_ylabel("Relative specific-energy drift (-)")
    axis.set_title("Two-body specific-energy conservation")
    axis.grid(True)
    axis.legend()
    created.extend(
        _save_figure(
            figure,
            directory,
            "relative_energy_drift_vs_time",
            save_png=save_png,
            save_pdf=save_pdf,
        )
    )

    figure = plt.figure()
    axis = figure.add_subplot(111)
    for diagnostic in diagnostics:
        axis.plot(
            np.asarray(diagnostic["elapsed_seconds"]) / 3600.0,
            diagnostic["relative_angular_momentum_drift"],
            label=diagnostic["model_name"],
        )
    axis.set_xlabel("Elapsed time (hours)")
    axis.set_ylabel("Relative angular-momentum drift (-)")
    axis.set_title("Two-body angular-momentum conservation")
    axis.grid(True)
    axis.legend()
    created.extend(
        _save_figure(
            figure,
            directory,
            "relative_angular_momentum_drift_vs_time",
            save_png=save_png,
            save_pdf=save_pdf,
        )
    )

    return created


def write_element_history_csv(
    output_path: str | Path,
    element_history: dict[str, Any],
) -> None:
    """Write an osculating orbital-element history."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array_fields = [
        "semi_major_axis_km",
        "eccentricity",
        "inclination_deg",
        "raan_deg",
        "raan_unwrapped_deg",
        "argument_of_perigee_deg",
        "true_anomaly_deg",
    ]
    fields = [
        "model",
        "frame",
        "elapsed_seconds",
        "timestamp_utc",
        *array_fields,
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, elapsed in enumerate(element_history["elapsed_seconds"]):
            row = {
                "model": element_history["model_name"],
                "frame": element_history["frame"],
                "elapsed_seconds": elapsed,
                "timestamp_utc": element_history["timestamps_utc"][index],
            }
            for field in array_fields:
                row[field] = element_history[field][index]
            writer.writerow(row)


def write_rtn_comparison_csv(
    output_path: str | Path,
    rtn_data: dict[str, Any],
) -> None:
    """Write model differences projected into the reference RTN frame."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array_fields = [
        "radial_position_difference_m",
        "along_track_position_difference_m",
        "cross_track_position_difference_m",
        "radial_velocity_difference_mm_s",
        "along_track_velocity_difference_mm_s",
        "cross_track_velocity_difference_mm_s",
    ]
    fields = [
        "reference_model",
        "comparison_model",
        "frame",
        "elapsed_seconds",
        "timestamp_utc",
        *array_fields,
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, elapsed in enumerate(rtn_data["elapsed_seconds"]):
            row = {
                "reference_model": rtn_data["reference_model"],
                "comparison_model": rtn_data["comparison_model"],
                "frame": rtn_data["frame"],
                "elapsed_seconds": elapsed,
                "timestamp_utc": rtn_data["timestamps_utc"][index],
            }
            for field in array_fields:
                row[field] = rtn_data[field][index]
            writer.writerow(row)


def write_j2_conservation_csv(
    output_path: str | Path,
    diagnostics: dict[str, Any],
) -> None:
    """Write J2 total-energy and z-angular-momentum conservation history."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "elapsed_seconds",
        "timestamp_utc",
        "total_specific_energy_km2_s2",
        "relative_total_energy_drift",
        "angular_momentum_z_km2_s",
        "relative_angular_momentum_z_drift",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, elapsed in enumerate(diagnostics["elapsed_seconds"]):
            writer.writerow(
                {
                    "model": diagnostics["model_name"],
                    "elapsed_seconds": elapsed,
                    "timestamp_utc": diagnostics["timestamps_utc"][index],
                    "total_specific_energy_km2_s2": diagnostics[
                        "total_specific_energy_km2_s2"
                    ][index],
                    "relative_total_energy_drift": diagnostics[
                        "relative_total_energy_drift"
                    ][index],
                    "angular_momentum_z_km2_s": diagnostics[
                        "angular_momentum_z_km2_s"
                    ][index],
                    "relative_angular_momentum_z_drift": diagnostics[
                        "relative_angular_momentum_z_drift"
                    ][index],
                }
            )


def write_j2_validation_csv(
    output_path: str | Path,
    validation_summary: dict[str, Any],
) -> None:
    """Write the main J2 validation values as one CSV row."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    excluded = {"conservation_diagnostics"}
    row = {
        key: value
        for key, value in validation_summary.items()
        if key not in excluded and np.isscalar(value)
    }
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def create_j2_figures(
    figures_directory: str | Path,
    *,
    j2_comparison: dict[str, Any],
    rtn_comparison: dict[str, Any],
    element_history: dict[str, Any],
    j2_validation: dict[str, Any],
    save_png: bool,
    save_pdf: bool,
) -> list[Path]:
    """Create the standard J2 analysis figure set."""
    directory = Path(figures_directory)
    elapsed_hours = np.asarray(j2_comparison["elapsed_seconds"]) / 3600.0
    elapsed_seconds = np.asarray(element_history["elapsed_seconds"], dtype=float)
    created: list[Path] = []

    figure = plt.figure()
    axis = figure.add_subplot(111)
    numerical_raan = np.asarray(element_history["raan_unwrapped_deg"], dtype=float)
    initial_raan = float(numerical_raan[0])
    analytical_raan = initial_raan + np.degrees(
        j2_validation["analytical_raan_rate_rad_s"] * elapsed_seconds
    )
    fitted_raan = np.degrees(
        j2_validation["fitted_intercept_rad"]
        + j2_validation["fitted_raan_rate_rad_s"] * elapsed_seconds
    )
    axis.plot(elapsed_hours, numerical_raan, label="Numerical osculating RAAN")
    axis.plot(elapsed_hours, analytical_raan, linestyle="--", label="Analytical secular RAAN")
    axis.plot(elapsed_hours, fitted_raan, linestyle=":", label="Fitted numerical trend")
    axis.set_xlabel("Elapsed time (hours)")
    axis.set_ylabel("Unwrapped RAAN (deg)")
    axis.set_title("J2 nodal precession")
    axis.grid(True)
    axis.legend()
    created.extend(
        _save_figure(
            figure,
            directory,
            "raan_evolution",
            save_png=save_png,
            save_pdf=save_pdf,
        )
    )

    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.plot(elapsed_hours, j2_comparison["position_difference_m"] / 1000.0)
    axis.set_xlabel("Elapsed time (hours)")
    axis.set_ylabel("Position separation (km)")
    axis.set_title("Numerical J2 versus numerical two-body separation")
    axis.grid(True)
    created.extend(
        _save_figure(
            figure,
            directory,
            "j2_two_body_position_separation",
            save_png=save_png,
            save_pdf=save_pdf,
        )
    )

    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.plot(
        elapsed_hours,
        np.asarray(rtn_comparison["cross_track_position_difference_m"]) / 1000.0,
    )
    axis.set_xlabel("Elapsed time (hours)")
    axis.set_ylabel("Cross-track separation (km)")
    axis.set_title("J2 cross-track separation in reference RTN frame")
    axis.grid(True)
    created.extend(
        _save_figure(
            figure,
            directory,
            "cross_track_separation",
            save_png=save_png,
            save_pdf=save_pdf,
        )
    )

    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.plot(elapsed_hours, element_history["semi_major_axis_km"])
    axis.set_xlabel("Elapsed time (hours)")
    axis.set_ylabel("Osculating semi-major axis (km)")
    axis.set_title("J2 osculating semi-major-axis variation")
    axis.grid(True)
    created.extend(
        _save_figure(
            figure,
            directory,
            "j2_semi_major_axis_evolution",
            save_png=save_png,
            save_pdf=save_pdf,
        )
    )

    figure = plt.figure()
    axis = figure.add_subplot(111)
    conservation = j2_validation["conservation_diagnostics"]
    axis.plot(
        elapsed_hours,
        conservation["relative_total_energy_drift"],
        label="Total J2 energy",
    )
    axis.plot(
        elapsed_hours,
        conservation["relative_angular_momentum_z_drift"],
        label="Angular momentum z-component",
    )
    axis.set_xlabel("Elapsed time (hours)")
    axis.set_ylabel("Relative drift (-)")
    axis.set_title("J2 conservative invariants")
    axis.grid(True)
    axis.legend()
    created.extend(
        _save_figure(
            figure,
            directory,
            "j2_conservation_diagnostics",
            save_png=save_png,
            save_pdf=save_pdf,
        )
    )

    return created


def write_drag_diagnostics_csv(
    output_path: str | Path,
    diagnostics: dict[str, Any],
) -> None:
    """Write simplified-atmosphere and dissipative-energy diagnostics."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array_fields = [
        "altitude_km",
        "density_kg_m3",
        "relative_speed_km_s",
        "drag_acceleration_m_s2",
        "drag_power_km2_s3",
        "drag_relative_power_km2_s3",
        "total_specific_energy_km2_s2",
        "total_specific_energy_change_km2_s2",
    ]
    fields = [
        "model",
        "frame",
        "elapsed_seconds",
        "timestamp_utc",
        *array_fields,
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, elapsed in enumerate(diagnostics["elapsed_seconds"]):
            row = {
                "model": diagnostics["model_name"],
                "frame": diagnostics["frame"],
                "elapsed_seconds": elapsed,
                "timestamp_utc": diagnostics["timestamps_utc"][index],
            }
            for field in array_fields:
                row[field] = diagnostics[field][index]
            writer.writerow(row)


def write_drag_validation_csv(
    output_path: str | Path,
    validation_summary: dict[str, Any],
) -> None:
    """Write scalar simplified-drag validation values."""
    excluded = {
        "comparison_error_summary",
        "semi_major_axis_difference_vs_j2_m",
    }
    row = {
        key: value
        for key, value in validation_summary.items()
        if key not in excluded and np.isscalar(value)
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_drag_sensitivity_csv(
    output_path: str | Path,
    sensitivity_results: list[dict[str, Any]],
) -> None:
    """Write one-at-a-time drag sensitivity cases."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not sensitivity_results:
        path.write_text("case_id\n", encoding="utf-8")
        return
    fields = list(sensitivity_results[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sensitivity_results)


def create_drag_figures(
    figures_directory: str | Path,
    *,
    j2_history: StateHistory,
    drag_history: StateHistory,
    drag_comparison: dict[str, Any],
    rtn_comparison: dict[str, Any],
    j2_element_history: dict[str, Any],
    drag_element_history: dict[str, Any],
    drag_diagnostics: dict[str, Any],
    sensitivity_results: list[dict[str, Any]],
    save_png: bool,
    save_pdf: bool,
) -> list[Path]:
    """Create the standard simplified-drag analysis figure set."""
    directory = Path(figures_directory)
    elapsed_hours = np.asarray(drag_history.elapsed_seconds, dtype=float) / 3600.0
    created: list[Path] = []

    j2_altitude = np.linalg.norm(j2_history.positions_km, axis=1)
    drag_altitude = np.linalg.norm(drag_history.positions_km, axis=1)
    drag_altitude_difference_m = (drag_altitude - j2_altitude) * 1000.0

    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.plot(elapsed_hours, j2_altitude, label="J2 radius")
    axis.plot(elapsed_hours, drag_altitude, label="J2 + drag radius")
    axis.set_xlabel("Elapsed time (hours)")
    axis.set_ylabel("Geocentric radius (km)")
    axis.set_title("J2 and simplified-drag radial histories")
    axis.grid(True)
    axis.legend()
    created.extend(
        _save_figure(
            figure,
            directory,
            "drag_radius_comparison",
            save_png=save_png,
            save_pdf=save_pdf,
        )
    )

    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.plot(elapsed_hours, drag_altitude_difference_m)
    axis.set_xlabel("Elapsed time (hours)")
    axis.set_ylabel("Radius difference from J2 (m)")
    axis.set_title("Simplified-drag radial difference from J2")
    axis.grid(True)
    created.extend(
        _save_figure(
            figure,
            directory,
            "drag_radius_difference_vs_j2",
            save_png=save_png,
            save_pdf=save_pdf,
        )
    )

    j2_a = np.asarray(j2_element_history["semi_major_axis_km"], dtype=float)
    drag_a = np.asarray(drag_element_history["semi_major_axis_km"], dtype=float)
    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.plot(elapsed_hours, (drag_a - j2_a) * 1000.0)
    axis.set_xlabel("Elapsed time (hours)")
    axis.set_ylabel("Semi-major-axis difference from J2 (m)")
    axis.set_title("Simplified-drag semi-major-axis effect")
    axis.grid(True)
    created.extend(
        _save_figure(
            figure,
            directory,
            "drag_semi_major_axis_difference",
            save_png=save_png,
            save_pdf=save_pdf,
        )
    )

    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.plot(
        elapsed_hours,
        np.asarray(rtn_comparison["along_track_position_difference_m"])
        / 1000.0,
    )
    axis.set_xlabel("Elapsed time (hours)")
    axis.set_ylabel("Along-track difference from J2 (km)")
    axis.set_title("Simplified-drag along-track separation")
    axis.grid(True)
    created.extend(
        _save_figure(
            figure,
            directory,
            "drag_along_track_separation",
            save_png=save_png,
            save_pdf=save_pdf,
        )
    )

    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.plot(elapsed_hours, drag_diagnostics["density_kg_m3"])
    axis.set_xlabel("Elapsed time (hours)")
    axis.set_ylabel("Density (kg/m³)")
    axis.set_title("Simplified exponential-atmosphere density")
    axis.grid(True)
    created.extend(
        _save_figure(
            figure,
            directory,
            "drag_density_history",
            save_png=save_png,
            save_pdf=save_pdf,
        )
    )

    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.plot(
        elapsed_hours,
        drag_diagnostics["total_specific_energy_change_km2_s2"],
    )
    axis.set_xlabel("Elapsed time (hours)")
    axis.set_ylabel("Total specific-energy change (km²/s²)")
    axis.set_title("Energy dissipation from simplified drag")
    axis.grid(True)
    created.extend(
        _save_figure(
            figure,
            directory,
            "drag_total_energy_change",
            save_png=save_png,
            save_pdf=save_pdf,
        )
    )

    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.plot(
        elapsed_hours,
        np.asarray(drag_comparison["position_difference_m"]) / 1000.0,
    )
    axis.set_xlabel("Elapsed time (hours)")
    axis.set_ylabel("Position separation from J2 (km)")
    axis.set_title("J2 + simplified drag versus J2 separation")
    axis.grid(True)
    created.extend(
        _save_figure(
            figure,
            directory,
            "drag_j2_position_separation",
            save_png=save_png,
            save_pdf=save_pdf,
        )
    )

    if sensitivity_results:
        labels = [item["case_id"] for item in sensitivity_results]
        values = [
            item["final_semi_major_axis_difference_vs_j2_m"]
            for item in sensitivity_results
        ]
        figure = plt.figure(figsize=(10, 5))
        axis = figure.add_subplot(111)
        positions = np.arange(len(labels))
        axis.bar(positions, values)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=45, ha="right")
        axis.set_ylabel("Final semi-major-axis difference from J2 (m)")
        axis.set_title("One-at-a-time simplified-drag sensitivity")
        axis.grid(True, axis="y")
        created.extend(
            _save_figure(
                figure,
                directory,
                "drag_sensitivity_semi_major_axis",
                save_png=save_png,
                save_pdf=save_pdf,
            )
        )

    return created
