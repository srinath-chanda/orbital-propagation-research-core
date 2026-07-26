"""Simplified atmospheric-drag diagnostics and sensitivity analysis."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from ..data_models import CartesianState, StateHistory
from ..propagators.numerical_drag import (
    atmospheric_relative_velocity_km_s,
    drag_acceleration_km_s2,
    exponential_atmospheric_density_kg_m3,
    propagate_numerical_j2_drag,
)
from .comparison import compare_state_histories, create_error_summary
from .j2 import j2_specific_potential_km2_s2


def drag_parameter_dict(drag_config: dict[str, Any]) -> dict[str, Any]:
    """Return only the parameters consumed by the drag propagator."""
    return {
        "mass_kg": float(drag_config["mass_kg"]),
        "cross_sectional_area_m2": float(
            drag_config["cross_sectional_area_m2"]
        ),
        "drag_coefficient": float(drag_config["drag_coefficient"]),
        "reference_altitude_km": float(drag_config["reference_altitude_km"]),
        "reference_density_kg_m3": float(
            drag_config["reference_density_kg_m3"]
        ),
        "scale_height_km": float(drag_config["scale_height_km"]),
        "co_rotating_atmosphere": bool(
            drag_config["co_rotating_atmosphere"]
        ),
    }


def create_drag_diagnostics(
    history: StateHistory,
    *,
    gravitational_parameter_km3_s2: float,
    earth_equatorial_radius_km: float,
    j2: float,
    earth_rotation_rate_rad_s: float,
    drag_config: dict[str, Any],
) -> dict[str, Any]:
    """Create atmospheric, acceleration, and dissipative-energy histories."""
    parameters = drag_parameter_dict(drag_config)
    altitudes = np.linalg.norm(history.positions_km, axis=1) - float(
        earth_equatorial_radius_km
    )
    density: list[float] = []
    relative_speed: list[float] = []
    drag_acceleration_m_s2: list[float] = []
    drag_power_km2_s3: list[float] = []
    drag_relative_power_km2_s3: list[float] = []

    for position, velocity in zip(history.positions_km, history.velocities_km_s):
        rho = exponential_atmospheric_density_kg_m3(
            position,
            earth_equatorial_radius_km,
            parameters["reference_altitude_km"],
            parameters["reference_density_kg_m3"],
            parameters["scale_height_km"],
        )
        relative_velocity = atmospheric_relative_velocity_km_s(
            position,
            velocity,
            earth_rotation_rate_rad_s,
            co_rotating_atmosphere=parameters["co_rotating_atmosphere"],
        )
        acceleration = drag_acceleration_km_s2(
            position,
            velocity,
            earth_equatorial_radius_km=earth_equatorial_radius_km,
            earth_rotation_rate_rad_s=earth_rotation_rate_rad_s,
            **parameters,
        )
        density.append(rho)
        relative_speed.append(float(np.linalg.norm(relative_velocity)))
        drag_acceleration_m_s2.append(float(np.linalg.norm(acceleration)) * 1000.0)
        drag_power_km2_s3.append(float(np.dot(velocity, acceleration)))
        drag_relative_power_km2_s3.append(
            float(np.dot(relative_velocity, acceleration))
        )

    speed_squared = np.sum(history.velocities_km_s**2, axis=1)
    total_specific_energy = 0.5 * speed_squared + j2_specific_potential_km2_s2(
        history.positions_km,
        gravitational_parameter_km3_s2,
        earth_equatorial_radius_km,
        j2,
    )
    energy_change = total_specific_energy - total_specific_energy[0]
    return {
        "model_name": history.model_name,
        "frame": history.frame,
        "epoch_utc": history.epoch_utc,
        "elapsed_seconds": history.elapsed_seconds,
        "timestamps_utc": history.timestamps_utc,
        "altitude_km": np.asarray(altitudes, dtype=float),
        "density_kg_m3": np.asarray(density, dtype=float),
        "relative_speed_km_s": np.asarray(relative_speed, dtype=float),
        "drag_acceleration_m_s2": np.asarray(
            drag_acceleration_m_s2,
            dtype=float,
        ),
        "drag_power_km2_s3": np.asarray(drag_power_km2_s3, dtype=float),
        "drag_relative_power_km2_s3": np.asarray(
            drag_relative_power_km2_s3,
            dtype=float,
        ),
        "total_specific_energy_km2_s2": total_specific_energy,
        "total_specific_energy_change_km2_s2": energy_change,
    }


def create_drag_validation_summary(
    *,
    initial_state: CartesianState,
    j2_history: StateHistory,
    drag_history: StateHistory,
    j2_element_history: dict[str, Any],
    drag_element_history: dict[str, Any],
    drag_diagnostics: dict[str, Any],
    gravitational_parameter_km3_s2: float,
    earth_equatorial_radius_km: float,
    earth_rotation_rate_rad_s: float,
    drag_config: dict[str, Any],
) -> dict[str, Any]:
    """Create scalar quantities used by drag validation and interpretation."""
    comparison = compare_state_histories(j2_history, drag_history)
    error_summary = create_error_summary(comparison)
    parameters = drag_parameter_dict(drag_config)
    initial_relative_velocity = atmospheric_relative_velocity_km_s(
        initial_state.position_km,
        initial_state.velocity_km_s,
        earth_rotation_rate_rad_s,
        co_rotating_atmosphere=parameters["co_rotating_atmosphere"],
    )
    initial_drag = drag_acceleration_km_s2(
        initial_state.position_km,
        initial_state.velocity_km_s,
        earth_equatorial_radius_km=earth_equatorial_radius_km,
        earth_rotation_rate_rad_s=earth_rotation_rate_rad_s,
        **parameters,
    )
    drag_magnitude = float(np.linalg.norm(initial_drag))
    relative_speed = float(np.linalg.norm(initial_relative_velocity))
    drag_direction_cosine = (
        float(np.dot(initial_drag, initial_relative_velocity))
        / (drag_magnitude * relative_speed)
        if drag_magnitude > 0.0 and relative_speed > 0.0
        else -1.0
    )
    central_magnitude = float(
        gravitational_parameter_km3_s2
        / np.linalg.norm(initial_state.position_km) ** 2
    )

    j2_a = np.asarray(j2_element_history["semi_major_axis_km"], dtype=float)
    drag_a = np.asarray(drag_element_history["semi_major_axis_km"], dtype=float)
    delta_a_vs_j2_m = (drag_a - j2_a) * 1000.0
    total_energy = np.asarray(
        drag_diagnostics["total_specific_energy_km2_s2"],
        dtype=float,
    )
    final_energy_change = float(total_energy[-1] - total_energy[0])
    density = np.asarray(drag_diagnostics["density_kg_m3"], dtype=float)
    acceleration = np.asarray(
        drag_diagnostics["drag_acceleration_m_s2"],
        dtype=float,
    )
    ballistic_coefficient = parameters["mass_kg"] / (
        parameters["drag_coefficient"]
        * parameters["cross_sectional_area_m2"]
    )

    return {
        "duration_hours": float(drag_history.elapsed_seconds[-1]) / 3600.0,
        "ballistic_coefficient_kg_m2": float(ballistic_coefficient),
        "initial_density_kg_m3": float(density[0]),
        "minimum_density_kg_m3": float(np.min(density)),
        "maximum_density_kg_m3": float(np.max(density)),
        "initial_relative_speed_km_s": relative_speed,
        "initial_drag_acceleration_m_s2": drag_magnitude * 1000.0,
        "maximum_drag_acceleration_m_s2": float(np.max(acceleration)),
        "initial_drag_to_central_acceleration_ratio": drag_magnitude
        / central_magnitude,
        "initial_drag_relative_velocity_direction_cosine": drag_direction_cosine,
        "initial_drag_relative_power_km2_s3": float(
            drag_diagnostics["drag_relative_power_km2_s3"][0]
        ),
        "final_total_specific_energy_change_km2_s2": final_energy_change,
        "total_specific_energy_loss_km2_s2": -final_energy_change,
        "final_semi_major_axis_change_from_initial_m": float(
            (drag_a[-1] - drag_a[0]) * 1000.0
        ),
        "final_semi_major_axis_difference_vs_j2_m": float(
            delta_a_vs_j2_m[-1]
        ),
        "minimum_semi_major_axis_difference_vs_j2_m": float(
            np.min(delta_a_vs_j2_m)
        ),
        "maximum_position_separation_from_j2_km": float(
            error_summary["position_difference_m"]["maximum_absolute"]
            / 1000.0
        ),
        "final_position_separation_from_j2_km": float(
            error_summary["position_difference_m"]["final"] / 1000.0
        ),
        "comparison_error_summary": error_summary,
        "semi_major_axis_difference_vs_j2_m": delta_a_vs_j2_m,
    }


def build_sensitivity_cases(drag_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build one-at-a-time drag parameter sensitivity cases."""
    sensitivity = drag_config.get("sensitivity", {})
    if not sensitivity.get("enabled", False):
        return []
    multipliers = sensitivity.get("multipliers", {})
    cases: list[dict[str, Any]] = []
    baseline = drag_parameter_dict(drag_config)
    cases.append(
        {
            "case_id": "baseline",
            "varied_parameter": "none",
            "multiplier": 1.0,
            "parameters": baseline,
        }
    )
    for parameter in (
        "mass_kg",
        "cross_sectional_area_m2",
        "drag_coefficient",
        "reference_density_kg_m3",
        "scale_height_km",
    ):
        values = multipliers.get(parameter, [])
        for multiplier in values:
            value = float(multiplier)
            parameters = deepcopy(baseline)
            parameters[parameter] *= value
            token = str(value).replace(".", "p")
            cases.append(
                {
                    "case_id": f"{parameter}_x{token}",
                    "varied_parameter": parameter,
                    "multiplier": value,
                    "parameters": parameters,
                }
            )
    return cases


def run_drag_sensitivity(
    *,
    initial_state: CartesianState,
    j2_history: StateHistory,
    j2_element_history: dict[str, Any],
    elapsed_seconds: np.ndarray,
    gravitational_parameter_km3_s2: float,
    earth_equatorial_radius_km: float,
    j2: float,
    earth_rotation_rate_rad_s: float,
    drag_config: dict[str, Any],
    integrator_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run one-at-a-time sensitivity cases and return scalar results."""
    cases = build_sensitivity_cases(drag_config)
    results: list[dict[str, Any]] = []
    j2_a = np.asarray(j2_element_history["semi_major_axis_km"], dtype=float)

    from .j2 import create_osculating_element_history

    for case in cases:
        history = propagate_numerical_j2_drag(
            initial_state,
            gravitational_parameter_km3_s2,
            earth_equatorial_radius_km,
            j2,
            earth_rotation_rate_rad_s,
            elapsed_seconds,
            method=integrator_config["method"],
            relative_tolerance=integrator_config["relative_tolerance"],
            absolute_tolerance=integrator_config["absolute_tolerance"],
            maximum_step_seconds=integrator_config["maximum_step_seconds"],
            **case["parameters"],
        )
        elements = create_osculating_element_history(
            history,
            gravitational_parameter_km3_s2,
        )
        comparison = compare_state_histories(j2_history, history)
        error_summary = create_error_summary(comparison)
        drag_a = np.asarray(elements["semi_major_axis_km"], dtype=float)
        results.append(
            {
                "case_id": case["case_id"],
                "varied_parameter": case["varied_parameter"],
                "multiplier": case["multiplier"],
                **case["parameters"],
                "ballistic_coefficient_kg_m2": case["parameters"]["mass_kg"]
                / (
                    case["parameters"]["drag_coefficient"]
                    * case["parameters"]["cross_sectional_area_m2"]
                ),
                "runtime_seconds": history.runtime_seconds,
                "function_evaluations": history.function_evaluations,
                "final_semi_major_axis_difference_vs_j2_m": float(
                    (drag_a[-1] - j2_a[-1]) * 1000.0
                ),
                "minimum_semi_major_axis_difference_vs_j2_m": float(
                    np.min((drag_a - j2_a) * 1000.0)
                ),
                "maximum_position_separation_from_j2_km": float(
                    error_summary["position_difference_m"]["maximum_absolute"]
                    / 1000.0
                ),
                "final_position_separation_from_j2_km": float(
                    error_summary["position_difference_m"]["final"] / 1000.0
                ),
            }
        )
    return results


def sensitivity_direction_checks(
    sensitivity_results: list[dict[str, Any]],
) -> dict[str, bool | None]:
    """Check expected one-at-a-time sensitivity directions."""
    by_id = {item["case_id"]: item for item in sensitivity_results}
    baseline = by_id.get("baseline")
    if baseline is None:
        return {
            "mass_direction_passed": None,
            "area_direction_passed": None,
            "drag_coefficient_direction_passed": None,
            "density_direction_passed": None,
        }
    baseline_decay = float(
        baseline["final_semi_major_axis_difference_vs_j2_m"]
    )

    def decay(case_id: str) -> float | None:
        item = by_id.get(case_id)
        if item is None:
            return None
        return float(item["final_semi_major_axis_difference_vs_j2_m"])

    mass_half = decay("mass_kg_x0p5")
    mass_double = decay("mass_kg_x2p0")
    area_half = decay("cross_sectional_area_m2_x0p5")
    area_double = decay("cross_sectional_area_m2_x2p0")
    cd_half = decay("drag_coefficient_x0p5")
    cd_double = decay("drag_coefficient_x2p0")
    rho_half = decay("reference_density_kg_m3_x0p5")
    rho_double = decay("reference_density_kg_m3_x2p0")

    def lower_mass_check() -> bool | None:
        if mass_half is None or mass_double is None:
            return None
        return mass_half < baseline_decay < mass_double

    def direct_check(low: float | None, high: float | None) -> bool | None:
        if low is None or high is None:
            return None
        return high < baseline_decay < low

    return {
        "mass_direction_passed": lower_mass_check(),
        "area_direction_passed": direct_check(area_half, area_double),
        "drag_coefficient_direction_passed": direct_check(cd_half, cd_double),
        "density_direction_passed": direct_check(rho_half, rho_double),
    }
