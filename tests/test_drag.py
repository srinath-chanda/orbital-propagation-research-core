from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.analysis.drag import create_drag_diagnostics
from research_core.analysis.j2 import create_osculating_element_history
from research_core.data_models import CartesianState
from research_core.orbital_elements import elements_from_config, elements_to_cartesian
from research_core.propagators import (
    atmospheric_relative_velocity_km_s,
    drag_acceleration_km_s2,
    exponential_atmospheric_density_kg_m3,
    propagate_numerical_j2,
    propagate_numerical_j2_drag,
)
from research_core.time_utils import build_time_grid


class DragTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = json.loads(
            (PROJECT_ROOT / "configs" / "case_leo400.json").read_text(
                encoding="utf-8"
            )
        )
        cls.config = config
        cls.earth = config["earth_model"]
        cls.drag = config["drag"]
        cls.elements = elements_from_config(config["initial_state"])
        position, velocity = elements_to_cartesian(
            cls.elements,
            cls.earth["gravitational_parameter_km3_s2"],
        )
        cls.initial_state = CartesianState(
            epoch_utc=config["initial_state"]["epoch_utc"],
            frame=config["initial_state"]["frame"],
            position_km=position,
            velocity_km_s=velocity,
        )
        cls.integrator = config["integrator"]

    def _propagate_drag(self, times: np.ndarray, **overrides: float):
        parameters = {
            "mass_kg": self.drag["mass_kg"],
            "cross_sectional_area_m2": self.drag["cross_sectional_area_m2"],
            "drag_coefficient": self.drag["drag_coefficient"],
            "reference_altitude_km": self.drag["reference_altitude_km"],
            "reference_density_kg_m3": self.drag["reference_density_kg_m3"],
            "scale_height_km": self.drag["scale_height_km"],
            "co_rotating_atmosphere": self.drag["co_rotating_atmosphere"],
        }
        parameters.update(overrides)
        return propagate_numerical_j2_drag(
            self.initial_state,
            self.earth["gravitational_parameter_km3_s2"],
            self.earth["equatorial_radius_km"],
            self.earth["j2"],
            self.earth["earth_rotation_rate_rad_s"],
            times,
            method=self.integrator["method"],
            relative_tolerance=self.integrator["relative_tolerance"],
            absolute_tolerance=self.integrator["absolute_tolerance"],
            maximum_step_seconds=self.integrator["maximum_step_seconds"],
            **parameters,
        )

    def test_density_matches_reference_and_decreases_with_altitude(self) -> None:
        radius = self.earth["equatorial_radius_km"]
        reference_position = np.array(
            [radius + self.drag["reference_altitude_km"], 0.0, 0.0]
        )
        high_position = reference_position + np.array([100.0, 0.0, 0.0])
        reference_density = exponential_atmospheric_density_kg_m3(
            reference_position,
            radius,
            self.drag["reference_altitude_km"],
            self.drag["reference_density_kg_m3"],
            self.drag["scale_height_km"],
        )
        high_density = exponential_atmospheric_density_kg_m3(
            high_position,
            radius,
            self.drag["reference_altitude_km"],
            self.drag["reference_density_kg_m3"],
            self.drag["scale_height_km"],
        )
        self.assertAlmostEqual(
            reference_density,
            self.drag["reference_density_kg_m3"],
            places=22,
        )
        self.assertLess(high_density, reference_density)

    def test_drag_acceleration_opposes_relative_velocity(self) -> None:
        relative_velocity = atmospheric_relative_velocity_km_s(
            self.initial_state.position_km,
            self.initial_state.velocity_km_s,
            self.earth["earth_rotation_rate_rad_s"],
            co_rotating_atmosphere=True,
        )
        acceleration = drag_acceleration_km_s2(
            self.initial_state.position_km,
            self.initial_state.velocity_km_s,
            earth_equatorial_radius_km=self.earth["equatorial_radius_km"],
            earth_rotation_rate_rad_s=self.earth["earth_rotation_rate_rad_s"],
            mass_kg=self.drag["mass_kg"],
            cross_sectional_area_m2=self.drag["cross_sectional_area_m2"],
            drag_coefficient=self.drag["drag_coefficient"],
            reference_altitude_km=self.drag["reference_altitude_km"],
            reference_density_kg_m3=self.drag["reference_density_kg_m3"],
            scale_height_km=self.drag["scale_height_km"],
            co_rotating_atmosphere=True,
        )
        cosine = np.dot(relative_velocity, acceleration) / (
            np.linalg.norm(relative_velocity) * np.linalg.norm(acceleration)
        )
        self.assertLess(cosine, -0.999999999)

    def test_zero_density_matches_j2(self) -> None:
        times = build_time_grid(1.0, 60.0)
        j2_history = propagate_numerical_j2(
            self.initial_state,
            self.earth["gravitational_parameter_km3_s2"],
            self.earth["equatorial_radius_km"],
            self.earth["j2"],
            times,
            method=self.integrator["method"],
            relative_tolerance=self.integrator["relative_tolerance"],
            absolute_tolerance=self.integrator["absolute_tolerance"],
            maximum_step_seconds=self.integrator["maximum_step_seconds"],
        )
        zero_density = self._propagate_drag(
            times,
            reference_density_kg_m3=0.0,
        )
        maximum_position_difference_m = float(
            np.max(
                np.linalg.norm(
                    zero_density.positions_km - j2_history.positions_km,
                    axis=1,
                )
            )
            * 1000.0
        )
        self.assertLess(maximum_position_difference_m, 1e-3)

    def test_drag_dissipates_energy_and_reduces_semi_major_axis(self) -> None:
        times = build_time_grid(6.0, 120.0)
        history = self._propagate_drag(times)
        j2_history = propagate_numerical_j2(
            self.initial_state,
            self.earth["gravitational_parameter_km3_s2"],
            self.earth["equatorial_radius_km"],
            self.earth["j2"],
            times,
            method=self.integrator["method"],
            relative_tolerance=self.integrator["relative_tolerance"],
            absolute_tolerance=self.integrator["absolute_tolerance"],
            maximum_step_seconds=self.integrator["maximum_step_seconds"],
        )
        diagnostics = create_drag_diagnostics(
            history,
            gravitational_parameter_km3_s2=self.earth[
                "gravitational_parameter_km3_s2"
            ],
            earth_equatorial_radius_km=self.earth["equatorial_radius_km"],
            j2=self.earth["j2"],
            earth_rotation_rate_rad_s=self.earth["earth_rotation_rate_rad_s"],
            drag_config=self.drag,
        )
        elements = create_osculating_element_history(
            history,
            self.earth["gravitational_parameter_km3_s2"],
        )
        j2_elements = create_osculating_element_history(
            j2_history,
            self.earth["gravitational_parameter_km3_s2"],
        )
        self.assertLess(
            diagnostics["total_specific_energy_change_km2_s2"][-1],
            0.0,
        )
        self.assertLess(
            elements["semi_major_axis_km"][-1],
            j2_elements["semi_major_axis_km"][-1],
        )

    def test_larger_area_produces_more_decay(self) -> None:
        times = build_time_grid(3.0, 180.0)
        low_area = self._propagate_drag(
            times,
            cross_sectional_area_m2=self.drag["cross_sectional_area_m2"] * 0.5,
        )
        high_area = self._propagate_drag(
            times,
            cross_sectional_area_m2=self.drag["cross_sectional_area_m2"] * 2.0,
        )
        low_elements = create_osculating_element_history(
            low_area,
            self.earth["gravitational_parameter_km3_s2"],
        )
        high_elements = create_osculating_element_history(
            high_area,
            self.earth["gravitational_parameter_km3_s2"],
        )
        low_change = (
            low_elements["semi_major_axis_km"][-1]
            - low_elements["semi_major_axis_km"][0]
        )
        high_change = (
            high_elements["semi_major_axis_km"][-1]
            - high_elements["semi_major_axis_km"][0]
        )
        self.assertLess(high_change, low_change)


if __name__ == "__main__":
    unittest.main()
