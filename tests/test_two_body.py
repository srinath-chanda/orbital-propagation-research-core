from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.analysis.comparison import compare_state_histories
from research_core.analysis.diagnostics import conservation_diagnostics
from research_core.data_models import CartesianState
from research_core.orbital_elements import elements_from_config, elements_to_cartesian
from research_core.propagators import (
    propagate_analytical_two_body,
    propagate_numerical_two_body,
)


class TwoBodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (PROJECT_ROOT / "configs" / "case_leo400.json").read_text(
                encoding="utf-8"
            )
        )
        cls.mu = cls.config["earth_model"]["gravitational_parameter_km3_s2"]
        cls.elements = elements_from_config(cls.config["initial_state"])
        position, velocity = elements_to_cartesian(cls.elements, cls.mu)
        cls.initial_state = CartesianState(
            epoch_utc=cls.config["initial_state"]["epoch_utc"],
            frame=cls.config["initial_state"]["frame"],
            position_km=position,
            velocity_km_s=velocity,
        )

    def test_analytical_initial_state_is_exact(self) -> None:
        times = np.array([0.0, 60.0])
        history = propagate_analytical_two_body(
            self.elements,
            self.mu,
            times,
            epoch_utc=self.initial_state.epoch_utc,
            frame=self.initial_state.frame,
        )
        np.testing.assert_allclose(
            history.positions_km[0], self.initial_state.position_km, atol=1e-11
        )
        np.testing.assert_allclose(
            history.velocities_km_s[0], self.initial_state.velocity_km_s, atol=1e-13
        )

    def test_analytical_one_period_closure(self) -> None:
        period = 2.0 * math.pi * math.sqrt(
            self.elements.semi_major_axis_km**3 / self.mu
        )
        history = propagate_analytical_two_body(
            self.elements,
            self.mu,
            np.array([0.0, period]),
            epoch_utc=self.initial_state.epoch_utc,
            frame=self.initial_state.frame,
        )
        self.assertLess(
            np.linalg.norm(history.positions_km[-1] - history.positions_km[0]),
            1e-8,
        )
        self.assertLess(
            np.linalg.norm(history.velocities_km_s[-1] - history.velocities_km_s[0]),
            1e-11,
        )

    def test_numerical_matches_analytical_for_three_hours(self) -> None:
        times = np.arange(0.0, 3.0 * 3600.0 + 60.0, 60.0)
        analytical = propagate_analytical_two_body(
            self.elements,
            self.mu,
            times,
            epoch_utc=self.initial_state.epoch_utc,
            frame=self.initial_state.frame,
        )
        numerical = propagate_numerical_two_body(
            self.initial_state,
            self.mu,
            times,
            method="DOP853",
            relative_tolerance=1e-10,
            absolute_tolerance=1e-12,
            maximum_step_seconds=30.0,
        )
        comparison = compare_state_histories(analytical, numerical)
        self.assertLess(np.max(comparison["position_difference_m"]), 0.01)
        self.assertLess(np.max(comparison["velocity_difference_mm_s"]), 0.01)

    def test_numerical_conservation_for_three_hours(self) -> None:
        times = np.arange(0.0, 3.0 * 3600.0 + 60.0, 60.0)
        numerical = propagate_numerical_two_body(
            self.initial_state,
            self.mu,
            times,
            method="DOP853",
            relative_tolerance=1e-10,
            absolute_tolerance=1e-12,
            maximum_step_seconds=30.0,
        )
        diagnostics = conservation_diagnostics(numerical, self.mu)
        self.assertLess(
            diagnostics["maximum_absolute_relative_energy_drift"], 1e-11
        )
        self.assertLess(
            diagnostics["maximum_absolute_relative_angular_momentum_drift"],
            1e-11,
        )


if __name__ == "__main__":
    unittest.main()
