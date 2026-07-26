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

from research_core.external_validation import initial_state_from_config
from research_core.gmat_eop import GMAT_R2026A_EOP_SHA256, GmatEopDataset
from research_core.gravity_harmonics import CofGravityField
from research_core.propagators.numerical_gravity import (
    propagate_spherical_harmonic_gravity,
    spherical_harmonic_derivative,
)
from research_core.propagators.numerical_j2 import propagate_numerical_j2_gmat_validated


class NumericalGravityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        baseline = json.loads(
            (PROJECT_ROOT / "configs/case_leo400_gmat_matched.json").read_text()
        )
        cls.state = initial_state_from_config(baseline)
        cls.field = CofGravityField.from_file(
            PROJECT_ROOT / "data/reference/gmat_r2026a/JGM2.cof"
        )
        cls.eop = GmatEopDataset.from_file(
            PROJECT_ROOT / "data/reference/gmat_r2026a/eopc04_08.62-now",
            expected_sha256=GMAT_R2026A_EOP_SHA256,
        )
        cls.kwargs = {
            "method": "DOP853",
            "relative_tolerance": 1e-11,
            "absolute_tolerance": 1e-13,
            "maximum_step_seconds": 10.0,
        }

    def test_degree_two_propagation_preserves_validated_j2_path(self):
        times = np.asarray([0.0, 10.0, 30.0, 60.0])
        harmonic = propagate_spherical_harmonic_gravity(
            self.state, self.field, self.eop, times, degree=2, order=0, **self.kwargs
        )
        validated = propagate_numerical_j2_gmat_validated(
            self.state,
            self.field.gravitational_parameter_km3_s2,
            self.field.reference_radius_km,
            self.field.j2,
            times,
            eop_dataset=self.eop,
            **self.kwargs,
        )
        np.testing.assert_allclose(harmonic.positions_km, validated.positions_km, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(harmonic.velocities_km_s, validated.velocities_km_s, rtol=0.0, atol=2e-15)

    def test_degree_twenty_history_is_finite_and_starts_exactly(self):
        times = np.asarray([0.0, 10.0, 30.0])
        history = propagate_spherical_harmonic_gravity(
            self.state, self.field, self.eop, times, degree=20, order=20, **self.kwargs
        )
        np.testing.assert_array_equal(history.positions_km[0], self.state.position_km)
        np.testing.assert_array_equal(history.velocities_km_s[0], self.state.velocity_km_s)
        self.assertTrue(np.all(np.isfinite(history.positions_km)))
        self.assertGreater(history.function_evaluations, 0)

    def test_invalid_order_and_time_grid_are_rejected(self):
        with self.assertRaises(ValueError):
            propagate_spherical_harmonic_gravity(
                self.state,
                self.field,
                self.eop,
                np.asarray([0.0, 10.0]),
                degree=4,
                order=5,
                **self.kwargs,
            )
        with self.assertRaises(ValueError):
            propagate_spherical_harmonic_gravity(
                self.state,
                self.field,
                self.eop,
                np.asarray([1.0, 10.0]),
                degree=4,
                order=4,
                **self.kwargs,
            )


if __name__ == "__main__":
    unittest.main()
