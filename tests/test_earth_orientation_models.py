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

from research_core.earth_orientation import (
    POLE_MODEL_DESCRIPTIONS,
    SUPPORTED_POLE_MODELS,
    earth_pole_unit_vector,
    iau_1976_1980_true_pole_unit_vector,
    pole_angular_separation_arcsec,
)
from research_core.external_validation import initial_state_from_config
from research_core.propagators.numerical_j2 import (
    propagate_numerical_j2_gmat_matched,
    propagate_numerical_j2_orientation_model,
)


class EarthOrientationModelTests(unittest.TestCase):
    def test_registry_models_return_finite_unit_axes(self):
        self.assertEqual(len(SUPPORTED_POLE_MODELS), 5)
        self.assertEqual(set(SUPPORTED_POLE_MODELS), set(POLE_MODEL_DESCRIPTIONS))
        for model in SUPPORTED_POLE_MODELS:
            axis = earth_pole_unit_vector("2026-01-01T00:00:00Z", 43200.0, model)
            self.assertEqual(axis.shape, (3,))
            self.assertTrue(np.all(np.isfinite(axis)))
            self.assertAlmostEqual(float(np.linalg.norm(axis)), 1.0, places=14)

    def test_legacy_true_pole_wrapper_is_exactly_preserved(self):
        expected = iau_1976_1980_true_pole_unit_vector(
            "2026-07-01T00:00:00Z", 12345.0
        )
        actual = earth_pole_unit_vector(
            "2026-07-01T00:00:00Z", 12345.0, "iau1976_1980"
        )
        np.testing.assert_array_equal(actual, expected)

    def test_pole_separations_have_expected_diagnostic_scale(self):
        epoch = "2026-01-01T00:00:00Z"
        baseline = earth_pole_unit_vector(epoch, 0.0, "iau1976_1980")
        self.assertEqual(pole_angular_separation_arcsec(baseline, baseline), 0.0)
        fixed = earth_pole_unit_vector(epoch, 0.0, "j2000_fixed")
        precession = earth_pole_unit_vector(epoch, 0.0, "iau1976_precession")
        modern = earth_pole_unit_vector(epoch, 0.0, "iau2006_2000a")
        self.assertGreater(pole_angular_separation_arcsec(baseline, fixed), 500.0)
        self.assertGreater(
            pole_angular_separation_arcsec(baseline, precession), 5.0
        )
        modern_separation = pole_angular_separation_arcsec(baseline, modern)
        self.assertGreater(modern_separation, 0.001)
        self.assertLess(modern_separation, 1.0)

    def test_invalid_model_and_axis_are_rejected(self):
        with self.assertRaises(ValueError):
            earth_pole_unit_vector("2026-01-01T00:00:00Z", 0.0, "unknown")
        with self.assertRaises(ValueError):
            pole_angular_separation_arcsec(np.zeros(3), np.ones(3))
        with self.assertRaises(ValueError):
            pole_angular_separation_arcsec(np.zeros(2), np.ones(3))

    def test_generic_baseline_propagator_preserves_validated_implementation(self):
        config = json.loads(
            (PROJECT_ROOT / "configs" / "case_leo400_gmat_matched.json").read_text(
                encoding="utf-8"
            )
        )
        state = initial_state_from_config(config)
        earth = config["earth_model"]
        integrator = config["integrator"]
        times = np.array([0.0, 60.0, 120.0, 300.0])
        kwargs = {
            "method": integrator["method"],
            "relative_tolerance": integrator["relative_tolerance"],
            "absolute_tolerance": integrator["absolute_tolerance"],
            "maximum_step_seconds": integrator["maximum_step_seconds"],
        }
        validated = propagate_numerical_j2_gmat_matched(
            state,
            earth["gravitational_parameter_km3_s2"],
            earth["equatorial_radius_km"],
            earth["j2"],
            times,
            **kwargs,
        )
        generic = propagate_numerical_j2_orientation_model(
            state,
            earth["gravitational_parameter_km3_s2"],
            earth["equatorial_radius_km"],
            earth["j2"],
            times,
            orientation_model="iau1976_1980",
            **kwargs,
        )
        np.testing.assert_array_equal(generic.positions_km, validated.positions_km)
        np.testing.assert_array_equal(generic.velocities_km_s, validated.velocities_km_s)


if __name__ == "__main__":
    unittest.main()
