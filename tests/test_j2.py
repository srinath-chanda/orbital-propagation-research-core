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

from research_core.analysis.j2 import (
    analytical_j2_raan_rate_rad_s,
    create_j2_validation_summary,
    create_osculating_element_history,
)
from research_core.data_models import CartesianState
from research_core.earth_orientation import (
    iau_1976_1980_precession_nutation_matrix,
    iau_1976_1980_true_pole_unit_vector,
)
from research_core.orbital_elements import elements_from_config, elements_to_cartesian
from research_core.propagators import (
    central_gravity_acceleration,
    j2_perturbing_acceleration,
    j2_perturbing_acceleration_about_axis,
    j2_perturbing_acceleration_gmat_matched,
    propagate_numerical_j2,
    propagate_numerical_j2_gmat_matched,
    propagate_numerical_two_body,
)
from research_core.time_utils import build_time_grid


class J2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = json.loads(
            (PROJECT_ROOT / "configs" / "case_leo400.json").read_text(
                encoding="utf-8"
            )
        )
        cls.config = config
        cls.earth = config["earth_model"]
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

    def test_j2_acceleration_is_small_and_finite(self) -> None:
        central = central_gravity_acceleration(
            self.initial_state.position_km,
            self.earth["gravitational_parameter_km3_s2"],
        )
        perturbing = j2_perturbing_acceleration(
            self.initial_state.position_km,
            self.earth["gravitational_parameter_km3_s2"],
            self.earth["equatorial_radius_km"],
            self.earth["j2"],
        )
        ratio = np.linalg.norm(perturbing) / np.linalg.norm(central)
        self.assertTrue(np.all(np.isfinite(perturbing)))
        self.assertGreater(ratio, 0.0)
        self.assertLess(ratio, 0.01)

    def test_arbitrary_axis_reduces_to_fixed_axis(self) -> None:
        fixed = j2_perturbing_acceleration(
            self.initial_state.position_km,
            self.earth["gravitational_parameter_km3_s2"],
            self.earth["equatorial_radius_km"],
            self.earth["j2"],
        )
        arbitrary = j2_perturbing_acceleration_about_axis(
            self.initial_state.position_km,
            np.array([0.0, 0.0, 1.0]),
            self.earth["gravitational_parameter_km3_s2"],
            self.earth["equatorial_radius_km"],
            self.earth["j2"],
        )
        np.testing.assert_allclose(arbitrary, fixed, rtol=2e-15, atol=1e-18)

    def test_pole_aware_vector_matches_explicit_frame_rotation(self) -> None:
        epoch = "2026-01-01T00:00:00Z"
        elapsed = 12345.0
        matrix = iau_1976_1980_precession_nutation_matrix(epoch, elapsed)
        rotated_position = matrix @ self.initial_state.position_km
        true_of_date = j2_perturbing_acceleration(
            rotated_position,
            self.earth["gravitational_parameter_km3_s2"],
            self.earth["equatorial_radius_km"],
            self.earth["j2"],
        )
        expected_inertial = matrix.T @ true_of_date
        direct_inertial = j2_perturbing_acceleration_gmat_matched(
            self.initial_state.position_km,
            epoch,
            elapsed,
            self.earth["gravitational_parameter_km3_s2"],
            self.earth["equatorial_radius_km"],
            self.earth["j2"],
        )
        np.testing.assert_allclose(
            direct_inertial,
            expected_inertial,
            rtol=2e-14,
            atol=1e-17,
        )

    def test_2026_true_pole_uses_correct_matrix_row(self) -> None:
        axis = iau_1976_1980_true_pole_unit_vector(
            "2026-01-01T00:00:00Z",
            0.0,
        )
        expected = np.array(
            [2.53698191e-3, 3.16659677e-5, 9.99996781e-1]
        )
        np.testing.assert_allclose(axis, expected, rtol=0.0, atol=6e-10)
        self.assertAlmostEqual(float(np.linalg.norm(axis)), 1.0, places=14)

    def test_zero_j2_matches_numerical_two_body(self) -> None:
        times = build_time_grid(1.0, 60.0)
        kwargs = {
            "method": "DOP853",
            "relative_tolerance": 1e-11,
            "absolute_tolerance": 1e-13,
            "maximum_step_seconds": 60.0,
        }
        two_body = propagate_numerical_two_body(
            self.initial_state,
            self.earth["gravitational_parameter_km3_s2"],
            times,
            **kwargs,
        )
        j2_zero = propagate_numerical_j2(
            self.initial_state,
            self.earth["gravitational_parameter_km3_s2"],
            self.earth["equatorial_radius_km"],
            0.0,
            times,
            **kwargs,
        )
        self.assertLess(
            float(np.max(np.linalg.norm(j2_zero.positions_km - two_body.positions_km, axis=1))),
            1e-9,
        )
        self.assertLess(
            float(np.max(np.linalg.norm(j2_zero.velocities_km_s - two_body.velocities_km_s, axis=1))),
            1e-12,
        )
        gmat_frame_state = CartesianState(
            epoch_utc=self.initial_state.epoch_utc,
            frame="EarthMJ2000Eq",
            position_km=self.initial_state.position_km,
            velocity_km_s=self.initial_state.velocity_km_s,
        )
        pole_aware_zero = propagate_numerical_j2_gmat_matched(
            gmat_frame_state,
            self.earth["gravitational_parameter_km3_s2"],
            self.earth["equatorial_radius_km"],
            0.0,
            times,
            **kwargs,
        )
        self.assertLess(
            float(
                np.max(
                    np.linalg.norm(
                        pole_aware_zero.positions_km - two_body.positions_km,
                        axis=1,
                    )
                )
            ),
            1e-9,
        )

    def test_analytical_raan_rate_sign_changes_with_inclination(self) -> None:
        prograde = analytical_j2_raan_rate_rad_s(
            self.elements,
            self.earth["gravitational_parameter_km3_s2"],
            self.earth["equatorial_radius_km"],
            self.earth["j2"],
        )
        retrograde_elements = type(self.elements)(
            semi_major_axis_km=self.elements.semi_major_axis_km,
            eccentricity=self.elements.eccentricity,
            inclination_rad=math.radians(180.0 - math.degrees(self.elements.inclination_rad)),
            raan_rad=self.elements.raan_rad,
            argument_of_perigee_rad=self.elements.argument_of_perigee_rad,
            true_anomaly_rad=self.elements.true_anomaly_rad,
        )
        retrograde = analytical_j2_raan_rate_rad_s(
            retrograde_elements,
            self.earth["gravitational_parameter_km3_s2"],
            self.earth["equatorial_radius_km"],
            self.earth["j2"],
        )
        self.assertLess(prograde, 0.0)
        self.assertGreater(retrograde, 0.0)

    def test_24_hour_raan_trend_and_invariants(self) -> None:
        times = build_time_grid(24.0, 120.0)
        history = propagate_numerical_j2(
            self.initial_state,
            self.earth["gravitational_parameter_km3_s2"],
            self.earth["equatorial_radius_km"],
            self.earth["j2"],
            times,
            method="DOP853",
            relative_tolerance=1e-11,
            absolute_tolerance=1e-13,
            maximum_step_seconds=60.0,
        )
        elements = create_osculating_element_history(
            history,
            self.earth["gravitational_parameter_km3_s2"],
        )
        summary = create_j2_validation_summary(
            initial_position_km=self.initial_state.position_km,
            elements=self.elements,
            element_history=elements,
            history=history,
            gravitational_parameter_km3_s2=self.earth[
                "gravitational_parameter_km3_s2"
            ],
            earth_equatorial_radius_km=self.earth["equatorial_radius_km"],
            j2=self.earth["j2"],
        )
        self.assertLess(summary["fitted_raan_rate_deg_day"], 0.0)
        self.assertLess(summary["relative_rate_difference"], 0.05)
        self.assertLess(
            summary["maximum_absolute_relative_total_energy_drift"],
            1e-9,
        )
        self.assertLess(
            summary["maximum_absolute_relative_angular_momentum_z_drift"],
            1e-9,
        )


if __name__ == "__main__":
    unittest.main()
