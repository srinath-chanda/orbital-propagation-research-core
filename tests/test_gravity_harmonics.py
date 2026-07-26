from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.gravity_harmonics import (
    CofGravityField,
    gravity_acceleration_fixed_km_s2,
    gravity_acceleration_inertial_km_s2,
)
from research_core.propagators.numerical_j2 import (
    central_gravity_acceleration,
    j2_perturbing_acceleration,
)


SYNTHETIC_COF = """CCCC synthetic normalized JGM2 subset
POTFIELD 4 4 1 3.98600441500000e+14 6.37813630000000e+06 1.00000000000000e+00
RECOEF 2 0 -4.84165390000000e-04
RECOEF 2 1 -1.86987640000000e-10 1.19528010000000e-09
RECOEF 2 2 2.43908370000000e-06-1.40010930000000e-06
RECOEF 3 0 9.57122390000000e-07
RECOEF 3 1 2.02839970000000e-06 2.48806640000000e-07
RECOEF 4 0 5.40143330000000e-07
RECOEF 4 4 -1.88488500000000e-07 3.08845320000000e-07
"""


class GravityHarmonicTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "JGM2.cof"
        self.path.write_text(SYNTHETIC_COF, encoding="ascii")
        self.field = CofGravityField.from_file(self.path)

    def tearDown(self):
        self.temporary.cleanup()

    def test_parser_reads_si_header_and_concatenated_coefficients(self):
        self.assertEqual(self.field.maximum_degree, 4)
        self.assertEqual(self.field.maximum_order, 4)
        self.assertAlmostEqual(self.field.gravitational_parameter_km3_s2, 398600.4415)
        self.assertAlmostEqual(self.field.reference_radius_km, 6378.1363)
        self.assertAlmostEqual(self.field.sine[2, 2], -1.4001093e-6)
        self.assertAlmostEqual(self.field.j2, 0.001082626724392697)

    def test_degree_zero_is_exact_point_mass(self):
        position = np.asarray([5000.0, -3000.0, 3500.0])
        actual = gravity_acceleration_fixed_km_s2(position, self.field, degree=0, order=0)
        expected = central_gravity_acceleration(position, 398600.4415)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-18)

    def test_degree_two_order_zero_is_existing_validated_j2_equation(self):
        position = np.asarray([4791.2448, 3981.8438, 2653.3345])
        actual = gravity_acceleration_fixed_km_s2(position, self.field, degree=2, order=0)
        expected = central_gravity_acceleration(position, 398600.4415) + j2_perturbing_acceleration(
            position, 398600.4415, 6378.1363, self.field.j2
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-17)

    def test_tesseral_terms_change_acceleration_and_rotation_is_respected(self):
        position = np.asarray([5500.0, 2200.0, 3000.0])
        zonal = gravity_acceleration_fixed_km_s2(position, self.field, degree=4, order=0)
        tesseral = gravity_acceleration_fixed_km_s2(position, self.field, degree=4, order=4)
        self.assertGreater(float(np.linalg.norm(tesseral - zonal)), 1.0e-9)
        angle = 0.7
        rotation = np.asarray(
            [[np.cos(angle), np.sin(angle), 0.0], [-np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
        )
        inertial = gravity_acceleration_inertial_km_s2(
            position, rotation, self.field, degree=4, order=4
        )
        expected = rotation.T @ gravity_acceleration_fixed_km_s2(
            rotation @ position, self.field, degree=4, order=4
        )
        np.testing.assert_allclose(inertial, expected, rtol=0.0, atol=2.0e-18)

    def test_invalid_field_and_degree_requests_are_rejected(self):
        invalid = Path(self.temporary.name) / "invalid.cof"
        invalid.write_text("POTFIELD 4 4 0 3.986e14 6.378e6 1\nRECOEF 2 0 -1e-3\n")
        with self.assertRaises(ValueError):
            CofGravityField.from_file(invalid)
        with self.assertRaises(ValueError):
            gravity_acceleration_fixed_km_s2(
                np.asarray([7000.0, 0.0, 0.0]), self.field, degree=5, order=0
            )


if __name__ == "__main__":
    unittest.main()
