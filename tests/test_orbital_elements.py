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

from research_core.orbital_elements import (
    cartesian_to_elements,
    elements_from_config,
    elements_to_cartesian,
    solve_kepler_elliptic,
)


class OrbitalElementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (PROJECT_ROOT / "configs" / "case_leo400.json").read_text(
                encoding="utf-8"
            )
        )
        cls.mu = cls.config["earth_model"]["gravitational_parameter_km3_s2"]
        cls.elements = elements_from_config(cls.config["initial_state"])

    def test_elements_cartesian_round_trip(self) -> None:
        position, velocity = elements_to_cartesian(self.elements, self.mu)
        reconstructed = cartesian_to_elements(position, velocity, self.mu)
        self.assertAlmostEqual(
            reconstructed["semi_major_axis_km"],
            self.elements.semi_major_axis_km,
            places=8,
        )
        self.assertAlmostEqual(
            reconstructed["eccentricity"], self.elements.eccentricity, places=11
        )
        self.assertAlmostEqual(
            reconstructed["inclination_deg"],
            math.degrees(self.elements.inclination_rad),
            places=9,
        )
        self.assertAlmostEqual(
            reconstructed["raan_deg"],
            math.degrees(self.elements.raan_rad),
            places=9,
        )
        self.assertAlmostEqual(
            reconstructed["argument_of_perigee_deg"],
            math.degrees(self.elements.argument_of_perigee_rad),
            places=8,
        )
        self.assertAlmostEqual(
            reconstructed["true_anomaly_deg"],
            math.degrees(self.elements.true_anomaly_rad),
            places=8,
        )

    def test_vis_viva(self) -> None:
        position, velocity = elements_to_cartesian(self.elements, self.mu)
        radius = np.linalg.norm(position)
        expected_speed = math.sqrt(
            self.mu * (2.0 / radius - 1.0 / self.elements.semi_major_axis_km)
        )
        self.assertAlmostEqual(np.linalg.norm(velocity), expected_speed, places=12)

    def test_kepler_solver_residual(self) -> None:
        mean_anomaly = 2.3
        eccentricity = 0.4
        eccentric_anomaly = solve_kepler_elliptic(mean_anomaly, eccentricity)
        residual = eccentric_anomaly - eccentricity * math.sin(eccentric_anomaly)
        wrapped_difference = math.atan2(
            math.sin(residual - mean_anomaly),
            math.cos(residual - mean_anomaly),
        )
        self.assertLess(abs(wrapped_difference), 1e-12)


if __name__ == "__main__":
    unittest.main()
