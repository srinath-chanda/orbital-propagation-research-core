from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.gmat_j2_diagnostics import (
    acceleration_sample_times,
    build_gmat_acceleration_diagnostic_script,
    compare_acceleration_samples,
    parse_gmat_acceleration_report,
    run_gmat_acceleration_validation,
)
from research_core.propagators import (
    central_gravity_acceleration,
    j2_perturbing_acceleration_gmat_matched,
)


class GmatJ2DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_path = (
            PROJECT_ROOT / "configs" / "case_leo400_gmat_matched.json"
        )
        cls.config = json.loads(cls.config_path.read_text(encoding="utf-8"))

    def test_script_uses_shared_state_force_model_parameters(self) -> None:
        script = build_gmat_acceleration_diagnostic_script(
            self.config,
            output_report=Path("diagnostic.csv"),
        )
        self.assertIn("DiagnosticSat.PointMassFM.AccelerationX", script)
        self.assertIn("DiagnosticSat.Degree2FM.AccelerationX", script)
        self.assertIn("PointMassFM.GravityField.Earth.Degree = 0", script)
        self.assertIn("Degree2FM.GravityField.Earth.Degree = 2", script)
        self.assertEqual(script.count("Report AccelerationReport"), 25)

    def test_sample_count_bounds_are_enforced(self) -> None:
        self.assertEqual(acceleration_sample_times(86400.0, 25).size, 25)
        with self.assertRaises(ValueError):
            acceleration_sample_times(86400.0, 19)
        with self.assertRaises(ValueError):
            acceleration_sample_times(86400.0, 51)

    def _write_synthetic_report(self, path: Path) -> None:
        earth = self.config["earth_model"]
        epoch = self.config["initial_state"]["epoch_utc"]
        mu = earth["gravitational_parameter_km3_s2"]
        radius = earth["equatorial_radius_km"]
        j2 = earth["j2"]
        times = acceleration_sample_times(86400.0, 25)
        base = np.array([5100.0, 3200.0, 3100.0])
        headers = [
            "DiagnosticSat.ElapsedSecs",
            "DiagnosticSat.EarthMJ2000Eq.X",
            "DiagnosticSat.EarthMJ2000Eq.Y",
            "DiagnosticSat.EarthMJ2000Eq.Z",
            "DiagnosticSat.PointMassFM.AccelerationX",
            "DiagnosticSat.PointMassFM.AccelerationY",
            "DiagnosticSat.PointMassFM.AccelerationZ",
            "DiagnosticSat.Degree2FM.AccelerationX",
            "DiagnosticSat.Degree2FM.AccelerationY",
            "DiagnosticSat.Degree2FM.AccelerationZ",
        ]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            for index, elapsed in enumerate(times):
                angle = 0.2 * index
                position = np.array(
                    [
                        base[0] * np.cos(angle) - base[1] * np.sin(angle),
                        base[0] * np.sin(angle) + base[1] * np.cos(angle),
                        base[2] * np.cos(0.1 * angle),
                    ]
                )
                point_mass = central_gravity_acceleration(position, mu)
                isolated = j2_perturbing_acceleration_gmat_matched(
                    position,
                    epoch,
                    float(elapsed),
                    mu,
                    radius,
                    j2,
                )
                writer.writerow(
                    [elapsed, *position, *point_mass, *(point_mass + isolated)]
                )

    def test_parser_and_end_to_end_synthetic_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "acceleration.csv"
            self._write_synthetic_report(report_path)
            samples = parse_gmat_acceleration_report(report_path)
            self.assertEqual(samples.elapsed_seconds.size, 25)
            earth = self.config["earth_model"]
            comparison = compare_acceleration_samples(
                samples,
                epoch_utc=self.config["initial_state"]["epoch_utc"],
                gravitational_parameter_km3_s2=earth[
                    "gravitational_parameter_km3_s2"
                ],
                earth_equatorial_radius_km=earth["equatorial_radius_km"],
                j2=earth["j2"],
            )
            self.assertLess(
                float(
                    np.max(
                        comparison["gmat_matched_vector_difference_km_s2"]
                    )
                ),
                1e-16,
            )
            self.assertGreater(
                float(np.max(comparison["fixed_axis_vector_difference_km_s2"])),
                1e-9,
            )
            config_copy = root / "config.json"
            config_copy.write_text(json.dumps(self.config), encoding="utf-8")
            result = run_gmat_acceleration_validation(
                config_copy,
                report_path,
                project_root=root,
            )
            self.assertEqual(result.validation_status, "passed_with_warnings")
            self.assertTrue(result.report_path.is_file())
            self.assertTrue(
                (result.result_directory / "RUN_MANIFEST.json").is_file()
            )

    def test_real_gmat_r2026a_raw_report_layout_is_parsed(self) -> None:
        report_path = (
            PROJECT_ROOT
            / "data"
            / "reference"
            / "gmat"
            / "output"
            / "CASE_LEO400_GMAT_ACCELERATION_DIAGNOSTIC.csv"
        )
        samples = parse_gmat_acceleration_report(report_path)
        self.assertEqual(samples.elapsed_seconds.size, 25)
        self.assertAlmostEqual(float(samples.elapsed_seconds[0]), 0.0, places=12)
        self.assertAlmostEqual(
            float(samples.elapsed_seconds[-1]),
            86400.0,
            places=5,
        )
        np.testing.assert_allclose(
            samples.positions_km[0],
            [4791.244801771759, 3981.843847094183, 2653.334545050801],
            rtol=0.0,
            atol=1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
