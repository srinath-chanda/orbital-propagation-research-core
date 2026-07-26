from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.external_validation import initial_state_from_config
from research_core.gmat_multicase import (
    load_gmat_matrix_spec,
    package_gmat_multicase_results,
    prepare_gmat_multicase_files,
    run_gmat_multicase_validation,
)
from research_core.propagators.numerical_j2 import propagate_numerical_j2_gmat_matched
from research_core.propagators.numerical_two_body import propagate_numerical_two_body


def write_stk(path: Path, history) -> None:
    epoch = datetime.fromisoformat(history.epoch_utc.replace("Z", "+00:00"))
    lines = [
        "stk.v.4.3",
        "",
        "BEGIN Ephemeris",
        f"NumberOfEphemerisPoints {history.elapsed_seconds.size}",
        f"ScenarioEpoch {epoch.strftime('%d %b %Y %H:%M:%S.%f')[:-3]}",
        "InterpolationMethod Lagrange",
        "InterpolationOrder 7",
        "CentralBody Earth",
        "CoordinateSystem J2000",
        "DistanceUnit Kilometers",
        "",
        "EphemerisTimePosVel",
    ]
    for t, r, v in zip(
        history.elapsed_seconds, history.positions_km, history.velocities_km_s
    ):
        lines.append(
            f"{t:.9f} {r[0]:.15f} {r[1]:.15f} {r[2]:.15f} "
            f"{v[0]:.15f} {v[1]:.15f} {v[2]:.15f}"
        )
    lines.extend(["END Ephemeris", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


class GmatMultiCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = json.loads(
            (PROJECT_ROOT / "configs" / "case_leo400_gmat_matched.json").read_text(
                encoding="utf-8"
            )
        )

    def _write_project(self, root: Path) -> Path:
        configs = root / "configs"
        configs.mkdir(parents=True)
        baseline_path = configs / "baseline.json"
        baseline_path.write_text(json.dumps(self.baseline), encoding="utf-8")
        spec = {
            "schema_version": "1B.0",
            "matrix_id": "EXP-GMAT-1B-TEST",
            "baseline_configuration": "configs/baseline.json",
            "reference_root": "data/reference/gmat_1b",
            "tool": "GMAT",
            "tool_version": "R2026a",
            "output_step_seconds": 60.0,
            "threshold_policy": {
                "status": "preregistered_before_gmat_execution",
                "initial_position_difference_m": 0.001,
                "initial_velocity_difference_mm_s": 0.001,
                "two_body_maximum_position_difference_m": 0.01,
                "two_body_maximum_velocity_difference_mm_s": 0.01,
                "j2_duration_tiers": [
                    {
                        "maximum_duration_hours": 1.0,
                        "maximum_position_difference_m": 0.01,
                        "maximum_velocity_difference_mm_s": 0.01,
                    }
                ],
                "rule": "Do not change after execution.",
            },
            "cases": [
                {
                    "case_id": "T01_CONTROL",
                    "factor": "baseline_control",
                    "epoch_utc": "2026-01-01T00:00:00Z",
                    "altitude_km": 400.0,
                    "inclination_deg": 51.6,
                    "duration_hours": 120.0 / 3600.0,
                },
                {
                    "case_id": "T02_INCLINATION",
                    "factor": "inclination",
                    "epoch_utc": "2026-04-01T00:00:00Z",
                    "altitude_km": 700.0,
                    "inclination_deg": 98.0,
                    "duration_hours": 120.0 / 3600.0,
                },
            ],
        }
        spec_path = configs / "matrix.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        return spec_path

    def test_real_matrix_spec_has_unique_preregistered_cases(self):
        spec = load_gmat_matrix_spec(
            PROJECT_ROOT / "configs" / "gmat_1b_multicase_matrix.json"
        )
        self.assertEqual(len(spec["cases"]), 10)
        self.assertEqual(len({case["case_id"] for case in spec["cases"]}), 10)
        self.assertEqual(
            {case["factor"] for case in spec["cases"]},
            {"baseline_control", "altitude", "inclination", "epoch", "duration"},
        )

    def test_prepare_creates_master_and_individual_scripts_and_archives_stale_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec_path = self._write_project(root)
            first = prepare_gmat_multicase_files(spec_path, project_root=root)
            self.assertEqual(first.case_count, 2)
            self.assertEqual(first.expected_output_count, 4)
            master = first.master_script.read_text(encoding="utf-8")
            self.assertEqual(master.count("Create Spacecraft"), 4)
            self.assertEqual(master.count("Propagate "), 4)
            self.assertIn("BeginMissionSequence", master)
            self.assertIn("GravityField.Earth.Degree = 0", master)
            self.assertIn("GravityField.Earth.Degree = 2", master)
            stale = first.reference_root / "output" / "T01_CONTROL_TWO_BODY.e"
            stale.write_text("stale", encoding="utf-8")
            second = prepare_gmat_multicase_files(spec_path, project_root=root)
            self.assertEqual(len(second.archived_outputs), 1)
            self.assertFalse(stale.exists())
            self.assertEqual(second.archived_outputs[0].read_text(), "stale")

    def test_synthetic_matrix_validation_and_results_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec_path = self._write_project(root)
            prepared = prepare_gmat_multicase_files(spec_path, project_root=root)
            with self.assertRaises(FileNotFoundError):
                run_gmat_multicase_validation(spec_path, project_root=root)

            manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
            for case in manifest["cases"]:
                config_path = root / case["configuration"]
                config = json.loads(config_path.read_text(encoding="utf-8"))
                times = np.array([0.0, 60.0, 120.0])
                state = initial_state_from_config(config)
                earth = config["earth_model"]
                integ = config["integrator"]
                two_body = propagate_numerical_two_body(
                    state,
                    earth["gravitational_parameter_km3_s2"],
                    times,
                    method=integ["method"],
                    relative_tolerance=integ["relative_tolerance"],
                    absolute_tolerance=integ["absolute_tolerance"],
                    maximum_step_seconds=integ["maximum_step_seconds"],
                )
                j2 = propagate_numerical_j2_gmat_matched(
                    state,
                    earth["gravitational_parameter_km3_s2"],
                    earth["equatorial_radius_km"],
                    earth["j2"],
                    times,
                    method=integ["method"],
                    relative_tolerance=integ["relative_tolerance"],
                    absolute_tolerance=integ["absolute_tolerance"],
                    maximum_step_seconds=integ["maximum_step_seconds"],
                )
                write_stk(root / case["two_body_output"], two_body)
                write_stk(root / case["j2_output"], j2)

            result = run_gmat_multicase_validation(spec_path, project_root=root)
            self.assertEqual(result.validation_status, "passed_with_warnings")
            self.assertEqual(result.passed_case_count, 2)
            self.assertTrue(result.report_path.is_file())
            archive = package_gmat_multicase_results(
                spec_path,
                project_root=root,
                output_path=root / "results.zip",
            )
            with zipfile.ZipFile(archive, "r") as stream:
                self.assertIsNone(stream.testzip())
                self.assertIn(
                    "GMAT_1B_RESULTS_PACKAGE_MANIFEST.json", stream.namelist()
                )


if __name__ == "__main__":
    unittest.main()
