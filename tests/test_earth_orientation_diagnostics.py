from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.earth_orientation_diagnostics import (
    load_earth_orientation_diagnostic_config,
    run_earth_orientation_diagnostics,
)
from research_core.external_validation import initial_state_from_config
from research_core.propagators.numerical_j2 import propagate_numerical_j2_gmat_matched


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
    for time, position, velocity in zip(
        history.elapsed_seconds, history.positions_km, history.velocities_km_s
    ):
        lines.append(
            f"{time:.9f} {position[0]:.15f} {position[1]:.15f} "
            f"{position[2]:.15f} {velocity[0]:.15f} {velocity[1]:.15f} "
            f"{velocity[2]:.15f}"
        )
    lines.extend(["END Ephemeris", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


class EarthOrientationDiagnosticTests(unittest.TestCase):
    def test_release_configuration_preregisters_five_model_comparison(self):
        config = load_earth_orientation_diagnostic_config(
            PROJECT_ROOT / "configs" / "earth_orientation_1c_diagnostic.json"
        )
        self.assertEqual(config["baseline_model"], "iau1976_1980")
        self.assertEqual(len(config["models"]), 5)
        candidates = [
            item["model_id"]
            for item in config["models"]
            if item["eligible_candidate"]
        ]
        self.assertEqual(candidates, ["iau2000a", "iau2006_2000a"])
        self.assertEqual(
            config["decision_rule"]["status"],
            "preregistered_before_full_matrix_diagnostic",
        )
        result_dir = (
            PROJECT_ROOT
            / "results"
            / "EXP-EARTH-ORIENTATION-1C-001"
            / "2026-07-19_155343_232971Z"
        )
        summary = json.loads(
            (result_dir / "earth_orientation_diagnostic_summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["comparison_count"], 50)
        self.assertEqual(
            summary["decision"],
            "baseline_retained_no_candidate_met_preregistered_rule",
        )
        self.assertEqual(summary["recommended_model"], "iau1976_1980")
        aggregates = {item["model_id"]: item for item in summary["model_aggregates"]}
        self.assertTrue(aggregates["iau1976_1980"]["all_existing_gates_passed"])
        self.assertTrue(aggregates["iau2000a"]["all_existing_gates_passed"])
        self.assertTrue(aggregates["iau2006_2000a"]["all_existing_gates_passed"])
        self.assertFalse(aggregates["iau2000a"]["decision_rule_passed"])
        self.assertFalse(aggregates["iau2006_2000a"]["decision_rule_passed"])
        manifest = json.loads(
            (result_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8")
        )
        for item in manifest["files"]:
            path = result_dir / item["path"]
            self.assertEqual(path.stat().st_size, item["size_bytes"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"]
            )

    def test_invalid_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "1C.0",
                        "baseline_model": "iau1976_1980",
                        "models": [
                            {
                                "model_id": "iau1976_1980",
                                "role": "baseline",
                                "eligible_candidate": False,
                            },
                            {
                                "model_id": "not_a_model",
                                "role": "candidate",
                                "eligible_candidate": True,
                            },
                        ],
                        "decision_rule": {
                            "maximum_median_position_ratio": 0.8,
                            "maximum_worst_position_ratio": 1.0,
                            "maximum_median_velocity_ratio": 0.8,
                            "maximum_worst_velocity_ratio": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_earth_orientation_diagnostic_config(path)

    def test_synthetic_diagnostic_creates_traceable_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configs = root / "configs"
            cases = root / "data" / "reference" / "gmat_1b" / "cases"
            output = root / "data" / "reference" / "gmat_1b" / "output"
            configs.mkdir(parents=True)
            cases.mkdir(parents=True)
            output.mkdir(parents=True)

            case_id = "T01_CONTROL"
            case_config = json.loads(
                (
                    PROJECT_ROOT
                    / "configs"
                    / "case_leo400_gmat_matched.json"
                ).read_text(encoding="utf-8")
            )
            case_config["external_validation"]["duration_seconds"] = 120.0
            case_config["external_validation"]["output_step_seconds"] = 60.0
            case_config["external_validation"]["thresholds"][
                "j2_maximum_position_difference_m"
            ] = 10.0
            case_config["external_validation"]["thresholds"][
                "j2_maximum_velocity_difference_mm_s"
            ] = 10.0
            case_path = cases / f"{case_id}.json"
            case_path.write_text(json.dumps(case_config), encoding="utf-8")

            state = initial_state_from_config(case_config)
            earth = case_config["earth_model"]
            integrator = case_config["integrator"]
            history = propagate_numerical_j2_gmat_matched(
                state,
                earth["gravitational_parameter_km3_s2"],
                earth["equatorial_radius_km"],
                earth["j2"],
                np.array([0.0, 60.0, 120.0]),
                method=integrator["method"],
                relative_tolerance=integrator["relative_tolerance"],
                absolute_tolerance=integrator["absolute_tolerance"],
                maximum_step_seconds=integrator["maximum_step_seconds"],
            )
            ephemeris = output / f"{case_id}_J2.e"
            write_stk(ephemeris, history)

            matrix = {
                "schema_version": "1B.0",
                "reference_root": "data/reference/gmat_1b",
                "cases": [
                    {
                        "case_id": case_id,
                        "factor": "baseline_control",
                        "epoch_utc": case_config["initial_state"]["epoch_utc"],
                        "altitude_km": 400.0,
                        "inclination_deg": 51.6,
                        "duration_hours": 120.0 / 3600.0,
                    }
                ],
            }
            matrix_path = configs / "matrix.json"
            matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
            diagnostic = {
                "schema_version": "1C.0",
                "experiment_id": "EXP-1C-TEST",
                "matrix_specification": "configs/matrix.json",
                "baseline_model": "iau1976_1980",
                "models": [
                    {
                        "model_id": "iau1976_1980",
                        "role": "baseline",
                        "eligible_candidate": False,
                    },
                    {
                        "model_id": "iau2006_2000a",
                        "role": "candidate",
                        "eligible_candidate": True,
                    },
                ],
                "decision_rule": {
                    "maximum_median_position_ratio": 0.8,
                    "maximum_worst_position_ratio": 1.0,
                    "maximum_median_velocity_ratio": 0.8,
                    "maximum_worst_velocity_ratio": 1.0,
                },
            }
            diagnostic_path = configs / "diagnostic.json"
            diagnostic_path.write_text(json.dumps(diagnostic), encoding="utf-8")

            result = run_earth_orientation_diagnostics(
                diagnostic_path, project_root=root
            )
            self.assertEqual(result.case_count, 1)
            self.assertEqual(result.model_count, 2)
            self.assertTrue(result.report_path.is_file())
            summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["comparison_count"], 2)
            self.assertEqual(summary["baseline_model"], "iau1976_1980")
            manifest = json.loads(
                (result.result_directory / "RUN_MANIFEST.json").read_text(
                    encoding="utf-8"
                )
            )
            for item in manifest["files"]:
                path = result.result_directory / item["path"]
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"]
                )


if __name__ == "__main__":
    unittest.main()
