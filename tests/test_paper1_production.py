from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.configuration import load_and_validate_config
from research_core.paper1_production import (
    _runtime_rows,
    load_paper1_matrix,
    prepare_paper1_production,
    verify_paper1_baseline,
)


MATRIX = PROJECT_ROOT / "configs/paper1_production_matrix.json"
CLOSURE = PROJECT_ROOT / "configs/paper1_baseline_closure.json"


class Paper1ProductionTests(unittest.TestCase):
    def test_original_research_matrix_is_frozen(self):
        matrix = load_paper1_matrix(MATRIX)
        self.assertEqual(matrix["primary_experiment_count"], 11)
        self.assertEqual(matrix["primary_model_run_count"], 43)
        self.assertEqual(matrix["executed_model_run_count"], 47)
        self.assertEqual(matrix["cases"][2]["durations_hours"], [6, 24, 72])

    def test_checksum_gated_baseline_is_complete(self):
        result = verify_paper1_baseline(CLOSURE, project_root=PROJECT_ROOT)
        self.assertEqual(result.evidence_count, 5)
        self.assertEqual(result.drag_scenario_count, 4)
        self.assertEqual(result.drag_check_count, 25)
        self.assertLess(result.maximum_drag_time_residual_seconds, 5e-6)

    def test_baseline_rejects_changed_evidence_checksum(self):
        payload = json.loads(CLOSURE.read_text(encoding="utf-8"))
        payload["required_evidence"]["drag_acceleration_summary"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "closure.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum failed"):
                verify_paper1_baseline(path, project_root=PROJECT_ROOT)

    def test_sso700_is_a_valid_near_polar_controlled_case(self):
        config, _ = load_and_validate_config(PROJECT_ROOT / "configs/case_sso700.json")
        self.assertEqual(config["experiment"]["case_id"], "CASE-SSO700")
        self.assertAlmostEqual(config["initial_state"]["semi_major_axis_km"], 7078.1363)
        self.assertAlmostEqual(config["initial_state"]["inclination_deg"], 98.18796410124214)
        self.assertFalse(config["drag"]["sensitivity"]["enabled"])

    def test_preparation_creates_exact_immutable_run_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = prepare_paper1_production(
                MATRIX,
                project_root=PROJECT_ROOT,
                output_directory=Path(temporary) / "configs",
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(result.experiment_count, 11)
            self.assertEqual(result.primary_model_run_count, 43)
            self.assertEqual(result.executed_model_run_count, 47)
            self.assertEqual(len(manifest["runs"]), 11)
            names = {Path(item["configuration"]).name for item in manifest["runs"]}
            self.assertIn("CASE-LEO400_168H.json", names)
            self.assertIn("CASE-SSO700_168H.json", names)
            self.assertIn("CASE-ISS-TLE_072H.json", names)

    def test_tle_runtime_summary_reads_nested_model_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary)
            (result / "sgp4_diagnostics.json").write_text(
                json.dumps({"sgp4_runtime_seconds": 0.25}), encoding="utf-8"
            )
            (result / "sgp4_model_error_summary.json").write_text(
                json.dumps(
                    {
                        "duration_hours": 6.0,
                        "reference_model": "sgp4",
                        "models": {
                            "numerical_j2": {
                                "runtime_seconds": 1.5,
                                "function_evaluations": 100,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            rows = _runtime_rows(
                {
                    "case_id": "CASE-ISS-TLE",
                    "duration_hours": 6,
                    "run_kind": "fixed_tle",
                    "result_directory": str(result),
                }
            )
        self.assertEqual([row["model"] for row in rows], ["sgp4", "numerical_j2"])
        self.assertEqual(rows[1]["function_evaluations"], 100)


if __name__ == "__main__":
    unittest.main()
