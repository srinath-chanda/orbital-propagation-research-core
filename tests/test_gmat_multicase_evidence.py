from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.external_validation import parse_stk_time_pos_vel


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GmatMultiCaseEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference_root = PROJECT_ROOT / "data" / "reference" / "gmat_1b"
        cls.package_manifest_path = (
            cls.reference_root / "GMAT_1B_RESULTS_PACKAGE_MANIFEST.json"
        )
        cls.package_manifest = json.loads(
            cls.package_manifest_path.read_text(encoding="utf-8")
        )
        cls.closure_path = (
            PROJECT_ROOT
            / "results"
            / "EXP-GMAT-1B-MULTICASE-001"
            / "GMAT_VALIDATION_CLOSURE_1B_1.json"
        )
        cls.closure = json.loads(cls.closure_path.read_text(encoding="utf-8"))
        cls.summary_path = (
            PROJECT_ROOT
            / cls.closure["evidence"]["aggregate_summary"]
        )
        cls.summary = json.loads(cls.summary_path.read_text(encoding="utf-8"))

    def test_official_evidence_inventory_matches_every_recorded_hash_and_size(self):
        files = self.package_manifest["files"]
        self.assertEqual(len(files), 60)
        ephemerides = [item for item in files if item["path"].endswith(".e")]
        self.assertEqual(len(ephemerides), 20)
        for item in files:
            path = PROJECT_ROOT / item["path"]
            self.assertTrue(path.is_file(), item["path"])
            self.assertEqual(path.stat().st_size, item["size_bytes"], item["path"])
            self.assertEqual(sha256(path), item["sha256"], item["path"])

        evidence = self.closure["evidence"]
        for name in (
            "package_manifest",
            "matrix_specification",
            "preparation_manifest",
            "aggregate_summary",
            "aggregate_report",
        ):
            path = PROJECT_ROOT / evidence[name]
            self.assertEqual(sha256(path), evidence[f"{name}_sha256"])

    def test_real_matrix_summary_and_raw_ephemerides_close_all_cases(self):
        self.assertEqual(self.summary["validation_status"], "passed_with_warnings")
        self.assertEqual(self.summary["total_case_count"], 10)
        self.assertEqual(self.summary["passed_case_count"], 10)
        self.assertEqual(self.summary["failed_case_count"], 0)
        self.assertEqual(self.summary["incomplete_case_count"], 0)
        self.assertTrue(self.summary["thresholds_preregistered"])
        self.assertEqual(self.closure["completed_check_count"], 120)
        self.assertEqual(self.closure["failed_check_count"], 0)
        self.assertFalse(self.closure["thresholds_relaxed"])

        maximum_position = 0.0
        maximum_velocity = 0.0
        for case in self.summary["cases"]:
            self.assertEqual(case["status"], "passed_with_warnings")
            config_path = self.reference_root / "cases" / f"{case['case_id']}.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            duration_seconds = float(config["external_validation"]["duration_seconds"])
            expected_count = round(duration_seconds / 60.0) + 1
            thresholds = config["external_validation"]["thresholds"]
            self.assertLessEqual(
                case["pole_aware_maximum_position_difference_m"],
                thresholds["j2_maximum_position_difference_m"],
            )
            self.assertLessEqual(
                case["pole_aware_maximum_velocity_difference_mm_s"],
                thresholds["j2_maximum_velocity_difference_mm_s"],
            )
            maximum_position = max(
                maximum_position,
                case["pole_aware_maximum_position_difference_m"],
            )
            maximum_velocity = max(
                maximum_velocity,
                case["pole_aware_maximum_velocity_difference_mm_s"],
            )
            histories = []
            for suffix in ("TWO_BODY", "J2"):
                path = self.reference_root / "output" / f"{case['case_id']}_{suffix}.e"
                history = parse_stk_time_pos_vel(path, model_name=suffix.lower())
                self.assertEqual(history.epoch_utc, case["epoch_utc"])
                self.assertEqual(history.elapsed_seconds.size, expected_count)
                self.assertLessEqual(
                    abs(float(history.elapsed_seconds[-1]) - duration_seconds),
                    1.0e-6,
                )
                histories.append(history)
            self.assertFalse(
                (histories[0].positions_km == histories[1].positions_km).all()
            )

        extrema = self.closure["validation_extrema"]
        self.assertAlmostEqual(
            maximum_position,
            extrema["maximum_matrix_pole_aware_position_difference_m"],
        )
        self.assertAlmostEqual(
            maximum_velocity,
            extrema["maximum_matrix_pole_aware_velocity_difference_mm_s"],
        )


if __name__ == "__main__":
    unittest.main()
