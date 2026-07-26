from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research_core.paper1_results_consolidation import MATRIX_ID, validate_production_summary


def valid_summary() -> dict:
    runs = []
    for case_id, hours in {
        "CASE-LEO400": (6, 24, 72, 168),
        "CASE-SSO700": (6, 24, 72, 168),
        "CASE-ISS-TLE": (6, 24, 72),
    }.items():
        for duration in hours:
            runs.append(
                {
                    "case_id": case_id,
                    "duration_hours": duration,
                    "validation_status": "passed_with_warnings",
                }
            )
    return {
        "matrix_id": MATRIX_ID,
        "status": "passed_with_warnings",
        "completed_experiment_count": 11,
        "expected_experiment_count": 11,
        "failed_experiment_count": 0,
        "primary_model_run_count": 43,
        "executed_model_run_count": 47,
        "failures": [],
        "scope_decision": "paper1_models_frozen_no_additional_force_models",
        "convergence": {
            "validation_status": "passed_with_warnings",
            "evaluated_setting_count": 36,
            "passing_candidate_count": 36,
        },
        "runs": runs,
    }


class ValidationTests(unittest.TestCase):
    def test_exact_frozen_matrix_passes(self) -> None:
        self.assertEqual(validate_production_summary(valid_summary()), [])

    def test_failed_run_is_rejected(self) -> None:
        summary = valid_summary()
        summary["runs"][0]["validation_status"] = "failed"
        errors = validate_production_summary(summary)
        self.assertTrue(any("run did not pass" in error for error in errors))

    def test_missing_case_is_rejected(self) -> None:
        summary = valid_summary()
        summary["runs"].pop()
        errors = validate_production_summary(summary)
        self.assertTrue(any("run matrix differs" in error for error in errors))

    def test_scope_drift_is_rejected(self) -> None:
        summary = valid_summary()
        summary["scope_decision"] = "add_more_models"
        errors = validate_production_summary(summary)
        self.assertTrue(any("scope-freeze" in error for error in errors))


if __name__ == "__main__":
    unittest.main()

