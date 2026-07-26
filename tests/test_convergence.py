from __future__ import annotations

import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.analysis.convergence import (
    generate_convergence_cases,
    pareto_frontier,
    select_recommendations,
)
from research_core.convergence_manager import run_convergence_study


class ConvergenceAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_path = PROJECT_ROOT / "configs" / "case_leo400.json"
        cls.config = json.loads(cls.config_path.read_text(encoding="utf-8"))

    def test_default_matrix_contains_36_unique_cases(self) -> None:
        cases = generate_convergence_cases(self.config["convergence"])
        self.assertEqual(36, len(cases))
        self.assertEqual(36, len({case.case_id for case in cases}))

    def test_pareto_frontier_excludes_dominated_case(self) -> None:
        rows = [
            {
                "case_id": "A",
                "median_runtime_seconds": 1.0,
                "maximum_position_difference_vs_analytical_m": 1.0,
            },
            {
                "case_id": "B",
                "median_runtime_seconds": 2.0,
                "maximum_position_difference_vs_analytical_m": 2.0,
            },
            {
                "case_id": "C",
                "median_runtime_seconds": 3.0,
                "maximum_position_difference_vs_analytical_m": 0.1,
            },
        ]
        identifiers = {row["case_id"] for row in pareto_frontier(rows)}
        self.assertEqual({"A", "C"}, identifiers)

    def test_selection_returns_balanced_recommendation(self) -> None:
        template = {
            "method": "DOP853",
            "relative_tolerance": 1e-9,
            "absolute_tolerance": 1e-11,
            "maximum_step_seconds": 30.0,
            "maximum_velocity_difference_vs_analytical_mm_s": 1e-5,
            "maximum_absolute_relative_energy_drift": 1e-12,
            "maximum_absolute_relative_angular_momentum_drift": 1e-12,
            "function_evaluations": 100,
        }
        rows = [
            {
                **template,
                "case_id": "fast",
                "median_runtime_seconds": 1.0,
                "maximum_position_difference_vs_analytical_m": 0.5,
            },
            {
                **template,
                "case_id": "balanced",
                "median_runtime_seconds": 2.0,
                "maximum_position_difference_vs_analytical_m": 0.01,
            },
            {
                **template,
                "case_id": "accurate",
                "median_runtime_seconds": 10.0,
                "maximum_position_difference_vs_analytical_m": 1e-5,
            },
        ]
        result = select_recommendations(rows, self.config["validation"])
        self.assertEqual(3, result["passing_case_count"])
        self.assertIsNotNone(result["balanced_recommendation"])

    def test_small_convergence_run_creates_required_outputs(self) -> None:
        config = deepcopy(self.config)
        config["convergence"].update(
            {
                "duration_hours": 0.1,
                "output_step_seconds": 30,
                "runtime_repetitions": 1,
                "relative_tolerances": [1e-7, 1e-9],
                "absolute_tolerances": [1e-9],
                "maximum_steps_seconds": [60, 10],
                "reference_maximum_step_seconds": 5,
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config["outputs"]["results_root"] = str(root / "results")
            path = root / "case.json"
            path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            result = run_convergence_study(
                path,
                project_root=PROJECT_ROOT,
                console_logging=False,
            )
            self.assertEqual(4, result.matrix_candidate_count)
            self.assertEqual(5, result.evaluated_setting_count)
            required = {
                "convergence_results.csv",
                "convergence_results.json",
                "numerical_reference_summary.json",
                "current_configuration_summary.json",
                "selected_integrator_settings.json",
                "convergence_validation_status.json",
                "CONVERGENCE_SUMMARY.md",
            }
            present = {file.name for file in result.created_files}
            self.assertTrue(required.issubset(present))


if __name__ == "__main__":
    unittest.main()
