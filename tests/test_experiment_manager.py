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

from research_core.experiment_manager import run_experiment


class ExperimentManagerTests(unittest.TestCase):
    def _temporary_configuration(self, temporary_path: Path) -> Path:
        source_config = PROJECT_ROOT / "configs" / "case_leo400.json"
        config = json.loads(source_config.read_text(encoding="utf-8"))
        config["outputs"]["results_root"] = str(temporary_path / "results")
        config["propagation"]["default_duration_hours"] = 0.25
        config["propagation"]["output_step_seconds"] = 60
        config["drag"]["sensitivity"]["enabled"] = False
        config["outputs"]["save_pdf"] = False
        temporary_config = temporary_path / "case.json"
        temporary_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return temporary_config

    def test_run_creates_scientific_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            result = run_experiment(
                self._temporary_configuration(temporary_path),
                project_root=PROJECT_ROOT,
                console_logging=False,
            )
            required = {
                "experiment_configuration.json",
                "environment_metadata.json",
                "initial_state.json",
                "initial_conditions.csv",
                "orbit_summary.json",
                "analytical_two_body_states.csv",
                "numerical_two_body_states.csv",
                "numerical_j2_states.csv",
                "numerical_j2_drag_states.csv",
                "two_body_comparison.csv",
                "j2_two_body_comparison.csv",
                "j2_rtn_comparison.csv",
                "j2_orbital_elements.csv",
                "j2_conservation_diagnostics.csv",
                "j2_validation.csv",
                "j2_validation_summary.json",
                "drag_j2_comparison.csv",
                "drag_rtn_comparison.csv",
                "j2_drag_orbital_elements.csv",
                "drag_diagnostics.csv",
                "drag_validation.csv",
                "drag_validation_summary.json",
                "drag_zero_density_limit.json",
                "drag_sensitivity.csv",
                "drag_sensitivity_summary.json",
                "conservation_diagnostics.csv",
                "model_error_summary.csv",
                "model_error_summary.json",
                "validation_status.json",
                "TECHNICAL_SUMMARY.md",
                "run_log.txt",
                "RESEARCH_REPORT.html",
                "RUN_MANIFEST.json",
                "FINAL_VALIDATION_SUMMARY.json",
            }
            created_names = {path.name for path in result.created_files}
            self.assertTrue(required.issubset(created_names))
            self.assertIn(result.validation_status, {"passed", "passed_with_warnings"})
            self.assertLess(result.maximum_position_difference_m, 1.0)
            self.assertGreater(result.maximum_j2_two_body_position_difference_km, 0.0)
            self.assertGreater(result.maximum_drag_j2_position_difference_km, 0.0)
            self.assertLess(result.final_drag_semi_major_axis_difference_vs_j2_m, 0.0)
            self.assertGreater(result.drag_total_specific_energy_loss_km2_s2, 0.0)

    def test_repeated_runs_create_unique_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            config_path = self._temporary_configuration(temporary_path)
            first = run_experiment(
                config_path,
                project_root=PROJECT_ROOT,
                console_logging=False,
            )
            second = run_experiment(
                config_path,
                project_root=PROJECT_ROOT,
                console_logging=False,
            )
            self.assertNotEqual(first.result_directory, second.result_directory)


if __name__ == "__main__":
    unittest.main()
