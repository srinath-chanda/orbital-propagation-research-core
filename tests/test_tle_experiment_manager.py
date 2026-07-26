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
from research_core.tle_experiment_manager import run_tle_experiment


class TLEExperimentManagerTests(unittest.TestCase):
    def test_fixed_tle_configuration_is_valid(self) -> None:
        config, warnings = load_and_validate_config(PROJECT_ROOT / "configs" / "case_iss_tle.json")
        self.assertEqual(config["initial_state"]["source_type"], "fixed_tle")
        self.assertGreaterEqual(len(warnings), 1)

    def test_short_tle_run_creates_scientific_outputs(self) -> None:
        source = PROJECT_ROOT / "configs" / "case_iss_tle.json"
        config = json.loads(source.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            config["propagation"]["default_duration_hours"] = 0.1
            config["propagation"]["output_step_seconds"] = 60
            config["outputs"]["results_root"] = str(temp / "results")
            # Resolve frozen files absolutely before moving the config.
            config["initial_state"]["tle_file"] = str(PROJECT_ROOT / "data" / "tle" / "iss_25544_2026-07-18_celestrak.tle")
            config["initial_state"]["tle_metadata_file"] = str(PROJECT_ROOT / "data" / "tle" / "iss_25544_2026-07-18_celestrak_metadata.json")
            path = temp / "case.json"
            path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            result = run_tle_experiment(path, project_root=PROJECT_ROOT)
            self.assertIn(result.validation_status, {"passed", "passed_with_warnings"})
            self.assertEqual(result.nonzero_sgp4_error_count, 0)
            self.assertTrue((result.result_directory / "SGP4_TECHNICAL_SUMMARY.md").is_file())
            self.assertTrue((result.result_directory / "sgp4_states.csv").is_file())
            self.assertTrue((result.result_directory / "sgp4_teme_states.csv").is_file())
            self.assertTrue((result.result_directory / "tle_provenance.json").is_file())
            self.assertTrue((result.result_directory / "sgp4_model_error_summary.csv").is_file())
            self.assertTrue((result.result_directory / "sgp4_ground_track.csv").is_file())
            self.assertTrue((result.result_directory / "ground_track_summary.csv").is_file())
            self.assertTrue((result.result_directory / "GROUND_TRACK_TECHNICAL_SUMMARY.md").is_file())
            self.assertTrue((result.result_directory / "figures" / "ground_track_comparison.png").is_file())
            self.assertTrue((result.result_directory / "pass_analysis_summary.json").is_file())
            self.assertTrue((result.result_directory / "gs_bremen_001_sgp4_visibility.csv").is_file())
            self.assertTrue((result.result_directory / "gs_bremen_001_sgp4_passes.csv").is_file())
            self.assertTrue((result.result_directory / "GS-BREMEN-001_PASS_TECHNICAL_SUMMARY.md").is_file())
            self.assertTrue((result.result_directory / "figures" / "bremen_elevation_history.png").is_file())
            self.assertTrue((result.result_directory / "RESEARCH_REPORT.html").is_file())
            self.assertTrue((result.result_directory / "RUN_MANIFEST.json").is_file())
            self.assertTrue((result.result_directory / "FINAL_VALIDATION_SUMMARY.json").is_file())
            self.assertEqual(result.research_report_path, result.result_directory / "RESEARCH_REPORT.html")
            self.assertLessEqual(result.geodetic_roundtrip_position_error_m, 1.0e-6)
            self.assertIn("numerical_j2", result.maximum_ground_track_separation_km_by_model)
            self.assertEqual(result.pass_station_id, "GS-BREMEN-001")
            self.assertIn("sgp4", result.pass_count_by_model)


if __name__ == "__main__":
    unittest.main()
