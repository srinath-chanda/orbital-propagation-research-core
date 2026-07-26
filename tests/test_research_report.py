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

from research_core.research_report import (
    build_final_validation_summary,
    build_run_manifest,
    write_controlled_research_report,
)


class ResearchReportTests(unittest.TestCase):
    def test_final_validation_summary_counts_statuses(self) -> None:
        config = {
            "experiment": {"experiment_id": "EXP-1", "case_id": "CASE-1"},
            "external_validation": {"enabled": False},
        }
        validation = {
            "overall_status": "passed_with_warnings",
            "checks": [
                {"status": "passed", "validation_id": "A", "name": "a"},
                {"status": "passed", "validation_id": "B", "name": "b"},
                {"status": "not_evaluated", "validation_id": "C", "name": "c"},
            ],
        }
        summary = build_final_validation_summary(
            config=config,
            validation=validation,
            warnings=["warning"],
            report_filename="RESEARCH_REPORT.html",
        )
        self.assertEqual(summary["validation_check_counts"]["passed"], 2)
        self.assertEqual(summary["validation_check_counts"]["not_evaluated"], 1)
        self.assertEqual(
            summary["scientific_claim_level"],
            "internally_verified_external_validation_pending",
        )

    def test_manifest_contains_checksum_inventory(self) -> None:
        config = {"experiment": {"experiment_id": "EXP-1", "case_id": "CASE-1"}}
        validation = {"overall_status": "passed", "checks": []}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
            (root / "notes.md").write_text("# Notes\n", encoding="utf-8")
            manifest = build_run_manifest(
                result_directory=root,
                config=config,
                validation=validation,
                warnings=[],
            )
            self.assertEqual(manifest["file_count_excluding_manifest"], 2)
            self.assertEqual(len(manifest["files"][0]["sha256"]), 64)
            self.assertIn("scientific_data_csv", manifest["category_counts"])

    def test_controlled_report_escapes_title_and_links_outputs(self) -> None:
        config = {
            "experiment": {
                "experiment_id": "EXP-1",
                "case_id": "CASE-1",
                "title": "Orbit <script>alert(1)</script>",
            },
            "scientific_cautions": ["Research only"],
        }
        validation = {"overall_status": "passed", "checks": []}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "states.csv"
            data.write_text("t,x\n0,1\n", encoding="utf-8")
            report = root / "RESEARCH_REPORT.html"
            write_controlled_research_report(
                report,
                config=config,
                orbit_summary={"orbital_period_minutes": 90.0},
                two_body_summary={
                    "position_difference_m": {"maximum_absolute": 1e-6}
                },
                j2_validation={"fitted_raan_rate_deg_day": -5.0},
                drag_validation={
                    "final_semi_major_axis_difference_vs_j2_m": -100.0
                },
                validation=validation,
                warnings=[],
                created_files=[data],
            )
            text = report.read_text(encoding="utf-8")
            self.assertNotIn("<script>alert(1)</script>", text)
            self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", text)
            self.assertIn('href="states.csv"', text)

    def test_manifest_json_is_serializable(self) -> None:
        config = {"experiment": {"experiment_id": "EXP-1", "case_id": "CASE-1"}}
        validation = {"overall_status": "passed", "checks": []}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "x.json").write_text("{}\n", encoding="utf-8")
            manifest = build_run_manifest(
                result_directory=root,
                config=config,
                validation=validation,
                warnings=["w"],
            )
            json.dumps(manifest)


if __name__ == "__main__":
    unittest.main()
