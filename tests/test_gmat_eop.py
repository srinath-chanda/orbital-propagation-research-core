from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from functools import partial
from pathlib import Path

import numpy as np
from astropy.time import Time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.earth_orientation import (
    earth_pole_unit_vector,
    pole_angular_separation_arcsec,
)
from research_core.external_validation import initial_state_from_config
from research_core.gmat_eop import (
    GMAT_R2026A_EOP_SHA256,
    GmatEopDataset,
    gmat_r2026a_eop_pole_unit_vector,
    gmat_r2026a_polar_motion_matrix,
)
from research_core.gmat_eop_diagnostics import load_gmat_eop_diagnostic_config
from research_core.propagators.numerical_j2 import (
    propagate_numerical_j2_gmat_matched,
    propagate_numerical_j2_pole_provider,
)


EOP_PATH = (
    PROJECT_ROOT / "data" / "reference" / "gmat_r2026a" / "eopc04_08.62-now"
)
OFFICIAL_RESULT = (
    PROJECT_ROOT
    / "results"
    / "EXP-GMAT-EOP-1C1-001"
    / "2026-07-19_173443_299216Z"
)


class GmatEopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = GmatEopDataset.from_file(
            EOP_PATH, expected_sha256=GMAT_R2026A_EOP_SHA256
        )

    def test_official_tagged_file_identity_and_coverage(self):
        self.assertEqual(self.dataset.source_sha256, GMAT_R2026A_EOP_SHA256)
        self.assertEqual(self.dataset.row_count, 23633)
        self.assertEqual(self.dataset.first_mjd_utc, 37665.0)
        self.assertEqual(self.dataset.last_mjd_utc, 61297.0)
        self.assertEqual(EOP_PATH.stat().st_size, 3687646)

    def test_release_configuration_preregisters_only_full_eop_candidate(self):
        config = load_gmat_eop_diagnostic_config(
            PROJECT_ROOT / "configs" / "gmat_eop_1c1_diagnostic.json"
        )
        self.assertEqual(config["baseline_model"], "iau1976_1980")
        self.assertEqual(len(config["models"]), 4)
        candidates = [
            item["model_id"] for item in config["models"] if item["eligible_candidate"]
        ]
        self.assertEqual(candidates, ["gmat_r2026a_eop_full"])
        self.assertEqual(
            config["decision_rule"]["status"],
            "preregistered_before_full_matrix_diagnostic",
        )
        self.assertEqual(
            config["eop_expected_sha256"], GMAT_R2026A_EOP_SHA256
        )

    def test_known_rows_and_october_clamping_match_gmat_behavior(self):
        january = self.dataset.sample(Time("2026-01-01T00:00:00", scale="utc"))
        self.assertEqual(january.coverage_status, "exact_source_row")
        self.assertAlmostEqual(january.x_arcsec, 0.110512, places=12)
        self.assertAlmostEqual(january.y_arcsec, 0.331193, places=12)
        self.assertAlmostEqual(january.ut1_utc_seconds, 0.0740869, places=12)
        self.assertAlmostEqual(january.lod_seconds, -0.0000217, places=12)
        october = self.dataset.sample(Time("2026-10-01T00:00:00", scale="utc"))
        self.assertEqual(october.coverage_status, "clamped_after_source_range")
        self.assertAlmostEqual(october.x_arcsec, 0.229781, places=12)
        self.assertAlmostEqual(october.y_arcsec, 0.326522, places=12)
        self.assertAlmostEqual(october.ut1_utc_seconds, 0.0915262, places=12)
        self.assertEqual(
            october.uncertainty_status,
            "placeholder_uncertainty_in_tagged_file",
        )

    def test_interpolation_and_left_row_lod_match_gmat(self):
        left = self.dataset.sample(Time(61041.0, format="mjd", scale="utc"))
        right = self.dataset.sample(Time(61042.0, format="mjd", scale="utc"))
        middle = self.dataset.sample(Time(61041.5, format="mjd", scale="utc"))
        self.assertEqual(
            middle.coverage_status, "interpolated_between_source_rows"
        )
        self.assertAlmostEqual(
            middle.x_arcsec, (left.x_arcsec + right.x_arcsec) / 2.0, places=12
        )
        self.assertAlmostEqual(
            middle.y_arcsec, (left.y_arcsec + right.y_arcsec) / 2.0, places=12
        )
        self.assertAlmostEqual(
            middle.ut1_utc_seconds,
            (left.ut1_utc_seconds + right.ut1_utc_seconds) / 2.0,
            places=10,
        )
        self.assertEqual(middle.lod_seconds, left.lod_seconds)

    def test_checksum_mismatch_and_malformed_data_are_rejected(self):
        with self.assertRaises(ValueError):
            GmatEopDataset.from_file(EOP_PATH, expected_sha256="0" * 64)
        with tempfile.TemporaryDirectory() as temporary:
            malformed = Path(temporary) / "eop.txt"
            malformed.write_text("2026 1 1 61041 0.1 0.2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                GmatEopDataset.from_file(malformed)

    def test_unsupported_diagnostic_model_is_rejected(self):
        release = json.loads(
            (
                PROJECT_ROOT / "configs" / "gmat_eop_1c1_diagnostic.json"
            ).read_text(encoding="utf-8")
        )
        release["models"][1]["model_id"] = "unsupported"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            path.write_text(json.dumps(release), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_gmat_eop_diagnostic_config(path)

    def test_polar_motion_matrix_is_proper_rotation_with_gmat_signs(self):
        matrix = gmat_r2026a_polar_motion_matrix(0.2, 0.4)
        np.testing.assert_allclose(matrix @ matrix.T, np.identity(3), atol=2.0e-16)
        self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=15)
        self.assertLess(matrix[2, 0], 0.0)
        self.assertGreater(matrix[2, 1], 0.0)

    def test_full_eop_axis_is_finite_and_has_expected_scale(self):
        epoch = "2026-07-01T00:00:00Z"
        baseline = earth_pole_unit_vector(epoch, 0.0, "iau1976_1980")
        full = gmat_r2026a_eop_pole_unit_vector(
            epoch, 0.0, self.dataset, "gmat_r2026a_eop_full"
        )
        self.assertAlmostEqual(float(np.linalg.norm(full)), 1.0, places=14)
        separation = pole_angular_separation_arcsec(baseline, full)
        self.assertGreater(separation, 0.3)
        self.assertLess(separation, 0.6)

    def test_pole_provider_preserves_closed_baseline_exactly(self):
        config = json.loads(
            (
                PROJECT_ROOT / "configs" / "case_leo400_gmat_matched.json"
            ).read_text(encoding="utf-8")
        )
        state = initial_state_from_config(config)
        earth = config["earth_model"]
        integrator = config["integrator"]
        times = np.asarray([0.0, 60.0, 120.0, 300.0])
        kwargs = {
            "method": integrator["method"],
            "relative_tolerance": integrator["relative_tolerance"],
            "absolute_tolerance": integrator["absolute_tolerance"],
            "maximum_step_seconds": integrator["maximum_step_seconds"],
        }
        closed = propagate_numerical_j2_gmat_matched(
            state,
            earth["gravitational_parameter_km3_s2"],
            earth["equatorial_radius_km"],
            earth["j2"],
            times,
            **kwargs,
        )
        provider = partial(earth_pole_unit_vector, model="iau1976_1980")
        generic = propagate_numerical_j2_pole_provider(
            state,
            earth["gravitational_parameter_km3_s2"],
            earth["equatorial_radius_km"],
            earth["j2"],
            times,
            pole_provider=provider,
            model_name="baseline_provider",
            **kwargs,
        )
        np.testing.assert_array_equal(generic.positions_km, closed.positions_km)
        np.testing.assert_array_equal(generic.velocities_km_s, closed.velocities_km_s)

    def test_official_diagnostic_result_is_complete_and_checksum_valid(self):
        summary = json.loads(
            (OFFICIAL_RESULT / "gmat_eop_diagnostic_summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["case_count"], 10)
        self.assertEqual(summary["model_count"], 4)
        self.assertEqual(summary["comparison_count"], 40)
        self.assertEqual(
            summary["decision"],
            "candidate_identified_requires_independent_validation",
        )
        self.assertEqual(summary["recommended_model"], "gmat_r2026a_eop_full")
        aggregates = {item["model_id"]: item for item in summary["model_aggregates"]}
        baseline = aggregates["iau1976_1980"]
        full = aggregates["gmat_r2026a_eop_full"]
        self.assertGreater(baseline["median_case_maximum_position_difference_m"], 3.0)
        self.assertLess(full["median_case_maximum_position_difference_m"], 0.01)
        self.assertLess(full["worst_case_maximum_position_difference_m"], 0.01)
        self.assertTrue(full["all_existing_gates_passed"])
        self.assertTrue(full["decision_rule_passed"])
        self.assertEqual(
            summary["eop_coverage"]["clamped_case_ids"],
            ["B08_EPOCH_OCT_400KM_I51P6_24H"],
        )
        manifest = json.loads(
            (OFFICIAL_RESULT / "RUN_MANIFEST.json").read_text(encoding="utf-8")
        )
        for item in manifest["files"]:
            path = OFFICIAL_RESULT / item["path"]
            self.assertEqual(path.stat().st_size, item["size_bytes"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"]
            )


if __name__ == "__main__":
    unittest.main()
