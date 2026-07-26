from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.external_validation import initial_state_from_config
from research_core.gmat_eop import (
    GMAT_R2026A_EOP_SHA256,
    GmatEopDataset,
    gmat_r2026a_eop_pole_unit_vector,
)
from research_core.gmat_eop_independent import (
    load_independent_matrix_spec,
    package_independent_results,
    prepare_independent_matrix,
    run_independent_validation,
    verify_gmat_eop_install,
)
from research_core.propagators.numerical_j2 import (
    propagate_numerical_j2_pole_provider,
)
from research_core.propagators.numerical_two_body import propagate_numerical_two_body


MATRIX_PATH = PROJECT_ROOT / "configs" / "gmat_eop_1c2_independent_matrix.json"
REFERENCE_ROOT = PROJECT_ROOT / "data" / "reference" / "gmat_1c2"
EOP_PATH = (
    PROJECT_ROOT / "data" / "reference" / "gmat_r2026a" / "eopc04_08.62-now"
)


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


class IndependentEopValidationTests(unittest.TestCase):
    def test_release_matrix_is_preregistered_and_independent(self):
        spec = load_independent_matrix_spec(MATRIX_PATH)
        self.assertEqual(spec["candidate_model"], "gmat_r2026a_eop_full")
        self.assertEqual(spec["closed_baseline_model"], "iau1976_1980")
        self.assertEqual(len(spec["cases"]), 6)
        self.assertTrue(spec["adoption_rule"]["all_cases_must_pass"])
        old = json.loads(
            (PROJECT_ROOT / "configs" / "gmat_1b_multicase_matrix.json").read_text(
                encoding="utf-8"
            )
        )
        old_signatures = {
            (
                item["epoch_utc"],
                float(item["altitude_km"]),
                float(item["inclination_deg"]),
                float(item["duration_hours"]),
            )
            for item in old["cases"]
        }
        for case in spec["cases"]:
            signature = (
                case["epoch_utc"],
                float(case["altitude_km"]),
                float(case["inclination_deg"]),
                float(case["duration_hours"]),
            )
            self.assertNotIn(signature, old_signatures)

    def test_release_preparation_inventory_is_complete_and_hash_valid(self):
        manifest = json.loads(
            (REFERENCE_ROOT / "GMAT_1C2_MATRIX_MANIFEST.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["case_count"], 6)
        self.assertEqual(manifest["expected_output_count"], 12)
        self.assertEqual(
            manifest["status"], "scripts_prepared_independent_gmat_execution_pending"
        )
        self.assertEqual(manifest["eop_file_sha256"], GMAT_R2026A_EOP_SHA256)
        for case in manifest["cases"]:
            for path_key, hash_key in (
                ("configuration", "configuration_sha256"),
                ("two_body_script", "two_body_script_sha256"),
                ("j2_script", "j2_script_sha256"),
            ):
                path = PROJECT_ROOT / case[path_key]
                self.assertTrue(path.is_file())
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), case[hash_key]
                )
            case_config = json.loads(
                (PROJECT_ROOT / case["configuration"]).read_text(encoding="utf-8")
            )
            self.assertIn(
                case_config["propagation"]["comparison_reference_model"],
                case_config["propagation"]["models"],
            )
            self.assertNotIn("clamped", case["eop_start_coverage"])
            self.assertNotIn("clamped", case["eop_end_coverage"])

    def test_master_script_contains_all_twelve_propagations(self):
        script = (
            REFERENCE_ROOT / "scripts" / "RUN_ALL_CASES_1C2.script"
        ).read_text(encoding="utf-8")
        self.assertIn("Research Core 1C.2 independent full-EOP validation", script)
        self.assertEqual(script.count("Propagate C"), 12)
        self.assertEqual(script.count("Create EphemerisFile"), 12)
        for case in load_independent_matrix_spec(MATRIX_PATH)["cases"]:
            self.assertIn(str(case["case_id"]), script)

    def test_gmat_install_eop_verifier_accepts_exact_file_and_rejects_change(self):
        result = verify_gmat_eop_install(
            EOP_PATH, expected_sha256=GMAT_R2026A_EOP_SHA256
        )
        self.assertTrue(result["matches_gmat_r2026a_tag"])
        self.assertTrue(result["byte_exact_match"])
        self.assertFalse(result["line_ending_equivalent"])
        with tempfile.TemporaryDirectory() as temporary:
            windows = Path(temporary) / "windows_eopc04_08.62-now"
            windows.write_bytes(EOP_PATH.read_bytes().replace(b"\n", b"\r\n"))
            result = verify_gmat_eop_install(
                windows, expected_sha256=GMAT_R2026A_EOP_SHA256
            )
            self.assertTrue(result["matches_gmat_r2026a_tag"])
            self.assertFalse(result["byte_exact_match"])
            self.assertTrue(result["line_ending_equivalent"])
            changed = Path(temporary) / "eopc04_08.62-now"
            changed.write_bytes(EOP_PATH.read_bytes() + b"\n")
            result = verify_gmat_eop_install(
                changed, expected_sha256=GMAT_R2026A_EOP_SHA256
            )
            self.assertFalse(result["matches_gmat_r2026a_tag"])

    def test_packager_ignores_newer_interrupted_result_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix_dir = root / "configs"
            reference = root / "data" / "reference" / "gmat_1c2"
            matrix_dir.mkdir(parents=True)
            for directory in (
                reference / "cases",
                reference / "scripts",
                reference / "output",
            ):
                directory.mkdir(parents=True)
            payload = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
            matrix = matrix_dir / "matrix.json"
            matrix.write_text(json.dumps(payload), encoding="utf-8")
            (reference / "GMAT_1C2_MATRIX_MANIFEST.json").write_text(
                "{}", encoding="utf-8"
            )
            (reference / "RUN_ORDER_1C2.txt").write_text("test", encoding="utf-8")
            for case in payload["cases"]:
                case_id = case["case_id"]
                (reference / "cases" / f"{case_id}.json").write_text(
                    "{}", encoding="utf-8"
                )
                for suffix in ("TWO_BODY", "J2"):
                    (reference / "scripts" / f"{case_id}_{suffix}.script").write_text(
                        "test", encoding="utf-8"
                    )
                    (reference / "output" / f"{case_id}_{suffix}.e").write_text(
                        "test", encoding="utf-8"
                    )
            result_root = root / "results" / payload["matrix_id"]
            complete = result_root / "2026-01-01_000000_000000Z"
            interrupted = result_root / "2026-01-02_000000_000000Z"
            complete.mkdir(parents=True)
            interrupted.mkdir(parents=True)
            for name in (
                "gmat_eop_1c2_matrix_summary.json",
                "GMAT_EOP_1C2_INDEPENDENT_REPORT.html",
                "RUN_MANIFEST.json",
                "complete_marker.txt",
            ):
                (complete / name).write_text("complete", encoding="utf-8")
            (interrupted / "interrupted_marker.txt").write_text(
                "partial", encoding="utf-8"
            )
            archive = package_independent_results(
                matrix, project_root=root, output_path=root / "results.zip"
            )
            with zipfile.ZipFile(archive) as bundle:
                names = bundle.namelist()
            self.assertTrue(any(name.endswith("complete_marker.txt") for name in names))
            self.assertFalse(
                any(name.endswith("interrupted_marker.txt") for name in names)
            )

    def test_validation_rejects_missing_new_gmat_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir(parents=True)
            eop_dir = root / "data" / "reference" / "gmat_r2026a"
            eop_dir.mkdir(parents=True)
            shutil.copy2(EOP_PATH, eop_dir / EOP_PATH.name)
            matrix = root / "configs" / "matrix.json"
            matrix.write_text(MATRIX_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                run_independent_validation(matrix, project_root=root)

    def test_invalid_adoption_safeguard_is_rejected(self):
        payload = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
        payload["adoption_rule"]["all_cases_must_pass"] = False
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_independent_matrix_spec(path)

    def test_synthetic_independent_matrix_passes_and_adopts_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir(parents=True)
            eop_dir = root / "data" / "reference" / "gmat_r2026a"
            eop_dir.mkdir(parents=True)
            shutil.copy2(EOP_PATH, eop_dir / EOP_PATH.name)
            shutil.copy2(
                PROJECT_ROOT / "configs" / "case_leo400_gmat_matched.json",
                root / "configs" / "baseline.json",
            )
            payload = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
            payload["baseline_configuration"] = "configs/baseline.json"
            payload["selection_matrix"] = "configs/unused.json"
            payload["selection_diagnostic"] = "configs/unused.json"
            payload["reference_root"] = "data/reference/gmat_1c2"
            payload["cases"] = [dict(item) for item in payload["cases"][:3]]
            for index, case in enumerate(payload["cases"], start=1):
                case["case_id"] = f"T0{index}_SYNTHETIC"
                case["epoch_utc"] = f"2026-02-{10 + index:02d}T00:00:00Z"
                case["duration_hours"] = 120.0 / 3600.0
            matrix = root / "configs" / "matrix.json"
            matrix.write_text(json.dumps(payload), encoding="utf-8")
            prepared = prepare_independent_matrix(matrix, project_root=root)
            self.assertEqual(prepared.case_count, 3)
            dataset = GmatEopDataset.from_file(
                eop_dir / EOP_PATH.name,
                expected_sha256=GMAT_R2026A_EOP_SHA256,
            )
            for case in payload["cases"]:
                case_id = case["case_id"]
                config = json.loads(
                    (
                        root
                        / "data"
                        / "reference"
                        / "gmat_1c2"
                        / "cases"
                        / f"{case_id}.json"
                    ).read_text(encoding="utf-8")
                )
                state = initial_state_from_config(config)
                earth = config["earth_model"]
                integrator = config["integrator"]
                times = np.asarray([0.0, 60.0, 120.0])
                kwargs = {
                    "method": integrator["method"],
                    "relative_tolerance": integrator["relative_tolerance"],
                    "absolute_tolerance": integrator["absolute_tolerance"],
                    "maximum_step_seconds": integrator["maximum_step_seconds"],
                }
                two_body = propagate_numerical_two_body(
                    state,
                    earth["gravitational_parameter_km3_s2"],
                    times,
                    **kwargs,
                )
                provider = partial(
                    gmat_r2026a_eop_pole_unit_vector,
                    dataset=dataset,
                    model="gmat_r2026a_eop_full",
                )
                candidate = propagate_numerical_j2_pole_provider(
                    state,
                    earth["gravitational_parameter_km3_s2"],
                    earth["equatorial_radius_km"],
                    earth["j2"],
                    times,
                    pole_provider=provider,
                    model_name="synthetic_gmat_j2",
                    **kwargs,
                )
                output = root / "data" / "reference" / "gmat_1c2" / "output"
                write_stk(output / f"{case_id}_TWO_BODY.e", two_body)
                write_stk(output / f"{case_id}_J2.e", candidate)
            result = run_independent_validation(matrix, project_root=root)
            self.assertEqual(result.validation_status, "passed_with_warnings")
            self.assertEqual(result.passed_case_count, 3)
            self.assertEqual(
                result.adoption_decision,
                "adopt_gmat_r2026a_eop_full_as_validated_baseline",
            )
            summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["failed_case_count"], 0)
            self.assertTrue(result.report_path.is_file())


if __name__ == "__main__":
    unittest.main()
