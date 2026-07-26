from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.external_validation import initial_state_from_config
from research_core.gmat_eop import GMAT_R2026A_EOP_SHA256, GmatEopDataset
from research_core.gmat_gravity_closure import verify_gravity_ladder_closure
from research_core.gmat_gravity_short_arc import (
    EXPECTED_MODELS,
    build_gravity_short_arc_master_script,
    load_gravity_short_arc_config,
    package_gravity_short_arc_results,
    run_gravity_short_arc_validation,
)
from research_core.gravity_harmonics import CofGravityField
from research_core.propagators.numerical_gravity import propagate_spherical_harmonic_gravity


CONFIG_PATH = PROJECT_ROOT / "configs/gmat_gravity_1d1_short_arc.json"
CLOSURE_PATH = PROJECT_ROOT / "configs/gmat_gravity_1d0_closure.json"


def _write_stk(path: Path, history) -> None:
    epoch = datetime.fromisoformat(history.epoch_utc.replace("Z", "+00:00"))
    scenario = epoch.astimezone(timezone.utc).strftime("%d %b %Y %H:%M:%S.%f")
    lines = [
        "stk.v.10.0",
        "BEGIN Ephemeris",
        f"ScenarioEpoch {scenario}",
        f"NumberOfEphemerisPoints {history.elapsed_seconds.size}",
        "DistanceUnit Kilometers",
        "EphemerisTimePosVel",
    ]
    for elapsed, position, velocity in zip(
        history.elapsed_seconds, history.positions_km, history.velocities_km_s
    ):
        values = [elapsed, *position, *velocity]
        lines.append(" ".join(f"{float(value):.16e}" for value in values))
    lines.extend(["END Ephemeris", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


class GmatGravityShortArcTests(unittest.TestCase):
    def test_release_configuration_preregisters_four_models(self):
        config = load_gravity_short_arc_config(CONFIG_PATH)
        models = tuple(
            (item["model_id"], item["degree"], item["order"])
            for item in config["models"]
        )
        self.assertEqual(models, EXPECTED_MODELS)
        self.assertEqual(config["duration_seconds"], 1800.0)
        self.assertEqual(config["output_step_seconds"], 10.0)
        self.assertTrue(config["decision_rule"]["all_four_models_must_pass_all_checks"])

    def test_official_1d0_evidence_authorizes_short_arc(self):
        closure = verify_gravity_ladder_closure(CLOSURE_PATH, project_root=PROJECT_ROOT)
        self.assertEqual(closure.model_count, 6)
        self.assertEqual(closure.sample_count, 25)
        self.assertLess(closure.largest_model_difference_km_s2, 2e-9)
        self.assertGreater(closure.smallest_adjacent_physical_increment_km_s2, 1e-8)

    def test_closure_rejects_wrong_official_hash(self):
        record = json.loads(CLOSURE_PATH.read_text())
        record["official_summary_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_gravity_ladder_closure(path, project_root=PROJECT_ROOT)

    def test_master_script_contains_four_independent_propagations(self):
        config = load_gravity_short_arc_config(CONFIG_PATH)
        baseline = json.loads(
            (PROJECT_ROOT / "configs/case_leo400_gmat_matched.json").read_text()
        )
        script = build_gravity_short_arc_master_script(
            config,
            baseline,
            gravity_file=Path("C:/project/JGM2.cof"),
            output_directory=Path("C:/project/output"),
        )
        for model_id, degree, order in EXPECTED_MODELS:
            self.assertIn(f"Create ForceModel {model_id}FM;", script)
            self.assertIn(f"{model_id}FM.GravityField.Earth.Degree = {degree};", script)
            self.assertIn(f"{model_id}FM.GravityField.Earth.Order = {order};", script)
            self.assertIn(f"Propagate {model_id}Prop({model_id}Sat)", script)
        self.assertEqual(script.count("Create EphemerisFile"), 4)

    def test_synthetic_short_arcs_validate_and_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            required = [
                "configs/case_leo400_gmat_matched.json",
                "configs/gmat_gravity_1d0_ladder.json",
                "configs/gmat_gravity_1d0_closure.json",
                "data/reference/gmat_r2026a/JGM2.cof",
                "data/reference/gmat_r2026a/JGM2_PROVENANCE_1D0.json",
                "data/reference/gmat_r2026a/eopc04_08.62-now",
                "data/reference/gmat_1d0/output/GMAT_GRAVITY_LADDER_1D0.csv",
                "results/EXP-GMAT-GRAVITY-1D0-001/2026-07-19_202029_369249Z/gravity_ladder_summary.json",
                "results/EXP-GMAT-GRAVITY-1D0-001/2026-07-19_202029_369249Z/gravity_ladder_differences.csv",
                "results/EXP-GMAT-GRAVITY-1D0-001/2026-07-19_202029_369249Z/GMAT_GRAVITY_LADDER_1D0_REPORT.html",
                "results/EXP-GMAT-GRAVITY-1D0-001/2026-07-19_202029_369249Z/RUN_MANIFEST.json",
            ]
            for relative in required:
                source = PROJECT_ROOT / relative
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            config = json.loads(CONFIG_PATH.read_text())
            config["duration_seconds"] = 60.0
            config["output_step_seconds"] = 30.0
            config["integrator"]["python_maximum_step_seconds"] = 10.0
            config_path = root / "configs/gmat_gravity_1d1_short_arc.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            baseline = json.loads(
                (root / "configs/case_leo400_gmat_matched.json").read_text()
            )
            initial = initial_state_from_config(baseline)
            field = CofGravityField.from_file(
                root / "data/reference/gmat_r2026a/JGM2.cof"
            )
            eop = GmatEopDataset.from_file(
                root / "data/reference/gmat_r2026a/eopc04_08.62-now",
                expected_sha256=GMAT_R2026A_EOP_SHA256,
            )
            times = np.asarray([0.0, 30.0, 60.0])
            for model in config["models"]:
                history = propagate_spherical_harmonic_gravity(
                    initial,
                    field,
                    eop,
                    times,
                    degree=model["degree"],
                    order=model["order"],
                    method="DOP853",
                    relative_tolerance=1e-11,
                    absolute_tolerance=1e-13,
                    maximum_step_seconds=10.0,
                )
                _write_stk(
                    root / f"data/reference/gmat_1d1/output/{model['model_id']}_SHORT_ARC.e",
                    history,
                )
            result = run_gravity_short_arc_validation(config_path, project_root=root)
            self.assertEqual(result.status, "passed_with_warnings")
            self.assertEqual(result.passed_model_count, 4)
            self.assertEqual(result.check_count, 16)
            archive = package_gravity_short_arc_results(
                config_path,
                project_root=root,
                output_path=root / "result.zip",
            )
            with zipfile.ZipFile(archive) as stream:
                self.assertIsNone(stream.testzip())
                self.assertIn(
                    "data/reference/gmat_1d1/output/G2020_SHORT_ARC.e",
                    stream.namelist(),
                )


if __name__ == "__main__":
    unittest.main()
