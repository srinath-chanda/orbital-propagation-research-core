from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.data_models import StateHistory
from research_core.external_validation import initial_state_from_config
from research_core.gmat_gravity_multicase import (
    EXPECTED_CASES,
    build_gravity_multicase_master_script,
    load_gravity_multicase_config,
    package_gravity_multicase_results,
    prepare_gravity_multicase,
    run_gravity_multicase_validation,
)
from research_core.gmat_gravity_short_arc import EXPECTED_MODELS
from research_core.gmat_gravity_short_arc_closure import verify_gravity_short_arc_closure
from research_core.time_utils import timestamps_from_epoch


CONFIG_PATH = PROJECT_ROOT / "configs/gmat_gravity_1d2_multicase.json"
CLOSURE_PATH = PROJECT_ROOT / "configs/gmat_gravity_1d1_closure.json"


def _write_stk(path: Path, history: StateHistory) -> None:
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
        lines.append(
            " ".join(
                f"{float(value):.16e}" for value in [elapsed, *position, *velocity]
            )
        )
    lines.extend(["END Ephemeris", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _linear_history(initial, times: np.ndarray, model_name: str = "synthetic") -> StateHistory:
    times = np.asarray(times, dtype=float)
    positions = initial.position_km[None, :] + times[:, None] * initial.velocity_km_s[None, :]
    velocities = np.repeat(initial.velocity_km_s[None, :], times.size, axis=0)
    return StateHistory(
        model_name=model_name,
        frame=initial.frame,
        epoch_utc=initial.epoch_utc,
        elapsed_seconds=times,
        timestamps_utc=timestamps_from_epoch(initial.epoch_utc, times),
        positions_km=positions,
        velocities_km_s=velocities,
        runtime_seconds=0.001,
        solver_status="synthetic",
        function_evaluations=1,
    )


def _copy_prerequisite(root: Path) -> None:
    paths = [
        "configs/case_leo400_gmat_matched.json",
        "configs/gmat_gravity_1d1_short_arc.json",
        "configs/gmat_gravity_1d1_closure.json",
        "data/reference/gmat_r2026a/JGM2.cof",
        "data/reference/gmat_r2026a/JGM2_PROVENANCE_1D0.json",
        "data/reference/gmat_r2026a/eopc04_08.62-now",
    ]
    paths.extend(
        f"data/reference/gmat_1d1/output/{model_id}_SHORT_ARC.e"
        for model_id, _degree, _order in EXPECTED_MODELS
    )
    result_relative = (
        "results/EXP-GMAT-GRAVITY-1D1-SHORT-ARC-001/"
        "2026-07-19_204643_500945Z"
    )
    paths.extend(
        str(path.relative_to(PROJECT_ROOT))
        for path in (PROJECT_ROOT / result_relative).rglob("*")
        if path.is_file()
    )
    for relative in paths:
        source = PROJECT_ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


class GmatGravityMulticaseTests(unittest.TestCase):
    def test_release_configuration_is_frozen_six_by_four_matrix(self):
        config = load_gravity_multicase_config(CONFIG_PATH)
        self.assertEqual(
            tuple((case["case_id"], case["duration_hours"]) for case in config["cases"]),
            EXPECTED_CASES,
        )
        self.assertEqual(
            tuple(
                (model["model_id"], model["degree"], model["order"])
                for model in config["models"]
            ),
            EXPECTED_MODELS,
        )
        self.assertEqual(len(config["cases"]) * len(config["models"]) * 4, 96)

    def test_official_1d1_evidence_authorizes_multicase(self):
        closure = verify_gravity_short_arc_closure(CLOSURE_PATH, project_root=PROJECT_ROOT)
        self.assertEqual(closure.model_count, 4)
        self.assertEqual(closure.check_count, 16)
        self.assertEqual(closure.sample_count_per_model, 181)
        self.assertLess(closure.maximum_position_difference_m, 0.05)
        self.assertLess(closure.maximum_velocity_difference_mm_s, 0.05)

    def test_closure_rejects_wrong_summary_hash(self):
        record = json.loads(CLOSURE_PATH.read_text())
        record["official_summary_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_gravity_short_arc_closure(path, project_root=PROJECT_ROOT)

    def test_master_script_contains_twenty_four_propagations(self):
        config = load_gravity_multicase_config(CONFIG_PATH)
        baseline = json.loads(
            (PROJECT_ROOT / "configs/case_leo400_gmat_matched.json").read_text()
        )
        script = build_gravity_multicase_master_script(
            config,
            baseline,
            gravity_file=Path("C:/project/JGM2.cof"),
            output_directory=Path("C:/project/output"),
        )
        self.assertEqual(script.count("Create EphemerisFile"), 24)
        self.assertEqual(script.count("Propagate "), 24)
        for case_id, _duration in EXPECTED_CASES:
            for model_id, degree, order in EXPECTED_MODELS:
                self.assertIn(f"{case_id}_{model_id}.e", script)
                self.assertIn(f"GravityField.Earth.Degree = {degree};", script)
                self.assertIn(f"GravityField.Earth.Order = {order};", script)

    def test_synthetic_matrix_validates_and_packages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _copy_prerequisite(root)
            config = json.loads(CONFIG_PATH.read_text())
            config["output_step_seconds"] = 21600.0
            config["integrator"]["python_maximum_step_seconds"] = 21600.0
            config["integrator"]["gmat_maximum_step_seconds"] = 21600.0
            config_path = root / "configs/gmat_gravity_1d2_multicase.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(config), encoding="utf-8")

            prepared = prepare_gravity_multicase(config_path, project_root=root)
            self.assertEqual(prepared.case_count, 6)
            self.assertEqual(prepared.expected_output_count, 24)
            self.assertEqual(prepared.expected_check_count, 96)

            baseline = json.loads(
                (root / "configs/case_leo400_gmat_matched.json").read_text()
            )
            radius = float(baseline["earth_model"]["equatorial_radius_km"])
            histories: dict[tuple[str, str], StateHistory] = {}
            for case in config["cases"]:
                case_config = json.loads(
                    (root / f"data/reference/gmat_1d2/cases/{case['case_id']}.json").read_text()
                )
                initial = initial_state_from_config(case_config)
                duration = float(case["duration_hours"]) * 3600.0
                times = np.arange(int(round(duration / 21600.0)) + 1) * 21600.0
                for model in config["models"]:
                    history = _linear_history(initial, times, str(model["model_id"]))
                    histories[(str(case["case_id"]), str(model["model_id"]))] = history
                    _write_stk(
                        root
                        / "data/reference/gmat_1d2/output"
                        / f"{case['case_id']}_{model['model_id']}.e",
                        history,
                    )
            self.assertGreater(radius, 6000.0)

            call_index = 0

            def fake_propagator(initial_state, _field, _eop, times, **_kwargs):
                nonlocal call_index
                case_index, model_index = divmod(call_index, 4)
                call_index += 1
                case_id = str(config["cases"][case_index]["case_id"])
                model_id = str(config["models"][model_index]["model_id"])
                expected = histories[(case_id, model_id)]
                self.assertTrue(np.array_equal(np.asarray(times), expected.elapsed_seconds))
                return _linear_history(initial_state, np.asarray(times), model_id)

            with patch(
                "research_core.gmat_gravity_multicase.propagate_spherical_harmonic_gravity",
                side_effect=fake_propagator,
            ):
                result = run_gravity_multicase_validation(config_path, project_root=root)
            self.assertEqual(result.status, "passed_with_warnings")
            self.assertEqual(result.passed_case_count, 6)
            self.assertEqual(result.passed_model_run_count, 24)
            self.assertEqual(result.passed_check_count, 96)

            archive = package_gravity_multicase_results(
                config_path,
                project_root=root,
                output_path=root / "result.zip",
            )
            with zipfile.ZipFile(archive) as stream:
                self.assertIsNone(stream.testzip())
                self.assertIn(
                    "data/reference/gmat_1d2/output/"
                    "D06_LONG_550KM_I45_SEP_72H_G2020.e",
                    stream.namelist(),
                )


if __name__ == "__main__":
    unittest.main()
