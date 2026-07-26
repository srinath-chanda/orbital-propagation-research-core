from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.external_validation import (
    _canonicalize_nominal_output_grid,
    build_gmat_script,
    initial_state_from_config,
    parse_stk_time_pos_vel,
    prepare_gmat_files,
    run_gmat_external_validation,
    run_gmat_j2_short_arc_validation,
)
from research_core.propagators.numerical_j2 import propagate_numerical_j2_gmat_matched
from research_core.propagators.numerical_two_body import propagate_numerical_two_body


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
    for t, r, v in zip(
        history.elapsed_seconds, history.positions_km, history.velocities_km_s
    ):
        lines.append(
            f"{t:.9f} {r[0]:.15f} {r[1]:.15f} {r[2]:.15f} "
            f"{v[0]:.15f} {v[1]:.15f} {v[2]:.15f}"
        )
    lines.extend(["END Ephemeris", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


class ExternalValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = PROJECT_ROOT / "configs" / "case_leo400_gmat_matched.json"
        cls.config = json.loads(cls.config_path.read_text(encoding="utf-8"))

    def test_initial_state_is_finite(self):
        state = initial_state_from_config(self.config)
        self.assertEqual(state.frame, "EarthMJ2000Eq")
        self.assertTrue(np.all(np.isfinite(state.position_km)))
        self.assertTrue(np.all(np.isfinite(state.velocity_km_s)))

    def test_gmat_scripts_encode_required_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "test.e"
            two_body = build_gmat_script(self.config, model="two_body", output_ephemeris=output)
            j2 = build_gmat_script(self.config, model="j2", output_ephemeris=output)
            self.assertIn("GravityField.Earth.Degree = 0", two_body)
            self.assertIn("GravityField.Earth.Degree = 2", j2)
            self.assertIn("GravityField.Earth.Order = 0", j2)
            self.assertIn("Drag = None", two_body)
            self.assertIn("SRP = Off", j2)
            self.assertIn("CoordinateSystem = EarthMJ2000Eq", j2)
            self.assertIn("FileFormat = STK-TimePosVel", j2)

    def test_prepare_files_creates_scripts_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_copy = root / "config.json"
            config_copy.write_text(json.dumps(self.config), encoding="utf-8")
            prepared = prepare_gmat_files(config_copy, project_root=root)
            self.assertTrue(prepared.two_body_script.is_file())
            self.assertTrue(prepared.j2_script.is_file())
            self.assertTrue(prepared.j2_short_arc_script.is_file())
            self.assertTrue(prepared.acceleration_diagnostic_script.is_file())
            self.assertTrue(prepared.metadata_file.is_file())
            metadata = json.loads(prepared.metadata_file.read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "scripts_prepared_gmat_execution_pending")

    def test_parse_stk_ephemeris(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.e"
            path.write_text(
                "\n".join(
                    [
                        "stk.v.4.3",
                        "BEGIN Ephemeris",
                        "ScenarioEpoch 01 Jan 2026 00:00:00.000",
                        "EphemerisTimePosVel",
                        "0 7000 0 0 0 7.5 1",
                        "60 6985 450 60 -0.5 7.48 1",
                        "END Ephemeris",
                    ]
                ),
                encoding="utf-8",
            )
            history = parse_stk_time_pos_vel(path, model_name="gmat_test")
            self.assertEqual(history.elapsed_seconds.size, 2)
            self.assertEqual(history.epoch_utc, "2026-01-01T00:00:00Z")
            self.assertAlmostEqual(history.positions_km[1, 1], 450.0)

    def test_parse_rejects_missing_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.e"
            path.write_text(
                "BEGIN Ephemeris\nEphemerisTimePosVel\n0 1 2 3 4 5 6\n60 1 2 3 4 5 6\nEND Ephemeris\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                parse_stk_time_pos_vel(path, model_name="bad")

    def test_canonicalize_accepts_sub_microsecond_epoch_noise(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "jittered.e"
            path.write_text(
                "\n".join(
                    [
                        "stk.v.4.3",
                        "BEGIN Ephemeris",
                        "ScenarioEpoch 01 Jan 2026 00:00:00.000",
                        "EphemerisTimePosVel",
                        "0 7000 0 0 0 7.5 1",
                        "60.0000004 6985 450 60 -0.5 7.48 1",
                        "119.9999996 6940 895 120 -0.99 7.43 1",
                        "END Ephemeris",
                    ]
                ),
                encoding="utf-8",
            )
            history = parse_stk_time_pos_vel(path, model_name="gmat_jittered")
            synchronized, diagnostics = _canonicalize_nominal_output_grid(
                history,
                expected_step_seconds=60.0,
                expected_duration_seconds=120.0,
                tolerance_seconds=1.0e-6,
            )
            np.testing.assert_array_equal(
                synchronized.elapsed_seconds,
                np.array([0.0, 60.0, 120.0]),
            )
            self.assertGreater(
                diagnostics["maximum_absolute_raw_time_residual_seconds"],
                0.0,
            )
            np.testing.assert_array_equal(
                synchronized.positions_km,
                history.positions_km,
            )

    def test_canonicalize_rejects_real_time_grid_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad_grid.e"
            path.write_text(
                "\n".join(
                    [
                        "stk.v.4.3",
                        "BEGIN Ephemeris",
                        "ScenarioEpoch 01 Jan 2026 00:00:00.000",
                        "EphemerisTimePosVel",
                        "0 7000 0 0 0 7.5 1",
                        "60.01 6985 450 60 -0.5 7.48 1",
                        "120 6940 895 120 -0.99 7.43 1",
                        "END Ephemeris",
                    ]
                ),
                encoding="utf-8",
            )
            history = parse_stk_time_pos_vel(path, model_name="gmat_bad_grid")
            with self.assertRaises(ValueError):
                _canonicalize_nominal_output_grid(
                    history,
                    expected_step_seconds=60.0,
                    expected_duration_seconds=120.0,
                    tolerance_seconds=1.0e-6,
                )

    def test_end_to_end_with_synthetic_gmat_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = json.loads(json.dumps(self.config))
            config["external_validation"]["duration_seconds"] = 600.0
            config["external_validation"]["output_step_seconds"] = 60.0
            config["external_validation"]["thresholds"] = {
                "initial_position_difference_m": 0.001,
                "initial_velocity_difference_mm_s": 0.001,
                "two_body_maximum_position_difference_m": 0.001,
                "two_body_maximum_velocity_difference_mm_s": 0.001,
                "j2_maximum_position_difference_m": 0.001,
                "j2_maximum_velocity_difference_mm_s": 0.001,
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            times = np.arange(0.0, 600.0 + 60.0, 60.0)
            state = initial_state_from_config(config)
            integ = config["integrator"]
            mu = config["earth_model"]["gravitational_parameter_km3_s2"]
            radius = config["earth_model"]["equatorial_radius_km"]
            j2 = config["earth_model"]["j2"]
            two_body = propagate_numerical_two_body(
                state,
                mu,
                times,
                method=integ["method"],
                relative_tolerance=integ["relative_tolerance"],
                absolute_tolerance=integ["absolute_tolerance"],
                maximum_step_seconds=integ["maximum_step_seconds"],
            )
            j2_history = propagate_numerical_j2_gmat_matched(
                state,
                mu,
                radius,
                j2,
                times,
                method=integ["method"],
                relative_tolerance=integ["relative_tolerance"],
                absolute_tolerance=integ["absolute_tolerance"],
                maximum_step_seconds=integ["maximum_step_seconds"],
            )
            tb_path = root / "tb.e"
            j2_path = root / "j2.e"
            write_stk(tb_path, two_body)
            write_stk(j2_path, j2_history)
            result = run_gmat_external_validation(
                config_path, tb_path, j2_path, project_root=root
            )
            self.assertEqual(result.validation_status, "passed_with_warnings")
            self.assertLess(result.two_body_maximum_position_difference_m, 1e-6)
            self.assertLess(
                result.j2_gmat_matched_maximum_position_difference_m,
                1e-6,
            )
            self.assertTrue(result.report_path.is_file())

    def test_short_arc_with_synthetic_pole_aware_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = json.loads(json.dumps(self.config))
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            short = config["external_validation"]["short_arc"]
            times = np.arange(
                0.0,
                short["duration_seconds"] + short["output_step_seconds"],
                short["output_step_seconds"],
            )
            state = initial_state_from_config(config)
            earth = config["earth_model"]
            integ = config["integrator"]
            history = propagate_numerical_j2_gmat_matched(
                state,
                earth["gravitational_parameter_km3_s2"],
                earth["equatorial_radius_km"],
                earth["j2"],
                times,
                method=integ["method"],
                relative_tolerance=integ["relative_tolerance"],
                absolute_tolerance=integ["absolute_tolerance"],
                maximum_step_seconds=1.0,
            )
            ephemeris = root / "short.e"
            write_stk(ephemeris, history)
            result = run_gmat_j2_short_arc_validation(
                config_path,
                ephemeris,
                project_root=root,
            )
            self.assertEqual(result.validation_status, "passed_with_warnings")
            self.assertLess(
                result.gmat_matched_maximum_position_difference_m,
                1e-6,
            )
            self.assertTrue(result.report_path.is_file())


if __name__ == "__main__":
    unittest.main()
