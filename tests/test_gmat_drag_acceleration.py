from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.time import Time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.external_validation import initial_state_from_config
from research_core.gmat_drag_acceleration import (
    EXPECTED_SCENARIOS,
    ExponentialAtmosphereTable,
    build_drag_acceleration_master_script,
    gmat_earth_angular_velocity_inertial_rad_s,
    gmat_exponential_drag_acceleration_km_s2,
    gmat_geodetic_height_km,
    load_drag_acceleration_config,
    parse_drag_acceleration_report,
)
from research_core.gmat_eop import GMAT_R2026A_EOP_SHA256, GmatEopDataset
from research_core.gmat_gravity_multicase_closure import (
    verify_gravity_multicase_closure,
)


CONFIG_PATH = PROJECT_ROOT / "configs/gmat_drag_1e0_acceleration.json"
BASELINE_PATH = PROJECT_ROOT / "configs/case_leo400_gmat_matched.json"
ATMOSPHERE_PATH = PROJECT_ROOT / "data/reference/gmat_r2026a/EarthExponentialAtmosphereData.txt"
EOP_PATH = PROJECT_ROOT / "data/reference/gmat_r2026a/eopc04_08.62-now"
CLOSURE_PATH = PROJECT_ROOT / "configs/gmat_gravity_1d2_closure.json"


class GmatDragAccelerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_drag_acceleration_config(CONFIG_PATH)
        cls.baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        cls.atmosphere = ExponentialAtmosphereTable.from_file(
            ATMOSPHERE_PATH,
            expected_sha256=cls.config["atmosphere_file_sha256"],
        )
        cls.eop = GmatEopDataset.from_file(
            EOP_PATH, expected_sha256=GMAT_R2026A_EOP_SHA256
        )

    def test_closed_1d2_evidence_authorizes_drag(self):
        closure = verify_gravity_multicase_closure(
            CLOSURE_PATH, project_root=PROJECT_ROOT
        )
        self.assertEqual(closure.case_count, 6)
        self.assertEqual(closure.model_run_count, 24)
        self.assertEqual(closure.check_count, 96)
        self.assertLess(closure.maximum_position_difference_m, 0.01)
        self.assertLess(closure.maximum_velocity_difference_mm_s, 0.02)
        self.assertLess(closure.maximum_time_residual_seconds, 5e-6)

    def test_configuration_has_exact_ballistic_scaling_matrix(self):
        observed = tuple(
            (
                item["scenario_id"],
                item["alias"],
                item["mass_kg"],
                item["drag_area_m2"],
                item["drag_coefficient"],
                item["expected_acceleration_scale"],
            )
            for item in self.config["scenarios"]
        )
        self.assertEqual(observed, EXPECTED_SCENARIOS)
        self.assertEqual(self.config["sample_count"], 25)
        self.assertEqual(self.config["time_grid_tolerance_seconds"], 5e-6)
        self.assertEqual(self.config["gravity_degree"], 20)
        self.assertEqual(self.config["gravity_order"], 20)

    def test_frozen_table_matches_key_r2026a_rows(self):
        self.assertEqual(self.atmosphere.reference_height_km.size, 28)
        self.assertAlmostEqual(self.atmosphere.density_kg_m3(350.0), 9.518e-12, places=23)
        self.assertAlmostEqual(self.atmosphere.density_kg_m3(400.0), 3.725e-12, places=23)
        expected_425 = 3.725e-12 * np.exp(-25.0 / 58.515)
        self.assertAlmostEqual(self.atmosphere.density_kg_m3(425.0), expected_425, places=23)

    def test_gmat_geodetic_height_uses_registered_ellipsoid(self):
        height = gmat_geodetic_height_km(
            np.array([6378.1363 + 400.0, 0.0, 0.0]),
            equatorial_radius_km=6378.1363,
            flattening=0.00335270,
        )
        self.assertAlmostEqual(height, 400.0, places=11)
        polar_radius = 6378.1363 * (1.0 - 0.00335270)
        polar_height = gmat_geodetic_height_km(
            np.array([0.0, 0.0, polar_radius + 400.0]),
            equatorial_radius_km=6378.1363,
            flattening=0.00335270,
        )
        self.assertAlmostEqual(polar_height, 400.0, places=10)

    def test_rotation_derivative_has_physical_earth_rate(self):
        omega = gmat_earth_angular_velocity_inertial_rad_s(
            Time("2026-01-01T00:00:00", scale="utc"), self.eop
        )
        self.assertGreater(np.linalg.norm(omega), 7.28e-5)
        self.assertLess(np.linalg.norm(omega), 7.31e-5)

    def test_matched_acceleration_is_antiparallel_and_scales(self):
        state = initial_state_from_config(self.baseline)
        epoch = Time(state.epoch_utc, scale="utc")
        nominal, density, height, relative = gmat_exponential_drag_acceleration_km_s2(
            state.position_km,
            state.velocity_km_s,
            epoch,
            eop=self.eop,
            atmosphere=self.atmosphere,
            equatorial_radius_km=6378.1363,
            flattening=0.00335270,
            mass_kg=500.0,
            drag_area_m2=4.0,
            drag_coefficient=2.2,
        )
        doubled, *_ = gmat_exponential_drag_acceleration_km_s2(
            state.position_km,
            state.velocity_km_s,
            epoch,
            eop=self.eop,
            atmosphere=self.atmosphere,
            equatorial_radius_km=6378.1363,
            flattening=0.00335270,
            mass_kg=500.0,
            drag_area_m2=8.0,
            drag_coefficient=2.2,
        )
        self.assertGreater(density, 0.0)
        self.assertGreater(height, 300.0)
        self.assertLess(float(np.dot(nominal, relative)), 0.0)
        np.testing.assert_allclose(doubled, 2.0 * nominal, rtol=0.0, atol=0.0)

    def test_generated_master_uses_official_exponential_syntax(self):
        script = build_drag_acceleration_master_script(
            self.config,
            self.baseline,
            gravity_file=Path("C:/project/JGM2.cof"),
            atmosphere_file=Path("C:/project/EarthExponentialAtmosphereData.txt"),
            output_directory=Path("C:/project/output"),
        )
        self.assertEqual(script.count("Drag.AtmosphereModel = Exponential;"), 4)
        self.assertEqual(script.count("Drag.DragModel = 'Spherical';"), 4)
        self.assertEqual(script.count("Drag.InputFile ="), 4)
        self.assertEqual(script.count("Report DGNReport"), 25)
        self.assertIn("DGNProp.FM = DGNGravityFM;", script)
        self.assertIn("DGNDragFM.GravityField.Earth.Degree = 20;", script)

    def test_c_delimited_report_with_repeated_header_is_parsed(self):
        scenario = self.config["scenarios"][0]
        alias = scenario["alias"]
        headers = [
            f"{alias}Sat.ElapsedSecs",
            f"{alias}Sat.EarthMJ2000Eq.X",
            f"{alias}Sat.EarthMJ2000Eq.Y",
            f"{alias}Sat.EarthMJ2000Eq.Z",
            f"{alias}Sat.EarthMJ2000Eq.VX",
            f"{alias}Sat.EarthMJ2000Eq.VY",
            f"{alias}Sat.EarthMJ2000Eq.VZ",
            f"{alias}Sat.{alias}GravityFM.AccelerationX",
            f"{alias}Sat.{alias}GravityFM.AccelerationY",
            f"{alias}Sat.{alias}GravityFM.AccelerationZ",
            f"{alias}Sat.{alias}DragFM.AccelerationX",
            f"{alias}Sat.{alias}DragFM.AccelerationY",
            f"{alias}Sat.{alias}DragFM.AccelerationZ",
        ]
        lines = ["   ".join(headers)]
        for index in range(25):
            elapsed = index * 50.0
            values = [elapsed, 6778.0, 10.0, 20.0, 0.0, 7.5, 1.0]
            values.extend([-0.008, 0.0, 0.0])
            values.extend([-0.008000001, -1e-9, 0.0])
            lines.append("C".join(str(value) for value in values))
            if index == 0:
                lines.append("   ".join(headers))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.csv"
            path.write_text("\n".join(lines), encoding="utf-8")
            elapsed, states, gravity, total, grid = parse_drag_acceleration_report(
                path, scenario, self.config
            )
        self.assertEqual(elapsed.shape, (25,))
        self.assertEqual(states.shape, (25, 6))
        self.assertEqual(gravity.shape, (25, 3))
        self.assertEqual(total.shape, (25, 3))
        self.assertEqual(grid["maximum_absolute_raw_time_residual_seconds"], 0.0)

    def test_real_gmat_microsecond_grid_residual_is_canonicalized(self):
        scenario = self.config["scenarios"][0]
        alias = scenario["alias"]
        headers = [
            f"{alias}Sat.ElapsedSecs",
            f"{alias}Sat.EarthMJ2000Eq.X",
            f"{alias}Sat.EarthMJ2000Eq.Y",
            f"{alias}Sat.EarthMJ2000Eq.Z",
            f"{alias}Sat.EarthMJ2000Eq.VX",
            f"{alias}Sat.EarthMJ2000Eq.VY",
            f"{alias}Sat.EarthMJ2000Eq.VZ",
            f"{alias}Sat.{alias}GravityFM.AccelerationX",
            f"{alias}Sat.{alias}GravityFM.AccelerationY",
            f"{alias}Sat.{alias}GravityFM.AccelerationZ",
            f"{alias}Sat.{alias}DragFM.AccelerationX",
            f"{alias}Sat.{alias}DragFM.AccelerationY",
            f"{alias}Sat.{alias}DragFM.AccelerationZ",
        ]
        lines = ["   ".join(headers)]
        for index in range(25):
            elapsed = index * 50.0 + index * 5.820766e-8
            values = [elapsed, 6778.0, 10.0, 20.0, 0.0, 7.5, 1.0]
            values.extend([-0.008, 0.0, 0.0])
            values.extend([-0.008000001, -1e-9, 0.0])
            lines.append("C".join(str(value) for value in values))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.csv"
            path.write_text("\n".join(lines), encoding="utf-8")
            elapsed, _, _, _, grid = parse_drag_acceleration_report(
                path, scenario, self.config
            )
        np.testing.assert_allclose(elapsed, np.arange(25) * 50.0, rtol=0.0, atol=0.0)
        self.assertGreater(grid["maximum_absolute_raw_time_residual_seconds"], 1e-6)
        self.assertLess(grid["maximum_absolute_raw_time_residual_seconds"], 5e-6)

    def test_time_grid_residual_above_closed_limit_is_rejected(self):
        scenario = self.config["scenarios"][0]
        alias = scenario["alias"]
        headers = [
            f"{alias}Sat.ElapsedSecs",
            f"{alias}Sat.EarthMJ2000Eq.X",
            f"{alias}Sat.EarthMJ2000Eq.Y",
            f"{alias}Sat.EarthMJ2000Eq.Z",
            f"{alias}Sat.EarthMJ2000Eq.VX",
            f"{alias}Sat.EarthMJ2000Eq.VY",
            f"{alias}Sat.EarthMJ2000Eq.VZ",
            f"{alias}Sat.{alias}GravityFM.AccelerationX",
            f"{alias}Sat.{alias}GravityFM.AccelerationY",
            f"{alias}Sat.{alias}GravityFM.AccelerationZ",
            f"{alias}Sat.{alias}DragFM.AccelerationX",
            f"{alias}Sat.{alias}DragFM.AccelerationY",
            f"{alias}Sat.{alias}DragFM.AccelerationZ",
        ]
        lines = ["   ".join(headers)]
        for index in range(25):
            elapsed = index * 50.0 + index * 5.0e-7
            values = [elapsed, 6778.0, 10.0, 20.0, 0.0, 7.5, 1.0]
            values.extend([-0.008, 0.0, 0.0])
            values.extend([-0.008000001, -1e-9, 0.0])
            lines.append("C".join(str(value) for value in values))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.csv"
            path.write_text("\n".join(lines), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exceeding the"):
                parse_drag_acceleration_report(path, scenario, self.config)


if __name__ == "__main__":
    unittest.main()
