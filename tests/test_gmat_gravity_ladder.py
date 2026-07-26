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

from research_core.gmat_eop import (
    GMAT_R2026A_EOP_SHA256,
    GmatEopDataset,
    gmat_r2026a_eop_pole_unit_vector,
    gmat_r2026a_inertial_to_fixed_matrix,
)
from research_core.gmat_gravity_ladder import (
    EXPECTED_LADDER,
    build_gravity_ladder_script,
    import_gmat_jgm2,
    load_gravity_ladder_config,
    parse_gravity_ladder_report,
)


CONFIG_PATH = PROJECT_ROOT / "configs/gmat_gravity_1d0_ladder.json"
EOP_PATH = PROJECT_ROOT / "data/reference/gmat_r2026a/eopc04_08.62-now"


def synthetic_jgm2() -> str:
    return (
        "POTFIELD 70 70 1 3.98600441500000e+14 6.37813630000000e+06 1.0\n"
        "RECOEF 2 0 -4.84165390000000e-04\n"
        "RECOEF 2 2 2.43908370000000e-06-1.40010930000000e-06\n"
    )


class GmatGravityLadderTests(unittest.TestCase):
    def test_preregistered_configuration_has_exact_ladder(self):
        config = load_gravity_ladder_config(CONFIG_PATH)
        ladder = tuple((item["degree"], item["order"]) for item in config["ladder"])
        self.assertEqual(ladder, EXPECTED_LADDER)
        self.assertEqual(config["sample_count"], 25)
        self.assertTrue(config["decision_rule"]["all_six_models_must_pass"])

    def test_full_rotation_exposes_same_pole_as_validated_axis(self):
        dataset = GmatEopDataset.from_file(EOP_PATH, expected_sha256=GMAT_R2026A_EOP_SHA256)
        evaluation = Time("2026-01-01T00:00:00", scale="utc")
        sample = dataset.sample(evaluation)
        rotation = gmat_r2026a_inertial_to_fixed_matrix(evaluation, sample)
        axis = gmat_r2026a_eop_pole_unit_vector(
            "2026-01-01T00:00:00Z", 0.0, dataset
        )
        np.testing.assert_allclose(rotation @ rotation.T, np.identity(3), atol=1.0e-15)
        np.testing.assert_allclose(rotation[2], axis, rtol=0.0, atol=2.0e-16)

    def test_import_is_read_only_copy_with_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "GMAT/data/gravity/earth/JGM2.cof"
            source.parent.mkdir(parents=True)
            source.write_text(synthetic_jgm2(), encoding="ascii")
            original = source.read_bytes()
            imported = import_gmat_jgm2(root / "GMAT", project_root=root / "project")
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(imported.destination.read_bytes(), original)
            provenance = json.loads(imported.provenance.read_text())
            self.assertEqual(provenance["maximum_degree"], 70)
            self.assertEqual(provenance["maximum_order"], 70)

    def test_generated_script_contains_all_six_force_models(self):
        config = load_gravity_ladder_config(CONFIG_PATH)
        baseline = json.loads(
            (PROJECT_ROOT / "configs/case_leo400_gmat_matched.json").read_text()
        )
        script = build_gravity_ladder_script(
            config,
            baseline,
            gravity_file=Path("C:/project/JGM2.cof"),
            output_report=Path("C:/project/output.csv"),
        )
        for item in config["ladder"]:
            self.assertIn(f"Create ForceModel {item['alias']}FM;", script)
            self.assertIn(
                f"{item['alias']}FM.GravityField.Earth.Degree = {item['degree']};",
                script,
            )
            self.assertIn(
                f"{item['alias']}FM.GravityField.Earth.Order = {item['order']};",
                script,
            )
        self.assertEqual(script.count("Report GravityLadderReport"), 25)

    def test_raw_c_delimited_report_with_repeated_headers_is_parsed(self):
        config = load_gravity_ladder_config(CONFIG_PATH)
        headers = [
            "LadderSat.ElapsedSecs",
            "LadderSat.EarthMJ2000Eq.X",
            "LadderSat.EarthMJ2000Eq.Y",
            "LadderSat.EarthMJ2000Eq.Z",
        ]
        for item in config["ladder"]:
            alias = item["alias"]
            headers.extend(
                [
                    f"LadderSat.{alias}FM.AccelerationX",
                    f"LadderSat.{alias}FM.AccelerationY",
                    f"LadderSat.{alias}FM.AccelerationZ",
                ]
            )
        lines = ["   ".join(headers)]
        for index in range(25):
            values = [index * 75.0, 7000.0, 10.0, 20.0]
            values.extend([-0.008, 0.0, 0.0] * 6)
            lines.append("C".join(str(value) for value in values))
            if index == 0:
                lines.append("   ".join(headers))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.csv"
            path.write_text("\n".join(lines), encoding="utf-8")
            elapsed, positions, accelerations = parse_gravity_ladder_report(path, config)
        self.assertEqual(elapsed.shape, (25,))
        self.assertEqual(positions.shape, (25, 3))
        self.assertEqual(set(accelerations), {item["alias"] for item in config["ladder"]})


if __name__ == "__main__":
    unittest.main()
