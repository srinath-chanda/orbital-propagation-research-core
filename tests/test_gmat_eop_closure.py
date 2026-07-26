from __future__ import annotations

import json
import sys
import tempfile
import unittest
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
    GMAT_VALIDATED_EOP_CLOSURE,
    GMAT_VALIDATED_EOP_MODEL,
    GmatEopDataset,
    gmat_r2026a_eop_pole_unit_vector,
    gmat_validated_eop_pole_unit_vector,
)
from research_core.gmat_eop_closure import (
    ADOPTION_DECISION,
    load_adoption_record,
    verify_gmat_eop_adoption,
)
from research_core.propagators.numerical_j2 import (
    propagate_numerical_j2_gmat_validated,
    propagate_numerical_j2_pole_provider,
)


ADOPTION_PATH = PROJECT_ROOT / "configs" / "gmat_eop_1c3_adoption.json"
EOP_PATH = PROJECT_ROOT / "data" / "reference" / "gmat_r2026a" / "eopc04_08.62-now"


class GmatEopClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = GmatEopDataset.from_file(
            EOP_PATH, expected_sha256=GMAT_R2026A_EOP_SHA256
        )

    def test_official_1c2_evidence_closes_1c3_adoption(self):
        result = verify_gmat_eop_adoption(ADOPTION_PATH, project_root=PROJECT_ROOT)
        self.assertEqual(result.closure_id, GMAT_VALIDATED_EOP_CLOSURE)
        self.assertEqual(result.validated_model, GMAT_VALIDATED_EOP_MODEL)
        self.assertEqual(result.adoption_decision, ADOPTION_DECISION)
        self.assertEqual(result.case_count, 6)
        self.assertEqual(result.check_count, 84)
        self.assertEqual(result.raw_ephemeris_count, 12)
        self.assertEqual(result.manifest_record_count, 149)
        self.assertTrue(result.official_report.is_file())

    def test_adoption_record_rejects_disabled_safeguard(self):
        payload = json.loads(ADOPTION_PATH.read_text(encoding="utf-8"))
        payload["requirements"]["all_cases_passed"] = False
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_adoption_record(path)

    def test_validated_pole_alias_is_exact_full_eop_model(self):
        epoch = "2026-06-17T06:30:00Z"
        elapsed = 12345.0
        adopted = gmat_validated_eop_pole_unit_vector(
            epoch, elapsed, self.dataset
        )
        explicit = gmat_r2026a_eop_pole_unit_vector(
            epoch, elapsed, self.dataset, model=GMAT_VALIDATED_EOP_MODEL
        )
        np.testing.assert_array_equal(adopted, explicit)

    def test_validated_propagator_is_exact_full_eop_path(self):
        config = json.loads(
            (PROJECT_ROOT / "data" / "reference" / "gmat_1c2" / "cases" /
             "C01_LOW_350KM_I1_FEB_6H.json").read_text(encoding="utf-8")
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
        adopted = propagate_numerical_j2_gmat_validated(
            state,
            earth["gravitational_parameter_km3_s2"],
            earth["equatorial_radius_km"],
            earth["j2"],
            times,
            eop_dataset=self.dataset,
            **kwargs,
        )
        explicit = propagate_numerical_j2_pole_provider(
            state,
            earth["gravitational_parameter_km3_s2"],
            earth["equatorial_radius_km"],
            earth["j2"],
            times,
            pole_provider=partial(
                gmat_r2026a_eop_pole_unit_vector,
                dataset=self.dataset,
                model=GMAT_VALIDATED_EOP_MODEL,
            ),
            model_name="explicit_full_eop",
            **kwargs,
        )
        np.testing.assert_array_equal(adopted.positions_km, explicit.positions_km)
        np.testing.assert_array_equal(adopted.velocities_km_s, explicit.velocities_km_s)


if __name__ == "__main__":
    unittest.main()
