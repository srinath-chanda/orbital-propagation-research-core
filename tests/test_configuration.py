from __future__ import annotations

import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.configuration import (
    ConfigValidationError,
    load_and_validate_config,
    validate_config,
)


class ConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config_path = PROJECT_ROOT / "configs" / "case_leo400.json"
        cls.valid_config, cls.warnings = load_and_validate_config(config_path)

    def test_valid_case_leo400_is_accepted(self) -> None:
        self.assertEqual(self.valid_config["experiment"]["case_id"], "CASE-LEO400")
        self.assertEqual(
            self.valid_config["propagation"]["models"],
            [
                "analytical_two_body",
                "numerical_two_body",
                "numerical_j2",
                "numerical_j2_drag",
            ],
        )
        self.assertGreaterEqual(len(self.warnings), 1)

    def test_missing_epoch_is_rejected(self) -> None:
        invalid = deepcopy(self.valid_config)
        del invalid["initial_state"]["epoch_utc"]
        with self.assertRaises(ConfigValidationError):
            validate_config(invalid)

    def test_invalid_model_is_rejected(self) -> None:
        invalid = deepcopy(self.valid_config)
        invalid["propagation"]["models"].append("imaginary_force_model")
        with self.assertRaises(ConfigValidationError):
            validate_config(invalid)

    def test_invalid_drag_sensitivity_multiplier_is_rejected(self) -> None:
        invalid = deepcopy(self.valid_config)
        invalid["drag"]["sensitivity"]["multipliers"]["mass_kg"] = [0.0]
        with self.assertRaises(ConfigValidationError):
            validate_config(invalid)

    def test_invalid_json_reports_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            path.write_text('{"broken": ', encoding="utf-8")
            with self.assertRaises(ConfigValidationError):
                load_and_validate_config(path)


if __name__ == "__main__":
    unittest.main()
