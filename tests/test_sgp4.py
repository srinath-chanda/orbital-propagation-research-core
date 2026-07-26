from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.frames import frame_roundtrip_error
from research_core.propagators.sgp4_propagator import propagate_sgp4_frozen_tle
from research_core.tle import load_frozen_tle, tle_checksum_is_valid, tle_checksum_value, tle_parameter_summary


class SGP4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tle_path = PROJECT_ROOT / "data" / "tle" / "iss_25544_2026-07-18_celestrak.tle"
        cls.metadata_path = PROJECT_ROOT / "data" / "tle" / "iss_25544_2026-07-18_celestrak_metadata.json"
        cls.tle = load_frozen_tle(cls.tle_path, cls.metadata_path, expected_catalog_number=25544)

    def test_tle_line_checksums_are_valid(self) -> None:
        self.assertTrue(tle_checksum_is_valid(self.tle.line1))
        self.assertTrue(tle_checksum_is_valid(self.tle.line2))
        self.assertEqual(tle_checksum_value(self.tle.line1), int(self.tle.line1[68]))

    def test_tle_epoch_and_catalog(self) -> None:
        summary = tle_parameter_summary(self.tle)
        self.assertEqual(summary["norad_catalog_number"], 25544)
        self.assertEqual(summary["tle_epoch_utc"], "2026-07-18T02:08:01.938048Z")

    def test_sgp4_short_history_is_finite(self) -> None:
        elapsed = np.array([0.0, 60.0, 120.0])
        teme, gcrs, diagnostics = propagate_sgp4_frozen_tle(self.tle, elapsed)
        self.assertEqual(diagnostics["nonzero_error_count"], 0)
        self.assertTrue(np.all(np.isfinite(teme.positions_km)))
        self.assertTrue(np.all(np.isfinite(gcrs.positions_km)))
        self.assertEqual(gcrs.frame, "GCRS_ASTROPY_FROM_TEME")

    def test_teme_gcrs_roundtrip_is_small(self) -> None:
        elapsed = np.array([0.0, 60.0])
        teme, _, _ = propagate_sgp4_frozen_tle(self.tle, elapsed)
        result = frame_roundtrip_error(teme.positions_km[0], teme.velocities_km_s[0], self.tle.epoch_utc)
        self.assertLessEqual(result["position_roundtrip_error_m"], 1e-6)
        self.assertLessEqual(result["velocity_roundtrip_error_mm_s"], 1e-5)

    def test_tampered_line_checksum_is_rejected(self) -> None:
        line = self.tle.line1[:20] + ("1" if self.tle.line1[20] != "1" else "2") + self.tle.line1[21:]
        self.assertFalse(tle_checksum_is_valid(line))


if __name__ == "__main__":
    unittest.main()
