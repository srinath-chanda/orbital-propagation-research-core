from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.ground_station import (
    GroundStation,
    VisibilityHistory,
    detect_passes,
    match_passes,
    station_topocentric_from_itrs,
)
from research_core.time_utils import timestamps_from_epoch


class GroundStationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.station = GroundStation(
            station_id="GS-TEST-001",
            name="Equator",
            latitude_deg=0.0,
            longitude_deg=0.0,
            altitude_m=0.0,
            minimum_elevation_deg=10.0,
        )

    def test_station_roundtrip_is_small(self) -> None:
        self.assertLessEqual(self.station.station_roundtrip_error_m(), 1.0e-6)

    def test_overhead_geometry(self) -> None:
        station_position = self.station.itrs_position_km()
        satellite_position = station_position + np.array([700.0, 0.0, 0.0])
        azimuth, elevation, range_km, range_rate = station_topocentric_from_itrs(
            satellite_position,
            np.zeros(3),
            self.station,
        )
        self.assertAlmostEqual(elevation[0], 90.0, places=10)
        self.assertAlmostEqual(range_km[0], 700.0, places=10)
        self.assertAlmostEqual(range_rate[0], 0.0, places=12)
        self.assertTrue(0.0 <= azimuth[0] < 360.0)

    def test_eastern_horizon_geometry(self) -> None:
        station_position = self.station.itrs_position_km()
        satellite_position = station_position + np.array([0.0, 1000.0, 0.0])
        azimuth, elevation, range_km, _ = station_topocentric_from_itrs(
            satellite_position,
            np.zeros(3),
            self.station,
        )
        self.assertAlmostEqual(azimuth[0], 90.0, places=10)
        self.assertAlmostEqual(elevation[0], 0.0, places=10)
        self.assertAlmostEqual(range_km[0], 1000.0, places=10)

    def test_synthetic_pass_is_refined(self) -> None:
        elapsed = np.arange(0.0, 101.0, 10.0)
        elevation = -10.0 + 40.0 * np.sin(np.pi * elapsed / 100.0)
        visibility = VisibilityHistory(
            model_name="synthetic",
            station=self.station,
            epoch_utc="2026-01-01T00:00:00Z",
            elapsed_seconds=elapsed,
            timestamps_utc=timestamps_from_epoch(
                "2026-01-01T00:00:00Z", elapsed
            ),
            azimuth_deg=np.linspace(30.0, 210.0, elapsed.size),
            elevation_deg=elevation,
            range_km=1200.0 - 500.0 * np.sin(np.pi * elapsed / 100.0),
            range_rate_km_s=np.zeros(elapsed.size),
        )
        passes = detect_passes(
            visibility,
            refinement_tolerance_seconds=0.1,
            calculate_closest_range=True,
        )
        self.assertEqual(len(passes), 1)
        event = passes[0]
        self.assertAlmostEqual(event.aos_elapsed_seconds, 16.67, delta=0.15)
        self.assertAlmostEqual(event.maximum_elevation_elapsed_seconds, 50.0, delta=0.2)
        self.assertAlmostEqual(event.los_elapsed_seconds, 83.33, delta=0.15)
        self.assertAlmostEqual(event.maximum_elevation_deg, 30.0, delta=0.05)
        self.assertAlmostEqual(event.closest_range_km, 700.0, delta=0.1)
        self.assertLessEqual(abs(event.aos_threshold_residual_deg), 1.0e-6)
        self.assertLessEqual(abs(event.los_threshold_residual_deg), 1.0e-6)

    def test_pass_matching_uses_maximum_elevation_time(self) -> None:
        elapsed = np.arange(0.0, 121.0, 2.0)
        timestamps = timestamps_from_epoch("2026-01-01T00:00:00Z", elapsed)

        def create(model: str, shift: float) -> list:
            shifted_elevation = -10.0 + 40.0 * np.sin(
                np.pi * (elapsed - shift) / 100.0
            )
            visibility = VisibilityHistory(
                model_name=model,
                station=self.station,
                epoch_utc="2026-01-01T00:00:00Z",
                elapsed_seconds=elapsed,
                timestamps_utc=timestamps,
                azimuth_deg=np.linspace(30.0, 210.0, elapsed.size),
                elevation_deg=shifted_elevation,
                range_km=1200.0 - 500.0 * np.maximum(
                    np.sin(np.pi * (elapsed - shift) / 100.0), 0.0
                ),
                range_rate_km_s=np.zeros(elapsed.size),
            )
            return detect_passes(
                visibility,
                refinement_tolerance_seconds=0.1,
            )

        reference = create("sgp4", 0.0)
        comparison = create("comparison", 2.0)
        rows = match_passes(
            reference,
            comparison,
            maximum_time_difference_seconds=20.0,
        )
        matched = [row for row in rows if row["match_status"] == "matched"]
        self.assertEqual(len(matched), 1)
        self.assertAlmostEqual(
            matched[0]["maximum_time_difference_seconds"],
            2.0,
            delta=0.5,
        )


if __name__ == "__main__":
    unittest.main()
