from __future__ import annotations

import sys
import unittest
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.ground_track import (
    GroundTrackHistory,
    compare_ground_tracks,
    geodetic_roundtrip_error,
    gcrs_state_history_to_ground_track,
    split_at_antimeridian,
    wrap_longitude_deg,
)
from research_core.propagators.sgp4_propagator import propagate_sgp4_frozen_tle
from research_core.tle import load_frozen_tle


class GroundTrackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tle_path = PROJECT_ROOT / "data" / "tle" / "iss_25544_2026-07-18_celestrak.tle"
        metadata_path = PROJECT_ROOT / "data" / "tle" / "iss_25544_2026-07-18_celestrak_metadata.json"
        cls.tle = load_frozen_tle(tle_path, metadata_path, expected_catalog_number=25544)
        _, cls.gcrs, _ = propagate_sgp4_frozen_tle(
            cls.tle, np.array([0.0, 60.0, 120.0, 180.0])
        )
        cls.track = gcrs_state_history_to_ground_track(cls.gcrs)

    def test_wrap_longitude_range(self) -> None:
        values = wrap_longitude_deg(np.array([-540.0, -181.0, 0.0, 181.0, 540.0]))
        self.assertTrue(np.all(values >= -180.0))
        self.assertTrue(np.all(values <= 180.0))
        np.testing.assert_allclose(values, np.array([-180.0, 179.0, 0.0, -179.0, 180.0]))

    def test_antimeridian_split(self) -> None:
        longitude = np.array([170.0, 179.0, -179.0, -170.0])
        latitude = np.array([0.0, 1.0, 2.0, 3.0])
        segments = split_at_antimeridian(longitude, latitude)
        self.assertEqual(len(segments), 2)
        np.testing.assert_allclose(segments[0][0], np.array([170.0, 179.0]))
        np.testing.assert_allclose(segments[1][0], np.array([-179.0, -170.0]))

    def test_ground_track_coordinates_are_finite_and_bounded(self) -> None:
        self.assertTrue(np.all(np.isfinite(self.track.latitude_deg)))
        self.assertTrue(np.all(np.isfinite(self.track.longitude_deg)))
        self.assertTrue(np.all(np.isfinite(self.track.altitude_km)))
        self.assertTrue(np.all(np.abs(self.track.latitude_deg) <= 90.0))
        self.assertTrue(np.all(np.abs(self.track.longitude_deg) <= 180.0))
        self.assertTrue(np.all(self.track.altitude_km > 100.0))

    def test_ground_track_uses_wgs84_and_itrs(self) -> None:
        self.assertEqual(self.track.ellipsoid, "WGS84")
        self.assertEqual(self.track.earth_fixed_frame, "ITRS_ASTROPY")
        self.assertEqual(self.track.source_frame, "GCRS_ASTROPY_FROM_TEME")

    def test_geodetic_roundtrip_is_small(self) -> None:
        result = geodetic_roundtrip_error(self.track, sample_count=4)
        self.assertLessEqual(result["maximum_position_residual_m"], 1.0e-6)

    def test_identical_track_comparison_is_zero(self) -> None:
        comparison = compare_ground_tracks(
            self.track, self.track, surface_radius_km=6378.137
        )
        self.assertEqual(comparison["maximum_surface_separation_km"], 0.0)
        self.assertEqual(comparison["maximum_absolute_altitude_difference_km"], 0.0)
        self.assertEqual(comparison["maximum_itrs_position_separation_km"], 0.0)

    def test_equatorial_wgs84_reference_point(self) -> None:
        location = EarthLocation.from_geocentric(
            6378.137 * u.km, 0.0 * u.km, 0.0 * u.km
        )
        longitude, latitude, height = location.to_geodetic(ellipsoid="WGS84")
        self.assertAlmostEqual(longitude.to_value(u.deg), 0.0, places=12)
        self.assertAlmostEqual(latitude.to_value(u.deg), 0.0, places=12)
        self.assertAlmostEqual(height.to_value(u.m), 0.0, places=6)

    def test_mismatched_time_grid_is_rejected(self) -> None:
        shorter = GroundTrackHistory(
            model_name="shorter",
            source_frame=self.track.source_frame,
            earth_fixed_frame=self.track.earth_fixed_frame,
            ellipsoid=self.track.ellipsoid,
            epoch_utc=self.track.epoch_utc,
            elapsed_seconds=self.track.elapsed_seconds[:-1],
            timestamps_utc=self.track.timestamps_utc[:-1],
            positions_itrs_km=self.track.positions_itrs_km[:-1],
            velocities_itrs_km_s=self.track.velocities_itrs_km_s[:-1],
            latitude_deg=self.track.latitude_deg[:-1],
            longitude_deg=self.track.longitude_deg[:-1],
            altitude_km=self.track.altitude_km[:-1],
        )
        with self.assertRaises(ValueError):
            compare_ground_tracks(self.track, shorter, surface_radius_km=6378.137)


if __name__ == "__main__":
    unittest.main()
