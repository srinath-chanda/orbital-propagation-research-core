"""Run one integrated Research Core experiment selected by configuration source."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core import BUILD_MARKER, RESEARCH_CORE_VERSION
from research_core.configuration import ConfigValidationError
from research_core.experiment_manager import run_experiment
from research_core.tle_experiment_manager import run_tle_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one integrated Research Core experiment.")
    parser.add_argument("configuration", help="Path to the experiment JSON configuration.")
    args = parser.parse_args()
    config_path = Path(args.configuration)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()

    print("=" * 78)
    print(f"Orbital Propagation Research Core — {RESEARCH_CORE_VERSION}")
    print(f"Build marker: {BUILD_MARKER}")
    print("=" * 78)

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        source_type = raw.get("initial_state", {}).get("source_type")
        if source_type == "fixed_tle":
            result = run_tle_experiment(config_path, project_root=PROJECT_ROOT)
            print("\nRESEARCH CORE 1A.7 COMPLETE RESEARCH PIPELINE COMPLETED")
            print(f"Experiment ID                         : {result.experiment_id}")
            print(f"Validation status                     : {result.validation_status}")
            print(f"Result folder                         : {result.result_directory}")
            print(f"TLE epoch UTC                         : {result.tle_epoch_utc}")
            print(f"End TLE age (hours)                   : {result.end_tle_age_hours:.9f}")
            print(f"SGP4 nonzero error count              : {result.nonzero_sgp4_error_count}")
            print(f"Frame roundtrip position error (m)    : {result.frame_roundtrip_position_error_m:.9e}")
            for model in sorted(result.maximum_separation_km_by_model):
                print(f"Max SGP4 separation — {model:22s}: {result.maximum_separation_km_by_model[model]:.9f} km")
                print(f"Final SGP4 separation — {model:20s}: {result.final_separation_km_by_model[model]:.9f} km")
            print(f"Geodetic roundtrip position error (m) : {result.geodetic_roundtrip_position_error_m:.9e}")
            for model in sorted(result.maximum_ground_track_separation_km_by_model):
                print(f"Max ground-track separation — {model:16s}: {result.maximum_ground_track_separation_km_by_model[model]:.9f} km")
                print(f"Final ground-track separation — {model:14s}: {result.final_ground_track_separation_km_by_model[model]:.9f} km")
            print(f"Pass station                          : {result.pass_station_name} ({result.pass_station_id})")
            print(f"Minimum elevation mask (deg)          : {result.pass_minimum_elevation_deg:.3f}")
            for model in sorted(result.pass_count_by_model):
                print(f"Pass count — {model:24s}: {result.pass_count_by_model[model]}")
            for model in sorted(result.matched_pass_count_by_model):
                aos = result.maximum_absolute_aos_difference_seconds_by_model[model]
                los = result.maximum_absolute_los_difference_seconds_by_model[model]
                aos_text = "n/a" if aos is None else f"{aos:.6f} s"
                los_text = "n/a" if los is None else f"{los:.6f} s"
                print(f"Matched passes vs SGP4 — {model:15s}: {result.matched_pass_count_by_model[model]}")
                print(f"Max |AOS difference| — {model:17s}: {aos_text}")
                print(f"Max |LOS difference| — {model:17s}: {los_text}")
            print(f"Warnings                              : {len(result.warnings)}")
            for warning in result.warnings:
                print(f"  - {warning}")
            print(f"\nCreated {len(result.created_files)} files.")
            print(f"Primary HTML report                   : {result.research_report_path}")
            print(f"Run manifest                          : {result.run_manifest_path}")
            print("Open RESEARCH_REPORT.html for the combined explanation, figures, validation, and data links.")
            return 0

        result = run_experiment(config_path, project_root=PROJECT_ROOT)
    except (FileNotFoundError, ConfigValidationError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\nRESEARCH CORE 1A.7 CONTROLLED-ORBIT PIPELINE COMPLETED")
    print(f"Experiment ID                            : {result.experiment_id}")
    print(f"Validation status                        : {result.validation_status}")
    print(f"Result folder                            : {result.result_directory}")
    print(f"Two-body max position difference (m)    : {result.maximum_position_difference_m:.9e}")
    print(f"J2 max separation from two-body (km)    : {result.maximum_j2_two_body_position_difference_km:.9f}")
    print(f"J2+drag max separation from J2 (km)      : {result.maximum_drag_j2_position_difference_km:.9f}")
    print(f"Final drag semi-major-axis Δ vs J2 (m)   : {result.final_drag_semi_major_axis_difference_vs_j2_m:.9f}")
    print(f"Warnings                                 : {len(result.warnings)}")
    for warning in result.warnings:
        print(f"  - {warning}")
    print(f"\nCreated {len(result.created_files)} files.")
    print(f"Primary HTML report                   : {result.result_directory / 'RESEARCH_REPORT.html'}")
    print(f"Run manifest                          : {result.result_directory / 'RUN_MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
