"""Compare GMAT point-mass and J2 ephemerides with Research Core."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core import BUILD_MARKER, RESEARCH_CORE_VERSION
from research_core.external_validation import run_gmat_external_validation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run external validation against GMAT STK-TimePosVel ephemerides."
    )
    parser.add_argument(
        "configuration",
        nargs="?",
        default="configs/case_leo400_gmat_matched.json",
    )
    parser.add_argument(
        "two_body_ephemeris",
        nargs="?",
        default="data/reference/gmat/output/CASE_LEO400_GMAT_TWO_BODY.e",
    )
    parser.add_argument(
        "j2_ephemeris",
        nargs="?",
        default="data/reference/gmat/output/CASE_LEO400_GMAT_J2.e",
    )
    args = parser.parse_args()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()

    print("=" * 78)
    print(f"Orbital Propagation Research Core — {RESEARCH_CORE_VERSION}")
    print(f"Build marker: {BUILD_MARKER}")
    print("=" * 78)

    try:
        result = run_gmat_external_validation(
            resolve(args.configuration),
            resolve(args.two_body_ephemeris),
            resolve(args.j2_ephemeris),
            project_root=PROJECT_ROOT,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\nRESEARCH CORE GMAT EXTERNAL VALIDATION COMPLETED")
    print(f"Experiment ID                         : {result.experiment_id}")
    print(f"Validation status                     : {result.validation_status}")
    print(f"Result folder                         : {result.result_directory}")
    print(f"Two-body max position difference (m) : {result.two_body_maximum_position_difference_m:.9e}")
    print(f"Two-body max velocity difference     : {result.two_body_maximum_velocity_difference_mm_s:.9e} mm/s")
    print(f"Fixed-axis J2 max position (m)       : {result.j2_maximum_position_difference_m:.9e}")
    print(f"Fixed-axis J2 max velocity           : {result.j2_maximum_velocity_difference_mm_s:.9e} mm/s")
    print(f"Pole-aware J2 max position (m)       : {result.j2_gmat_matched_maximum_position_difference_m:.9e}")
    print(f"Pole-aware J2 max velocity           : {result.j2_gmat_matched_maximum_velocity_difference_mm_s:.9e} mm/s")
    print(f"Warnings                              : {len(result.warnings)}")
    for warning in result.warnings:
        print(f"  - {warning}")
    print(f"Created files                         : {len(result.created_files)}")
    print(f"Open report                           : {result.report_path}")
    return 0 if result.validation_status == "passed_with_warnings" else 3


if __name__ == "__main__":
    raise SystemExit(main())
