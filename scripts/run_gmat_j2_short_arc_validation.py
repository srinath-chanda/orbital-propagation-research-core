"""Validate the 10-minute GMAT J2 short arc against both Python models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core import BUILD_MARKER, RESEARCH_CORE_VERSION
from research_core.external_validation import run_gmat_j2_short_arc_validation


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare the GMAT J2 short arc with fixed-axis and pole-aware Python J2."
    )
    parser.add_argument(
        "configuration",
        nargs="?",
        default="configs/case_leo400_gmat_matched.json",
    )
    parser.add_argument(
        "j2_ephemeris",
        nargs="?",
        default="data/reference/gmat/output/CASE_LEO400_GMAT_J2_SHORT_ARC.e",
    )
    args = parser.parse_args()
    print("=" * 78)
    print(f"Orbital Propagation Research Core — {RESEARCH_CORE_VERSION}")
    print(f"Build marker: {BUILD_MARKER}")
    print("=" * 78)
    try:
        result = run_gmat_j2_short_arc_validation(
            _resolve(args.configuration),
            _resolve(args.j2_ephemeris),
            project_root=PROJECT_ROOT,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("\nGMAT J2 SHORT-ARC VALIDATION COMPLETED")
    print(f"Status                              : {result.validation_status}")
    print(f"Fixed-axis maximum position         : {result.fixed_axis_maximum_position_difference_m:.9e} m")
    print(f"Fixed-axis maximum velocity         : {result.fixed_axis_maximum_velocity_difference_mm_s:.9e} mm/s")
    print(f"Pole-aware maximum position         : {result.gmat_matched_maximum_position_difference_m:.9e} m")
    print(f"Pole-aware maximum velocity         : {result.gmat_matched_maximum_velocity_difference_mm_s:.9e} mm/s")
    print(f"Result folder                       : {result.result_directory}")
    print(f"Open report                         : {result.report_path}")
    return 0 if result.validation_status == "passed_with_warnings" else 3


if __name__ == "__main__":
    raise SystemExit(main())
