"""Validate Python J2 accelerations against a shared-state GMAT report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core import BUILD_MARKER, RESEARCH_CORE_VERSION
from research_core.gmat_j2_diagnostics import run_gmat_acceleration_validation


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare isolated GMAT J2 acceleration with both Python J2 models."
    )
    parser.add_argument(
        "configuration",
        nargs="?",
        default="configs/case_leo400_gmat_matched.json",
    )
    parser.add_argument(
        "acceleration_report",
        nargs="?",
        default=(
            "data/reference/gmat/output/"
            "CASE_LEO400_GMAT_ACCELERATION_DIAGNOSTIC.csv"
        ),
    )
    args = parser.parse_args()
    print("=" * 78)
    print(f"Orbital Propagation Research Core — {RESEARCH_CORE_VERSION}")
    print(f"Build marker: {BUILD_MARKER}")
    print("=" * 78)
    try:
        result = run_gmat_acceleration_validation(
            _resolve(args.configuration),
            _resolve(args.acceleration_report),
            project_root=PROJECT_ROOT,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("\nGMAT J2 ACCELERATION VALIDATION COMPLETED")
    print(f"Status                              : {result.validation_status}")
    print(f"Shared-state samples                : {result.sample_count}")
    print(f"Fixed-axis maximum difference       : {result.fixed_axis_maximum_difference_km_s2:.9e} km/s²")
    print(f"Pole-aware maximum difference       : {result.gmat_matched_maximum_difference_km_s2:.9e} km/s²")
    print(f"Pole-aware maximum relative error   : {result.gmat_matched_maximum_relative_difference:.9e}")
    print(f"Result folder                       : {result.result_directory}")
    print(f"Open report                         : {result.report_path}")
    return 0 if result.validation_status == "passed_with_warnings" else 3


if __name__ == "__main__":
    raise SystemExit(main())
