"""Validate the four Research Core 1E.0 GMAT drag reports."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core import BUILD_MARKER, RESEARCH_CORE_VERSION
from research_core.gmat_drag_acceleration import run_drag_acceleration_validation


def main() -> int:
    print("=" * 78)
    print(f"Orbital Propagation Research Core — {RESEARCH_CORE_VERSION}")
    print(f"Build marker: {BUILD_MARKER}")
    print("=" * 78)
    try:
        result = run_drag_acceleration_validation(
            PROJECT_ROOT / "configs/gmat_drag_1e0_acceleration.json",
            project_root=PROJECT_ROOT,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("\nRESEARCH CORE 1E.0 GMAT DRAG ACCELERATION VALIDATION COMPLETED")
    print(f"Status                  : {result.status}")
    print(f"Passed scenarios        : {result.passed_scenario_count}/{result.scenario_count}")
    print(f"Passed checks           : {result.passed_check_count}/{result.check_count}")
    print(f"Maximum vector diff     : {result.maximum_drag_difference_km_s2:.9e} km/s²")
    print(f"Maximum relative diff   : {result.maximum_drag_relative_difference:.9e}")
    print(f"Maximum density rel diff: {result.maximum_density_relative_difference:.9e}")
    print(f"Decision                : {result.decision}")
    print(f"Result folder           : {result.result_directory}")
    print(f"Open report             : {result.report_path}")
    return 0 if result.status == "passed_with_warnings" else 1


if __name__ == "__main__":
    raise SystemExit(main())
