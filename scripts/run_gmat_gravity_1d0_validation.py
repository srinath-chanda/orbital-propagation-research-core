"""Import the GMAT 1D.0 report and compare all gravity levels in Python."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core import BUILD_MARKER, RESEARCH_CORE_VERSION
from research_core.gmat_gravity_ladder import run_gravity_ladder_validation


def main() -> int:
    print("=" * 78)
    print(f"Orbital Propagation Research Core — {RESEARCH_CORE_VERSION}")
    print(f"Build marker: {BUILD_MARKER}")
    print("=" * 78)
    try:
        result = run_gravity_ladder_validation(
            PROJECT_ROOT / "configs/gmat_gravity_1d0_ladder.json",
            project_root=PROJECT_ROOT,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("\nRESEARCH CORE 1D.0 GMAT GRAVITY LADDER COMPLETED")
    print(f"Status             : {result.status}")
    print(f"Models             : {result.model_count}")
    print(f"Shared samples     : {result.sample_count}")
    print(f"Maximum difference : {result.maximum_difference_km_s2:.9e} km/s²")
    print(f"Decision           : {result.decision}")
    print(f"Result folder      : {result.result_directory}")
    print(f"Open report        : {result.report_path}")
    return 0 if result.status == "passed_with_warnings" else 1


if __name__ == "__main__":
    raise SystemExit(main())
