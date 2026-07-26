"""Verify the frozen Paper 1 validation baseline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core import BUILD_MARKER, RESEARCH_CORE_VERSION
from research_core.paper1_production import verify_paper1_baseline


def main() -> int:
    print("=" * 78)
    print(f"Orbital Propagation Research Core — {RESEARCH_CORE_VERSION}")
    print(f"Build marker: {BUILD_MARKER}")
    print("=" * 78)
    try:
        result = verify_paper1_baseline(
            PROJECT_ROOT / "configs/paper1_baseline_closure.json",
            project_root=PROJECT_ROOT,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("\nPAPER 1 PRODUCTION BASELINE VERIFIED")
    print(f"Closure ID                : {result.closure_id}")
    print(f"Checksum-gated evidence   : {result.evidence_count}")
    print(f"GMAT drag scenarios       : {result.drag_scenario_count}/4")
    print(f"GMAT drag checks          : {result.drag_check_count}/25")
    print(f"Maximum drag time residual: {result.maximum_drag_time_residual_seconds:.9e} s")
    print(f"Decision                  : {result.decision}")
    print("Meaning                   : physics scope is frozen; production runs are authorized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
