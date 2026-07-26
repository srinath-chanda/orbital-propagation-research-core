"""Verify the completed 1D.2 evidence before starting drag validation."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core import BUILD_MARKER, RESEARCH_CORE_VERSION
from research_core.gmat_gravity_multicase_closure import (
    verify_gravity_multicase_closure,
)


def main() -> int:
    print("=" * 78)
    print(f"Orbital Propagation Research Core — {RESEARCH_CORE_VERSION}")
    print(f"Build marker: {BUILD_MARKER}")
    print("=" * 78)
    try:
        result = verify_gravity_multicase_closure(
            PROJECT_ROOT / "configs/gmat_gravity_1d2_closure.json",
            project_root=PROJECT_ROOT,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("\nRESEARCH CORE 1D.2 GRAVITY CLOSURE VERIFIED")
    print(f"Closure ID          : {result.closure_id}")
    print(f"Passed cases        : {result.case_count}/{result.case_count}")
    print(f"Passed model runs   : {result.model_run_count}/{result.model_run_count}")
    print(f"Passed checks       : {result.check_count}/{result.check_count}")
    print(f"Worst position      : {result.maximum_position_difference_m:.9e} m")
    print(f"Worst velocity      : {result.maximum_velocity_difference_mm_s:.9e} mm/s")
    print(f"Worst time residual : {result.maximum_time_residual_seconds:.9e} s")
    print("Decision            : 1E.0 drag acceleration preparation is authorized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
