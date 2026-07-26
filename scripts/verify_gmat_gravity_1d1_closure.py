"""Verify the successful 1D.1 evidence before 1D.2 preparation."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core import BUILD_MARKER, RESEARCH_CORE_VERSION
from research_core.gmat_gravity_short_arc_closure import verify_gravity_short_arc_closure


def main() -> int:
    print("=" * 78)
    print(f"Orbital Propagation Research Core — {RESEARCH_CORE_VERSION}")
    print(f"Build marker: {BUILD_MARKER}")
    print("=" * 78)
    try:
        result = verify_gravity_short_arc_closure(
            PROJECT_ROOT / "configs/gmat_gravity_1d1_closure.json",
            project_root=PROJECT_ROOT,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("\nRESEARCH CORE 1D.1 SHORT-ARC CLOSURE VERIFIED")
    print(f"Closure ID                 : {result.closure_id}")
    print(f"Models                     : {result.model_count}")
    print(f"Validation checks          : {result.check_count}")
    print(f"Samples per model          : {result.sample_count_per_model}")
    print(f"Maximum position difference: {result.maximum_position_difference_m:.9e} m")
    print(f"Maximum velocity difference: {result.maximum_velocity_difference_mm_s:.9e} mm/s")
    print("Decision                   : 1D.2 preparation is authorized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
