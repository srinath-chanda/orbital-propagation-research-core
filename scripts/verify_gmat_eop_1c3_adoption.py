"""Verify the closed 1C.2 evidence and Research Core 1C.3 baseline adoption."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core import BUILD_MARKER, RESEARCH_CORE_VERSION
from research_core.gmat_eop_closure import verify_gmat_eop_adoption


def main() -> int:
    record = PROJECT_ROOT / "configs" / "gmat_eop_1c3_adoption.json"
    print("=" * 78)
    print(f"Orbital Propagation Research Core — {RESEARCH_CORE_VERSION}")
    print(f"Build marker: {BUILD_MARKER}")
    print("=" * 78)
    try:
        result = verify_gmat_eop_adoption(record, project_root=PROJECT_ROOT)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("\nRESEARCH CORE 1C.3 GMAT EOP BASELINE ADOPTION VERIFIED")
    print(f"Closure ID            : {result.closure_id}")
    print(f"Matrix ID             : {result.matrix_id}")
    print(f"Validated model       : {result.validated_model}")
    print(f"Cases                 : {result.case_count}")
    print(f"Validation checks     : {result.check_count}")
    print(f"Raw GMAT ephemerides  : {result.raw_ephemeris_count}")
    print(f"Manifest records      : {result.manifest_record_count}")
    print(f"Maximum position      : {result.maximum_position_difference_m:.9e} m")
    print(f"Maximum velocity      : {result.maximum_velocity_difference_mm_s:.9e} mm/s")
    print(f"Adoption decision     : {result.adoption_decision}")
    print(f"Official report       : {result.official_report}")
    print("Meaning               : the full-EOP model is the validated GMAT J2 baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
