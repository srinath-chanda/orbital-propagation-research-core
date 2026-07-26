"""Run the Research Core 1C.2 independent full-EOP validation."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core import BUILD_MARKER, RESEARCH_CORE_VERSION
from research_core.gmat_eop_independent import run_independent_validation


def main() -> int:
    matrix = PROJECT_ROOT / "configs" / "gmat_eop_1c2_independent_matrix.json"
    print("=" * 78)
    print(f"Orbital Propagation Research Core — {RESEARCH_CORE_VERSION}")
    print(f"Build marker: {BUILD_MARKER}")
    print("=" * 78)
    print("\nImporting 12 new GMAT ephemerides and validating 6 holdout cases...")
    try:
        result = run_independent_validation(matrix, project_root=PROJECT_ROOT)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("\nRESEARCH CORE 1C.2 INDEPENDENT GMAT VALIDATION COMPLETED")
    print(f"Status             : {result.validation_status}")
    print(f"Cases              : {result.case_count}")
    print(f"Passed cases       : {result.passed_case_count}")
    print(f"Failed cases       : {result.failed_case_count}")
    print(f"Incomplete cases   : {result.incomplete_case_count}")
    print(f"Adoption decision  : {result.adoption_decision}")
    print(f"Result folder      : {result.result_directory}")
    print(f"Open report        : {result.report_path}")
    return 0 if result.validation_status == "passed_with_warnings" else 2


if __name__ == "__main__":
    raise SystemExit(main())
