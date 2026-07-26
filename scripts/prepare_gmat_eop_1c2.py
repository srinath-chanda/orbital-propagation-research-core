"""Prepare the preregistered Research Core 1C.2 GMAT holdout matrix."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core import BUILD_MARKER, RESEARCH_CORE_VERSION
from research_core.gmat_eop_independent import prepare_independent_matrix


def main() -> int:
    matrix = PROJECT_ROOT / "configs" / "gmat_eop_1c2_independent_matrix.json"
    print("=" * 78)
    print(f"Orbital Propagation Research Core — {RESEARCH_CORE_VERSION}")
    print(f"Build marker: {BUILD_MARKER}")
    print("=" * 78)
    try:
        result = prepare_independent_matrix(matrix, project_root=PROJECT_ROOT)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("\nGMAT 1C.2 INDEPENDENT VALIDATION FILES PREPARED")
    print(f"Matrix ID              : {result.matrix_id}")
    print(f"Independent cases      : {result.case_count}")
    print(f"Expected GMAT outputs  : {result.expected_output_count}")
    print(f"Archived old outputs   : {len(result.archived_outputs)}")
    print(f"Master GMAT script     : {result.master_script}")
    print(f"Run order              : {result.run_order_path}")
    print(f"Manifest               : {result.manifest_path}")
    print("\nNext: verify the GMAT EOP file, then run RUN_ALL_CASES_1C2.script once in GMAT R2026a.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
