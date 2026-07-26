"""Prepare the preregistered Research Core 1B GMAT matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core import BUILD_MARKER, RESEARCH_CORE_VERSION
from research_core.gmat_multicase import prepare_gmat_multicase_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the Research Core 1B multi-case GMAT scripts."
    )
    parser.add_argument(
        "matrix",
        nargs="?",
        default="configs/gmat_1b_multicase_matrix.json",
    )
    args = parser.parse_args()
    matrix = Path(args.matrix)
    if not matrix.is_absolute():
        matrix = (Path.cwd() / matrix).resolve()

    print("=" * 78)
    print(f"Orbital Propagation Research Core — {RESEARCH_CORE_VERSION}")
    print(f"Build marker: {BUILD_MARKER}")
    print("=" * 78)
    try:
        prepared = prepare_gmat_multicase_files(matrix, project_root=PROJECT_ROOT)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\nGMAT 1B MULTI-CASE FILES PREPARED")
    print(f"Matrix ID              : {prepared.matrix_id}")
    print(f"Cases                  : {prepared.case_count}")
    print(f"Expected GMAT outputs  : {prepared.expected_output_count}")
    print(f"Archived old outputs   : {len(prepared.archived_outputs)}")
    print(f"Master GMAT script     : {prepared.master_script}")
    print(f"Run order              : {prepared.run_order_path}")
    print(f"Manifest               : {prepared.manifest_path}")
    print("\nNext: open RUN_ALL_CASES_1B.script in GMAT R2026a and run it once.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
