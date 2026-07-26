"""Analyze all preregistered Research Core 1B GMAT cases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core import BUILD_MARKER, RESEARCH_CORE_VERSION
from research_core.gmat_multicase import run_gmat_multicase_validation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Research Core 1B multi-case GMAT comparison."
    )
    parser.add_argument(
        "matrix",
        nargs="?",
        default="configs/gmat_1b_multicase_matrix.json",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Create an explicitly incomplete report instead of stopping.",
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
        result = run_gmat_multicase_validation(
            matrix,
            project_root=PROJECT_ROOT,
            allow_missing=args.allow_missing,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\nRESEARCH CORE 1B GMAT MULTI-CASE VALIDATION COMPLETED")
    print(f"Matrix ID          : {result.matrix_id}")
    print(f"Status             : {result.validation_status}")
    print(f"Passed cases       : {result.passed_case_count}/{result.total_case_count}")
    print(f"Failed cases       : {result.failed_case_count}")
    print(f"Incomplete cases   : {result.incomplete_case_count}")
    print(f"Result folder      : {result.result_directory}")
    print(f"Open report        : {result.report_path}")
    if result.validation_status == "passed_with_warnings":
        print("Meaning            : all preregistered cases passed; warnings document scope limits.")
        return 0
    if result.validation_status == "incomplete":
        print("Meaning            : one or more required GMAT outputs are missing.")
        return 2
    print("Meaning            : preserve the outputs and investigate; do not change thresholds.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
