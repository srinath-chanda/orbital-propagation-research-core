"""Verify hashes and readiness status for the latest consolidation result."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_core.paper1_results_consolidation import (
    BUILD_MARKER,
    CONSOLIDATION_ID,
    verify_result_directory,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, help="Specific consolidation result directory.")
    args = parser.parse_args()
    if args.result_dir is None:
        base = PROJECT_ROOT / "results" / CONSOLIDATION_ID
        candidates = sorted(path.parent for path in base.glob("*/RUN_MANIFEST.json"))
        if not candidates:
            print("ERROR: no consolidation result was found")
            return 1
        result_dir = candidates[-1]
    else:
        result_dir = args.result_dir.resolve()
    errors = verify_result_directory(result_dir)
    print("=" * 78)
    print("Paper 1 Results Consolidation — Integrity Verification")
    print(f"Build marker: {BUILD_MARKER}")
    print("=" * 78)
    if errors:
        print("Status: FAILED")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Status             : PASSED")
    print(f"Verified directory : {result_dir}")
    print("Meaning            : every consolidated artifact matches its registered hash.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

