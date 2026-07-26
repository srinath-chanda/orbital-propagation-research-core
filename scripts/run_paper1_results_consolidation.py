"""Create Paper 1 publication tables, figures, and results narrative."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_core.paper1_results_consolidation import BUILD_MARKER, run_consolidation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--production-dir",
        type=Path,
        help="Specific EXP-PAPER1-PRODUCTION-001 timestamp directory; defaults to latest.",
    )
    args = parser.parse_args()
    print("=" * 78)
    print("Orbital Propagation Research Core — PAPER 1 RESULTS CONSOLIDATION")
    print(f"Build marker: {BUILD_MARKER}")
    print("=" * 78)
    try:
        result = run_consolidation(PROJECT_ROOT, production_dir=args.production_dir)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 1
    print()
    print("PAPER 1 RESULTS CONSOLIDATION COMPLETED")
    print(f"Status             : {result.status}")
    print(f"Publication figures: {result.figure_count} PNG + {result.figure_count} PDF")
    print(f"Publication tables : {result.table_count}")
    print(f"Result folder      : {result.result_directory}")
    print(f"Open report        : {result.report_path}")
    print("Meaning            : production evidence is ready for manuscript drafting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

