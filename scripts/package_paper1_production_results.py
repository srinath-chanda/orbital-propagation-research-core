"""Package the compact Paper 1 production review evidence."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.paper1_production import package_paper1_production_results


def main() -> int:
    try:
        archive = package_paper1_production_results(
            PROJECT_ROOT / "configs/paper1_production_matrix.json",
            project_root=PROJECT_ROOT,
            output_path=PROJECT_ROOT / "RESEARCH_CORE_PAPER1_PRODUCTION_RESULTS.zip",
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("PAPER 1 PRODUCTION REVIEW PACKAGE CREATED")
    print(f"ZIP: {archive}")
    print("Upload this ZIP for production-result review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
