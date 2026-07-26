"""Prepare the exact preregistered Paper 1 production configurations."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core import BUILD_MARKER, RESEARCH_CORE_VERSION
from research_core.paper1_production import prepare_paper1_production


def main() -> int:
    print("=" * 78)
    print(f"Orbital Propagation Research Core — {RESEARCH_CORE_VERSION}")
    print(f"Build marker: {BUILD_MARKER}")
    print("=" * 78)
    try:
        result = prepare_paper1_production(
            PROJECT_ROOT / "configs/paper1_production_matrix.json",
            project_root=PROJECT_ROOT,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("\nPAPER 1 PRODUCTION MATRIX PREPARED")
    print(f"Matrix ID              : {result.matrix_id}")
    print(f"Experiments            : {result.experiment_count}")
    print(f"Primary model runs     : {result.primary_model_run_count}")
    print(f"Executed model runs    : {result.executed_model_run_count}")
    print(f"Configuration folder   : {result.configuration_directory}")
    print(f"Preparation manifest   : {result.manifest_path}")
    print("Next                    : run scripts\\run_paper1_production.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
