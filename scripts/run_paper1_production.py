"""Run convergence plus the 11 frozen Paper 1 production experiments."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core import BUILD_MARKER, RESEARCH_CORE_VERSION
from research_core.paper1_production import run_paper1_production


def main() -> int:
    print("=" * 78)
    print(f"Orbital Propagation Research Core — {RESEARCH_CORE_VERSION}")
    print(f"Build marker: {BUILD_MARKER}")
    print("=" * 78)
    print("This run performs one convergence study and 11 production experiments.")
    print("Keep PowerShell open; no GMAT run is required.\n")
    try:
        result = run_paper1_production(
            PROJECT_ROOT / "configs/paper1_production_matrix.json",
            project_root=PROJECT_ROOT,
            progress=lambda message: print(message, flush=True),
        )
    except KeyboardInterrupt:
        print("\nSTOPPED: production was interrupted; existing results were preserved.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("\nPAPER 1 PRODUCTION MATRIX COMPLETED")
    print(f"Matrix ID              : {result.matrix_id}")
    print(f"Status                 : {result.status}")
    print(f"Completed experiments  : {result.completed_experiment_count}/{result.expected_experiment_count}")
    print(f"Failed experiments     : {result.failed_experiment_count}")
    print(f"Result folder          : {result.result_directory}")
    print(f"Summary                : {result.summary_path}")
    print(f"Open report            : {result.report_path}")
    return 0 if result.status == "passed_with_warnings" else 1


if __name__ == "__main__":
    raise SystemExit(main())
