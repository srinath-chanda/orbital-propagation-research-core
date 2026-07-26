"""Run Research Core 1C.1 exact GMAT R2026a EOP attribution."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core import BUILD_MARKER, RESEARCH_CORE_VERSION
from research_core.gmat_eop_diagnostics import (
    load_gmat_eop_diagnostic_config,
    run_gmat_eop_diagnostics,
)
from research_core.gmat_multicase import load_gmat_matrix_spec


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Attribute the closed GMAT residual to exact R2026a EOP polar motion."
    )
    parser.add_argument(
        "configuration",
        nargs="?",
        default="configs/gmat_eop_1c1_diagnostic.json",
    )
    args = parser.parse_args()
    configuration = Path(args.configuration)
    if not configuration.is_absolute():
        configuration = (Path.cwd() / configuration).resolve()

    print("=" * 78)
    print(f"Orbital Propagation Research Core — {RESEARCH_CORE_VERSION}")
    print(f"Build marker: {BUILD_MARKER}")
    print("=" * 78)
    try:
        config = load_gmat_eop_diagnostic_config(configuration)
        matrix_path = PROJECT_ROOT / str(config["matrix_specification"])
        matrix = load_gmat_matrix_spec(matrix_path)
        model_count = len(config["models"])
        case_count = len(matrix["cases"])
        print(
            f"\nRunning {model_count} EOP models across "
            f"{case_count} saved GMAT cases..."
        )

        def show_progress(
            completed: int, total: int, case_id: str, _model_id: str
        ) -> None:
            if completed % model_count == 0 or completed == total:
                print(
                    f"  Completed {completed}/{total} comparisons: {case_id}",
                    flush=True,
                )

        result = run_gmat_eop_diagnostics(
            configuration,
            project_root=PROJECT_ROOT,
            progress_callback=show_progress,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\nRESEARCH CORE 1C.1 GMAT EOP DIAGNOSTIC COMPLETED")
    print(f"Status                : {result.diagnostic_status}")
    print(f"Cases                 : {result.case_count}")
    print(f"Models                : {result.model_count}")
    print(f"Comparisons           : {result.comparison_count}")
    print(f"Decision              : {result.decision}")
    print(f"Closed 1B baseline    : {result.baseline_model}")
    print(f"Recommended candidate : {result.recommended_model}")
    print(f"Result folder         : {result.result_directory}")
    print(f"Open report           : {result.report_path}")
    print(
        "Meaning               : candidate identified; independent GMAT "
        "validation is required before adoption."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
