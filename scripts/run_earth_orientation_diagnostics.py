"""Run Research Core 1C Earth-orientation residual attribution."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core import BUILD_MARKER, RESEARCH_CORE_VERSION
from research_core.earth_orientation_diagnostics import (
    run_earth_orientation_diagnostics,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Earth-orientation realizations against the closed GMAT matrix."
    )
    parser.add_argument(
        "configuration",
        nargs="?",
        default="configs/earth_orientation_1c_diagnostic.json",
    )
    args = parser.parse_args()
    configuration = Path(args.configuration)
    if not configuration.is_absolute():
        configuration = (Path.cwd() / configuration).resolve()

    print("=" * 78)
    print(f"Orbital Propagation Research Core — {RESEARCH_CORE_VERSION}")
    print(f"Build marker: {BUILD_MARKER}")
    print("=" * 78)
    print("\nRunning 5 orientation models across 10 saved GMAT cases...")
    def show_progress(completed: int, total: int, case_id: str, _model_id: str) -> None:
        if completed % 5 == 0 or completed == total:
            print(f"  Completed {completed}/{total} comparisons: {case_id}", flush=True)

    try:
        result = run_earth_orientation_diagnostics(
            configuration,
            project_root=PROJECT_ROOT,
            progress_callback=show_progress,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\nRESEARCH CORE 1C EARTH-ORIENTATION DIAGNOSTIC COMPLETED")
    print(f"Status             : {result.diagnostic_status}")
    print(f"Cases              : {result.case_count}")
    print(f"Models             : {result.model_count}")
    print(f"Comparisons        : {result.case_count * result.model_count}")
    print(f"Decision           : {result.decision}")
    print(f"1B baseline        : {result.baseline_model}")
    print(f"Recommended model  : {result.recommended_model}")
    print(f"Result folder      : {result.result_directory}")
    print(f"Open report        : {result.report_path}")
    print("Meaning            : diagnostic completed; the closed 1B claim is unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
