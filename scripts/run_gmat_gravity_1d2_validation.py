"""Validate the Research Core 1D.2 multi-case full-arc GMAT matrix."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core import BUILD_MARKER, RESEARCH_CORE_VERSION
from research_core.gmat_gravity_multicase import run_gravity_multicase_validation


def _progress(completed: int, total: int, case_id: str, model_id: str) -> None:
    print(f"  Completed {completed}/{total}: {case_id} / {model_id}", flush=True)


def main() -> int:
    print("=" * 78)
    print(f"Orbital Propagation Research Core — {RESEARCH_CORE_VERSION}")
    print(f"Build marker: {BUILD_MARKER}")
    print("=" * 78)
    print("\nValidating 6 cases x 4 gravity models. This can take 20–40 minutes.")
    try:
        result = run_gravity_multicase_validation(
            PROJECT_ROOT / "configs/gmat_gravity_1d2_multicase.json",
            project_root=PROJECT_ROOT,
            progress_callback=_progress,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("\nRESEARCH CORE 1D.2 GMAT MULTI-CASE VALIDATION COMPLETED")
    print(f"Status                       : {result.status}")
    print(f"Passed cases                 : {result.passed_case_count}/{result.case_count}")
    print(
        f"Passed model runs            : "
        f"{result.passed_model_run_count}/{result.model_run_count}"
    )
    print(f"Passed checks                : {result.passed_check_count}/{result.check_count}")
    print(f"Maximum position difference  : {result.maximum_position_difference_m:.9e} m")
    print(f"Maximum velocity difference  : {result.maximum_velocity_difference_mm_s:.9e} mm/s")
    print(f"Decision                     : {result.decision}")
    print(f"Result folder                : {result.result_directory}")
    print(f"Open report                  : {result.report_path}")
    return 0 if result.status == "passed_with_warnings" else 1


if __name__ == "__main__":
    raise SystemExit(main())
