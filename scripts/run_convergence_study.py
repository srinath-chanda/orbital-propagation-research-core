"""Run the Research Core 1A.1C numerical convergence study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core import BUILD_MARKER, RESEARCH_CORE_VERSION
from research_core.configuration import ConfigValidationError
from research_core.convergence_manager import run_convergence_study


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Run the full numerical two-body tolerance and maximum-step "
            "convergence matrix."
        )
    )
    result.add_argument("configuration", help="Path to experiment JSON configuration")
    return result


def main() -> int:
    arguments = parser().parse_args()
    configuration = Path(arguments.configuration)
    if not configuration.is_absolute():
        configuration = (Path.cwd() / configuration).resolve()

    print("=" * 78)
    print(f"Orbital Propagation Research Core — {RESEARCH_CORE_VERSION}")
    print(f"Build marker: {BUILD_MARKER}")
    print("=" * 78)
    try:
        result = run_convergence_study(
            configuration,
            project_root=PROJECT_ROOT,
        )
    except (FileNotFoundError, ConfigValidationError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print()
    print("NUMERICAL CONVERGENCE STUDY COMPLETED")
    print(f"Experiment ID             : {result.experiment_id}")
    print(f"Validation status         : {result.validation_status}")
    print(f"Result folder             : {result.result_directory}")
    print(f"Matrix candidate settings : {result.matrix_candidate_count}")
    print(f"Evaluated settings        : {result.evaluated_setting_count}")
    print(f"Passing settings          : {result.passing_candidate_count}")
    print(f"Balanced recommendation  : {result.balanced_case_id}")
    print(f"Created files             : {len(result.created_files)}")
    print(f"Warnings                  : {len(result.warnings)}")
    for warning in result.warnings:
        print(f"  - {warning}")
    print()
    print("Open CONVERGENCE_SUMMARY.md and selected_integrator_settings.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
