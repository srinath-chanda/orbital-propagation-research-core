"""Prepare the Research Core 1D.0 GMAT gravity ladder."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core import BUILD_MARKER, RESEARCH_CORE_VERSION
from research_core.gmat_gravity_ladder import prepare_gravity_ladder


def main() -> int:
    print("=" * 78)
    print(f"Orbital Propagation Research Core — {RESEARCH_CORE_VERSION}")
    print(f"Build marker: {BUILD_MARKER}")
    print("=" * 78)
    try:
        result = prepare_gravity_ladder(
            PROJECT_ROOT / "configs/gmat_gravity_1d0_ladder.json",
            project_root=PROJECT_ROOT,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("\nGMAT 1D.0 GRAVITY LADDER PREPARED")
    print(f"Experiment ID      : {result.experiment_id}")
    print(f"Gravity models     : {result.model_count}")
    print(f"Shared samples     : {result.sample_count}")
    print(f"Archived outputs   : {len(result.archived_outputs)}")
    print(f"Master GMAT script : {result.master_script}")
    print(f"Expected output    : {result.output_report}")
    print(f"Manifest           : {result.manifest}")
    print("\nNext: run RUN_GRAVITY_LADDER_1D0.script once in GMAT R2026a.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
