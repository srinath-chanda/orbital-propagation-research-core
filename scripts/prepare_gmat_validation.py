"""Generate reproducible GMAT R2026a scripts for CASE-LEO400 validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core import BUILD_MARKER, RESEARCH_CORE_VERSION
from research_core.external_validation import prepare_gmat_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the retained single-case GMAT J2 validation ladder."
    )
    parser.add_argument(
        "configuration",
        nargs="?",
        default="configs/case_leo400_gmat_matched.json",
        help="GMAT-matched JSON configuration.",
    )
    args = parser.parse_args()
    config = Path(args.configuration)
    if not config.is_absolute():
        config = (Path.cwd() / config).resolve()

    print("=" * 78)
    print(f"Orbital Propagation Research Core — {RESEARCH_CORE_VERSION}")
    print(f"Build marker: {BUILD_MARKER}")
    print("=" * 78)
    try:
        prepared = prepare_gmat_files(config, project_root=PROJECT_ROOT)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\nGMAT VALIDATION FILES PREPARED")
    print(f"Two-body GMAT script : {prepared.two_body_script}")
    print(f"J2 GMAT script       : {prepared.j2_script}")
    print(f"J2 short-arc script  : {prepared.j2_short_arc_script}")
    print(f"Acceleration script  : {prepared.acceleration_diagnostic_script}")
    print(f"Two-body output      : {prepared.two_body_ephemeris}")
    print(f"J2 output            : {prepared.j2_ephemeris}")
    print(f"J2 short-arc output  : {prepared.j2_short_arc_ephemeris}")
    print(f"Acceleration output  : {prepared.acceleration_diagnostic_report}")
    print(f"Metadata             : {prepared.metadata_file}")
    print("\nRun order in GMAT R2026a:")
    print("  1. Acceleration diagnostic")
    print("  2. J2 short arc")
    print("  3. Two-body and J2 full arcs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
