"""Import and freeze the installed GMAT R2026a JGM2 gravity file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core import BUILD_MARKER, RESEARCH_CORE_VERSION
from research_core.gmat_gravity_ladder import import_gmat_jgm2


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy GMAT R2026a JGM2.cof into frozen project evidence.")
    parser.add_argument("gmat_root", help=r"GMAT root, for example C:\GMAT\R2026a")
    args = parser.parse_args()
    print("=" * 78)
    print(f"Orbital Propagation Research Core — {RESEARCH_CORE_VERSION}")
    print(f"Build marker: {BUILD_MARKER}")
    print("=" * 78)
    try:
        result = import_gmat_jgm2(args.gmat_root, project_root=PROJECT_ROOT)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("\nGMAT R2026a JGM2 FILE IMPORTED AND VERIFIED")
    print(f"Frozen file       : {result.destination}")
    print(f"SHA-256           : {result.sha256}")
    print(f"Degree/order      : {result.maximum_degree}/{result.maximum_order}")
    print(f"Provenance        : {result.provenance}")
    print("GMAT install      : unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
