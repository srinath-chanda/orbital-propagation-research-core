"""Package raw and processed Research Core 1C.2 evidence."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.gmat_eop_independent import package_independent_results


def main() -> int:
    matrix = PROJECT_ROOT / "configs" / "gmat_eop_1c2_independent_matrix.json"
    output = PROJECT_ROOT / "RESEARCH_CORE_1C2_GMAT_RESULTS.zip"
    try:
        archive = package_independent_results(
            matrix, project_root=PROJECT_ROOT, output_path=output
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("RESEARCH CORE 1C.2 RESULTS PACKAGE CREATED")
    print(f"ZIP: {archive}")
    print("Upload this ZIP if the result needs independent review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
