"""Package raw and processed Research Core 1D.2 evidence."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.gmat_gravity_multicase import package_gravity_multicase_results


def main() -> int:
    try:
        archive = package_gravity_multicase_results(
            PROJECT_ROOT / "configs/gmat_gravity_1d2_multicase.json",
            project_root=PROJECT_ROOT,
            output_path=PROJECT_ROOT / "RESEARCH_CORE_1D2_GMAT_RESULTS.zip",
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("RESEARCH CORE 1D.2 RESULTS PACKAGE CREATED")
    print(f"ZIP: {archive}")
    print("Upload this ZIP for independent review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
