"""Package the latest Paper 1 consolidation result for review or archiving."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_core.paper1_results_consolidation import CONSOLIDATION_ID, verify_result_directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, help="Specific consolidation result directory.")
    args = parser.parse_args()
    if args.result_dir is None:
        base = PROJECT_ROOT / "results" / CONSOLIDATION_ID
        candidates = sorted(path.parent for path in base.glob("*/RUN_MANIFEST.json"))
        if not candidates:
            print("ERROR: no consolidation result was found")
            return 1
        result_dir = candidates[-1]
    else:
        result_dir = args.result_dir.resolve()
    errors = verify_result_directory(result_dir)
    if errors:
        print("ERROR: the result failed integrity verification; ZIP was not created.")
        for error in errors:
            print(f"  - {error}")
        return 1
    output = PROJECT_ROOT / "RESEARCH_CORE_PAPER1_CONSOLIDATED_RESULTS.zip"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(result_dir.iterdir()):
            if path.is_file():
                archive.write(path, arcname=f"PAPER1_CONSOLIDATED_RESULTS/{path.name}")
    print("PAPER 1 CONSOLIDATED RESULTS PACKAGE CREATED")
    print(f"ZIP: {output}")
    print("Upload this ZIP for final manuscript and GitHub preparation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

