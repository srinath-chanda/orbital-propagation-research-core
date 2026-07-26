"""Package raw and aggregate Research Core 1B GMAT evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.gmat_multicase import package_gmat_multicase_results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a verified ZIP containing Research Core 1B GMAT evidence."
    )
    parser.add_argument(
        "matrix",
        nargs="?",
        default="configs/gmat_1b_multicase_matrix.json",
    )
    parser.add_argument(
        "--output",
        default="RESEARCH_CORE_1B_GMAT_RESULTS.zip",
    )
    args = parser.parse_args()
    matrix = Path(args.matrix)
    output = Path(args.output)
    if not matrix.is_absolute():
        matrix = (Path.cwd() / matrix).resolve()
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    try:
        archive = package_gmat_multicase_results(
            matrix,
            project_root=PROJECT_ROOT,
            output_path=output,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("RESEARCH CORE 1B GMAT RESULTS PACKAGE CREATED")
    print(f"ZIP file: {archive}")
    print("Next: upload this ZIP for review and explainable result interpretation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
