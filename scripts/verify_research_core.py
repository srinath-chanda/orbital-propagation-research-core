"""Run imports and the automated Research Core test suite."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    print("Checking scientific dependencies...")
    for package in ("numpy", "scipy", "matplotlib", "sgp4", "astropy"):
        module = __import__(package)
        print(f"  {package}: {module.__version__}")

    print("\nRunning automated tests...")
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
