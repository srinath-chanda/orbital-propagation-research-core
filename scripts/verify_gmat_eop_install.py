"""Verify that a local GMAT installation uses the frozen R2026a EOP file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.gmat_eop import GMAT_R2026A_EOP_SHA256
from research_core.gmat_eop_independent import verify_gmat_eop_install


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify GMAT R2026a's eopc04_08.62-now checksum."
    )
    parser.add_argument("gmat_path", help="GMAT root folder or exact EOP file path")
    args = parser.parse_args()
    try:
        result = verify_gmat_eop_install(
            args.gmat_path, expected_sha256=GMAT_R2026A_EOP_SHA256
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("GMAT EOP FILE CHECK")
    print(f"File        : {result['path']}")
    print(f"Size        : {result['size_bytes']} bytes")
    print(f"Raw SHA-256 : {result['sha256']}")
    print(f"LF SHA-256  : {result['canonical_lf_sha256']}")
    print(f"Expected    : {result['expected_sha256']}")
    print(f"Byte exact  : {result['byte_exact_match']}")
    print(f"CRLF/LF only: {result['line_ending_equivalent']}")
    print(f"Content pass: {result['matches_gmat_r2026a_tag']}")
    if not result["matches_gmat_r2026a_tag"]:
        print("STOP: do not run the 1C.2 GMAT matrix with this EOP file.")
        return 2
    if result["line_ending_equivalent"]:
        print("PASS: scientific text is exact; only Windows line endings differ.")
    else:
        print("PASS: this GMAT installation byte-matches the frozen R2026a EOP evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
