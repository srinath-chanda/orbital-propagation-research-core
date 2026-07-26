#!/usr/bin/env python3
"""Verify the portable public-release tree and its SHA-256 manifest."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"

REQUIRED_PATHS = (
    "README.md",
    "LICENSE",
    "COPYRIGHT.md",
    "CITATION.cff",
    "THIRD_PARTY_NOTICES.md",
    "RELEASE_NOTES.md",
    "pyproject.toml",
    "requirements.txt",
    "configs/paper1_baseline_closure.json",
    "configs/paper1_production_matrix.json",
    "data/tle/iss_25544_2026-07-18_celestrak.tle",
    "docs/INSTALLATION.md",
    "docs/REPRODUCIBILITY.md",
    "docs/VALIDATION_SCOPE.md",
    "docs/GMAT_VALIDATION.md",
    "paper/manuscript/PAPER1_MANUSCRIPT_PREPRINT.pdf",
    "paper/assets/figures/figure_02_raan_rate_comparison.png",
    "paper/assets/tables/table_05_raan_rates.csv",
    "src/research_core/__init__.py",
    "tests/test_two_body.py",
    "MANIFEST.sha256",
)

FORBIDDEN_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "venv",
}

TEXT_SUFFIXES = {
    ".cff",
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".script",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

PRIVATE_PATH_PATTERNS = (
    re.compile(r"C:[/\\]Users[/\\]", re.IGNORECASE),
    re.compile("/" + "Users/"),
    re.compile("/workspace/" + "scratch/"),
    re.compile("/home/" + r"[^/\s]+/"),
)

SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{50,}\b"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_files() -> list[Path]:
    return sorted(path for path in ROOT.rglob("*") if path.is_file())


def check_required(errors: list[str]) -> None:
    for relative in REQUIRED_PATHS:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")


def check_tree(files: list[Path], errors: list[str]) -> None:
    for path in files:
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_DIRECTORY_NAMES for part in relative.parts[:-1]):
            errors.append(f"forbidden directory in release: {relative}")
        if path.suffix.lower() in {".pyc", ".pyo", ".zip"}:
            errors.append(f"forbidden generated/archive file: {relative}")
        if path.name.endswith(".egg-info"):
            errors.append(f"forbidden build metadata: {relative}")
        if path.stat().st_size >= 95 * 1024 * 1024:
            errors.append(f"file exceeds 95 MiB GitHub safety limit: {relative}")


def check_text(files: list[Path], errors: list[str]) -> None:
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="strict")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(ROOT)
        for pattern in PRIVATE_PATH_PATTERNS:
            if pattern.search(content):
                errors.append(f"private machine path found in: {relative}")
                break
        for pattern in SECRET_PATTERNS:
            if pattern.search(content):
                errors.append(f"possible secret found in: {relative}")
                break


def read_manifest(errors: list[str]) -> dict[str, str]:
    if not MANIFEST.is_file():
        return {}
    entries: dict[str, str] = {}
    for number, line in enumerate(
        MANIFEST.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", maxsplit=1)
        except ValueError:
            errors.append(f"invalid manifest line {number}")
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"invalid manifest digest on line {number}")
            continue
        if relative in entries:
            errors.append(f"duplicate manifest entry: {relative}")
            continue
        entries[relative] = digest
    return entries


def check_manifest(files: list[Path], errors: list[str]) -> int:
    entries = read_manifest(errors)
    if not entries:
        errors.append("manifest has no entries")
        return 0
    expected_files = {
        path.relative_to(ROOT).as_posix()
        for path in files
        if path != MANIFEST
    }
    registered_files = set(entries)
    for relative in sorted(expected_files - registered_files):
        errors.append(f"file missing from manifest: {relative}")
    for relative in sorted(registered_files - expected_files):
        errors.append(f"manifest references missing file: {relative}")
    for relative in sorted(expected_files & registered_files):
        if sha256(ROOT / relative) != entries[relative]:
            errors.append(f"manifest hash mismatch: {relative}")
    return len(entries)


def main() -> int:
    errors: list[str] = []
    files = repository_files()
    check_required(errors)
    check_tree(files, errors)
    check_text(files, errors)
    manifest_count = check_manifest(files, errors)

    print("=" * 72)
    print("ORBITAL PROPAGATION RESEARCH CORE - PUBLIC RELEASE VERIFICATION")
    print("=" * 72)
    print(f"Repository files : {len(files)}")
    print(f"Manifest entries : {manifest_count}")
    if errors:
        print(f"Status           : FAILED ({len(errors)} issue(s))")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Private-path scan: passed")
    print("Secret scan      : passed")
    print("Manifest hashes  : passed")
    print("Status           : PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
