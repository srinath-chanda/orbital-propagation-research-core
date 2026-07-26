"""Environment and checksum metadata for reproducible experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import BUILD_MARKER, RESEARCH_CORE_VERSION
from .constants import ENVIRONMENT_PACKAGES


def utc_now_iso() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(file_path: str | Path) -> str:
    """Calculate the SHA-256 checksum of a file."""
    path = Path(file_path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    """Return installed versions for selected scientific packages."""
    versions: dict[str, str] = {}
    for package_name in ENVIRONMENT_PACKAGES:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = "not installed"
    return versions


def collect_environment_metadata(
    *,
    config_path: str | Path,
    result_directory: str | Path,
    run_started_utc: str,
) -> dict[str, Any]:
    """Collect reproducibility and execution-environment metadata."""
    config = Path(config_path).expanduser().resolve()
    result = Path(result_directory).expanduser().resolve()

    return {
        "research_core_version": RESEARCH_CORE_VERSION,
        "build_marker": BUILD_MARKER,
        "run_started_utc": run_started_utc,
        "metadata_created_utc": utc_now_iso(),
        "configuration_source": str(config),
        "configuration_sha256": sha256_file(config),
        "result_directory": str(result),
        "python": {
            "version": sys.version,
            "version_info": list(sys.version_info[:5]),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "operating_system": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "host": {
            "hostname": socket.gethostname(),
            "cpu_count": os.cpu_count(),
        },
        "packages": package_versions(),
    }


def write_json(data: dict[str, Any], output_path: str | Path) -> None:
    """Write JSON with stable readable formatting."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
