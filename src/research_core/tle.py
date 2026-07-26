"""Frozen TLE loading, validation, and provenance utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sgp4.api import Satrec
from sgp4.conveniences import sat_epoch_datetime


@dataclass(frozen=True)
class FrozenTLE:
    """One immutable three-line element snapshot and its provenance."""

    name: str
    line1: str
    line2: str
    file_path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    file_sha256: str
    epoch_utc: str
    norad_catalog_number: int


def tle_checksum_value(line: str) -> int:
    """Calculate the standard TLE checksum over columns 1-68."""
    if len(line) < 69:
        raise ValueError("A TLE line must contain at least 69 characters.")
    total = 0
    for character in line[:68]:
        if character.isdigit():
            total += int(character)
        elif character == "-":
            total += 1
    return total % 10


def tle_checksum_is_valid(line: str) -> bool:
    """Return whether the final checksum digit matches the line contents."""
    return len(line) >= 69 and line[68].isdigit() and tle_checksum_value(line) == int(line[68])


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def load_frozen_tle(
    tle_file: str | Path,
    metadata_file: str | Path,
    *,
    expected_catalog_number: int | None = None,
) -> FrozenTLE:
    """Load a frozen TLE and verify line checksums and metadata checksum."""
    tle_path = Path(tle_file).expanduser().resolve()
    metadata_path = Path(metadata_file).expanduser().resolve()
    if not tle_path.is_file():
        raise FileNotFoundError(f"TLE file not found: {tle_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"TLE metadata file not found: {metadata_path}")

    lines = [line.rstrip("\r\n") for line in tle_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 3:
        raise ValueError("The frozen TLE file must contain name, line 1, and line 2.")
    name, line1, line2 = lines
    if not line1.startswith("1 ") or not line2.startswith("2 "):
        raise ValueError("The frozen TLE does not contain valid line-number prefixes.")
    if not tle_checksum_is_valid(line1):
        raise ValueError("TLE line 1 checksum is invalid.")
    if not tle_checksum_is_valid(line2):
        raise ValueError("TLE line 2 checksum is invalid.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("TLE metadata must be a JSON object.")
    file_sha = sha256_file(tle_path)
    expected_sha = str(metadata.get("tle_sha256", "")).lower()
    if not expected_sha or file_sha.lower() != expected_sha:
        raise ValueError("Frozen TLE file checksum does not match its metadata.")

    satellite = Satrec.twoline2rv(line1, line2)
    catalog_number = int(satellite.satnum)
    if expected_catalog_number is not None and catalog_number != int(expected_catalog_number):
        raise ValueError(
            f"Expected NORAD catalog {expected_catalog_number}, found {catalog_number}."
        )
    metadata_catalog = metadata.get("norad_catalog_number")
    if metadata_catalog is not None and int(metadata_catalog) != catalog_number:
        raise ValueError("TLE catalog number does not match metadata.")

    epoch_utc = _format_utc(sat_epoch_datetime(satellite))
    metadata_epoch = metadata.get("tle_epoch_utc")
    if metadata_epoch and str(metadata_epoch) != epoch_utc:
        raise ValueError(
            f"TLE epoch {epoch_utc} does not match metadata epoch {metadata_epoch}."
        )

    return FrozenTLE(
        name=name,
        line1=line1,
        line2=line2,
        file_path=tle_path,
        metadata_path=metadata_path,
        metadata=metadata,
        file_sha256=file_sha,
        epoch_utc=epoch_utc,
        norad_catalog_number=catalog_number,
    )


def tle_parameter_summary(tle: FrozenTLE) -> dict[str, Any]:
    """Return parsed SGP4/TLE parameters in readable units."""
    satellite = Satrec.twoline2rv(tle.line1, tle.line2)
    import math

    mean_motion_rev_day = float(satellite.no_kozai) * 1440.0 / (2.0 * math.pi)
    return {
        "object_name": tle.name,
        "norad_catalog_number": int(satellite.satnum),
        "classification": tle.line1[7],
        "international_designator_raw": tle.line1[9:17].strip(),
        "tle_epoch_utc": tle.epoch_utc,
        "bstar_inverse_earth_radii": float(satellite.bstar),
        "inclination_deg": math.degrees(float(satellite.inclo)),
        "raan_deg": math.degrees(float(satellite.nodeo)) % 360.0,
        "eccentricity": float(satellite.ecco),
        "argument_of_perigee_deg": math.degrees(float(satellite.argpo)) % 360.0,
        "mean_anomaly_deg": math.degrees(float(satellite.mo)) % 360.0,
        "mean_motion_rev_day": mean_motion_rev_day,
        "element_set_number": int(satellite.elnum),
        "revolution_number_at_epoch": int(satellite.revnum),
        "tle_line1_checksum_valid": tle_checksum_is_valid(tle.line1),
        "tle_line2_checksum_valid": tle_checksum_is_valid(tle.line2),
        "tle_file_sha256": tle.file_sha256,
        "tle_file": str(tle.file_path),
        "metadata_file": str(tle.metadata_path),
        "source_organisation": tle.metadata.get("source_organisation"),
        "source_query": tle.metadata.get("source_query"),
        "retrieved_utc": tle.metadata.get("retrieved_utc"),
        "snapshot_policy": tle.metadata.get("snapshot_policy"),
    }
