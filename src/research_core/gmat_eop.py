"""Frozen GMAT R2026a Earth-orientation data and rotation helpers.

The implementation follows the Earth path in GMAT R2026a ``BodyFixedAxes``:

``PM * ST * NUT * PREC``

Only the third row of that inertial-to-body-fixed rotation is needed for a
degree-2/order-0 (axisymmetric) gravity field.  The bundled EOP file is the
exact file from the official NASA GMAT R2026a source tag; it is intentionally
frozen so the saved GMAT evidence remains reproducible.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import erfa
import numpy as np
from astropy.time import Time, TimeDelta


GMAT_R2026A_EOP_SHA256 = (
    "52c95d6871066e892463328424ca34f5e9cedde92c747465a40f7cd0ecd86ed3"
)
GMAT_R2026A_EOP_FIRST_MJD = 37665.0
GMAT_R2026A_EOP_LAST_MJD = 61297.0
GMAT_VALIDATED_EOP_MODEL = "gmat_r2026a_eop_full"
GMAT_VALIDATED_EOP_CLOSURE = "CLOSURE-GMAT-EOP-1C3-001"
SECONDS_PER_DAY = 86400.0
RADIANS_PER_ARCSECOND = np.pi / (180.0 * 3600.0)

GMAT_EOP_POLE_MODELS = (
    "gmat_r2026a_eop_x_only",
    "gmat_r2026a_eop_y_only",
    "gmat_r2026a_eop_full",
)

GMAT_EOP_MODEL_DESCRIPTIONS = {
    "gmat_r2026a_eop_x_only": "GMAT R2026a EOP x polar-motion ablation",
    "gmat_r2026a_eop_y_only": "GMAT R2026a EOP y polar-motion ablation",
    "gmat_r2026a_eop_full": "Validated GMAT R2026a full x/y EOP/polar motion",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class GmatEopSample:
    """One GMAT-compatible EOP sample at a requested UTC epoch."""

    mjd_utc: float
    x_arcsec: float
    y_arcsec: float
    ut1_utc_seconds: float
    lod_seconds: float
    coverage_status: str
    uncertainty_status: str
    left_source_mjd: float
    right_source_mjd: float


@dataclass(frozen=True)
class GmatEopDataset:
    """Parsed old-format IERS C04 data as consumed by GMAT R2026a."""

    source_path: Path
    source_sha256: str
    mjd_utc: np.ndarray
    tai_mjd: np.ndarray
    x_arcsec: np.ndarray
    y_arcsec: np.ndarray
    ut1_utc_seconds: np.ndarray
    lod_seconds: np.ndarray
    uncertainty_placeholder: np.ndarray

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> "GmatEopDataset":
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"GMAT EOP file not found: {source}")
        actual_sha256 = _sha256(source)
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ValueError(
                "GMAT EOP checksum mismatch: expected "
                f"{expected_sha256}, found {actual_sha256}."
            )

        rows: list[tuple[float, ...]] = []
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            fields = line.split()
            if not fields or len(fields[0]) != 4 or not fields[0].isdigit():
                continue
            if len(fields) < 14:
                raise ValueError(
                    f"Malformed GMAT EOP data row at line {line_number}."
                )
            try:
                # Old C04 columns: year, month, day, MJD, x, y, UT1-UTC,
                # LOD, dPsi, dEps, x_err, y_err, UT1_err, LOD_err, ...
                rows.append(
                    (
                        float(fields[3]),
                        float(fields[4]),
                        float(fields[5]),
                        float(fields[6]),
                        float(fields[7]),
                        float(fields[10]),
                        float(fields[11]),
                        float(fields[12]),
                        float(fields[13]),
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric GMAT EOP value at line {line_number}."
                ) from exc
        if len(rows) < 2:
            raise ValueError("The GMAT EOP file contains fewer than two data rows.")

        values = np.asarray(rows, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("The GMAT EOP file contains non-finite values.")
        mjd = values[:, 0]
        if np.any(np.diff(mjd) <= 0.0):
            raise ValueError("GMAT EOP MJD rows must be strictly increasing.")
        # GMAT interpolates UT1-UTC on a TAI grid to remain continuous through
        # leap seconds.  Astropy supplies the same UTC-to-TAI transformation.
        tai_mjd = np.asarray(Time(mjd, format="mjd", scale="utc").tai.mjd)
        placeholders = np.any(values[:, 5:9] >= 0.99, axis=1)
        return cls(
            source_path=source,
            source_sha256=actual_sha256,
            mjd_utc=mjd,
            tai_mjd=tai_mjd,
            x_arcsec=values[:, 1],
            y_arcsec=values[:, 2],
            ut1_utc_seconds=values[:, 3],
            lod_seconds=values[:, 4],
            uncertainty_placeholder=placeholders,
        )

    @property
    def first_mjd_utc(self) -> float:
        return float(self.mjd_utc[0])

    @property
    def last_mjd_utc(self) -> float:
        return float(self.mjd_utc[-1])

    @property
    def row_count(self) -> int:
        return int(self.mjd_utc.size)

    def sample(self, evaluation_time_utc: Time) -> GmatEopSample:
        """Sample with GMAT R2026a interpolation and boundary behavior.

        GMAT linearly interpolates x and y on the UTC grid, linearly
        interpolates UT1-UTC on a TAI grid, takes LOD from the left row, and
        clamps requests outside the data range to the nearest endpoint.
        """
        if evaluation_time_utc.scale != "utc":
            evaluation_time_utc = evaluation_time_utc.utc
        query_mjd = float(evaluation_time_utc.mjd)
        if not np.isfinite(query_mjd):
            raise ValueError("The EOP evaluation epoch must be finite.")

        if query_mjd <= self.first_mjd_utc:
            index = 0
            coverage = (
                "first_source_row"
                if query_mjd == self.first_mjd_utc
                else "clamped_before_source_range"
            )
            return self._endpoint_sample(query_mjd, index, coverage)
        if query_mjd >= self.last_mjd_utc:
            index = self.row_count - 1
            coverage = (
                "last_source_row"
                if query_mjd == self.last_mjd_utc
                else "clamped_after_source_range"
            )
            return self._endpoint_sample(query_mjd, index, coverage)

        exact = int(np.searchsorted(self.mjd_utc, query_mjd, side="left"))
        if exact < self.row_count and abs(self.mjd_utc[exact] - query_mjd) <= 1.0e-12:
            return self._endpoint_sample(query_mjd, exact, "exact_source_row")

        left = int(np.searchsorted(self.mjd_utc, query_mjd, side="right") - 1)
        right = left + 1
        utc_ratio = (query_mjd - self.mjd_utc[left]) / (
            self.mjd_utc[right] - self.mjd_utc[left]
        )
        query_tai_mjd = float(evaluation_time_utc.tai.mjd)
        tai_span = float(self.tai_mjd[right] - self.tai_mjd[left])
        tai_ratio = (query_tai_mjd - self.tai_mjd[left]) / tai_span
        offset_difference = float(
            self.ut1_utc_seconds[right] - self.ut1_utc_seconds[left]
        )
        leap_error_seconds = (tai_span - 1.0) * SECONDS_PER_DAY
        if abs(leap_error_seconds) > 0.6:
            offset_difference -= round(leap_error_seconds)
        ut1_utc = self.ut1_utc_seconds[left] + tai_ratio * offset_difference
        placeholder = bool(
            self.uncertainty_placeholder[left]
            or self.uncertainty_placeholder[right]
        )
        return GmatEopSample(
            mjd_utc=query_mjd,
            x_arcsec=float(
                self.x_arcsec[left]
                + utc_ratio * (self.x_arcsec[right] - self.x_arcsec[left])
            ),
            y_arcsec=float(
                self.y_arcsec[left]
                + utc_ratio * (self.y_arcsec[right] - self.y_arcsec[left])
            ),
            ut1_utc_seconds=float(ut1_utc),
            lod_seconds=float(self.lod_seconds[left]),
            coverage_status="interpolated_between_source_rows",
            uncertainty_status=(
                "placeholder_uncertainty_in_tagged_file"
                if placeholder
                else "reported_uncertainty_in_tagged_file"
            ),
            left_source_mjd=float(self.mjd_utc[left]),
            right_source_mjd=float(self.mjd_utc[right]),
        )

    def _endpoint_sample(
        self, query_mjd: float, index: int, coverage: str
    ) -> GmatEopSample:
        placeholder = bool(self.uncertainty_placeholder[index])
        source_mjd = float(self.mjd_utc[index])
        return GmatEopSample(
            mjd_utc=query_mjd,
            x_arcsec=float(self.x_arcsec[index]),
            y_arcsec=float(self.y_arcsec[index]),
            ut1_utc_seconds=float(self.ut1_utc_seconds[index]),
            lod_seconds=float(self.lod_seconds[index]),
            coverage_status=coverage,
            uncertainty_status=(
                "placeholder_uncertainty_in_tagged_file"
                if placeholder
                else "reported_uncertainty_in_tagged_file"
            ),
            left_source_mjd=source_mjd,
            right_source_mjd=source_mjd,
        )


def gmat_r2026a_polar_motion_matrix(
    x_arcsec: float,
    y_arcsec: float,
) -> np.ndarray:
    """Return GMAT R2026a's exact polar-motion matrix convention."""
    x = float(x_arcsec)
    y = float(y_arcsec)
    if not np.isfinite(x) or not np.isfinite(y):
        raise ValueError("Polar-motion coordinates must be finite.")
    cos_x = np.cos(-x * RADIANS_PER_ARCSECOND)
    sin_x = np.sin(-x * RADIANS_PER_ARCSECOND)
    cos_y = np.cos(-y * RADIANS_PER_ARCSECOND)
    sin_y = np.sin(-y * RADIANS_PER_ARCSECOND)
    return np.asarray(
        [
            [cos_x, sin_x * sin_y, -sin_x * cos_y],
            [0.0, cos_y, sin_y],
            [sin_x, -cos_x * sin_y, cos_x * cos_y],
        ],
        dtype=float,
    )


def gmat_r2026a_apparent_sidereal_angle(
    evaluation_time_utc: Time,
    ut1_utc_seconds: float,
) -> float:
    """Return GMAT-compatible Greenwich apparent sidereal angle in radians."""
    utc = evaluation_time_utc.utc
    offset = float(ut1_utc_seconds)
    if not np.isfinite(offset):
        raise ValueError("UT1-UTC must be finite.")
    # ERFA GMST82 implements the same IAU-1982/Vallado mean-sidereal model
    # used in GMAT's AxisSystem. EQEQ94 adds IAU-1980 dPsi*cos(epsilon) and
    # the post-1997 lunar-node complementary terms used by GMAT.
    gmst = erfa.gmst82(utc.jd1, utc.jd2 + offset / SECONDS_PER_DAY)
    equation_of_equinoxes = erfa.eqeq94(utc.tt.jd1, utc.tt.jd2)
    return float((gmst + equation_of_equinoxes) % (2.0 * np.pi))


def gmat_r2026a_eop_pole_unit_vector(
    epoch_utc: str,
    elapsed_seconds: float,
    dataset: GmatEopDataset,
    model: str = "gmat_r2026a_eop_full",
) -> np.ndarray:
    """Return the GMAT EOP-corrected J2 symmetry axis in EarthMJ2000Eq."""
    if model not in GMAT_EOP_POLE_MODELS:
        raise ValueError(
            f"Unsupported GMAT EOP pole model {model!r}; expected one of "
            f"{', '.join(GMAT_EOP_POLE_MODELS)}."
        )
    elapsed = float(elapsed_seconds)
    if not np.isfinite(elapsed):
        raise ValueError("elapsed_seconds must be finite.")
    evaluation_time = Time(str(epoch_utc), scale="utc") + TimeDelta(
        elapsed, format="sec"
    )
    sample = dataset.sample(evaluation_time.utc)
    inertial_to_fixed = gmat_r2026a_inertial_to_fixed_matrix(
        evaluation_time.utc,
        sample,
        model=model,
    )
    axis = inertial_to_fixed[2, :].copy()
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("GMAT EOP pole calculation returned an invalid axis.")
    return axis / norm


def gmat_r2026a_inertial_to_fixed_matrix(
    evaluation_time_utc: Time,
    sample: GmatEopSample,
    *,
    model: str = "gmat_r2026a_eop_full",
) -> np.ndarray:
    """Return GMAT R2026a's EarthMJ2000Eq-to-body-fixed rotation.

    Unlike the pole-only helper, this exposes all three body-fixed axes needed
    for tesseral and sectorial gravity.  The validated 1C.3 EOP realization is
    reused without changing the historical degree-2/order-0 path.
    """
    if model not in GMAT_EOP_POLE_MODELS:
        raise ValueError(
            f"Unsupported GMAT EOP pole model {model!r}; expected one of "
            f"{', '.join(GMAT_EOP_POLE_MODELS)}."
        )
    evaluation_time = evaluation_time_utc.utc
    x = sample.x_arcsec if model != "gmat_r2026a_eop_y_only" else 0.0
    y = sample.y_arcsec if model != "gmat_r2026a_eop_x_only" else 0.0
    precession_nutation = np.asarray(
        erfa.pnm80(evaluation_time.tt.jd1, evaluation_time.tt.jd2), dtype=float
    )
    angle = gmat_r2026a_apparent_sidereal_angle(
        evaluation_time.utc, sample.ut1_utc_seconds
    )
    cosine = np.cos(angle)
    sine = np.sin(angle)
    sidereal = np.asarray(
        [[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    rotation = (
        gmat_r2026a_polar_motion_matrix(x, y)
        @ sidereal
        @ precession_nutation
    )
    if not np.all(np.isfinite(rotation)):
        raise RuntimeError("GMAT EOP rotation calculation returned non-finite values.")
    return rotation


def gmat_validated_eop_pole_unit_vector(
    epoch_utc: str,
    elapsed_seconds: float,
    dataset: GmatEopDataset,
) -> np.ndarray:
    """Return the independently validated GMAT-matched J2 pole.

    Research Core 1C.3 adopts the full R2026a x/y EOP realization after the
    preregistered six-case 1C.2 holdout matrix passed all 84 checks.  The
    explicit name separates the adopted model from the historical
    IAU-1976/1980 compatibility path retained for reproducing older results.
    """
    return gmat_r2026a_eop_pole_unit_vector(
        epoch_utc,
        elapsed_seconds,
        dataset,
        model=GMAT_VALIDATED_EOP_MODEL,
    )
