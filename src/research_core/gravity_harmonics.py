"""Normalized spherical-harmonic gravity for the 1D.0 GMAT ladder."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import lpmv


_FLOAT_PATTERN = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CofGravityField:
    """A normalized GMAT ``.cof`` gravity field in kilometre units."""

    source_path: Path
    source_sha256: str
    maximum_degree: int
    maximum_order: int
    normalized: bool
    gravitational_parameter_km3_s2: float
    reference_radius_km: float
    cosine: np.ndarray
    sine: np.ndarray

    @classmethod
    def from_file(cls, path: str | Path) -> "CofGravityField":
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"GMAT gravity file not found: {source}")
        header: tuple[int, int, bool, float, float] | None = None
        rows: dict[tuple[int, int], tuple[float, float]] = {}
        for number, raw_line in enumerate(
            source.read_text(encoding="ascii").splitlines(), start=1
        ):
            line = raw_line.strip()
            if line.startswith("POTFIELD"):
                values = _FLOAT_PATTERN.findall(line[len("POTFIELD") :])
                if len(values) < 6:
                    raise ValueError(f"Malformed POTFIELD header at line {number}.")
                degree = int(float(values[0]))
                order = int(float(values[1]))
                normalized = int(float(values[2])) == 1
                mu = float(values[3].replace("D", "E").replace("d", "e"))
                radius = float(values[4].replace("D", "E").replace("d", "e"))
                # Official GMAT .cof files store SI header units.
                if mu > 1.0e9:
                    mu /= 1.0e9
                if radius > 1.0e5:
                    radius /= 1000.0
                header = (degree, order, normalized, mu, radius)
            elif line.startswith("RECOEF"):
                values = _FLOAT_PATTERN.findall(line[len("RECOEF") :])
                if len(values) < 3:
                    raise ValueError(f"Malformed RECOEF row at line {number}.")
                degree = int(float(values[0]))
                order = int(float(values[1]))
                cosine = float(values[2].replace("D", "E").replace("d", "e"))
                sine = (
                    float(values[3].replace("D", "E").replace("d", "e"))
                    if len(values) >= 4
                    else 0.0
                )
                key = (degree, order)
                if key in rows:
                    raise ValueError(f"Duplicate RECOEF {degree},{order} at line {number}.")
                rows[key] = (cosine, sine)
        if header is None:
            raise ValueError("GMAT gravity file has no POTFIELD header.")
        max_degree, max_order, normalized, mu, radius = header
        if not normalized:
            raise ValueError("Research Core 1D.0 requires normalized .cof coefficients.")
        if max_degree < 2 or max_order < 0 or max_order > max_degree:
            raise ValueError("Invalid gravity-field degree/order in POTFIELD header.")
        cosine = np.zeros((max_degree + 1, max_degree + 1), dtype=float)
        sine = np.zeros_like(cosine)
        for (degree, order), (c_value, s_value) in rows.items():
            if degree > max_degree or order > min(degree, max_order):
                raise ValueError(
                    f"RECOEF {degree},{order} exceeds the POTFIELD limits."
                )
            cosine[degree, order] = c_value
            sine[degree, order] = s_value
        if not rows or (2, 0) not in rows:
            raise ValueError("Gravity file does not contain the required C20 coefficient.")
        return cls(
            source_path=source,
            source_sha256=_sha256(source),
            maximum_degree=max_degree,
            maximum_order=max_order,
            normalized=normalized,
            gravitational_parameter_km3_s2=mu,
            reference_radius_km=radius,
            cosine=cosine,
            sine=sine,
        )

    @property
    def j2(self) -> float:
        """Return J2 implied by the fully normalized C20 coefficient."""
        return float(-math.sqrt(5.0) * self.cosine[2, 0])


def _normalization(degree: int, order: int) -> float:
    multiplier = 1.0 if order == 0 else 2.0
    logarithm = (
        math.log(multiplier * (2 * degree + 1))
        + math.lgamma(degree - order + 1)
        - math.lgamma(degree + order + 1)
    )
    return math.exp(0.5 * logarithm)


def gravity_acceleration_fixed_km_s2(
    position_fixed_km: np.ndarray,
    field: CofGravityField,
    *,
    degree: int,
    order: int,
) -> np.ndarray:
    """Evaluate full gravity acceleration in the body-fixed frame.

    The normalized coefficients use the no-Condon-Shortley geodesy
    convention. Degree/order zero returns the exact point-mass field.
    """
    maximum_degree = int(degree)
    maximum_order = int(order)
    if maximum_degree < 0 or maximum_degree > field.maximum_degree:
        raise ValueError("Requested gravity degree is outside the field limits.")
    if maximum_order < 0 or maximum_order > min(maximum_degree, field.maximum_order):
        raise ValueError("Requested gravity order is outside the field limits.")
    position = np.asarray(position_fixed_km, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("position_fixed_km must contain three finite values.")
    radius = float(np.linalg.norm(position))
    if radius <= field.reference_radius_km * 0.1:
        raise ValueError("Position radius is too small for Earth gravity evaluation.")
    x, y, z = position
    longitude = math.atan2(y, x)
    sin_latitude = float(np.clip(z / radius, -1.0, 1.0))
    cos_latitude = math.hypot(x, y) / radius
    if cos_latitude < 1.0e-10 and maximum_order > 0:
        raise ValueError("Tesseral acceleration is undefined numerically at the pole.")

    radial_sum = 1.0
    latitude_sum = 0.0
    longitude_sum = 0.0
    for n in range(2, maximum_degree + 1):
        radius_power = (field.reference_radius_km / radius) ** n
        for m in range(0, min(n, maximum_order) + 1):
            norm = _normalization(n, m)
            # scipy includes (-1)^m; geodesy harmonics do not.
            p_nm = ((-1.0) ** m) * norm * float(lpmv(m, n, sin_latitude))
            if n - 1 >= m:
                p_previous = ((-1.0) ** m) * norm * float(
                    lpmv(m, n - 1, sin_latitude)
                )
            else:
                p_previous = 0.0
            denominator = sin_latitude * sin_latitude - 1.0
            derivative_x = (
                n * sin_latitude * p_nm - (n + m) * p_previous
            ) / denominator
            derivative_latitude = cos_latitude * derivative_x
            angle = m * longitude
            cosine_angle = math.cos(angle)
            sine_angle = math.sin(angle)
            c_nm = float(field.cosine[n, m])
            s_nm = float(field.sine[n, m])
            harmonic = c_nm * cosine_angle + s_nm * sine_angle
            longitude_harmonic = m * (-c_nm * sine_angle + s_nm * cosine_angle)
            term = radius_power * p_nm * harmonic
            radial_sum += (n + 1) * term
            latitude_sum += radius_power * derivative_latitude * harmonic
            longitude_sum += radius_power * p_nm * longitude_harmonic

    scale = field.gravitational_parameter_km3_s2 / (radius * radius)
    radial_acceleration = -scale * radial_sum
    latitude_acceleration = scale * latitude_sum
    longitude_acceleration = (
        scale * longitude_sum / cos_latitude if maximum_order > 0 else 0.0
    )
    cos_longitude = math.cos(longitude)
    sin_longitude = math.sin(longitude)
    radial_direction = np.asarray(
        [cos_latitude * cos_longitude, cos_latitude * sin_longitude, sin_latitude]
    )
    latitude_direction = np.asarray(
        [-sin_latitude * cos_longitude, -sin_latitude * sin_longitude, cos_latitude]
    )
    longitude_direction = np.asarray([-sin_longitude, cos_longitude, 0.0])
    return (
        radial_acceleration * radial_direction
        + latitude_acceleration * latitude_direction
        + longitude_acceleration * longitude_direction
    )


def gravity_acceleration_inertial_km_s2(
    position_inertial_km: np.ndarray,
    inertial_to_fixed: np.ndarray,
    field: CofGravityField,
    *,
    degree: int,
    order: int,
) -> np.ndarray:
    """Rotate a state to body-fixed, evaluate gravity, and rotate it back."""
    rotation = np.asarray(inertial_to_fixed, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("inertial_to_fixed must be a finite 3x3 rotation.")
    fixed_position = rotation @ np.asarray(position_inertial_km, dtype=float)
    fixed_acceleration = gravity_acceleration_fixed_km_s2(
        fixed_position, field, degree=degree, order=order
    )
    return rotation.T @ fixed_acceleration
