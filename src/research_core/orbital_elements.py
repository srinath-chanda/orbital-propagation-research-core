"""Classical orbital-element and Cartesian-state conversions."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .data_models import ClassicalElements

_TWO_PI = 2.0 * math.pi


def wrap_to_2pi(angle_rad: float) -> float:
    """Wrap an angle to [0, 2π), collapsing round-off near 2π to zero."""
    wrapped = float(angle_rad % _TWO_PI)
    if math.isclose(wrapped, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return 0.0
    if math.isclose(wrapped, _TWO_PI, rel_tol=0.0, abs_tol=1e-12):
        return 0.0
    return wrapped


def elements_from_config(initial_state: dict[str, Any]) -> ClassicalElements:
    """Create a ClassicalElements object from a validated configuration section."""
    return ClassicalElements(
        semi_major_axis_km=float(initial_state["semi_major_axis_km"]),
        eccentricity=float(initial_state["eccentricity"]),
        inclination_rad=math.radians(float(initial_state["inclination_deg"])),
        raan_rad=math.radians(float(initial_state["raan_deg"])),
        argument_of_perigee_rad=math.radians(
            float(initial_state["argument_of_perigee_deg"])
        ),
        true_anomaly_rad=math.radians(float(initial_state["true_anomaly_deg"])),
    )


def perifocal_to_inertial_matrix(
    raan_rad: float,
    inclination_rad: float,
    argument_of_perigee_rad: float,
) -> np.ndarray:
    """Return the standard perifocal-to-inertial rotation matrix."""
    cos_o = math.cos(raan_rad)
    sin_o = math.sin(raan_rad)
    cos_i = math.cos(inclination_rad)
    sin_i = math.sin(inclination_rad)
    cos_w = math.cos(argument_of_perigee_rad)
    sin_w = math.sin(argument_of_perigee_rad)

    return np.array(
        [
            [
                cos_o * cos_w - sin_o * sin_w * cos_i,
                -cos_o * sin_w - sin_o * cos_w * cos_i,
                sin_o * sin_i,
            ],
            [
                sin_o * cos_w + cos_o * sin_w * cos_i,
                -sin_o * sin_w + cos_o * cos_w * cos_i,
                -cos_o * sin_i,
            ],
            [sin_w * sin_i, cos_w * sin_i, cos_i],
        ],
        dtype=float,
    )


def elements_to_cartesian(
    elements: ClassicalElements,
    gravitational_parameter_km3_s2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert elliptical osculating classical elements to Cartesian state."""
    mu = float(gravitational_parameter_km3_s2)
    a = float(elements.semi_major_axis_km)
    e = float(elements.eccentricity)
    nu = float(elements.true_anomaly_rad)

    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("Gravitational parameter must be positive and finite.")
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError("Semi-major axis must be positive and finite.")
    if not (0.0 <= e < 1.0):
        raise ValueError("This release supports elliptical orbits with 0 <= e < 1.")

    semi_latus_rectum = a * (1.0 - e * e)
    denominator = 1.0 + e * math.cos(nu)
    if denominator <= 0.0:
        raise ValueError("Invalid orbital geometry for the selected true anomaly.")

    radius_km = semi_latus_rectum / denominator
    position_perifocal = np.array(
        [radius_km * math.cos(nu), radius_km * math.sin(nu), 0.0],
        dtype=float,
    )
    speed_factor = math.sqrt(mu / semi_latus_rectum)
    velocity_perifocal = speed_factor * np.array(
        [-math.sin(nu), e + math.cos(nu), 0.0],
        dtype=float,
    )

    rotation = perifocal_to_inertial_matrix(
        elements.raan_rad,
        elements.inclination_rad,
        elements.argument_of_perigee_rad,
    )
    return rotation @ position_perifocal, rotation @ velocity_perifocal


def cartesian_to_elements(
    position_km: np.ndarray,
    velocity_km_s: np.ndarray,
    gravitational_parameter_km3_s2: float,
    *,
    singularity_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Convert a Cartesian state to osculating classical elements.

    The returned mapping explicitly identifies circular/equatorial singularities.
    """
    r = np.asarray(position_km, dtype=float)
    v = np.asarray(velocity_km_s, dtype=float)
    mu = float(gravitational_parameter_km3_s2)
    if r.shape != (3,) or v.shape != (3,):
        raise ValueError("Position and velocity must each have shape (3,).")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)):
        raise ValueError("Cartesian state contains non-finite values.")
    if mu <= 0.0 or not np.isfinite(mu):
        raise ValueError("Gravitational parameter must be positive and finite.")

    r_mag = float(np.linalg.norm(r))
    v_mag = float(np.linalg.norm(v))
    if r_mag <= 0.0:
        raise ValueError("Position magnitude must be non-zero.")

    h_vec = np.cross(r, v)
    h_mag = float(np.linalg.norm(h_vec))
    if h_mag <= 0.0:
        raise ValueError("Angular-momentum magnitude must be non-zero.")

    k_hat = np.array([0.0, 0.0, 1.0])
    node_vec = np.cross(k_hat, h_vec)
    node_mag = float(np.linalg.norm(node_vec))
    eccentricity_vec = np.cross(v, h_vec) / mu - r / r_mag
    eccentricity = float(np.linalg.norm(eccentricity_vec))
    specific_energy = 0.5 * v_mag * v_mag - mu / r_mag
    if abs(specific_energy) <= np.finfo(float).eps:
        semi_major_axis = math.inf
    else:
        semi_major_axis = -mu / (2.0 * specific_energy)

    inclination = math.acos(float(np.clip(h_vec[2] / h_mag, -1.0, 1.0)))
    circular = eccentricity < singularity_tolerance
    equatorial = node_mag < singularity_tolerance

    raan: float | None = None
    argument_of_perigee: float | None = None
    true_anomaly: float | None = None
    argument_of_latitude: float | None = None
    true_longitude: float | None = None
    longitude_of_periapsis: float | None = None

    if not equatorial:
        raan = wrap_to_2pi(math.atan2(node_vec[1], node_vec[0]))

    if not circular and not equatorial:
        sin_argument = float(
            np.dot(np.cross(node_vec, eccentricity_vec), h_vec)
            / (node_mag * eccentricity * h_mag)
        )
        cos_argument = float(
            np.dot(node_vec, eccentricity_vec) / (node_mag * eccentricity)
        )
        argument_of_perigee = wrap_to_2pi(math.atan2(sin_argument, cos_argument))

    if not circular:
        sin_true = float(
            np.dot(np.cross(eccentricity_vec, r), h_vec)
            / (eccentricity * r_mag * h_mag)
        )
        cos_true = float(np.dot(eccentricity_vec, r) / (eccentricity * r_mag))
        true_anomaly = wrap_to_2pi(math.atan2(sin_true, cos_true))

    if circular and not equatorial:
        sin_u = float(
            np.dot(np.cross(node_vec, r), h_vec) / (node_mag * r_mag * h_mag)
        )
        cos_u = float(np.dot(node_vec, r) / (node_mag * r_mag))
        argument_of_latitude = wrap_to_2pi(math.atan2(sin_u, cos_u))

    if circular and equatorial:
        true_longitude = wrap_to_2pi(math.atan2(r[1], r[0]))

    if not circular and equatorial:
        longitude_of_periapsis = wrap_to_2pi(
            math.atan2(eccentricity_vec[1], eccentricity_vec[0])
        )

    def degrees_or_none(value: float | None) -> float | None:
        return None if value is None else float(math.degrees(value) % 360.0)

    return {
        "semi_major_axis_km": float(semi_major_axis),
        "eccentricity": eccentricity,
        "inclination_deg": float(math.degrees(inclination)),
        "raan_deg": degrees_or_none(raan),
        "argument_of_perigee_deg": degrees_or_none(argument_of_perigee),
        "true_anomaly_deg": degrees_or_none(true_anomaly),
        "argument_of_latitude_deg": degrees_or_none(argument_of_latitude),
        "true_longitude_deg": degrees_or_none(true_longitude),
        "longitude_of_periapsis_deg": degrees_or_none(longitude_of_periapsis),
        "specific_orbital_energy_km2_s2": float(specific_energy),
        "angular_momentum_vector_km2_s": h_vec.tolist(),
        "angular_momentum_magnitude_km2_s": h_mag,
        "eccentricity_vector": eccentricity_vec.tolist(),
        "is_circular": circular,
        "is_equatorial": equatorial,
        "element_type": "osculating",
    }


def true_to_eccentric_anomaly(true_anomaly_rad: float, eccentricity: float) -> float:
    """Convert true anomaly to eccentric anomaly for an elliptical orbit."""
    e = float(eccentricity)
    nu = float(true_anomaly_rad)
    if not (0.0 <= e < 1.0):
        raise ValueError("Eccentricity must satisfy 0 <= e < 1.")
    if e == 0.0:
        return wrap_to_2pi(nu)
    return wrap_to_2pi(
        2.0
        * math.atan2(
            math.sqrt(1.0 - e) * math.sin(nu / 2.0),
            math.sqrt(1.0 + e) * math.cos(nu / 2.0),
        )
    )


def eccentric_to_true_anomaly(eccentric_anomaly_rad: float, eccentricity: float) -> float:
    """Convert eccentric anomaly to true anomaly for an elliptical orbit."""
    e = float(eccentricity)
    eccentric_anomaly = float(eccentric_anomaly_rad)
    if not (0.0 <= e < 1.0):
        raise ValueError("Eccentricity must satisfy 0 <= e < 1.")
    if e == 0.0:
        return wrap_to_2pi(eccentric_anomaly)
    return wrap_to_2pi(
        2.0
        * math.atan2(
            math.sqrt(1.0 + e) * math.sin(eccentric_anomaly / 2.0),
            math.sqrt(1.0 - e) * math.cos(eccentric_anomaly / 2.0),
        )
    )


def solve_kepler_elliptic(
    mean_anomaly_rad: float,
    eccentricity: float,
    *,
    tolerance: float = 1e-13,
    maximum_iterations: int = 50,
) -> float:
    """Solve M = E - e sin(E) using safeguarded Newton iteration."""
    e = float(eccentricity)
    if not (0.0 <= e < 1.0):
        raise ValueError("Eccentricity must satisfy 0 <= e < 1.")
    mean_anomaly = wrap_to_2pi(float(mean_anomaly_rad))
    if e == 0.0:
        return mean_anomaly

    eccentric_anomaly = mean_anomaly if e < 0.8 else math.pi
    for _ in range(maximum_iterations):
        residual = eccentric_anomaly - e * math.sin(eccentric_anomaly) - mean_anomaly
        derivative = 1.0 - e * math.cos(eccentric_anomaly)
        correction = residual / derivative
        eccentric_anomaly -= correction
        if abs(correction) <= tolerance:
            return wrap_to_2pi(eccentric_anomaly)
    raise RuntimeError("Kepler equation did not converge within the iteration limit.")
