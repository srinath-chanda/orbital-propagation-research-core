"""Project-wide constants for Research Core 1A.0."""

from __future__ import annotations

SUPPORTED_SCHEMA_VERSIONS = {"1A.0", "1A.8.2"}

SUPPORTED_INITIAL_STATE_SOURCES = {
    "classical_elements",
    "cartesian_state",
    "fixed_tle",
}

SUPPORTED_MODELS = {
    "analytical_two_body",
    "numerical_two_body",
    "numerical_j2",
    "numerical_j2_gmat_matched",
    "numerical_j2_drag",
    "sgp4",
}

SUPPORTED_INTEGRATORS = {
    "DOP853",
    "RK45",
    "RK23",
    "Radau",
    "BDF",
    "LSODA",
}

ENVIRONMENT_PACKAGES = (
    "numpy",
    "scipy",
    "matplotlib",
    "sgp4",
    "pandas",
    "astropy",
    "plotly",
)
