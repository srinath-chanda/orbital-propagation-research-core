"""Analysis routines for model comparison and scientific validation."""

from .comparison import compare_state_histories, create_error_summary
from .diagnostics import conservation_diagnostics, create_orbit_summary
from .j2 import (
    analytical_j2_raan_rate_rad_s,
    compare_in_reference_rtn,
    create_j2_validation_summary,
    create_osculating_element_history,
    fit_raan_rate,
    j2_conservation_diagnostics,
)

__all__ = [
    "analytical_j2_raan_rate_rad_s",
    "compare_in_reference_rtn",
    "compare_state_histories",
    "conservation_diagnostics",
    "create_error_summary",
    "create_j2_validation_summary",
    "create_orbit_summary",
    "create_osculating_element_history",
    "fit_raan_rate",
    "j2_conservation_diagnostics",
]
