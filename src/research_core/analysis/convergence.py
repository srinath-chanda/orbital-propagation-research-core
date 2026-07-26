"""Numerical-convergence case generation and selection logic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class ConvergenceCase:
    """One candidate numerical integration setting."""

    case_id: str
    method: str
    relative_tolerance: float
    absolute_tolerance: float
    maximum_step_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "method": self.method,
            "relative_tolerance": self.relative_tolerance,
            "absolute_tolerance": self.absolute_tolerance,
            "maximum_step_seconds": self.maximum_step_seconds,
        }


def _scientific_token(value: float) -> str:
    return f"{float(value):.0e}".replace("+", "")


def generate_convergence_cases(config: dict[str, Any]) -> list[ConvergenceCase]:
    """Generate a deterministic Cartesian product of convergence settings."""
    method = str(config["method"])
    cases: list[ConvergenceCase] = []
    for relative_tolerance in config["relative_tolerances"]:
        for absolute_tolerance in config["absolute_tolerances"]:
            for maximum_step_seconds in config["maximum_steps_seconds"]:
                rtol = float(relative_tolerance)
                atol = float(absolute_tolerance)
                step = float(maximum_step_seconds)
                case_id = (
                    f"{method}_rtol{_scientific_token(rtol)}_"
                    f"atol{_scientific_token(atol)}_maxstep{step:g}s"
                )
                cases.append(
                    ConvergenceCase(
                        case_id=case_id,
                        method=method,
                        relative_tolerance=rtol,
                        absolute_tolerance=atol,
                        maximum_step_seconds=step,
                    )
                )
    identifiers = [case.case_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Generated convergence case IDs are not unique.")
    return cases


def reference_case(config: dict[str, Any]) -> ConvergenceCase:
    """Return the configured high-accuracy numerical reference case."""
    method = str(config["reference_method"])
    return ConvergenceCase(
        case_id="NUMERICAL_REFERENCE",
        method=method,
        relative_tolerance=float(config["reference_relative_tolerance"]),
        absolute_tolerance=float(config["reference_absolute_tolerance"]),
        maximum_step_seconds=float(config["reference_maximum_step_seconds"]),
    )


def passing_rows(
    rows: Iterable[dict[str, Any]],
    validation_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return rows satisfying all provisional two-body thresholds."""
    position_limit = float(
        validation_config["two_body_maximum_position_difference_m"]
    )
    velocity_limit = float(
        validation_config["two_body_maximum_velocity_difference_mm_s"]
    )
    energy_limit = float(
        validation_config["two_body_maximum_relative_energy_drift"]
    )
    h_limit = float(
        validation_config[
            "two_body_maximum_relative_angular_momentum_drift"
        ]
    )
    return [
        row
        for row in rows
        if row["maximum_position_difference_vs_analytical_m"] <= position_limit
        and row["maximum_velocity_difference_vs_analytical_mm_s"] <= velocity_limit
        and row["maximum_absolute_relative_energy_drift"] <= energy_limit
        and row["maximum_absolute_relative_angular_momentum_drift"] <= h_limit
    ]


def pareto_frontier(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return non-dominated rows minimising runtime and position error."""
    candidates = list(rows)
    frontier: list[dict[str, Any]] = []
    for row in candidates:
        runtime = float(row["median_runtime_seconds"])
        error = float(row["maximum_position_difference_vs_analytical_m"])
        dominated = False
        for other in candidates:
            if other is row:
                continue
            other_runtime = float(other["median_runtime_seconds"])
            other_error = float(
                other["maximum_position_difference_vs_analytical_m"]
            )
            no_worse = other_runtime <= runtime and other_error <= error
            strictly_better = other_runtime < runtime or other_error < error
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return sorted(
        frontier,
        key=lambda item: (
            float(item["median_runtime_seconds"]),
            float(item["maximum_position_difference_vs_analytical_m"]),
        ),
    )


def _normalise(values: np.ndarray) -> np.ndarray:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1e-15):
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def select_recommendations(
    rows: list[dict[str, Any]],
    validation_config: dict[str, Any],
) -> dict[str, Any]:
    """Create transparent fastest, most-accurate, and balanced recommendations."""
    passed = passing_rows(rows, validation_config)
    if not passed:
        return {
            "selection_status": "no_candidate_passed_provisional_thresholds",
            "passing_case_count": 0,
            "pareto_case_ids": [],
            "fastest_passing": None,
            "most_accurate_passing": None,
            "balanced_recommendation": None,
            "selection_method": (
                "No recommendation because no candidate satisfied every "
                "provisional validation threshold."
            ),
        }

    fastest = min(
        passed,
        key=lambda item: (
            float(item["median_runtime_seconds"]),
            float(item["maximum_position_difference_vs_analytical_m"]),
        ),
    )
    accurate = min(
        passed,
        key=lambda item: (
            float(item["maximum_position_difference_vs_analytical_m"]),
            float(item["median_runtime_seconds"]),
        ),
    )
    frontier = pareto_frontier(passed)

    runtimes = np.array(
        [max(float(row["median_runtime_seconds"]), 1e-15) for row in frontier]
    )
    errors = np.array(
        [
            max(
                float(row["maximum_position_difference_vs_analytical_m"]),
                1e-15,
            )
            for row in frontier
        ]
    )
    runtime_score = _normalise(np.log10(runtimes))
    error_score = _normalise(np.log10(errors))
    combined = np.sqrt(runtime_score**2 + error_score**2)
    balanced_index = int(np.argmin(combined))
    balanced = dict(frontier[balanced_index])
    balanced["balanced_log_space_score"] = float(combined[balanced_index])

    def compact(row: dict[str, Any]) -> dict[str, Any]:
        return {
            key: row[key]
            for key in (
                "case_id",
                "method",
                "relative_tolerance",
                "absolute_tolerance",
                "maximum_step_seconds",
                "median_runtime_seconds",
                "maximum_position_difference_vs_analytical_m",
                "maximum_velocity_difference_vs_analytical_mm_s",
                "maximum_absolute_relative_energy_drift",
                "maximum_absolute_relative_angular_momentum_drift",
                "function_evaluations",
            )
            if key in row
        }

    balanced_compact = compact(balanced)
    balanced_compact["balanced_log_space_score"] = balanced[
        "balanced_log_space_score"
    ]

    return {
        "selection_status": "provisional_recommendation_available",
        "passing_case_count": len(passed),
        "pareto_case_ids": [row["case_id"] for row in frontier],
        "fastest_passing": compact(fastest),
        "most_accurate_passing": compact(accurate),
        "balanced_recommendation": balanced_compact,
        "selection_method": (
            "Candidates must first pass every provisional two-body threshold. "
            "The balanced recommendation is the Pareto-frontier case closest "
            "to the ideal point after log10 runtime and log10 maximum-position-"
            "error normalisation. It is a transparent engineering recommendation, "
            "not a final scientific truth."
        ),
    }
