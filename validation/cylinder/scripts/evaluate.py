#!/usr/bin/env python3
"""Evaluate the expected low-k convergence of 1LPT and 2LPT."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from validation._common.evaluation import archive_provenance, malformed_result, write_result
from validation._common.result import Metric, ValidationResult, upper_bound_check


def _load_power(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path, comments="#", ndmin=2)
    if values.ndim != 2 or values.shape[1] < 2 or len(values) < 4:
        raise ValueError(f"{path} must contain at least four k, P(k) rows")
    k = np.asarray(values[:, 0], dtype=np.float64)
    power = np.asarray(values[:, 1], dtype=np.float64)
    if (
        not np.all(np.isfinite(k))
        or not np.all(np.isfinite(power))
        or np.any(k <= 0.0)
        or np.any(power <= 0.0)
    ):
        raise ValueError(f"{path} contains non-finite or non-positive data")
    order = np.argsort(k)
    k = k[order]
    power = power[order]
    if np.any(np.diff(k) <= 0.0):
        raise ValueError(f"{path} k bins must be unique")
    return k, power


def evaluate(
    one_lpt: str | Path,
    two_lpt: str | Path,
    *,
    figures: list[str | Path],
) -> ValidationResult:
    try:
        k_one, power_one = _load_power(one_lpt)
        k_two, power_two = _load_power(two_lpt)
        overlap = (k_one >= k_two[0]) & (k_one <= k_two[-1])
        if np.count_nonzero(overlap) < 4:
            raise ValueError("1LPT and 2LPT spectra have fewer than four common bins")
        k = k_one[overlap]
        reference = np.exp(
            np.interp(np.log(k), np.log(k_two), np.log(power_two))
        )
        deviation = power_one[overlap] / reference - 1.0
        midpoint = len(deviation) // 2
        low_rms = float(np.sqrt(np.mean(deviation[:midpoint] ** 2)))
        high_rms = float(np.sqrt(np.mean(deviation[midpoint:] ** 2)))
        check = upper_bound_check(
            name="low-k LPT convergence",
            observed=low_rms,
            limit=high_rms,
            unit="fraction",
            rationale=(
                "Second-order Lagrangian corrections vanish in the linear "
                "large-scale limit, so the lower-k half must not disagree "
                "more strongly than the upper-k half."
            ),
            source="Lagrangian perturbation-theory convergence",
        )
        return ValidationResult(
            campaign="cylinder",
            parameters={
                "common_bin_count": len(k),
                "low_k_max_inv_mpc": float(k[midpoint - 1]),
                "high_k_min_inv_mpc": float(k[midpoint]),
            },
            provenance={
                "one_lpt": archive_provenance(one_lpt),
                "two_lpt": archive_provenance(two_lpt),
            },
            metrics={
                "low_k_ratio_rms": Metric(low_rms, "fraction"),
                "high_k_ratio_rms": Metric(high_rms, "fraction"),
            },
            checks=[check],
            numerical_archive=Path(one_lpt).parent,
            figures=figures,
        )
    except (IndexError, OSError, TypeError, ValueError) as error:
        return malformed_result(
            campaign="cylinder",
            archive=one_lpt,
            figures=figures,
            error=error,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--one-lpt", required=True)
    parser.add_argument("--two-lpt", required=True)
    parser.add_argument("--figure", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return write_result(
        evaluate(args.one_lpt, args.two_lpt, figures=args.figure),
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
