#!/usr/bin/env python3
"""Evaluate padding convergence independently of plot limits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from validation._common.evaluation import (
    archive_provenance,
    load_npz,
    malformed_result,
    metadata_parameters,
    write_result,
)
from validation._common.result import Metric, ValidationResult, upper_bound_check


def evaluate(archive: str | Path, *, figure: str | Path) -> ValidationResult:
    try:
        data = load_npz(archive)
        alphas = np.asarray(data["alphas"], dtype=np.float64)
        rms_error = np.asarray(data["prof_rms_dx"], dtype=np.float64)
        reference_rms = np.asarray(data["ref_rms_psi"], dtype=np.float64)
        if alphas.ndim != 1 or alphas.size < 3 or np.any(np.diff(alphas) <= 0):
            raise ValueError("padding factors must contain reference plus refinements")
        if rms_error.shape[1] != alphas.size - 1:
            raise ValueError("profile errors must omit only the reference padding")
        relative = rms_error / reference_rms[:, None, :]
        finite = np.isfinite(relative)
        finite_counts = np.sum(finite, axis=(0, 2))
        if np.any(finite_counts == 0):
            raise ValueError("each padding factor needs a finite profile sample")
        mean_errors = (
            np.sum(np.where(finite, relative, 0.0), axis=(0, 2))
            / finite_counts
        )
        if not np.all(np.isfinite(mean_errors)) or np.any(mean_errors < 0.0):
            raise ValueError("relative displacement errors must be finite")
        convergence_ratio = float(mean_errors[-1] / mean_errors[0])
        return ValidationResult(
            campaign="padding",
            parameters=metadata_parameters(data),
            provenance=archive_provenance(archive),
            metrics={
                "coarse_relative_displacement_error": Metric(
                    float(mean_errors[0]), "dimensionless"
                ),
                "refined_relative_displacement_error": Metric(
                    float(mean_errors[-1]), "dimensionless"
                ),
                "convergence_ratio": Metric(
                    convergence_ratio, "dimensionless"
                ),
            },
            checks=[
                upper_bound_check(
                    name="padding convergence",
                    observed=convergence_ratio,
                    limit=1.0,
                    unit="dimensionless",
                    rationale=(
                        "Increasing the Fourier embedding toward the reference "
                        "must not increase the RMS displacement truncation error."
                    ),
                    source="nested-domain convergence argument",
                )
            ],
            numerical_archive=archive,
            figures=[figure],
        )
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ) as error:
        return malformed_result(
            campaign="padding",
            archive=archive,
            figures=[figure],
            error=error,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return write_result(
        evaluate(args.archive, figure=args.figure),
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
