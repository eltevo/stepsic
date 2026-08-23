#!/usr/bin/env python3
"""Evaluate the slab's analytically missing line-of-sight long modes."""

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
        slab_variance = np.asarray(data["var_slab"], dtype=np.float64)
        cut_variance = np.asarray(data["var_cut"], dtype=np.float64)
        if (
            slab_variance.ndim != 2
            or slab_variance.shape != cut_variance.shape
            or slab_variance.shape[1] != 3
        ):
            raise ValueError("variance samples must have shape (N, 3)")
        if (
            not np.all(np.isfinite(slab_variance))
            or not np.all(np.isfinite(cut_variance))
            or np.any(cut_variance <= 0.0)
        ):
            raise ValueError("component variances must be finite and positive")
        line_of_sight_ratio = float(
            np.mean(slab_variance[:, 2]) / np.mean(cut_variance[:, 2])
        )
        return ValidationResult(
            campaign="slab-sampling-fairness",
            parameters=metadata_parameters(data),
            provenance=archive_provenance(archive),
            metrics={
                "line_of_sight_variance_ratio": Metric(
                    line_of_sight_ratio, "dimensionless"
                )
            },
            checks=[
                upper_bound_check(
                    name="short-axis long-mode suppression",
                    observed=line_of_sight_ratio,
                    limit=1.0,
                    unit="dimensionless",
                    rationale=(
                        "A native periodic slab lacks the line-of-sight modes "
                        "longer than its short side that remain in the region cut from the cube."
                    ),
                    source="discrete Fourier support of a periodic slab",
                )
            ],
            numerical_archive=archive,
            figures=[figure],
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        return malformed_result(
            campaign="slab-sampling-fairness",
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
