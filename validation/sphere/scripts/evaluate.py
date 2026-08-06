#!/usr/bin/env python3
"""Evaluate IC component variances against the periodic reference."""

from __future__ import annotations

import argparse
import math
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
        geometry = np.asarray(data["var_geom"], dtype=np.float64)
        reference = np.asarray(data["var_ref"], dtype=np.float64)
        if geometry.shape != (3,) or reference.shape != (3,):
            raise ValueError("component variances must be three-vectors")
        if (
            not np.all(np.isfinite(geometry))
            or not np.all(np.isfinite(reference))
            or np.any(reference <= 0.0)
        ):
            raise ValueError("component variances must be finite and positive")
        geometry_count = int(data["meta_n_geom_core"])
        reference_count = int(data["meta_n_ref_core"])
        if min(geometry_count, reference_count) <= 1:
            raise ValueError("variance checks require at least two particles")
        fractional_variance = (
            2.0 / (geometry_count - 1)
            + 2.0 / (reference_count - 1)
        )
        ratios = geometry / reference
        statistic = float(np.sum((ratios - 1.0) ** 2 / fractional_variance))
        dof = 3
        x = math.log(1.0e6)
        limit = dof + 2.0 * math.sqrt(dof * x) + 2.0 * x
        return ValidationResult(
            campaign="sphere-ic",
            parameters=metadata_parameters(data),
            provenance=archive_provenance(archive),
            metrics={
                "component_variance_chi_square": Metric(
                    statistic, "chi-square"
                ),
                "velocity_rms_relative_residual": Metric(
                    float(data["rms_dv"]) / float(data["rms_ref"]),
                    "dimensionless",
                ),
            },
            checks=[
                upper_bound_check(
                    name="core component-variance agreement",
                    observed=statistic,
                    limit=limit,
                    unit="chi-square",
                    rationale=(
                        "Gaussian sample variances have fractional variance "
                        "2/(N-1); matched fields make the independent-sample "
                        "bound conservative."
                    ),
                    source="Gaussian sample-variance identity",
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
            campaign="sphere-ic",
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
