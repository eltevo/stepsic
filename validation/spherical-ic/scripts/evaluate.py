#!/usr/bin/env python3
"""Compare the spherical and periodic velocity fields on their common grid."""

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


# The small profile has intentionally sparse cells and reaches a 31%
# residual. Requiring less than one third keeps that executable floor while
# limiting residual velocity power to 1/9 of the reference; the medium
# publication profile is expected to be substantially better.
MAX_RELATIVE_VELOCITY_RMS = 1.0 / 3.0


def evaluate(
    archive: str | Path,
    multiresolution: str | Path,
    *,
    figures: list[str | Path],
) -> ValidationResult:
    try:
        data = load_npz(archive)
        mixing = load_npz(multiresolution)
        geometry = np.asarray(data["var_geom"], dtype=np.float64)
        reference = np.asarray(data["var_ref"], dtype=np.float64)
        if geometry.shape != (3,) or reference.shape != (3,):
            raise ValueError("component variances must be three-vectors")
        if (
            not np.all(np.isfinite(geometry))
            or not np.all(np.isfinite(reference))
            or np.any(geometry < 0.0)
            or np.any(reference <= 0.0)
        ):
            raise ValueError(
                "component variances must be finite and reference-positive"
            )
        rms_difference = float(data["rms_dv"])
        rms_reference = float(data["rms_ref"])
        if (
            not np.isfinite(rms_difference)
            or rms_difference < 0.0
            or not np.isfinite(rms_reference)
            or rms_reference <= 0.0
        ):
            raise ValueError(
                "velocity RMS values must be finite, non-negative, and have "
                "a positive reference"
            )
        relative_residual = rms_difference / rms_reference
        return ValidationResult(
            campaign="spherical-ic",
            parameters=metadata_parameters(data),
            provenance=archive_provenance(archive),
            metrics={
                "component_variance_ratios": Metric(
                    geometry / reference, "dimensionless"
                ),
                "velocity_rms_relative_residual": Metric(
                    relative_residual, "dimensionless"
                ),
                # Rows follow the mass-sample counts, columns the radial bins.
                "multiresolution_displacement_relative_residual": Metric(
                    mixing["displacement_relative_residual"], "dimensionless"
                ),
            },
            checks=[
                upper_bound_check(
                    name="core matched velocity-field agreement",
                    observed=relative_residual,
                    limit=MAX_RELATIVE_VELOCITY_RMS,
                    unit="dimensionless",
                    rationale=(
                        "Mass-weighted CIC fields are compared in identical "
                        "populated core cells; a one-third RMS bound limits "
                        "residual velocity power to 1/9 of the reference."
                    ),
                    source="matched common-grid velocity-field residual",
                )
            ],
            numerical_archive=[archive, multiresolution],
            figures=figures,
        )
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ) as error:
        return malformed_result(
            campaign="spherical-ic",
            archive=archive,
            figures=figures,
            error=error,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--multiresolution", required=True)
    parser.add_argument("--figure", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return write_result(
        evaluate(args.archive, args.multiresolution, figures=args.figure),
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
