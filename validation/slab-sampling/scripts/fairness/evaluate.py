#!/usr/bin/env python3
"""Evaluate displacement and velocity variance in a slab and cube cut."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from validation._common.evaluation import (
    average_realization_members,
    archive_provenance,
    load_npz,
    malformed_result,
    metadata_parameters,
    ratio_of_means,
    write_result,
)
from validation._common.result import Metric, ValidationResult, upper_bound_check


def evaluate(archive: str | Path, *, figure: str | Path) -> ValidationResult:
    try:
        data = load_npz(archive)
        variance_members = {
            name: np.asarray(data[name], dtype=np.float64)
            for name in (
                "var_disp_slab", "var_disp_cut",
                "var_vel_slab", "var_vel_cut",
            )
        }
        member_shape = variance_members["var_disp_slab"].shape
        if member_shape[1:] != (3,) or any(
            values.shape != member_shape for values in variance_members.values()
        ):
            raise ValueError("variance samples must have shape (N, 3)")
        if any(
            not np.all(np.isfinite(values)) or np.any(values <= 0.0)
            for values in variance_members.values()
        ):
            raise ValueError("component variances must be finite and positive")
        nreal_value = np.asarray(data["meta_nreal"])
        paired_value = np.asarray(data["meta_paired"])
        if nreal_value.size != 1 or nreal_value.dtype.kind not in "iu":
            raise ValueError("meta_nreal must be a positive integer")
        if paired_value.size != 1 or paired_value.dtype.kind != "b":
            raise ValueError("meta_paired must be boolean")
        nreal = int(nreal_value.item())
        paired = bool(paired_value.item())
        realizations = {
            name: average_realization_members(
                values, nreal=nreal, paired=paired,
            )
            for name, values in variance_members.items()
        }
        displacement_ratios, displacement_standard_errors = ratio_of_means(
            realizations["var_disp_slab"], realizations["var_disp_cut"],
        )
        velocity_ratios, velocity_standard_errors = ratio_of_means(
            realizations["var_vel_slab"], realizations["var_vel_cut"],
        )
        cut_displacement_means = np.mean(
            realizations["var_disp_cut"], axis=0, dtype=np.float64
        )
        linearized_displacement_ratios = (
            realizations["var_disp_slab"]
            - displacement_ratios * realizations["var_disp_cut"]
        ) / cut_displacement_means
        transverse_difference = float(
            displacement_ratios[0] - displacement_ratios[1]
        )
        transverse_difference_standard_error = float(
            np.std(
                linearized_displacement_ratios[:, 0]
                - linearized_displacement_ratios[:, 1],
                ddof=1,
                dtype=np.float64,
            )
            / np.sqrt(nreal)
        )
        line_of_sight_displacement_ratio = float(displacement_ratios[2])
        line_of_sight_velocity_ratio = float(velocity_ratios[2])
        return ValidationResult(
            campaign="slab-sampling-fairness",
            parameters=metadata_parameters(data),
            provenance=archive_provenance(archive),
            metrics={
                "displacement_component_variance_ratios": Metric(
                    displacement_ratios, "dimensionless"
                ),
                "displacement_component_variance_ratio_standard_errors": Metric(
                    displacement_standard_errors, "dimensionless"
                ),
                "velocity_component_variance_ratios": Metric(
                    velocity_ratios, "dimensionless"
                ),
                "velocity_component_variance_ratio_standard_errors": Metric(
                    velocity_standard_errors, "dimensionless"
                ),
                "transverse_displacement_variance_ratio_difference": Metric(
                    transverse_difference, "dimensionless"
                ),
                "transverse_displacement_variance_ratio_difference_standard_error": Metric(
                    transverse_difference_standard_error, "dimensionless"
                ),
                "line_of_sight_displacement_variance_ratio": Metric(
                    line_of_sight_displacement_ratio, "dimensionless"
                ),
                "line_of_sight_velocity_variance_ratio": Metric(
                    line_of_sight_velocity_ratio, "dimensionless"
                )
            },
            checks=[
                upper_bound_check(
                    name="short-axis displacement long-mode suppression",
                    observed=line_of_sight_displacement_ratio,
                    limit=1.0,
                    unit="dimensionless",
                    rationale=(
                        "A native periodic slab lacks the line-of-sight modes "
                        "longer than its short side that remain in the region cut from the cube."
                    ),
                    source="discrete Fourier support of a periodic slab",
                ),
                upper_bound_check(
                    name="short-axis velocity long-mode suppression",
                    observed=line_of_sight_velocity_ratio,
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
