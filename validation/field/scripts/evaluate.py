#!/usr/bin/env python3
"""Evaluate histogram conservation and cubic-field isotropy."""

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


def evaluate(
    archive: str | Path,
    *,
    figures: list[str | Path],
) -> ValidationResult:
    try:
        data = load_npz(archive)
        n_part = int(data["meta_npart"])
        n_samples = n_part * int(data["meta_n_total"])
        second_moments = []
        conservation_errors = []
        for axis in "xyz":
            centres = np.asarray(
                data[f"hist_disp_{axis}_centres"], dtype=np.float64
            )
            counts = np.asarray(
                data[f"hist_disp_{axis}_counts"], dtype=np.float64
            )
            if centres.shape != counts.shape or counts.size == 0:
                raise ValueError("component histogram arrays must be peers")
            if np.any(counts < 0.0) or not np.all(np.isfinite(centres)):
                raise ValueError("component histograms contain invalid values")
            count = float(np.sum(counts))
            conservation_errors.append(abs(count - n_part))
            second_moments.append(float(np.sum(counts * centres**2) / count))

        moments = np.asarray(second_moments)
        mean_moment = float(np.mean(moments))
        if not math.isfinite(mean_moment) or mean_moment <= 0.0:
            raise ValueError("component second moments must be finite and positive")
        # A Gaussian component sample variance has fractional variance
        # 2/(N-1). Summing the three conservative standardized deviations
        # gives a chi-square-like statistic.
        isotropy_statistic = float(
            np.sum(
                ((moments / mean_moment) - 1.0) ** 2
                / (2.0 / (n_samples - 1))
            )
        )
        dof = 3
        x = math.log(1.0e6)
        isotropy_limit = dof + 2.0 * math.sqrt(dof * x) + 2.0 * x
        conservation_error = max(conservation_errors)
        conservation_limit = (
            np.finfo(np.float64).eps
            * n_part
            * int(data["meta_nbins"])
        )
        return ValidationResult(
            campaign="field",
            parameters=metadata_parameters(data),
            provenance=archive_provenance(archive),
            metrics={
                "component_isotropy": Metric(
                    isotropy_statistic, "chi-square"
                ),
                "histogram_count_error": Metric(
                    conservation_error, "particles"
                ),
            },
            checks=[
                upper_bound_check(
                    name="cubic displacement-component isotropy",
                    observed=isotropy_statistic,
                    limit=isotropy_limit,
                    unit="chi-square",
                    rationale=(
                        "Gaussian sample-variance law 2/(N-1), with the "
                        "Laurent-Massart 1e-6 upper-tail bound."
                    ),
                    source="isotropy of a homogeneous Gaussian field",
                ),
                upper_bound_check(
                    name="component histogram particle conservation",
                    observed=conservation_error,
                    limit=conservation_limit,
                    unit="particles",
                    rationale="Each component contributes one value per particle.",
                    source="histogram conservation identity",
                ),
            ],
            numerical_archive=archive,
            figures=figures,
        )
    except (KeyError, OSError, TypeError, ValueError, ZeroDivisionError) as error:
        return malformed_result(
            campaign="field",
            archive=archive,
            figures=figures,
            error=error,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--figure", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return write_result(
        evaluate(args.archive, figures=args.figure),
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
