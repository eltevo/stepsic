#!/usr/bin/env python3
"""Check histogram counts and rotational symmetry in a cubic field."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import f as f_distribution

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
        nreal = int(data["meta_nreal"])
        n_total = int(data["meta_n_total"])
        if nreal < 4 or n_total % nreal != 0:
            raise ValueError("isotropy evaluation requires at least four realizations")
        moments = np.asarray(
            data["stat_disp_component_second_moments"], dtype=np.float64,
        )
        if moments.shape != (n_total, 3) or not np.all(np.isfinite(moments)):
            raise ValueError("component second moments have the wrong shape or values")
        moments = moments.reshape(nreal, n_total // nreal, 3).mean(axis=1)
        contrasts = np.column_stack(
            (moments[:, 0] - moments[:, 2], moments[:, 1] - moments[:, 2])
        )
        covariance = np.cov(contrasts, rowvar=False, ddof=1)
        if covariance.shape != (2, 2) or np.linalg.matrix_rank(covariance) != 2:
            raise ValueError("isotropy contrast covariance is singular")
        mean_contrast = np.mean(contrasts, axis=0)
        hotelling_t2 = float(
            nreal * mean_contrast @ np.linalg.solve(covariance, mean_contrast)
        )
        isotropy_statistic = float(
            (nreal - 2) * hotelling_t2 / (2 * (nreal - 1))
        )
        isotropy_limit = float(f_distribution.ppf(1.0 - 1.0e-6, 2, nreal - 2))
        conservation_error = max(conservation_errors)
        conservation_limit = (
            np.finfo(np.float64).eps
            * n_part
            * int(data["meta_nbins"])
        )
        return ValidationResult(
            campaign="field-statistics",
            parameters=metadata_parameters(data),
            provenance=archive_provenance(archive),
            metrics={
                "component_isotropy": Metric(
                    isotropy_statistic, "F statistic"
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
                    unit="F statistic",
                    rationale=(
                        "A Hotelling test compares two independent component-variance differences across random fields. The limit is the 1e-6 upper tail of its F distribution."
                    ),
                    source="rotational symmetry of homogeneous Gaussian fields",
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
            campaign="field-statistics",
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
