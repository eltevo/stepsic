#!/usr/bin/env python3
"""Compare periodic 1LPT and 2LPT runs evolved with Gadget-4."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from validation._common.evaluation import (
    archive_provenance,
    gaussian_power_check,
    load_npz,
    malformed_result,
    metadata_parameters,
    write_result,
)
from validation._common.result import ValidationResult


def evaluate(
    archive: str | Path,
    *,
    figure: str | Path,
) -> ValidationResult:
    try:
        data = load_npz(archive)
        metric, check = gaussian_power_check(
            name="large-scale LPT convergence",
            ratios=np.asarray(data["ratio_1lpt_2lpt"], dtype=np.float64),
            mode_counts=np.asarray(data["nmodes"], dtype=np.float64),
            sample_count=1,
            source="independent Gaussian Fourier-mode variance bound",
        )
        return ValidationResult(
            campaign="periodic-lpt",
            parameters=metadata_parameters(data),
            provenance=archive_provenance(archive),
            metrics={"lpt_convergence_chi_square": metric},
            checks=[check],
            numerical_archive=archive,
            figures=[figure],
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        return malformed_result(
            campaign="periodic-lpt",
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
