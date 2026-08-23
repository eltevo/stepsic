#!/usr/bin/env python3
"""Evaluate matched-phase power recovery against monofonIC."""

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
        ratio = np.asarray(data["pk_ratio"], dtype=np.float64)
        modes = np.asarray(data["pk_nmodes"], dtype=np.float64)
        metric, check = gaussian_power_check(
            name="mode-counted power agreement",
            ratios=ratio,
            mode_counts=modes,
            sample_count=1,
            source="independent Gaussian Fourier-mode variance bound",
        )
        return ValidationResult(
            campaign="monofonic-agreement",
            parameters=metadata_parameters(data),
            provenance=archive_provenance(archive),
            metrics={"power_agreement_chi_square": metric},
            checks=[check],
            numerical_archive=archive,
            figures=[figure],
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        return malformed_result(
            campaign="monofonic-agreement",
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
