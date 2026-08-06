#!/usr/bin/env python3
"""Evaluate mode-counted power recovery across box aspect ratios."""

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


def evaluate(archive: str | Path, *, figure: str | Path) -> ValidationResult:
    try:
        data = load_npz(archive)
        ratios = []
        modes = []
        sample_count = int(data["meta_n_total"])
        for length in np.asarray(data["meta_lz_values"], dtype=np.float64):
            tag = f"lz{length:g}"
            reference = np.asarray(data[f"ref_{tag}_pk"], dtype=np.float64)
            measured = np.asarray(data[f"lpt_{tag}_pk"], dtype=np.float64)
            if np.any(reference <= 0.0):
                raise ValueError("reference power must be positive")
            ratios.append(measured / reference)
            modes.append(
                np.asarray(data[f"lpt_{tag}_nmodes"], dtype=np.float64)
                * sample_count
            )
        metric, check = gaussian_power_check(
            name="aspect-ratio power recovery",
            ratios=np.concatenate(ratios),
            mode_counts=np.concatenate(modes),
            sample_count=1,
            source="Gaussian Fourier-mode sampling variance",
        )
        return ValidationResult(
            campaign="squish",
            parameters=metadata_parameters(data),
            provenance=archive_provenance(archive),
            metrics={"power_recovery_chi_square": metric},
            checks=[check],
            numerical_archive=archive,
            figures=[figure],
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        return malformed_result(
            campaign="squish",
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
