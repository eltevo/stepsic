#!/usr/bin/env python3
"""Evaluate matched-phase power recovery against monofonIC."""

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


_COMPARISON_KMAX_FRACTION_NYQUIST = 0.8
_MAX_POWER_DEVIATION = 1.0e-3


def evaluate(
    archive: str | Path,
    *,
    figure: str | Path,
) -> ValidationResult:
    try:
        data = load_npz(archive)
        k = np.asarray(data["pk_k"], dtype=np.float64)
        ratio = np.asarray(data["pk_ratio"], dtype=np.float64)
        k_nyquist = float(data["meta_k_ny"])
        if k.ndim != 1 or k.shape != ratio.shape:
            raise ValueError("pk_k and pk_ratio must be one-dimensional peers")
        supported = k <= _COMPARISON_KMAX_FRACTION_NYQUIST * k_nyquist
        if not np.any(supported):
            raise ValueError("no power bins below 0.8 k_Ny")
        max_deviation = float(np.max(np.abs(ratio[supported] - 1.0)))
        check = upper_bound_check(
            name="power agreement below 0.8 k_Ny",
            observed=max_deviation,
            limit=_MAX_POWER_DEVIATION,
            unit="fraction",
            rationale=(
                "Same white noise, spectrum and cosmology; Fig. 8 shades +-0.1%."
            ),
            source="max |P_stepsic/P_monofonic - 1| for k <= 0.8 k_Ny",
        )
        return ValidationResult(
            campaign="monofonic-agreement",
            parameters=metadata_parameters(data),
            provenance=archive_provenance(archive),
            metrics={
                "power_agreement_max_relative_residual": Metric(
                    max_deviation, "fraction",
                ),
            },
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
