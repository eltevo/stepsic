#!/usr/bin/env python3
"""Check power transfer against the cubic box at each aspect ratio."""

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
from validation._common.result import Check, Metric, ValidationResult
from common import cubic_normalized_transfer_curves


_MAX_TRANSFER_DEVIATION = np.float64(0.01)
_FLOAT64_BOUNDARY_TOLERANCE = 8.0 * np.finfo(np.float64).eps


def evaluate(archive: str | Path, *, figure: str | Path) -> ValidationResult:
    try:
        data = load_npz(archive)
        sample_count_value = np.asarray(data["meta_n_total"])
        if sample_count_value.size != 1:
            raise ValueError("meta_n_total must be a positive integer")
        sample_count_float = float(sample_count_value.reshape(-1)[0])
        if (
            not np.isfinite(sample_count_float)
            or sample_count_float < 1.0
            or not sample_count_float.is_integer()
        ):
            raise ValueError("meta_n_total must be a positive integer")
        sample_count = int(sample_count_float)

        curves = cubic_normalized_transfer_curves(data)
        anisotropic = [curve for curve in curves if not curve.is_cubic]
        ratios = [curve.transfer for curve in anisotropic]
        effective_modes = []
        for curve in anisotropic:
            with np.errstate(over="ignore"):
                aspect_modes = curve.aspect_mode_counts * sample_count
                cubic_modes = curve.cubic_mode_counts * sample_count
            if not np.all(np.isfinite(aspect_modes)) or not np.all(
                np.isfinite(cubic_modes)
            ):
                raise ValueError("sample-scaled mode counts must be finite")
            smaller_modes = np.minimum(aspect_modes, cubic_modes)
            larger_modes = np.maximum(aspect_modes, cubic_modes)
            effective_modes.append(
                smaller_modes / (1.0 + smaller_modes / larger_modes)
            )

        chi_square_metric, chi_square_check = gaussian_power_check(
            name="power transfer agrees with the cubic box",
            ratios=np.concatenate(ratios),
            mode_counts=np.concatenate(effective_modes),
            sample_count=1,
            source="combined anisotropic and cubic Gaussian sampling variance",
        )
        max_deviation = float(
            np.max(np.abs(np.concatenate(ratios) - 1.0))
        )
        deviation_metric = Metric(max_deviation, "fraction")
        deviation_check = Check(
            name="largest power-transfer difference from the cubic box",
            observed=max_deviation,
            comparison="<=",
            limit=float(_MAX_TRANSFER_DEVIATION),
            unit="fraction",
            rationale=(
                "Power transfer for every slab must remain within 1% of the cubic result. Eight float64 epsilons allow for rounding exactly at the limit."
            ),
            source="slab power-transfer acceptance threshold",
            passed=bool(
                np.isfinite(max_deviation)
                and max_deviation
                <= _MAX_TRANSFER_DEVIATION + _FLOAT64_BOUNDARY_TOLERANCE
            ),
        )
        return ValidationResult(
            campaign="slab-sampling-transfer",
            parameters=metadata_parameters(data),
            provenance=archive_provenance(archive),
            metrics={
                "aspect_invariance_chi_square": chi_square_metric,
                "max_aspect_transfer_deviation": deviation_metric,
            },
            checks=[chi_square_check, deviation_check],
            numerical_archive=archive,
            figures=[figure],
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        return malformed_result(
            campaign="slab-sampling-transfer",
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
