#!/usr/bin/env python3
"""Evaluate the grid campaign's mode-counted power-recovery claim."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

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


LPT_POWER_KEY = re.compile(
    r"^lpt(?P<order>\d+)_(?P<method>[a-z]+)_(?P<nmesh>\d+)_"
    r"z(?P<redshift>[^_]+)_pk$"
)


def evaluate(
    archives: list[str | Path],
    *,
    figure: str | Path,
) -> ValidationResult:
    try:
        all_ratios: list[np.ndarray] = []
        all_modes: list[np.ndarray] = []
        parameters: dict[str, object] = {"panels": []}
        sample_counts: list[int] = []
        for archive in archives:
            data = load_npz(archive)
            parameters["panels"].append(metadata_parameters(data))
            sample_count = int(data["meta_n_total"])
            sample_counts.append(sample_count)
            for key, measured in data.items():
                match = LPT_POWER_KEY.match(key)
                if match is None:
                    continue
                ref_key = (
                    f"ref_{match['method']}_{match['nmesh']}_"
                    f"z{match['redshift']}_pk"
                )
                modes_key = key.removesuffix("_pk") + "_nmodes"
                reference = np.asarray(data[ref_key], dtype=np.float64)
                measured = np.asarray(measured, dtype=np.float64)
                modes = np.asarray(data[modes_key], dtype=np.float64)
                if np.any(reference <= 0.0):
                    raise ValueError(f"{ref_key} contains non-positive power")
                all_ratios.append(measured / reference)
                all_modes.append(modes * sample_count)
        if not all_ratios:
            raise ValueError("no LPT power curves found")
        ratios = np.concatenate(all_ratios)
        effective_modes = np.concatenate(all_modes)
        metric, check = gaussian_power_check(
            name="mode-counted LPT power recovery",
            ratios=ratios,
            mode_counts=effective_modes,
            sample_count=1,
            source="Gaussian Fourier-mode sampling variance",
        )
        return ValidationResult(
            campaign="grid",
            parameters=parameters,
            provenance={
                "archives": [
                    archive_provenance(path) for path in archives
                ],
            },
            metrics={
                "power_recovery_chi_square": metric,
            },
            checks=[check],
            numerical_archive=Path(archives[0]).parent,
            figures=[figure],
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        return malformed_result(
            campaign="grid",
            archive=archives[0] if archives else "",
            figures=[figure],
            error=error,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", action="append", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(args.archive, figure=args.figure)
    return write_result(result, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
