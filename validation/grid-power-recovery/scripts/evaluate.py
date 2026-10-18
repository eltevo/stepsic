#!/usr/bin/env python3
"""Check large-scale LPT power recovery against Gaussian sampling variance."""

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
# The figure deliberately exposes near-Nyquist LPT/MAS transfer. Restrict the
# statistical recovery claim to the common large-scale regime, while retaining
# the first complete Fourier shell for the low-resolution small profile.
_COMPARISON_KMAX_FRACTION_NYQUIST = 0.1
_MINIMUM_COMPARISON_KFUND_MULTIPLE = 1.5


def evaluate(
    archives: list[str | Path],
    *,
    figure: str | Path,
) -> ValidationResult:
    try:
        all_ratios: list[np.ndarray] = []
        all_modes: list[np.ndarray] = []
        parameters: dict[str, object] = {
            "panels": [],
            "comparison_kmax_fraction_nyquist": (
                _COMPARISON_KMAX_FRACTION_NYQUIST
            ),
            "comparison_minimum_kfund_multiple": (
                _MINIMUM_COMPARISON_KFUND_MULTIPLE
            ),
        }
        for archive in archives:
            data = load_npz(archive)
            parameters["panels"].append(metadata_parameters(data))
            sample_count = int(data["meta_nreal"])
            lbox_value = np.asarray(data["meta_lbox"], dtype=np.float64)
            if lbox_value.size != 1:
                raise ValueError("meta_lbox must be a positive scalar")
            lbox = float(lbox_value.reshape(-1)[0])
            if not np.isfinite(lbox) or lbox <= 0.0:
                raise ValueError("meta_lbox must be a positive scalar")
            for key, measured in data.items():
                match = LPT_POWER_KEY.match(key)
                if match is None:
                    continue
                ref_key = (
                    f"ref_{match['method']}_{match['nmesh']}_"
                    f"z{match['redshift']}_pk"
                )
                k_key = key.removesuffix("_pk") + "_k"
                modes_key = key.removesuffix("_pk") + "_nmodes"
                k = np.asarray(data[k_key], dtype=np.float64)
                reference = np.asarray(data[ref_key], dtype=np.float64)
                measured = np.asarray(measured, dtype=np.float64)
                modes = np.asarray(data[modes_key], dtype=np.float64)
                if not (
                    k.ndim == reference.ndim == measured.ndim == modes.ndim == 1
                    and k.shape == reference.shape == measured.shape == modes.shape
                ):
                    raise ValueError(f"{key} arrays must be one-dimensional peers")
                if not np.all(np.isfinite(k)) or np.any(k <= 0.0):
                    raise ValueError(f"{k_key} contains invalid wavenumbers")
                if np.any(reference <= 0.0):
                    raise ValueError(f"{ref_key} contains non-positive power")
                nmesh = int(match["nmesh"])
                k_nyquist = np.pi * nmesh / lbox
                k_fundamental = 2.0 * np.pi / lbox
                comparison_kmax = max(
                    _COMPARISON_KMAX_FRACTION_NYQUIST * k_nyquist,
                    _MINIMUM_COMPARISON_KFUND_MULTIPLE * k_fundamental,
                )
                supported = k <= comparison_kmax
                if np.any(supported):
                    all_ratios.append(measured[supported] / reference[supported])
                    all_modes.append(modes[supported] * sample_count)
        if not all_ratios:
            raise ValueError("no LPT power bins in common low-k support")
        ratios = np.concatenate(all_ratios)
        effective_modes = np.concatenate(all_modes)
        parameters["comparison_bin_count"] = int(ratios.size)
        metric, check = gaussian_power_check(
            name="low-k mode-counted LPT power recovery",
            ratios=ratios,
            mode_counts=effective_modes,
            sample_count=1,
            source=(
                "Gaussian Fourier-mode sampling variance on common low-k "
                "support"
            ),
        )
        return ValidationResult(
            campaign="grid-power-recovery",
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
            campaign="grid-power-recovery",
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
