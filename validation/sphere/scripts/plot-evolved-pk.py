#!/usr/bin/env python3
"""Plot full-domain spectra and diagnostics against common Halofit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile

import numpy as np

from validation import setup_matplotlib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    setup_matplotlib()
    with np.load(args.input, allow_pickle=False) as data:
        k_steps = data["k_steps_native_h_mpc"]
        steps = data["pk_steps_native_mpc_h3"]
        k_periodic = data["k_periodic_native_h_mpc"]
        periodic = data["pk_periodic_native_mpc_h3"]
        k_common = data["k_common_h_mpc"]
        halofit = data["pk_halofit_common_mpc_h3"]
        ratio_steps = data["ratio_steps_halofit"]
        ratio_periodic = data["ratio_periodic_halofit"]
        direct_ratio = data["ratio_steps_periodic"]
    figure, axes = plt.subplots(2, 1, figsize=(4.2, 5.0), sharex=True)
    axes[0].loglog(k_steps, steps, label="StePS (FKP)")
    axes[0].loglog(k_periodic, periodic, label="periodic (FFT)")
    axes[0].loglog(k_common, halofit, color="0.2", ls="--", label="CAMB/Halofit")
    axes[0].set_ylabel(r"$P(k)\ [({\rm Mpc}/h)^3]$")
    axes[0].legend(frameon=False)
    axes[1].semilogx(k_common, ratio_steps, label="StePS/Halofit")
    axes[1].semilogx(k_common, ratio_periodic, label="periodic/Halofit")
    axes[1].semilogx(k_common, direct_ratio, ls=":", label="StePS/periodic")
    axes[1].axhline(1.0, color="0.3", ls="--", lw=0.8)
    axes[1].set_xlabel(r"$k\ [h/{\rm Mpc}]$")
    axes[1].set_ylabel("ratio")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-", dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(
            temporary,
            format=output.suffix.lstrip("."),
            bbox_inches="tight",
            metadata={"CreationDate": datetime(2000, 1, 1, tzinfo=timezone.utc)},
        )
        file_descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)


if __name__ == "__main__":
    main()
