#!/usr/bin/env python3
"""Plot matched-shell angular spectra and their signed differences."""

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
        ell = data["ell"]
        shells = data["shell_bounds_mpc_h"]
        steps = data["cl_steps"]
        periodic = data["cl_periodic"]
        difference = data["cl_signed_difference"]
    figure, axes = plt.subplots(len(shells), 2, figsize=(6.8, 2.4 * len(shells)))
    axes = np.atleast_2d(axes)
    for index, bounds in enumerate(shells):
        label = f"{bounds[0]:g}–{bounds[1]:g} Mpc/h"
        axes[index, 0].semilogy(ell[1:], steps[index, 1:], label="StePS")
        axes[index, 0].semilogy(ell[1:], periodic[index, 1:], label="periodic")
        axes[index, 0].set_title(label)
        axes[index, 0].set_ylabel(r"$C_\ell$")
        axes[index, 1].plot(ell[1:], difference[index, 1:])
        axes[index, 1].axhline(0.0, color="0.3", ls="--", lw=0.8)
        axes[index, 1].set_ylabel(r"$C_\ell^{\rm StePS}-C_\ell^{\rm periodic}$")
    axes[0, 0].legend(frameon=False)
    axes[-1, 0].set_xlabel(r"$\ell$")
    axes[-1, 1].set_xlabel(r"$\ell$")
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
