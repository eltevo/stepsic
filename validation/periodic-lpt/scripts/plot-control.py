#!/usr/bin/env python3
"""Plot the periodic Gadget-4 1LPT/2LPT control ratio."""

from __future__ import annotations

import argparse
import numpy as np

from validation._common.plotting import atomic_savefig, setup_matplotlib  # noqa: E402

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    setup_matplotlib()
    with np.load(args.input) as data:
        k_h_mpc = data["k_h_mpc"]
        ratio = data["ratio_1lpt_2lpt"]

    figure, axis = plt.subplots(figsize=(3.5, 2.5))
    axis.semilogx(k_h_mpc, ratio, color="tab:blue", lw=1.2)
    axis.axhline(1.0, color="0.35", ls="--", lw=0.8)
    axis.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    axis.set_ylabel(r"$P_{\rm 1LPT}/P_{\rm 2LPT}$")
    figure.tight_layout()
    atomic_savefig(figure, args.output)
    plt.close(figure)


if __name__ == "__main__":
    main()
