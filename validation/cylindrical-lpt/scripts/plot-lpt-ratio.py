#!/usr/bin/env python3
'''Plot the 1LPT-to-2LPT power ratio after cylindrical StePS evolution.'''

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

from validation._common.plotting import atomic_savefig, setup_matplotlib


# A&A single-column width [inches]
AA_COL_WIDTH = 3.5


def load_pk_measured(path: str) -> tuple[np.ndarray, np.ndarray]:
    '''Load the k, P(k) columns from a StePS_Pk ASCII output file.

    Parameters
    ----------
    path : str
        Path to the whitespace-delimited ASCII file with columns
        k [1/Mpc] and P(k) [Mpc^3].

    Returns
    -------
    k : ndarray
        Wavenumber array (positive, finite entries only).
    pk : ndarray
        Power spectrum values (positive, finite entries only).
    '''
    data = np.atleast_2d(np.loadtxt(path, comments="#"))
    k = data[:, 0]
    pk = data[:, 1]
    mask = np.isfinite(k) & np.isfinite(pk) & (k > 0) & (pk > 0)
    return k[mask], pk[mask]


def plot_pk_ratio(
    pk_1lpt_file: str,
    pk_2lpt_file: str,
    output: str,
    h: float = 0.6774,
    k_max_invMpc: float = 1.0,
) -> None:
    r'''
    Plot P_1LPT(k) / P_2LPT(k).

    The 2LPT spectrum is log-log interpolated onto the 1LPT k-bins
    so both can live on different grids without drama.

    Input k values are assumed to be in [1/Mpc] and are converted
    to [h/Mpc] for consistency with the other validation figures.

    Parameters
    ----------
    pk_1lpt_file : str
        Path to the 1LPT StePS_Pk output file.
    pk_2lpt_file : str
        Path to the 2LPT StePS_Pk output file.
    output : str
        Output figure path (e.g. ``validation_pk_ratio.pdf``).
    h : float
        Dimensionless Hubble parameter for unit conversion.
    k_max_invMpc : float
        Maximum wavenumber to plot, in [1/Mpc] (before conversion).
    '''
    import matplotlib.pyplot as plt

    setup_matplotlib()

    k_1, pk_1 = load_pk_measured(pk_1lpt_file)
    k_2, pk_2 = load_pk_measured(pk_2lpt_file)
    log.info("Loaded %d k-bins (1LPT) and %d k-bins (2LPT)", len(k_1), len(k_2))

    # --- Unit conversion: k [1/Mpc] -> k [h/Mpc] ---
    # Physical distance: r [Mpc] = r [h^-1 Mpc] / h
    # Therefore:         k [h/Mpc] = k [1/Mpc] / h
    k_1 = k_1 / h
    k_2 = k_2 / h
    k_max = k_max_invMpc / h

    # Cut to k < k_max before interpolation
    cut_1 = k_1 < k_max
    k_1, pk_1 = k_1[cut_1], pk_1[cut_1]
    cut_2 = k_2 < k_max
    k_2, pk_2 = k_2[cut_2], pk_2[cut_2]

    # Interpolate 2LPT onto 1LPT k-bins in log-log space.  A
    # singleton grid cannot be interpolated, but matching bin centres can be
    # compared directly.
    if len(k_1) == 1 and len(k_2) == 1:
        if not np.isclose(k_1[0], k_2[0], rtol=1e-3):
            raise ValueError("singleton spectrum bins do not overlap")
        ratio = pk_1 / pk_2
    else:
        log_pk_2_interp = np.interp(
            np.log(k_1), np.log(k_2), np.log(pk_2),
            left=np.nan, right=np.nan,
        )
        ratio = pk_1 / np.exp(log_pk_2_interp)

    valid = np.isfinite(ratio)
    if not np.any(valid):
        raise ValueError("power spectra have no overlapping finite bins")

    fig, ax = plt.subplots(1, 1, figsize=(AA_COL_WIDTH, 2.2))

    # Shaded +-1% band (matching Fig. 5 caption style)
    ax.axhspan(0.99, 1.01, color='grey', alpha=0.10, zorder=0)

    # Unity reference
    ax.axhline(1.0, color="gray", ls="--", lw=0.5, zorder=1)

    ax.plot(k_1[valid], ratio[valid], lw=1.2, color="k", zorder=2)

    ax.set_xscale("log")
    ax.set_xlim(k_1[valid].min(), k_max)
    ax.set_ylim(0.988, 1.042)
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel(r"$k\;[h\,\mathrm{Mpc}^{-1}]$", fontsize=9)
    ax.set_title(
        r"$P_{\rm 1LPT}(k) \,/\, P_{\rm 2LPT}(k)$",
        loc="left", fontsize=9,
    )

    fig.tight_layout()
    atomic_savefig(fig, output, dpi=300, bbox_inches="tight")
    log.info("Figure saved to %s", output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot P(k) ratio 1LPT / 2LPT for cylindrical StePS "
            "simulations."
        ),
    )
    parser.add_argument(
        "--pk-1lpt", type=str, required=True,
        help="Path to the 1LPT StePS_Pk output ASCII file.",
    )
    parser.add_argument(
        "--pk-2lpt", type=str, required=True,
        help="Path to the 2LPT StePS_Pk output ASCII file.",
    )
    parser.add_argument(
        "--h", type=float, default=0.6766,
        help="Dimensionless Hubble parameter (default: 0.6766).",
    )
    parser.add_argument(
        "-o", "--output", type=str,
        default="cylindrical-pk-ratio.pdf",
        help="Output figure path (default: cylindrical-pk-ratio.pdf).",
    )
    args = parser.parse_args()

    plot_pk_ratio(
        pk_1lpt_file=args.pk_1lpt,
        pk_2lpt_file=args.pk_2lpt,
        output=args.output,
        h=args.h,
    )


if __name__ == "__main__":
    main()
