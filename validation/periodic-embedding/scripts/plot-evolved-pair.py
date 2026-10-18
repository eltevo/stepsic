#!/usr/bin/env python3
"""Plot the power spectrum of the evolved box of two diameters over that of one diameter."""

import argparse

import matplotlib.pyplot as plt

from validation._common.evaluation import load_npz
from validation._common.plotting import atomic_savefig, setup_matplotlib

# A&A single-column width [inches]
WIDTH = 3.46
BAND = 0.01


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Archive of evolved-pair.py.')
    parser.add_argument('--output', required=True, help='Output figure.')
    args = parser.parse_args()
    data = load_npz(args.input)
    setup_matplotlib()

    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    ax.axhline(1., color='k', lw=1, ls=':', zorder=1)
    ax.axhspan(1 - BAND, 1 + BAND, color='grey', alpha=0.10, zorder=0)
    ax.semilogx(data['k_h'], data['ratio_same_randoms'], lw=1.2, label='same volume')
    ax.semilogx(data['k_h'], data['ratio_moved_randoms'], lw=1.2, label='moved back')
    ax.set_title(r'$P_{\alpha=2}/P_{\alpha=1}$', loc='left', fontsize=9)
    ax.set_xlabel(r'$k$ [$h\,\mathrm{Mpc}^{-1}$]', fontsize=9)
    ax.set_ylim(0.97, 1.01)
    ax.legend(fontsize=6, loc='lower right', frameon=False)
    fig.tight_layout()
    atomic_savefig(fig, args.output, bbox_inches='tight')


if __name__ == '__main__':
    main()
