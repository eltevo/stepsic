#!/usr/bin/env python3
'''
Plot the agreement of stepsic and monofonIC order by order.

Reads the archives of compare-orders.py, one per mesh size, and draws for the
isolated first-order and second-order displacement modes

    (a) the stepsic / monofonIC amplitude ratio per wavenumber shell,
    (b) their cross-correlation coefficient,

and for the total 2LPT displacement

    (c) the RMS difference per shell relative to the monofonIC displacement,

against k / k_Ny.

Usage
-----
::

    python plot-isolated-orders.py isolated-orders-64.npz \\
        isolated-orders-128.npz isolated-orders-256.npz \\
        --output isolated-orders.pdf
'''

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from validation._common.plotting import atomic_savefig, setup_matplotlib

ORDERS = (('first', 'first order', '-'), ('second', 'second order', '--'))
COLORS = ('#E69F00', '#0072B2', '#D55E00')


def load(path):
    '''Return mesh size, k / k_Ny and the statistics of one archive.'''
    with np.load(path) as archive:
        mesh = int(archive['meta_nmesh'])
        k_ny = np.pi * mesh / float(archive['meta_box_mpc_h'])
        return mesh, archive['k_h_mpc'] / k_ny, {
            'difference': archive['total_displacement_relative_difference'],
            **{
                order: {
                    'ratio': archive[f'{order}_displacement_transfer'],
                    'correlation': archive[
                        f'{order}_displacement_cross_correlation'
                    ],
                }
                for order, _, _ in ORDERS
            },
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('archives', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    setup_matplotlib()

    runs = sorted((load(path) for path in args.archives), key=lambda r: r[0])

    fig, axes = plt.subplots(
        3, 1, figsize=(3.46, 5.0), sharex=True, gridspec_kw={'hspace': 0.35},
    )
    for (mesh, x, stats), color in zip(runs, COLORS):
        for order, _, style in ORDERS:
            axes[0].semilogx(x, stats[order]['ratio'], style, color=color, lw=1.2)
            axes[1].semilogx(
                x, stats[order]['correlation'], style, color=color, lw=1.2,
            )
        axes[2].loglog(x, stats['difference'], color=color, lw=1.2)
    axes[0].axhline(1, color='k', lw=1, ls=':', zorder=1)
    axes[0].set_title(
        r'$|\Psi_\mathrm{stepsic}| / |\Psi_\mathrm{monofonIC}|$',
        loc='left', fontsize=9,
    )
    axes[1].axhline(1, color='k', lw=1, ls=':', zorder=1)
    axes[1].set_ylim(0.75, 1.01)
    axes[1].set_title(r'Correlation coefficient $r$', loc='left', fontsize=9)
    axes[2].set_title(
        r'$|\Psi_\mathrm{stepsic}-\Psi_\mathrm{monofonIC}| / '
        r'|\Psi_\mathrm{monofonIC}|$',
        loc='left', fontsize=9,
    )
    axes[2].set_xlabel(r'$k / k_\mathrm{Ny}$', fontsize=9)
    # Panel (b) has data along its top edge, so its letter sits at the bottom.
    for ax, letter, y, va in zip(
        axes, ('(a)', '(b)', '(c) 2LPT'), (0.91, 0.09, 0.91),
        ('top', 'bottom', 'top'),
    ):
        ax.text(
            0.03, y, letter, transform=ax.transAxes, fontsize=8,
            va=va, ha='left',
        )
    # Two legend rows below panel (c): line styles above, meshes below.
    rows = (
        [
            plt.Line2D([], [], color='k', ls=style, lw=1.2, label=label)
            for _, label, style in ORDERS
        ],
        [
            plt.Line2D([], [], color=color, lw=1.2, label=rf'${mesh}^3$')
            for (mesh, _, _), color in zip(runs, COLORS)
        ],
    )
    for handles, y in zip(rows, (-0.36, -0.52)):
        fig.legend(
            handles=handles, fontsize=7, loc='upper center', frameon=False,
            ncol=len(handles), bbox_to_anchor=(0.5, y),
            bbox_transform=axes[2].transAxes,
        )
    atomic_savefig(fig, args.output, bbox_inches='tight')


if __name__ == '__main__':
    main()
