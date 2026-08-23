#!/usr/bin/env python3
'''
Plot a native slab beside an equally sized region cut from a cube.

The panels compare isotropic power, transverse and line-of-sight power, and displacement variance. Shaded bands show variation between random fields.

Usage
-----
::

    python plot.py -i slab_data.npz -o slab.pdf
'''

from __future__ import annotations



import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

from validation._common.plotting import atomic_savefig, setup_matplotlib
from validation._common.artifacts import load_archive

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _ratio(num: np.ndarray, den: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = num / den
    finite = np.isfinite(ratio)
    count = np.sum(finite, axis=0)
    total = np.sum(np.where(finite, ratio, 0.0), axis=0)
    mean = np.divide(
        total, count, out=np.full_like(total, np.nan), where=count > 0
    )
    squared = np.sum(
        np.where(finite, (ratio - mean) ** 2, 0.0), axis=0
    )
    standard_deviation = np.sqrt(
        np.divide(
            squared, count, out=np.full_like(total, np.nan), where=count > 0
        )
    )
    return mean, standard_deviation


def main() -> None:
    p = argparse.ArgumentParser(
        description='Compare a native slab with an equally sized region cut from a cube.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-i', '--input', type=str, required=True)
    p.add_argument('-o', '--output', type=str, default=None)
    args = p.parse_args()

    d = load_archive(args.input)
    k = d['pk_k']
    aspect = int(d['meta_aspect'])
    mu_split = float(d['meta_mu_split'])

    setup_matplotlib()
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(7.1, 2.3))

    # -- (a) isotropic ratio -------------------------------------------------
    m, s = _ratio(d['pk_slab'], d['pk_cut'])
    ax_a.axhline(1.0, color='k', lw=1, ls=':', zorder=1)
    ax_a.semilogx(k, m, '-', lw=1.4, zorder=3)
    ax_a.fill_between(k, m - s, m + s, alpha=0.15, lw=0, zorder=2)
    ax_a.set_xlabel(r'$k$ [$h\,\mathrm{Mpc}^{-1}$]')
    ax_a.set_title(
        rf'$P_{{\rm slab}}(k)/P_{{\rm cut}}(k)$ ({aspect}:1, isotropic)',
        loc='left', fontsize=8,
    )

    # -- (b) direction-resolved ratios ---------------------------------------
    ax_b.axhline(1.0, color='k', lw=1, ls=':', zorder=1)
    for num, den, ls, lbl in (
        (d['pk_slab_t'], d['pk_cut_t'], '-',
         rf'$\mu < {mu_split:g}$ (transverse)'),
        (d['pk_slab_l'], d['pk_cut_l'], '--',
         rf'$\mu \geq {mu_split:g}$ (line of sight)'),
    ):
        m, s = _ratio(num, den)
        ax_b.semilogx(k, m, ls, lw=1.4, zorder=3, label=lbl)
        ax_b.fill_between(k, m - s, m + s, alpha=0.15, lw=0, zorder=2)
    ax_b.set_xlabel(r'$k$ [$h\,\mathrm{Mpc}^{-1}$]')
    ax_b.set_title('direction-resolved ratio', loc='left', fontsize=8)
    ax_b.legend(fontsize=6, loc='best', frameon=False)

    # -- (c) displacement variance ratio -------------------------------------
    r = d['var_slab'] / d['var_cut']
    x = np.arange(3)
    ax_c.axhline(1.0, color='k', lw=1, ls=':', zorder=1)
    ax_c.errorbar(x, r.mean(axis=0), yerr=r.std(axis=0),
                  fmt='o', ms=5, capsize=3, zorder=3)
    ax_c.set_xticks(x, [r'$\Psi_x$', r'$\Psi_y$', r'$\Psi_z$'])
    ax_c.set_title(r'$\mathrm{var}(\Psi_i)$: slab / cut',
                   loc='left', fontsize=8)

    fig.tight_layout()
    if args.output:
        atomic_savefig(fig, args.output, bbox_inches='tight')
        log.info('Figure saved to %s', args.output)
    else:
        plt.show()


if __name__ == '__main__':
    main()
