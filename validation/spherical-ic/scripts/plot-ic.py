#!/usr/bin/env python3
'''
Plot the embedded-to-periodic initial-condition comparison.

The panels show relative velocity difference by radius, the core power ratio, and velocity variance along each axis.

Usage
-----
::

    python plot-ic.py -i compare.npz -o sphere-ic.pdf
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


def main() -> None:
    p = argparse.ArgumentParser(
        description='Plot embedded and periodic initial conditions side by side.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-i', '--input', type=str, required=True)
    p.add_argument('-o', '--output', type=str, default=None)
    args = p.parse_args()

    d = load_archive(args.input)
    geometry = str(d['meta_geometry'])
    rcore = float(d['meta_rcore'])
    rel = float(d['rms_dv']) / float(d['rms_ref'])
    log.info('%s vs cube: overall relative velocity residual %.4e '
             '(%d cells)', geometry, rel, int(d['meta_n_cells']))

    setup_matplotlib()
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(7.1, 2.3))

    r_mid = 0.5 * (d['prof_edges'][:-1] + d['prof_edges'][1:]) / rcore
    ax_a.semilogy(r_mid, d['prof_rel'], 'o-', lw=1.2, ms=3)
    ax_a.set_xlabel(r'$r / r_{\mathrm{core}}$')
    ax_a.set_title(
        r'RMS $|\mathbf{v}_{\rm geom} - \mathbf{v}_{\rm ref}|\,/\,'
        r'\mathrm{RMS}\,|\mathbf{v}_{\rm ref}|$',
        loc='left', fontsize=8,
    )

    ratio = d['pk_geom'] / d['pk_ref']
    ax_b.axhline(1.0, color='k', lw=1, ls=':', zorder=1)
    ax_b.semilogx(d['pk_k'], ratio, '-', lw=1.4, zorder=3)
    ax_b.set_xlabel(r'$k$ [$h\,\mathrm{Mpc}^{-1}$]')
    ax_b.set_title(r'$P_{\rm geom}(k) / P_{\rm ref}(k)$ (core window)',
                   loc='left', fontsize=8)

    x = np.arange(3)
    w = 0.35
    ax_c.bar(x - w / 2, d['var_geom'], w, label='geometry IC')
    ax_c.bar(x + w / 2, d['var_ref'], w, label='reference cube')
    ax_c.set_xticks(x, [r'$v_x$', r'$v_y$', r'$v_z$'])
    ax_c.set_title(r'core $\mathrm{var}(v_i)$ [$\mathrm{km^2\,s^{-2}}$]',
                   loc='left', fontsize=8)
    ax_c.legend(fontsize=6, loc='lower right', frameon=False)

    fig.tight_layout()
    if args.output:
        atomic_savefig(fig, args.output, bbox_inches='tight')
        log.info('Figure saved to %s', args.output)
    else:
        plt.show()


if __name__ == '__main__':
    main()
