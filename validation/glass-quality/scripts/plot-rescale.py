#!/usr/bin/env python3
'''
Plot residual force after compressing a toroidal glass to several aspect ratios.

The 1:1 glass gives the relaxed reference and the Poisson twin gives the force expected from random particle positions. The rescaled measurements show how ``q = |F| d^2 / m`` changes with aspect ratio.

Usage
-----
::

    python plot-rescale.py --diag 1:diag_s1.npz --diag 2:diag_s2.npz \
        --diag 5:diag_s5.npz --twin diag_twin.npz -o rescale.pdf
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
        description='Plot residual force after compressing a toroidal glass.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--diag', action='append', required=True,
                   metavar='ASPECT:NPZ',
                   help='Aspect ratio and diagnose.py archive; repeatable.')
    p.add_argument('--twin', type=str, default=None,
                   help='Archive containing the Poisson comparison.')
    p.add_argument('-o', '--output', type=str, default=None)
    args = p.parse_args()

    aspects, q_rms, q_med, q_p99 = [], [], [], []
    for spec in args.diag:
        aspect_s, path = spec.split(':', 1)
        d = load_archive(path)
        if not bool(d['meta_has_force']):
            raise ValueError(f'{path} carries no force data.')
        aspects.append(float(aspect_s))
        q_rms.append(float(d['q_all'][0]))
        q_med.append(float(d['q_all'][1]))
        q_p99.append(float(d['q_all'][2]))
    order = np.argsort(aspects)
    aspects = np.asarray(aspects)[order]
    q_rms = np.asarray(q_rms)[order]
    q_med = np.asarray(q_med)[order]
    q_p99 = np.asarray(q_p99)[order]

    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.46, 2.4))
    ax.semilogy(aspects, q_rms, 'o-', lw=1.4, ms=4, label='RMS')
    ax.semilogy(aspects, q_med, 's--', lw=1.2, ms=4, label='median')
    ax.semilogy(aspects, q_p99, '^:', lw=1.2, ms=4,
                label=r'$99^{\mathrm{th}}$ pct.')
    if args.twin:
        t = load_archive(args.twin)
        ax.axhline(float(t['q_all'][0]), color='grey', lw=1.0, ls='-.',
                   label='Poisson twin (RMS)')
    ax.set_xlabel('aspect ratio $s$ ($z$ compressed by $1/s$)')
    ax.set_title(r'residual force $q = |F|\,d^2/m$ (G=1)',
                 loc='left', fontsize=8)
    ax.legend(fontsize=6, loc='best', frameon=False)
    fig.tight_layout()
    if args.output:
        atomic_savefig(fig, args.output, bbox_inches='tight')
        log.info('Figure saved to %s', args.output)
    else:
        plt.show()


if __name__ == '__main__':
    main()
