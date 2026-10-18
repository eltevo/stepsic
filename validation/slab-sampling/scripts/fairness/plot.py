#!/usr/bin/env python3
'''Plot slab/cut component-variance ratios across independent seeds.'''

from __future__ import annotations



import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

from validation._common.plotting import atomic_savefig, setup_matplotlib
from validation._common.artifacts import load_archive
from validation._common.evaluation import average_realization_members

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _variance_ratio_samples(
    slab_members: np.ndarray,
    cut_members: np.ndarray,
    *,
    nreal: int,
    paired: bool,
) -> np.ndarray:
    slab_realizations = average_realization_members(
        slab_members, nreal=nreal, paired=paired,
    )
    cut_realizations = average_realization_members(
        cut_members, nreal=nreal, paired=paired,
    )
    return slab_realizations / cut_realizations


def _draw_ratio_distributions(
    ax: plt.Axes,
    ratio_samples: np.ndarray,
    *,
    title: str,
    xlabel: str,
) -> None:
    labels = (r'$x$', r'$y$', r'$z$')
    for axis, label in enumerate(labels):
        values = np.sort(ratio_samples[:, axis])
        cumulative_probability = np.arange(1, values.size + 1) / values.size
        ax.step(
            np.concatenate(([values[0]], values)),
            np.concatenate(([0.0], cumulative_probability)),
            where='post', lw=1.3, label=label,
        )
    ax.axvline(1.0, color='k', lw=1, ls=':', zorder=1)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc='left', fontsize=8)


def main() -> None:
    p = argparse.ArgumentParser(
        description='Compare a native slab with an equally sized region cut from a cube.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-i', '--input', type=str, required=True)
    p.add_argument('-o', '--output', type=str, default=None)
    args = p.parse_args()

    d = load_archive(args.input)
    nreal = int(d['meta_nreal'])
    paired = bool(d['meta_paired'])

    setup_matplotlib()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.1, 2.3), sharey=True,
    )
    displacement_ratios = _variance_ratio_samples(
        d['var_disp_slab'], d['var_disp_cut'],
        nreal=nreal, paired=paired,
    )
    velocity_ratios = _variance_ratio_samples(
        d['var_vel_slab'], d['var_vel_cut'],
        nreal=nreal, paired=paired,
    )
    _draw_ratio_distributions(
        ax_a, displacement_ratios,
        title='(a) Displacement variance',
        xlabel=r'$\mathrm{var}(\Psi_i)_{\rm slab}/\mathrm{var}(\Psi_i)_{\rm cut}$',
    )
    _draw_ratio_distributions(
        ax_b, velocity_ratios,
        title='(b) Velocity variance',
        xlabel=r'$\mathrm{var}(v_i)_{\rm slab}/\mathrm{var}(v_i)_{\rm cut}$',
    )
    ax_a.set_ylabel('Empirical cumulative probability')
    ax_b.legend(fontsize=7, loc='lower right', frameon=False)

    fig.tight_layout()
    if args.output:
        atomic_savefig(fig, args.output, bbox_inches='tight')
        log.info('Figure saved to %s', args.output)
    else:
        plt.show()


if __name__ == '__main__':
    main()
