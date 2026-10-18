#!/usr/bin/env python3
"""Plot the relative displacement and the opposite-point correlation against the padding.

Left panels show the root mean square relative displacement sigma_12 of two
points against their separation s in units of the domain diameter 2R, along a
box axis (solid) and along a diagonal (dashed), for open space and for periodic
boxes of 1, 2 and 3 domain diameters. Right panels show the correlation of the
displacements at opposite boundary points against the angle of their axis from
a box axis. Lines are the mode sums of two-point.py; points are the seed
ensemble of monopole.py, the ratio of the summed product to the summed square
of the two displacements, with the standard deviation over 2000 bootstrap
resamples of the seeds as error bars. The black line is the open-space value.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from validation._common.evaluation import load_npz
from validation._common.plotting import atomic_savefig, setup_matplotlib

PADDINGS = (1., 2., 3.)
CUTS = {'spherical': (('axis', '-'), ('body_diagonal', '--')),
        'cylindrical': (('axis', '-'), ('diagonal', '--'))}
FAMILIES = {'spherical': 'face', 'cylindrical': 'transverse'}
NAMES = {'spherical': 'Sphere', 'cylindrical': 'Cylinder'}
BOOTSTRAP = 2000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--two-point', type=Path, required=True, help='archive written by two-point.py')
    parser.add_argument('--ensemble', type=Path, required=True, help='archive written by monopole.py')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data, ensemble = load_npz(args.two_point), load_npz(args.ensemble)
    paddings = list(data['paddings'])
    missing = [a for a in PADDINGS if a not in paddings]
    if missing:
        raise ValueError(f'paddings {missing} are missing from {args.two_point}; it has {paddings}')
    seeds = len(ensemble['seeds'])
    draws = np.random.default_rng(1).integers(seeds, size=(BOOTSTRAP, seeds))
    setup_matplotlib()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    shown = dict(zip(PADDINGS, (colors[6], colors[5], colors[3])))

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.8), sharex='col')
    for row, geometry in enumerate(CUTS):
        ax = axes[row, 0]
        diameter = data[f'{geometry}_axis_separation_mpc_h'][-1]
        for name, ls in CUTS[geometry]:
            s = data[f'{geometry}_{name}_separation_mpc_h'] / diameter
            if ls == '-':
                ax.plot(s, data[f'{geometry}_{name}_sqrt_s_reference'], color='k', lw=2.2,
                        label='infinite Universe')
            for alpha, color in shown.items():
                label = rf'box of {alpha:g} diameter' + 's' * (alpha > 1) if ls == '-' else None
                ax.plot(s, data[f'{geometry}_{name}_sqrt_s'][paddings.index(alpha)],
                        color=color, lw=1.1, ls=ls, label=label)
        ax.set_ylim(0, .68)
        ax.set_xlim(0, 1.02)
        ax.text(.03, .94, f'({"ac"[row]}) {NAMES[geometry]}', transform=ax.transAxes,
                fontsize=8, va='top', ha='left')
        ax.tick_params(labelsize=8)

        ax = axes[row, 1]
        family = FAMILIES[geometry]
        degrees = data[f'{geometry}_opposite_{family}_degrees']
        ax.axhline(data[f'{geometry}_opposite_correlation_reference'], color='k', lw=2.2)
        for offset, (alpha, color) in zip((-.4, 0., .4), shown.items()):
            ax.plot(degrees, data[f'{geometry}_opposite_{family}_correlation'][paddings.index(alpha)],
                    color=color, lw=1.1)
            key = f'{geometry}_a{int(alpha)}_opposite_{family}'
            if f'{key}_product' not in ensemble:
                raise ValueError(f'padding {alpha:g} is missing from {args.ensemble}')
            product, square = ensemble[f'{key}_product'], ensemble[f'{key}_square']
            boot = product[draws].sum(axis=1) / square[draws].sum(axis=1)
            ax.errorbar(degrees + offset, product.sum(axis=0) / square.sum(axis=0), boot.std(axis=0),
                        color=color, fmt='o', ms=2.5, lw=.8, capsize=1.5)
        ax.set_ylim(-.12, 1.05)
        ax.text(.97, .94, f'({"bd"[row]}) {NAMES[geometry]}', transform=ax.transAxes,
                fontsize=8, va='top', ha='right')
        ax.tick_params(labelsize=8)
    axes[0, 0].set_title(r'$\sigma_{12}$ [$h^{-1}\,$Mpc]', loc='left', fontsize=9)
    axes[0, 1].set_title('Correlation of opposite points', loc='left', fontsize=9)
    axes[1, 0].set_xlabel(r'separation $s/(2R)$', fontsize=9)
    axes[1, 1].set_xlabel('angle from the box axis [deg]', fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, .05, 1, 1))
    atomic_savefig(fig, args.output)
    plt.close(fig)
    print(f'Relative displacement and opposite-point correlation figure: {args.output}')


if __name__ == '__main__':
    main()
