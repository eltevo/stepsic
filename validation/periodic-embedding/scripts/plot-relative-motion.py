#!/usr/bin/env python3
"""Plot measured and predicted changes in RMS relative particle displacement.

Four panels (sphere and cylinder, central region and whole domain) show the
change of the RMS relative displacement of particle pairs, in percent, against
the pair separation in units of the domain diameter, when the periodic box grows
from 1 to 2 and from 2 to 3 domain diameters. Solid lines are the measurement,
with its confidence band; crosses mark bins whose result depends on the pair
sampling. Dashed lines in the same colours are the change predicted by the mode
sums of two-point.py for the same particle pairs.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from validation._common.evaluation import load_npz
from validation._common.plotting import atomic_savefig, setup_matplotlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement', type=Path, required=True,
                        help='relative-motion.npz written by relative-motion.py measure')
    parser.add_argument('--prediction', type=Path, required=True,
                        help='archive written by two-point.py with the glass inputs and pairs')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = load_npz(args.measurement)
    prediction = load_npz(args.prediction)
    if list(prediction['pair_paddings']) != [1, 2, 3]:
        raise ValueError(f'{args.prediction} has pair paddings {prediction["pair_paddings"].tolist()}, expected [1, 2, 3]')
    setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.6), sharex=True)
    for gi, geometry in enumerate(('Sphere', 'Cylinder')):
        for ri, region in enumerate(('Central region', 'Whole domain')):
            ax = axes[gi, ri]
            geometry_key = ('spherical', 'cylindrical')[gi]
            region_key = ('central', 'whole')[ri]
            separation = data[f'{geometry_key}_{region_key}_separation_mpc_h']/(2*data[f'{geometry_key}_radius_mpc_h'])
            for comparison, label in enumerate((r'box width $1\to2$ diameters', r'box width $2\to3$ diameters')):
                values = data['change_percent'][comparison, gi, ri]
                lower = data['lower_percent'][comparison, gi, ri]
                upper = data['upper_percent'][comparison, gi, ri]
                line, = ax.plot(separation, values, lw=1.3, label=label)
                ax.fill_between(separation, lower, upper, color=line.get_color(), alpha=.18, lw=0)
                modes = prediction[f'{geometry_key}_{region_key}_pair_s']
                ax.plot(separation, 100 * (np.sqrt(modes[comparison + 1] / modes[comparison]) - 1),
                        color=line.get_color(), lw=1.1, ls='--')
                sensitive = data['sampling_sensitive'][comparison, gi, ri]
                ax.plot(separation[sensitive], values[sensitive], 'x', color=line.get_color(), ms=5)
            ax.axhline(0, color='.5', lw=.7, ls=':')
            ax.set_xscale('log')
            ax.set_title(f'{geometry}: {region.lower()}', fontsize=9)
            ax.tick_params(labelsize=8)
            if gi == 1:
                ax.set_xlabel(r'Pair separation $s/(2R)$', fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([], [], color='.3', lw=1.1, ls='--'))
    labels.append('dashed: sums over the modes of each box')
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .015),
               ncol=3, fontsize=7, frameon=False, handlelength=1.5,
               columnspacing=1.2, handletextpad=.5)
    fig.supylabel(r'Change in RMS relative displacement [\%]', fontsize=9, x=.012)
    fig.tight_layout(rect=(.025, .085, 1, 1))
    atomic_savefig(fig, args.output)
    plt.close(fig)
    print(f'Four-panel relative-displacement figure with mode-sum predictions: {args.output}')


if __name__ == '__main__':
    main()
