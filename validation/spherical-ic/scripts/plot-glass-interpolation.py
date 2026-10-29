#!/usr/bin/env python3
"""Plot the interpolation to glass particles against sums over modes.

Panel (a) shows the relative RMS difference between the fixed-mesh
displacements and the direct sum of direct-reference.py against the cut-off
of the input spectrum, for both meshes and both glasses. Symbols use all
selected particles and their bars span the radial bins; lines are the error
predicted for CIC interpolation at uniformly distributed positions. Dotted
lines give the size of the second-order term relative to the displacement.

Panel (b) shows the RMS difference between the displacements that two
adjacent multiresolution levels give the same particle, from
periodic-embedding/scripts/mixing-variance.py, against the cell of the
coarser level, with and without the origin shift. Symbols are means
over the seeds with their standard errors; lines are the mode sums.

The strip under each panel is the ratio of measurement to prediction.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from validation._common.evaluation import load_npz
from validation._common.plotting import atomic_savefig, setup_matplotlib

MARKERS = {'spherical': 'o', 'cylindrical': 's'}
NAMES = {'spherical': 'Sphere', 'cylindrical': 'Cylinder'}
MESH_COLORS = {128: '#000000', 256: '#0072B2'}
SHIFT = {'shifted': ('#009E73', 'with shift'),
         'unshifted': ('#D55E00', 'without shift')}


def relative_rms(case, use=slice(None)):
    measured, reference = case['measured_displacement'][use], case['reference_displacement'][use]
    return np.sqrt(np.sum((measured - reference)**2) / np.sum(reference**2))


def plot_fixed_mesh(top, strip, cases):
    cutoffs = sorted({cutoff for _, _, cutoff in cases})
    for mesh, color in MESH_COLORS.items():
        predicted = np.array([cases['spherical', mesh, cutoff]['predicted_displacement_relative_rms']
                              for cutoff in cutoffs])
        top.plot(cutoffs, predicted, color=color, lw=1.1)
        for offset, (geometry, marker) in zip((.97, 1.03), MARKERS.items()):
            for cutoff, prediction in zip(cutoffs, predicted):
                case = cases[geometry, mesh, cutoff]
                value, bins = relative_rms(case), case['displacement_relative_rms']
                bar = np.array([[value - bins.min()], [bins.max() - value]])
                for axis, scale in ((top, 1.), (strip, 1 / prediction)):
                    axis.errorbar(cutoff * offset, value * scale, yerr=bar * scale, color=color,
                                  marker=marker, ms=3.5, lw=.8, capsize=1.5, ls='none')
    for geometry in MARKERS:
        top.plot(cutoffs, [cases[geometry, 256, cutoff]['second_order_relative_rms'] for cutoff in cutoffs],
                 color='0.55', lw=.9, ls=':')
    top.set_xscale('log')
    top.set_yscale('log')
    strip.set_xlabel(r'Cut-off $k_{\rm t}$ [$h\,\mathrm{Mpc}^{-1}$]', fontsize=9)
    strip.set_xticks(cutoffs, labels=[f'{cutoff:.2f}' for cutoff in cutoffs], minor=False)
    strip.tick_params(which='minor', labelbottom=False)
    handles = [Line2D([], [], color=color, lw=1.1, label=rf'${mesh}^3$ mesh') for mesh, color in MESH_COLORS.items()]
    handles += [Line2D([], [], color='0.4', marker=MARKERS[g], ms=3.5, ls='none', label=NAMES[g]) for g in MARKERS]
    handles.append(Line2D([], [], color='0.55', lw=.9, ls=':', label='second-order term'))
    top.legend(handles=handles, fontsize=7, frameon=False, loc='lower right')


def plot_levels(top, strip, mixing):
    for geometry, marker in MARKERS.items():
        cells = mixing[f'{geometry}_cells_mpc_h']
        for label, (color, _) in SHIFT.items():
            predicted = np.array([mixing[f'{geometry}_pair{pair}_{label}_predicted'].sum()
                                  for pair in range(len(cells) - 1)])
            samples = np.array([mixing[f'{geometry}_pair{pair}_{label}_measured'].sum(axis=1)
                                for pair in range(len(cells) - 1)])
            mean = samples.mean(axis=1)
            error = samples.std(axis=1, ddof=1) / np.sqrt(samples.shape[1])
            fill = color if label == 'shifted' else 'white'
            style = dict(color=color, marker=marker, ms=3.5, mfc=fill, lw=.8, capsize=1.5, ls='none')
            top.plot(cells[:-1], np.sqrt(predicted), color=color, lw=1.1)
            top.errorbar(cells[:-1], np.sqrt(mean), yerr=error / (2 * np.sqrt(mean)), **style)
            strip.errorbar(cells[:-1], mean / predicted, yerr=error / predicted, **style)
    top.set_xscale('log')
    top.set_yscale('log')
    top.set_xticks((7, 10, 20, 40, 70), labels=('7', '10', '20', '40', '70'), minor=False)
    top.set_yticks((.05, .1, .2), labels=('0.05', '0.1', '0.2'), minor=False)
    top.tick_params(which='minor', labelleft=False, labelbottom=False)
    strip.tick_params(which='minor', labelbottom=False)
    strip.set_xlabel(r'Cell size of the coarser level [$h^{-1}\mathrm{Mpc}$]', fontsize=9)
    handles = [Line2D([], [], color=color, lw=1.1, label=label) for color, label in SHIFT.values()]
    handles += [Line2D([], [], color='0.4', marker=MARKERS[g], ms=3.5, ls='none', label=NAMES[g]) for g in MARKERS]
    top.legend(handles=handles, fontsize=7, frameon=False, loc='lower right')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', nargs=4, action='append', required=True,
                        metavar=('GEOMETRY', 'NMESH', 'CUTOFF', 'ARCHIVE'))
    parser.add_argument('--mixing-variance', type=Path, required=True,
                        help='archive written by periodic-embedding/scripts/mixing-variance.py')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = {(geometry, int(nmesh), float(cutoff)): load_npz(archive)
             for geometry, nmesh, cutoff, archive in args.case}
    setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 3.5), sharex='col',
                             gridspec_kw=dict(height_ratios=(3, 1), hspace=.3, wspace=.16))
    fig.subplots_adjust(left=.06, right=.985, bottom=.13, top=.94)
    plot_fixed_mesh(axes[0, 0], axes[1, 0], cases)
    plot_levels(axes[0, 1], axes[1, 1], load_npz(args.mixing_variance))
    panels = (('Relative RMS residual', '(a) Fixed mesh'),
              (r'RMS level difference [$h^{-1}\mathrm{Mpc}$]', '(b) Multires. levels'))
    for column, (quantity, label) in enumerate(panels):
        top = axes[0, column]
        top.set_title(quantity, loc='left', fontsize=9)
        top.text(.03, .94, label, transform=top.transAxes, fontsize=8, va='top', ha='left')
        strip = axes[1, column]
        strip.axhline(1, color='0.3', lw=.8, zorder=0)
        strip.set_title('Measured / predicted', loc='left', fontsize=9)
        for axis in axes[:, column]:
            axis.tick_params(labelsize=8)
    axes[1, 1].axhspan(.99, 1.01, color='0.85', lw=0, zorder=-1)
    atomic_savefig(fig, args.output)
    plt.close(fig)


if __name__ == '__main__':
    main()
