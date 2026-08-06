#!/usr/bin/env python3
'''
Plot stepsic displacement and velocity field histograms from ``.npz``
archives produced by :mod:`validate-fields-run`.

Produces a single, vertical, three-panel figure:

    Panel (a): Per-component displacement  (|Psi_x|, |Psi_y|, |Psi_z|)
    Panel (b): Total displacement magnitude |Psi|
    Panel (c): Total velocity magnitude     |v|

Multiple archives can be overlaid on the same figure (e.g. 1LPT vs
2LPT, different redshifts, different box geometries).

Usage
-----
::

    # Single realisation
    python validate_fields_plot.py -i fields_data.npz -o fields.pdf

    # Overlay two realisations (e.g. 1LPT vs 2LPT)
    python validate_fields_plot.py \\
        -i fields_1lpt.npz fields_2lpt.npz \\
        --labels "1LPT" "2LPT" \\
        -o fields_comparison.pdf
'''

from __future__ import annotations

import sys


import argparse
import logging
from typing import Sequence, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from validation import setup_matplotlib, load_archive

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


ArrayF: TypeAlias = NDArray[np.float64]


def _make_label(data: dict, user_label: str | None = None) -> str:
    '''Build a default label from archive metadata if none is given.'''
    if user_label is not None:
        return user_label
    lpt = int(data['meta_lpt_order'])
    z = float(data['meta_redshift'])
    method = str(data['meta_method']).upper()
    lbox = data['meta_lbox']
    return rf'{lpt}LPT, $z={z:g}$, {method}, ${lbox[0]:g}^3$'


def _step_style_histogram(
    ax: plt.Axes,
    centres: ArrayF,
    counts: ArrayF,
    widths: ArrayF,
    **kwargs,
) -> None:
    '''
    Draw a histogram as a step curve, trimming zero-count bins so log-y
    plots do not show artificial vertical drops at the tail.
    '''
    edges = np.empty(centres.size + 1, dtype=np.float64)
    edges[:-1] = centres - 0.5 * widths
    edges[-1] = centres[-1] + 0.5 * widths[-1]

    positive = counts > 0.0
    if not np.any(positive):
        return

    idx = np.flatnonzero(positive)
    i0 = int(idx[0])
    i1 = int(idx[-1]) + 1

    ax.stairs(counts[i0:i1], edges[i0:i1 + 1], **kwargs)


def _draw_panel_components(
    ax: plt.Axes,
    datasets: list[dict],
    labels: list[str],
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    *,
    show_xlabel: bool = False,
    show_ylabel: bool = False,
) -> None:
    '''
    Per-component displacement histograms.

    For each dataset, plot |Psi_x|, |Psi_y|, |Psi_z| as three curves.
    When only one dataset is loaded, the three components use three
    distinct colors. When multiple datasets are overlaid, each
    dataset gets a single color with the components distinguished by
    linestyle (solid/dashed/dotted) to avoid palette exhaustion.

    Used for panel (a) of the figure.
    '''
    comp_labels = [r'$|\Psi_x|$', r'$|\Psi_y|$', r'$|\Psi_z|$']
    comp_suffixes = ['x', 'y', 'z']
    comp_linestyles = ['-', '--', ':']

    multi = len(datasets) > 1
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i_ds, (data, ds_label) in enumerate(zip(datasets, labels)):
        color = prop_cycle[i_ds % len(prop_cycle)]
        for j, (suffix, comp_lbl, ls) in enumerate(
            zip(comp_suffixes, comp_labels, comp_linestyles)
        ):
            c = data[f'hist_disp_{suffix}_centres']
            h = data[f'hist_disp_{suffix}_counts']
            w = data[f'hist_disp_{suffix}_widths']

            if multi:
                label = f'{ds_label} {comp_lbl}'
                _step_style_histogram(
                    ax, c, h, w,
                    color=color, ls=ls, lw=1.0, label=label,
                )
            else:
                _step_style_histogram(
                    ax, c, h, w,
                    ls='-', lw=1.0, label=comp_lbl,
                )

    ax.set_yscale('log')
    if show_ylabel and ylabel is not None:
        ax.set_title(ylabel, loc='left', fontsize=9)
    if show_xlabel and xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=9)

    if title:
        ax.text(
            0.03, 0.94, title,
            transform=ax.transAxes, fontsize=9,
            va='top', ha='left',
        )
    ax.legend(
        fontsize=5, loc='upper right', frameon=False,
        ncol=1, handlelength=1.5, columnspacing=0.8,
    )


def _draw_panel_total(
    ax: plt.Axes,
    datasets: list[dict],
    labels: list[str],
    hist_prefix: str,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    color: str | None = None,
    *,
    show_xlabel: bool = False,
    show_ylabel: bool = False,
    annotate_stats: bool = True,
    stat_prefix: str = 'stat_disp',
) -> None:
    '''
    Draw a single total-magnitude histogram panel.

    Used for total displacement (panel b) and total velocity (panel c).
    '''
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i_ds, (data, ds_label) in enumerate(zip(datasets, labels)):
        if not color:
            color = prop_cycle[i_ds % len(prop_cycle)]
        c = data[f'{hist_prefix}_centres']
        h = data[f'{hist_prefix}_counts']
        w = data[f'{hist_prefix}_widths']
        _step_style_histogram(
            ax, c, h, w,
            color=color, ls='-', lw=1.2, label=ds_label,
        )

        # Annotate summary statistics (median line)
        if annotate_stats and f'{stat_prefix}_median' in data:
            median_val = float(data[f'{stat_prefix}_median'])
            ax.axvline(
                median_val, color='black', ls=':', lw=0.8, alpha=0.6,
            )

    ax.set_yscale('log')
    if show_ylabel and ylabel is not None:
        ax.set_title(ylabel, loc='left', fontsize=9)
    if show_xlabel and xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=9)

    if title:
        ax.text(
            0.03, 0.94, title, fontsize=9,
            va='top', ha='left', transform=ax.transAxes,
        )
    if len(datasets) > 1:
        ax.legend(
            fontsize=5, loc='upper right', frameon=False,
            ncol=1, handlelength=1.5, columnspacing=0.8,
        )


def plot_fields(
    datasets: list[dict],
    labels: list[str],
    output: str | None = None,
) -> None:
    '''Produce the three-panel displacement/velocity histogram figure.

    Parameters
    ----------
    datasets : list of dict
        Archive contents from :func:`load_archive`, one per overlay.
    labels : list of str
        Display labels, one per dataset.
    output : str or None
        Save path. Shows interactively if ``None``.
    '''
    if not datasets:
        log.error('No datasets provided.')
        return

    setup_matplotlib()

    n_panels = 3
    panel_height = 1.1  # inches per panel
    fig_width = 3.46    # inches (A&A single column)
    fig_height = n_panels * (panel_height + 0.65)  # + bottom labels

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(fig_width, fig_height),
        gridspec_kw={'hspace': 0.65},
    )

    # Panel (a): per-component displacements
    _draw_panel_components(
        axes[0], datasets, labels,
        # title=r'(a) Per-component displacement',
        xlabel=r'$|\mathbf{\Psi}_i|$ [$h^{-1}\,\mathrm{kpc}$]',
        ylabel=r'Mean counts $\langle N \rangle$',
        show_xlabel=True,
        show_ylabel=True,
    )
    # axes[0].set_xlabel('')
    # axes[0].tick_params(labelbottom=False)

    # Panel (b): total displacement magnitude
    _draw_panel_total(
        axes[1], datasets, labels,
        hist_prefix='hist_disp_tot',
        # title=r'(b) Total displacement',
        xlabel=r'$|\mathbf{\Psi}|$ [$h^{-1}\,\mathrm{kpc}$]',
        ylabel=r'Mean counts $\langle N \rangle$',
        color='tab:red',
        show_xlabel=True,
        show_ylabel=True,
        annotate_stats=True,
        stat_prefix='stat_disp',
    )
    # axes[1].set_xlabel('')
    # axes[1].tick_params(labelbottom=False)

    # Panel (c): total velocity magnitude
    _draw_panel_total(
        axes[2], datasets, labels,
        hist_prefix='hist_vel_tot',
        # title=r'(c) Total velocity',
        xlabel=r'$|\mathbf{v}|$ [km\,s$^{-1}$]',
        ylabel=r'Mean counts $\langle N \rangle$',
        color='tab:green',
        show_xlabel=True,
        show_ylabel=True,
        annotate_stats=True,
        stat_prefix='stat_vel',
    )

    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.15)

    if output:
        fig.savefig(output, bbox_inches='tight')
        log.info('Figure saved to %s', output)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Plot stepsic displacement/velocity histograms.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '-i', '--input', type=str, nargs='+', required=True,
        help='Input .npz archive(s) from validate_fields_run.py. '
             'Multiple files are overlaid on the same figure.',
    )
    p.add_argument(
        '--labels', type=str, nargs='+', default=None,
        help='Display labels for each input archive. '
             'Defaults to auto-generated labels from metadata.',
    )
    p.add_argument(
        '-o', '--output', type=str, default=None,
        help='Output figure path (e.g. fields.pdf). '
             'Shows interactively if omitted.',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    archives = [load_archive(p) for p in args.input]

    # Build labels
    user_labels = args.labels
    if user_labels is not None and len(user_labels) != len(archives):
        log.error(
            'Number of labels (%d) does not match number of input '
            'files (%d).', len(user_labels), len(archives),
        )
        sys.exit(1)

    labels = [
        _make_label(data, lbl)
        for data, lbl in zip(
            archives,
            user_labels if user_labels else [None] * len(archives),
        )
    ]

    # Log what we loaded
    for path, data, lbl in zip(args.input, archives, labels):
        lbox = data['meta_lbox']
        nmesh = int(data['meta_nmesh'])
        nvox = data['meta_nvox']
        npart = int(data['meta_npart'])
        lpt_order = int(data['meta_lpt_order'])
        z = float(data['meta_redshift'])
        method = str(data['meta_method'])
        log.info(
            'Loaded %s: "%s" | box = %.0fx%.0fx%.0f | '
            'mesh %dx%dx%d | %dLPT %s z=%g | N = %d',
            path, lbl,
            lbox[0], lbox[1], lbox[2],
            nvox[0], nvox[1], nvox[2],
            lpt_order, method.upper(), z, npart,
        )
        if 'stat_disp_median' in data:
            log.info(
                '  |Psi|: median = %.2f, max = %.2f kpc/h',
                float(data['stat_disp_median']),
                float(data['stat_disp_max']),
            )
        if 'stat_vel_median' in data:
            log.info(
                '  |v|: median = %.2f, max = %.2f km/s',
                float(data['stat_vel_median']),
                float(data['stat_vel_max']),
            )

    plot_fields(archives, labels, output=args.output)