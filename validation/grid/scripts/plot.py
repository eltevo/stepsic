#!/usr/bin/env python3
'''
Plot stepsic P(k) validation results from ``.npz`` archives produced
by :mod:`validate_run`.

Produces a single four-panel figure (vertical stack, shared x-axis)
designed for a single-column A&A layout:

    Panel A, Resolution dependence (multiple nmesh, fixed z/LPT/method)
    Panel B, Redshift dependence   (multiple z, fixed nmesh/LPT/method)
    Panel C, LPT order comparison  (1LPT vs 2LPT, fixed nmesh/z/method)
    Panel D, MAS scheme comparison (NGP/CIC/TSC, fixed nmesh/z/LPT)

Each panel can be loaded from a separate ``.npz`` file, or all four
can live in a single archive if the run covered the full parameter
space.

Usage
-----
::

    python validate_pk.py \\
        --panel-a panel_a.npz \\
        --panel-b panel_b.npz \\
        --panel-c panel_c.npz \\
        --panel-d panel_d.npz \\
        -o validation_figure.pdf
'''

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass, field
from typing import Sequence, TypeAlias


from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from validation import setup_matplotlib, load_archive

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


ArrayF: TypeAlias = NDArray[np.float64]
ArrayI: TypeAlias = NDArray[np.int64]

# MAS display names
MAS_LABELS = {'ngp': 'NGP', 'cic': 'CIC', 'tsc': 'TSC'}


def _z_tag(z: float) -> str:
    '''Reproduce the key-formatting convention from validate_run.'''
    if np.isclose(z, round(z)):
        return str(int(round(z)))
    return f'{z:.1f}'


@dataclass
class CurveData:
    '''One P(k) ratio curve ready for plotting.'''
    k: ArrayF
    ratio: ArrayF
    nmesh: int
    label: str
    k_ny: float  # Nyquist wavenumber [h/Mpc]


def _load_lpt_ratio(
    data: dict,
    lpt_order: int,
    method: str,
    nmesh: int,
    z: float,
    lbox: float,
) -> CurveData | None:
    '''Load one LPT P(k) and its matching reference, return the ratio.

    Returns ``None`` if the keys are not found in ``data``.
    '''
    zt = _z_tag(z)
    lpt_key = f'lpt{lpt_order}_{method}_{nmesh}_z{zt}'
    ref_key = f'ref_{method}_{nmesh}_z{zt}'

    try:
        k = data[f'{lpt_key}_k']
        pk = data[f'{lpt_key}_pk']
        pk_ref = data[f'{ref_key}_pk']
    except KeyError:
        log.warning('Keys not found for %s (ref: %s)', lpt_key, ref_key)
        return None

    ratio = pk / pk_ref

    # Nyquist wavenumber
    from stepsic.field import cubic_voxels
    boxsize = np.array([lbox] * 3, dtype=np.float64)
    _, dk = cubic_voxels(nmesh, boxsize)
    k_ny = np.pi / dk

    return CurveData(k=k, ratio=ratio, label='', k_ny=k_ny, nmesh=nmesh)


def _get_lbox(data: dict) -> float:
    '''Extract box size from archive metadata.'''
    return float(data['meta_lbox'])


# Each function returns a list of CurveData with labels set.
def build_panel_a(
    data: dict,
    nmesh_list: Sequence[int],
    lpt_order: int = 2,
    method: str = 'cic',
    z: float = 31.0,
) -> list[CurveData]:
    '''Panel A: resolution dependence.'''
    lbox = _get_lbox(data)
    curves = []
    for nmesh in nmesh_list:
        c = _load_lpt_ratio(data, lpt_order, method, nmesh, z, lbox)
        if c is not None:
            c.label = rf'${nmesh}^3$'
            curves.append(c)
    return curves


def build_panel_b(
    data: dict,
    z_list: Sequence[float],
    nmesh: int = 128,
    lpt_order: int = 2,
    method: str = 'cic',
) -> list[CurveData]:
    '''Panel B: redshift dependence.'''
    lbox = _get_lbox(data)
    curves = []
    for z in z_list:
        c = _load_lpt_ratio(data, lpt_order, method, nmesh, z, lbox)
        if c is not None:
            c.label = rf'$z = {z:g}$'
            curves.append(c)
    return curves


def build_panel_c(
    data: dict,
    lpt_orders: Sequence[int] = (1, 2),
    nmesh: int = 128,
    z: float = 15.0,
    method: str = 'cic',
) -> list[CurveData]:
    '''Panel C: 1LPT vs 2LPT.'''
    lbox = _get_lbox(data)
    curves = []
    for order in lpt_orders:
        c = _load_lpt_ratio(data, order, method, nmesh, z, lbox)
        if c is not None:
            c.label = rf'${order}$LPT'
            curves.append(c)
    return curves


def build_panel_d(
    data: dict,
    methods: Sequence[str] = ('ngp', 'cic', 'tsc'),
    nmesh: int = 256,
    lpt_order: int = 2,
    z: float = 31.0,
) -> list[CurveData]:
    '''Panel D: MAS scheme comparison.'''
    lbox = _get_lbox(data)
    curves = []
    for method in methods:
        c = _load_lpt_ratio(data, lpt_order, method, nmesh, z, lbox)
        if c is not None:
            c.label = MAS_LABELS.get(method, method.upper())
            curves.append(c)
    return curves


def _draw_panel(
    ax: plt.Axes,
    curves: list[CurveData],
    title: str,
    *,
    show_xlabel: bool = False,
    show_ylabel: bool = False,
    ylim: tuple[float, float] = (0.99, 1.01),
    percent_band: float = 0.005,
) -> None:
    '''Draw one ratio panel.'''
    # Reference line and +-0.5% band
    ax.axhline(1.0, color='k', lw=1, ls=':', zorder=1)
    ax.axhspan(
        1.0 - percent_band, 1.0 + percent_band,
        color='grey', alpha=0.10, zorder=0,
        #label=rf'$\pm\,{percent_band*100:.1f}\%$',
    )

    # Data curves
    for curve in curves:
        ax.semilogx(
            curve.k, curve.ratio, '-', lw=1.2, label=curve.label, zorder=3,
        )

    # Nyquist markers, collect unique values
    seen_ny: set[float] = set()
    for curve in curves:
        k_ny_rounded = round(curve.k_ny, 4)
        if k_ny_rounded not in seen_ny:
            seen_ny.add(k_ny_rounded)
            ax.axvline(
                curve.k_ny,
                color='grey',
                ls=':',
                lw=0.6,
                alpha=0.5,
                zorder=2,
                label='_nolegend_',
            )
            ax.annotate(
                fr'${curve.nmesh}^3$',
                xy=(curve.k_ny, 0.1),
                xycoords=ax.get_xaxis_transform(),
                fontsize=6,
                color='grey',
                rotation=90,
                ha='right',
                va='bottom',
            )

    ax.set_ylim(*ylim)
    if show_xlabel:
        ax.set_xlabel(r'$k$ [$h\,\mathrm{Mpc}^{-1}$]', fontsize=9)
    if show_ylabel:
        ax.set_title(r'$P_{\mathrm{meas}} (k) / P_{\mathrm{ref}} (k)$', loc='left', fontsize=9)

    ax.text(
        0.03, 0.91, title,
        transform=ax.transAxes, fontsize=8,
        va='top', ha='left',
    )

    handles, labels = ax.get_legend_handles_labels()

    # Add one proxy handle for all Nyquist lines
    # nyquist_handle = Line2D(
    #     [0], [0], color='grey', ls=':', lw=0.6, alpha=0.5,
    #     label=r'$k_\mathrm{Ny}$',
    # )
    # handles.append(nyquist_handle)
    # labels.append(r'$k_\mathrm{Ny}$')

    ax.legend(
        handles, labels,
        fontsize=6,
        loc='lower left',
        frameon=False,
        ncol=len(curves) + 2,
        handlelength=1.5,
        columnspacing=1.0,
    )


def plot_validation(
    panels: dict[str, list[CurveData]],
    output: str | None = None,
    *,
    ylim: tuple[float, float] = (0.99, 1.01),
) -> None:
    '''
    Produce the four-panel validation figure.

    Parameters
    ----------
    panels : dict
        Mapping ``{'a': [...], 'b': [...], 'c': [...], 'd': [...]}``
        where each value is a list of :class:`CurveData`.
    output : str or None
        Save path. Shows interactively if ``None``.
    ylim : tuple
        Common y-axis limits for all panels.
    '''
    setup_matplotlib()

    panel_keys = ['a', 'b', 'c', 'd']
    titles = {
        'a': r'(a) Resolution',
        'b': r'(b) Redshift',
        'c': r'(c) LPT order',
        'd': r'(d) MAS scheme',
    }
    # Fallback: skip panels that have no data
    active_keys = [k for k in panel_keys if panels.get(k)]
    n_panels = len(active_keys)
    if n_panels == 0:
        log.error('No panels have data, nothing to plot.')
        return

    # Figure sizing: single A&A column ~ 88 mm ~ 3.46 in
    # Each panel height ~ 1.1 in (half-height, ~2:1 aspect)
    panel_height = 1.1  # inches
    fig_width = 3.46    # inches (A&A single column)
    fig_height = n_panels * panel_height + 0.35  # +0.35 for bottom label

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(fig_width, fig_height),
        sharex=True,
        gridspec_kw={'hspace': 0.15},
    )
    if n_panels == 1:
        axes = [axes]

    for i, key in enumerate(active_keys):
        is_bottom = (i == n_panels - 1)
        _draw_panel(
            axes[i], panels[key], titles.get(key, ''),
            show_xlabel=is_bottom, show_ylabel=(i == 0),
            ylim=ylim,
        )

    # Suppress x tick labels on all but the bottom panel
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.15)

    if output:
        fig.savefig(output, bbox_inches='tight')
        log.info('Figure saved to %s', output)
    else:
        plt.show()


def _discover_keys(data: dict, prefix: str) -> list[str]:
    '''Find all keys matching ``prefix*_pk`` and extract the varying part.'''
    pattern = re.compile(rf'^{re.escape(prefix)}(.+)_pk$')
    return sorted(m.group(1) for k in data if (m := pattern.match(k)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Four-panel P(k) validation in the `stepsic` article.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--panel-a', type=str, default=None,
        help='Archive for panel A (resolution dependence).',
    )
    p.add_argument(
        '--panel-b', type=str, default=None,
        help='Archive for panel B (redshift dependence).',
    )
    p.add_argument(
        '--panel-c', type=str, default=None,
        help='Archive for panel C (LPT order comparison).',
    )
    p.add_argument(
        '--panel-d', type=str, default=None,
        help='Archive for panel D (MAS scheme comparison).',
    )
    p.add_argument(
        '--nmesh-a', type=int, nargs='+', default=[64, 128, 256],
        help='Mesh resolutions for panel A.',
    )
    p.add_argument(
        '--z-a', type=float, default=31.0,
        help='Redshift for panel A.',
    )
    p.add_argument(
        '--nmesh-b', type=int, default=128,
        help='Mesh resolution for panel B.',
    )
    p.add_argument(
        '--z-b', type=float, nargs='+', default=[63.0, 31.0, 15.0],
        help='Redshifts for panel B.',
    )
    p.add_argument(
        '--nmesh-c', type=int, default=128,
        help='Mesh resolution for panel C.',
    )
    p.add_argument(
        '--z-c', type=float, default=15.0,
        help='Redshift for panel C.',
    )
    p.add_argument(
        '--nmesh-d', type=int, default=256,
        help='Mesh resolution for panel D.',
    )
    p.add_argument(
        '--z-d', type=float, default=31.0,
        help='Redshift for panel D.',
    )
    p.add_argument(
        '--methods-d', type=str, nargs='+', default=['ngp', 'cic', 'tsc'],
        help='MAS methods for panel D.',
    )
    p.add_argument(
        '--ylim', type=float, nargs=2, default=[0.99, 1.01],
        help='Y-axis limits.',
    )
    p.add_argument(
        '-o', '--output', type=str, default=None,
        help='Output figure path (e.g. validation.pdf).',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    panels: dict[str, list[CurveData]] = {}

    if args.panel_a:
        data_a = load_archive(args.panel_a)
        panels['a'] = build_panel_a(data_a, args.nmesh_a, z=args.z_a)
        log.info('Panel A: %d curves loaded', len(panels['a']))

    if args.panel_b:
        data_b = load_archive(args.panel_b)
        panels['b'] = build_panel_b(data_b, args.z_b, nmesh=args.nmesh_b)
        log.info('Panel B: %d curves loaded', len(panels['b']))

    if args.panel_c:
        data_c = load_archive(args.panel_c)
        panels['c'] = build_panel_c(
            data_c, nmesh=args.nmesh_c, z=args.z_c,
        )
        log.info('Panel C: %d curves loaded', len(panels['c']))

    if args.panel_d:
        data_d = load_archive(args.panel_d)
        panels['d'] = build_panel_d(
            data_d, methods=args.methods_d, nmesh=args.nmesh_d, z=args.z_d,
        )
        log.info('Panel D: %d curves loaded', len(panels['d']))

    if not panels:
        log.error('No panel archives provided. Use --panel-a, --panel-b, etc.')
        sys.exit(1)

    plot_validation(panels, output=args.output, ylim=tuple(args.ylim))