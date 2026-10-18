#!/usr/bin/env python3
'''
Plot power transfer relative to the cubic box as aspect ratio changes.

The figure has one panel for each non-cubic z length. Each panel shows the
ratio in the native k bins and its mode-weighted average in logarithmic bins.

Usage
-----
::

    python squish_pk.py -i squish_data.npz -o squish_figure.pdf
'''

from __future__ import annotations



import argparse
import logging
from dataclasses import dataclass
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from validation._common.plotting import atomic_savefig, setup_matplotlib
from validation._common.artifacts import load_archive
from common import LOG_BINS_PER_DECADE, cubic_normalized_transfer_curves

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


ArrayF: TypeAlias = NDArray[np.float64]


@dataclass
class CurveData:
    '''One power-transfer curve relative to the cubic box.'''
    k: ArrayF
    transfer: ArrayF
    k_binned: ArrayF
    transfer_binned: ArrayF
    nmesh_max: int
    lz: float
    label: str
    k_ny: float  # Nyquist wavenumber [h/Mpc]


def build_curves(data: dict, bins_per_decade: int) -> list[CurveData]:
    '''Extract a curve for each non-cubic L_z value, relative to the cubic box.

    Parameters
    ----------
    data : dict
        Archive contents from :func:`load_archive`.
    bins_per_decade : int
        Number of logarithmic k bins per decade for the averaged curve.

    Returns
    -------
    list of CurveData
        One entry per non-cubic L_z step, sorted from largest to smallest L_z.
    '''
    l_cube = float(data['meta_l_cube'])
    nmesh = int(data['meta_nmesh'])

    dk = float(data['meta_dk'])

    # The fixed cell size gives every curve the same Nyquist wavenumber.
    k_ny = np.pi / dk

    curves = []
    binned_curves = cubic_normalized_transfer_curves(data, bins_per_decade)
    for normalized, binned in zip(
        cubic_normalized_transfer_curves(data), binned_curves, strict=True,
    ):
        if normalized.is_cubic:
            continue
        lz = normalized.lz_mpc_h
        aspect = l_cube / lz
        label = rf'$L_z = {lz:g}\,h^{{-1}}\mathrm{{Mpc}}$ ({aspect:.1f}:1)'

        nmesh_max = np.round(l_cube / dk)
        curves.append(CurveData(
            k=normalized.k_h_mpc,
            transfer=normalized.transfer,
            k_binned=binned.k_h_mpc,
            transfer_binned=binned.transfer,
            nmesh_max=nmesh_max,
            lz=lz,
            label=label,
            k_ny=k_ny,
        ))
    return curves


_MARKERS = ['o', 's', '^', 'D', 'v']


def plot_squish(
    curves: list[CurveData],
    output: str | None = None,
    *,
    ylim: tuple[float, float] = (0.994, 1.006),
    percent_band: float = 0.001,
    title: str | None = None,
) -> None:
    '''Plot power transfer relative to the cubic box.

    Parameters
    ----------
    curves : list of CurveData
        Cubic-normalized transfer curves, one per non-cubic L_z value.
    output : str or None
        Save path. Shows interactively if ``None``.
    ylim : tuple
        Y-axis limits shared by every panel.
    percent_band : float
        Half-width of the shaded reference band as a fraction.
    title : str or None
        Optional title text (top-right annotation in the first panel).
    '''
    if not curves:
        log.error('No curves to plot.')
        return

    setup_matplotlib()

    # Figure sizing: single A&A column ~ 88 mm ~ 3.46 in
    fig_width = 3.46  # inches
    panel_height = 0.9  # inches
    fig_height = len(curves) * panel_height + 0.5
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(
        len(curves), 1, figsize=(fig_width, fig_height), sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    for i, (ax, curve) in enumerate(zip(axes, curves)):
        ax.axhline(1.0, color='k', lw=1, ls=':', zorder=1)
        ax.axhspan(
            1.0 - percent_band, 1.0 + percent_band,
            color='grey', alpha=0.10, zorder=0,
        )
        ax.semilogx(curve.k, curve.transfer, color='0.7', lw=0.6, zorder=2)
        ax.semilogx(
            curve.k_binned, curve.transfer_binned,
            marker=_MARKERS[i % len(_MARKERS)], ms=3, lw=1,
            color=colors[(i + 1) % len(colors)], zorder=3,
        )
        ax.axvline(curve.k_ny, color='grey', ls=':', lw=0.6, alpha=0.5, zorder=2)
        ax.set_ylim(*ylim)
        ax.text(
            0.03, 0.92, f'({chr(97 + i)}) {curve.label}',
            transform=ax.transAxes, fontsize=7, va='top', ha='left',
        )
    axes[-1].annotate(
        fr'${curves[0].nmesh_max:g}^3$',
        xy=(curves[0].k_ny, 0.1),
        xycoords=axes[-1].get_xaxis_transform(),
        fontsize=6,
        color='grey',
        rotation=90,
        ha='right',
        va='bottom',
    )

    axes[-1].set_xlabel(r'$k$ [$h\,\mathrm{Mpc}^{-1}$]')
    axes[0].set_title(
        r'$\left(P_{\rm meas}/P_{\rm ref}\right)_{L_z}\,/\,'
        r'\left(P_{\rm meas}/P_{\rm ref}\right)_{\rm cube}$',
        loc='left', fontsize=8,
    )

    if title:
        axes[0].text(
            0.97, 0.92, title,
            transform=axes[0].transAxes, fontsize=8,
            va='top', ha='right',
        )

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.12)

    if output:
        atomic_savefig(fig, output, bbox_inches='tight')
        log.info('Figure saved to %s', output)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Plot power transfer relative to the cubic box at each aspect ratio.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '-i', '--input', type=str, required=True,
        help='Input NumPy archive from the transfer measurement.',
    )
    p.add_argument(
        '-o', '--output', type=str, default=None,
        help='Output figure path. '
             'Shows interactively if omitted.',
    )
    p.add_argument(
        '--ylim', type=float, nargs=2, default=[0.994, 1.006],
        help='Y-axis limits (ratio).',
    )
    p.add_argument(
        '--percent-band', type=float, default=0.001,
        dest='percent_band',
        help='Half-width of the reference band (fraction, e.g. 0.001).',
    )
    p.add_argument(
        '--bins-per-decade', type=int, default=LOG_BINS_PER_DECADE,
        help='Logarithmic k bins per decade for the averaged curve.',
    )
    p.add_argument(
        '--title', type=str, default=None,
        help='Title text (top-right annotation in the first panel).',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    data = load_archive(args.input)
    curves = build_curves(data, args.bins_per_decade)

    # Log what we found
    l_cube = float(data['meta_l_cube'])
    nmesh = int(data['meta_nmesh'])
    lpt_order = int(data['meta_lpt_order'])
    z = float(data['meta_redshift'])
    method = str(data['meta_method'])
    dk = float(data['meta_dk'])
    log.info(
        'Loaded %d curves: L_cube = %.0f, nmesh = %d, dk = %.6f, '
        '%dLPT, %s, z = %g',
        len(curves), l_cube, nmesh, dk,
        lpt_order, method.upper(), z,
    )
    for c in curves:
        log.info(
            '  L_z = %7.1f | %3d bins | k = [%.4e, %.4e] | '
            'k_Ny = %.4f | transfer = [%.6f, %.6f] | '
            'binned = [%.6f, %.6f]',
            c.lz, len(c.k), c.k[0], c.k[-1], c.k_ny,
            c.transfer.min(), c.transfer.max(),
            c.transfer_binned.min(), c.transfer_binned.max(),
        )

    plot_squish(
        curves,
        output=args.output,
        ylim=tuple(args.ylim),
        percent_band=args.percent_band,
        title=args.title,
    )
