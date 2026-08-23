#!/usr/bin/env python3
'''
Plot power transfer relative to the cubic box as aspect ratio changes.

The figure contains one curve for each z length.

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
from common import cubic_normalized_transfer_curves

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


ArrayF: TypeAlias = NDArray[np.float64]


@dataclass
class CurveData:
    '''One power-transfer curve relative to the cubic box.'''
    k: ArrayF
    transfer: ArrayF
    nmesh_max: int
    lz: float
    label: str
    k_ny: float  # Nyquist wavenumber [h/Mpc]


def build_curves(data: dict) -> list[CurveData]:
    '''Extract a curve for each L_z value, relative to the cubic box.

    Parameters
    ----------
    data : dict
        Archive contents from :func:`load_archive`.

    Returns
    -------
    list of CurveData
        One entry per L_z step, sorted from largest to smallest L_z.
    '''
    l_cube = float(data['meta_l_cube'])
    nmesh = int(data['meta_nmesh'])

    dk = float(data['meta_dk'])

    # The fixed cell size gives every curve the same Nyquist wavenumber.
    k_ny = np.pi / dk

    curves = []
    for normalized in cubic_normalized_transfer_curves(data):
        lz = normalized.lz_mpc_h
        aspect = l_cube / lz
        label = rf'$L_z = {lz:g}$ ({aspect:.1f}:1)'

        nmesh_max = np.round(l_cube / dk)
        curves.append(CurveData(
            k=normalized.k_h_mpc,
            transfer=normalized.transfer,
            nmesh_max=nmesh_max,
            lz=lz,
            label=label,
            k_ny=k_ny,
        ))
    return curves


# Distinct linestyles so overlapping curves remain distinguishable.
# Cycled in the same order as the colour palette.
_LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1))]


def plot_squish(
    curves: list[CurveData],
    output: str | None = None,
    *,
    ylim: tuple[float, float] = (0.9949, 1.0051),
    percent_band: float = 0.005,
    title: str | None = None,
) -> None:
    '''Plot power transfer relative to the cubic box.

    Parameters
    ----------
    curves : list of CurveData
        Cubic-normalized transfer curves, one per L_z value.
    output : str or None
        Save path. Shows interactively if ``None``.
    ylim : tuple
        Y-axis limits for the ratio panel.
    percent_band : float
        Half-width of the shaded reference band as a fraction.
    title : str or None
        Optional panel title text (top-left annotation).
    '''
    if not curves:
        log.error('No curves to plot.')
        return

    setup_matplotlib()

    # Figure sizing: single A&A column ~ 88 mm ~ 3.46 in
    fig_width = 3.46  # inches
    fig_height = 2.2  # inches (single panel, slightly taller than stacked)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Reference line and +/- band
    ax.axhline(1.0, color='k', lw=1, ls=':', zorder=1)
    ax.axhspan(
        1.0 - percent_band, 1.0 + percent_band,
        color='grey', alpha=0.10, zorder=0,
        #label=rf'$\pm\,{percent_band*100:.1f}\%$',
    )

    # Data curves - distinct linestyles to resolve overlaps
    for i, curve in enumerate(curves):
        ls = _LINESTYLES[i % len(_LINESTYLES)]
        ax.semilogx(
            curve.k, curve.transfer, ls=ls, lw=1.4,
            label=curve.label, zorder=3 + i,
        )

    # Nyquist marker - single value since dk is constant across all curves
    k_ny_values = {round(c.k_ny, 4) for c in curves}
    for k_ny in k_ny_values:
        ax.axvline(
            k_ny, color='grey', ls=':', lw=0.6,
            alpha=0.5, zorder=2,
        )
    ax.annotate(
        fr'${curves[0].nmesh_max:g}^3$',
        xy=(curve.k_ny, 0.1),
        xycoords=ax.get_xaxis_transform(),
        fontsize=6,
        color='grey',
        rotation=90,
        ha='right',
        va='bottom',
    )

    ax.set_ylim(*ylim)
    ax.set_xlabel(r'$k$ [$h\,\mathrm{Mpc}^{-1}$]')
    ax.set_title(
        r'$T_{L_z}(k)$',
        loc='left', fontsize=8,
    )

    # Title annotation (top-left, inside panel)
    if title:
        ax.text(
            0.03, 0.94, title,
            transform=ax.transAxes, fontsize=8,
            va='top', ha='left',
        )

    # Legend - compact, no frame
    ax.legend(
        fontsize=6, loc='lower left', frameon=False,
        ncol=2, handlelength=1.5, columnspacing=1.0,
    )

    fig.tight_layout()

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
        '--ylim', type=float, nargs=2, default=[0.9949, 1.0051],
        help='Y-axis limits (ratio).',
    )
    p.add_argument(
        '--percent-band', type=float, default=0.005,
        dest='percent_band',
        help='Half-width of the reference band (fraction, e.g. 0.005).',
    )
    p.add_argument(
        '--title', type=str, default=None,
        help='Panel title text (top-left annotation).',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    data = load_archive(args.input)
    curves = build_curves(data)

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
            'k_Ny = %.4f | transfer = [%.6f, %.6f]',
            c.lz, len(c.k), c.k[0], c.k[-1], c.k_ny,
            c.transfer.min(), c.transfer.max(),
        )

    plot_squish(
        curves,
        output=args.output,
        ylim=tuple(args.ylim),
        percent_band=args.percent_band,
        title=args.title,
    )
