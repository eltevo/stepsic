#!/usr/bin/env python3
'''
Plot how padding changes displacement, power, boundary correlation, and radial drift.

The four panels show:

(a) relative displacement error vs radius: RMS |x_a - x_ref| divided by
    the reference RMS displacement, one curve per alpha (seed mean, with
    a seed-scatter band);
(b) central-region fractional power error P_alpha / P_ref - 1
    (seed mean +/- scatter);
(c) antipodal boundary-shell correlation excess C_alpha - C_ref vs alpha;
(d) monopole diagnostics vs alpha: mass-fraction change |1 - M_in/M_tot|,
    the surface-flux estimate S<|Psi_r|>/V, and the mean boundary radial
    drift <Delta r>/R_3D (all dimensionless).

A second figure repeats (a) for both geometries and two field meshes, taken
from one archive per geometry and mesh.

Usage
-----
::

    python plot.py -i padding_data.npz -o padding.pdf \
        --comparison other-geometry.npz half-mesh.npz other-half-mesh.npz \
        --comparison-output padding-comparison.pdf
'''

from __future__ import annotations



import argparse
import logging

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from validation._common.plotting import atomic_savefig, setup_matplotlib
from validation._common.artifacts import load_archive

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Cycle line styles as well as colours so overlapping curves remain visible.
_LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1))]


def _relative_error(data: dict) -> tuple[np.ndarray, np.ndarray]:
    '''Shell radii over R_3D and RMS |x_a - x_ref| over the reference RMS
    displacement, indexed [seed, alpha, shell].'''
    r_edges = data['r_edges']
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:]) / float(data['meta_r3d'])
    # Small runs may leave some radial shells empty.
    ref = np.where(data['ref_rms_psi'] > 0.0, data['ref_rms_psi'], np.nan)
    return r_mid, data['prof_rms_dx'] / ref[:, None, :]


def plot_comparison(archives: list[dict], output: str) -> None:
    '''Draw the displacement error for both geometries at two meshes.'''
    setup_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(3.46, 4.6), sharex=True, sharey=True)
    alphas = archives[0]['alphas']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ax, geometry in zip(axes, ('spherical', 'cylindrical')):
        coarse, fine = sorted(
            (data for data in archives if data['meta_geometry'] == geometry),
            key=lambda data: int(data['meta_ngrid0']),
        )
        r_mid, rel_coarse = _relative_error(coarse)
        r_mid, rel_fine = _relative_error(fine)
        for i_a in range(len(alphas) - 1):
            color = colors[i_a % len(colors)]
            ax.semilogy(r_mid, np.nanmean(rel_fine[:, i_a], axis=0),
                        color=color, lw=1.4)
            ax.fill_between(
                r_mid, np.nanmin(rel_fine[:, i_a], axis=0),
                np.nanmax(rel_fine[:, i_a], axis=0),
                color=color, alpha=0.18, lw=0,
            )
            ax.semilogy(r_mid, np.nanmean(rel_coarse[:, i_a], axis=0),
                        color=color, ls='--', lw=1.1)
        ax.set_title(f'{geometry} domain', loc='left', fontsize=8)
    fig.supylabel(
        r'RMS $|\mathbf{x}_\alpha - \mathbf{x}_{\mathrm{ref}}|\, /\,'
        r'\mathrm{RMS}\,|\boldsymbol{\Psi}_{\mathrm{ref}}|$',
        fontsize=plt.rcParams['axes.labelsize'],
    )
    axes[1].set_xlabel(r'$r / R_{\mathrm{3D}}$')
    handles = [
        Line2D([], [], color=colors[i_a % len(colors)], lw=1.4,
               label=rf'$\alpha = {alphas[i_a]:g}$')
        for i_a in range(len(alphas) - 1)
    ] + [
        Line2D([], [], color='0.4', lw=1.4,
               label=rf"$N = {int(fine['meta_ngrid0'])}$"),
        Line2D([], [], color='0.4', ls='--', lw=1.1,
               label=rf"$N = {int(coarse['meta_ngrid0'])}$"),
        Patch(facecolor='0.4', alpha=0.3, edgecolor='none', label='seed range'),
    ]
    fig.legend(
        handles=handles, fontsize=7, ncol=3, frameon=False, handlelength=1.5,
        loc='lower center',
        title=rf'reference: $\alpha = {alphas[-1]:g}$', title_fontsize=7,
    )
    fig.tight_layout(rect=(0.0, 0.05 + 0.035 * np.ceil(len(handles) / 3), 1.0, 1.0))
    atomic_savefig(fig, output)
    plt.close(fig)
    log.info('Figure saved to %s', output)


def plot_padding(data: dict, output: str | None = None) -> None:
    setup_matplotlib()

    alphas = data['alphas']
    n_a = len(alphas)
    r3d = float(data['meta_r3d'])

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # -- (a) relative displacement error vs radius ---------------------------
    r_mid, rel = _relative_error(data)                # [seed, alpha, shell]
    rel_mean = np.nanmean(rel, axis=0)
    rel_std = np.nanstd(rel, axis=0)
    for i_a in range(n_a - 1):
        ls = _LINESTYLES[i_a % len(_LINESTYLES)]
        line, = ax_a.semilogy(
            r_mid, rel_mean[i_a], ls=ls, lw=1.4,
            label=rf'$\alpha = {alphas[i_a]:g}$', zorder=3 + i_a,
        )
        ax_a.fill_between(
            r_mid, rel_mean[i_a] - rel_std[i_a], rel_mean[i_a] + rel_std[i_a],
            color=line.get_color(), alpha=0.15, lw=0, zorder=2,
        )
    ax_a.set_xlabel(r'$r / R_{\mathrm{3D}}$')
    ax_a.set_ylabel(
        r'RMS $|\mathbf{x}_\alpha - \mathbf{x}_{\mathrm{ref}}|\, /\,'
        r'\mathrm{RMS}\,|\boldsymbol{\Psi}_{\mathrm{ref}}|$'
    )
    ax_a.set_title('(a) Displacement error', loc='left', fontsize=8)
    ax_a.legend(fontsize=6, loc='upper left', frameon=False,
                handlelength=1.8)

    # -- (b) central-region P(k) ratio ---------------------------------------
    pk = data['pk']                                   # [seed, alpha, k]
    k = data['pk_k']
    fractional_error = pk[:, :-1, :] / pk[:, -1:, :] - 1.0
    error_mean = fractional_error.mean(axis=0)
    error_std = fractional_error.std(axis=0)
    ax_b.axhline(0.0, color='k', lw=1, ls=':', zorder=1)
    for i_a in range(n_a - 1):
        ls = _LINESTYLES[i_a % len(_LINESTYLES)]
        line, = ax_b.semilogx(
            k, error_mean[i_a], ls=ls, lw=1.4, zorder=3 + i_a,
        )
        ax_b.fill_between(
            k, error_mean[i_a] - error_std[i_a],
            error_mean[i_a] + error_std[i_a],
            color=line.get_color(), alpha=0.15, lw=0, zorder=2,
        )
    ax_b.set_xlabel(r'$k$ [$h\,\mathrm{Mpc}^{-1}$]')
    ax_b.set_ylabel(
        r'$P_\alpha(k) / P_{\mathrm{ref}}(k) - 1$'
    )
    ax_b.set_title('(b) Central power-spectrum error',
                   loc='left', fontsize=8)

    # -- (c) antipodal correlation excess ------------------------------------
    anti = data['anti_c']                             # [seed, alpha]
    excess = anti[:, :-1] - anti[:, -1:]
    ax_c.axhline(0.0, color='k', lw=1, ls=':', zorder=1)
    ax_c.errorbar(
        alphas[:-1], excess.mean(axis=0), yerr=excess.std(axis=0),
        fmt='o-', lw=1.4, ms=4, capsize=2, zorder=3,
    )
    ax_c.set_xlabel(r'$\alpha = L_{\mathrm{box}} / 2 R_{\mathrm{3D}}$')
    ax_c.set_ylabel(r'$C_\alpha - C_{\mathrm{ref}}$')
    ax_c.set_title('(c) Antipodal correlation excess',
                   loc='left', fontsize=8)

    # -- (d) monopole diagnostics --------------------------------------------
    massdef = np.abs(1.0 - data['mono_massfrac'])     # [seed, alpha]
    flux = data['mono_pred_flux']
    drift = np.abs(data['mono_dr_mean']) / r3d
    for series, label, marker in (
        (massdef, r'$|1 - M_{\mathrm{in}}/M_{\mathrm{tot}}|$', 'o'),
        (flux, r'$S \langle|\Psi_r|\rangle / V$', 's'),
        (drift, r'$|\langle \Delta r \rangle_{\mathrm{bnd}}| / R_{\mathrm{3D}}$', '^'),
    ):
        ax_d.errorbar(
            alphas, series.mean(axis=0), yerr=series.std(axis=0),
            fmt=f'{marker}-', lw=1.2, ms=4, capsize=2, label=label,
        )
    ax_d.set_yscale('log')
    ax_d.set_xlabel(r'$\alpha = L_{\mathrm{box}} / 2 R_{\mathrm{3D}}$')
    ax_d.set_title('(d) Monopole diagnostics', loc='left', fontsize=8)
    ax_d.legend(fontsize=6, loc='best', frameon=False, handlelength=1.8)

    fig.tight_layout()
    if output:
        atomic_savefig(fig, output, bbox_inches='tight')
        log.info('Figure saved to %s', output)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Plot the periodic-embedding padding sweep.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-i', '--input', type=str, required=True,
                   help='Input .npz archive from run.py.')
    p.add_argument('-o', '--output', type=str, default=None,
                   help='Output figure path; shows interactively if omitted.')
    p.add_argument('--comparison', type=str, nargs='+', required=True,
                   help='run.py archives for the other geometry and mesh.')
    p.add_argument('--comparison-output', type=str, required=True,
                   help='Output path of the geometry and mesh comparison.')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data = load_archive(args.input)
    log.info(
        'Loaded sweep: %s, R_3D=%g, alphas=%s, %d seeds, %d particles',
        data['meta_geometry'], float(data['meta_r3d']),
        data['alphas'].tolist(), int(data['meta_nseeds']),
        int(data['meta_n_part']),
    )
    plot_padding(data, output=args.output)
    plot_comparison(
        [data] + [load_archive(path) for path in args.comparison],
        args.comparison_output,
    )
