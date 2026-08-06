#!/usr/bin/env python3
'''
Plot the periodic-embedding padding sweep from ``run.py`` archives.

Four panels (A&A two-column layout):

(a) relative displacement error vs radius: RMS |x_a - x_ref| divided by
    the reference RMS displacement, one curve per alpha (seed mean, with
    a seed-scatter band);
(b) central-region P(k) ratio P_alpha / P_ref (seed mean +/- scatter);
(c) antipodal boundary-shell correlation excess C_alpha - C_ref vs alpha;
(d) monopole diagnostics vs alpha: mass-fraction change |1 - M_in/M_tot|,
    the surface-flux estimate S<|Psi_r|>/V, and the mean boundary radial
    drift <Delta r>/R_3D (all dimensionless).

Usage
-----
::

    python plot.py -i padding_data.npz -o padding.pdf
'''

from __future__ import annotations



import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

from validation import setup_matplotlib, load_archive

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Distinct linestyles cycled with the colour palette (house convention).
_LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1))]


def plot_padding(data: dict, output: str | None = None) -> None:
    setup_matplotlib()

    alphas = data['alphas']
    n_a = len(alphas)
    r3d = float(data['meta_r3d'])
    r_edges = data['r_edges']
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:]) / r3d

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # -- (a) relative displacement error vs radius ---------------------------
    # Guard the ratio: shells can be empty at tiny smoke resolutions.
    ref = np.where(data['ref_rms_psi'] > 0.0, data['ref_rms_psi'], np.nan)
    rel = data['prof_rms_dx'] / ref[:, None, :]      # [seed, alpha, shell]
    rel_mean = np.nanmean(rel, axis=0)
    rel_std = np.nanstd(rel, axis=0)
    for i_a in range(n_a - 1):
        ls = _LINESTYLES[i_a % len(_LINESTYLES)]
        ax_a.semilogy(
            r_mid, rel_mean[i_a], ls=ls, lw=1.4,
            label=rf'$\alpha = {alphas[i_a]:g}$', zorder=3 + i_a,
        )
        ax_a.fill_between(
            r_mid, rel_mean[i_a] - rel_std[i_a], rel_mean[i_a] + rel_std[i_a],
            alpha=0.15, lw=0, zorder=2,
        )
    ax_a.set_xlabel(r'$r / R_{\mathrm{3D}}$')
    ax_a.set_title(
        r'RMS $|\mathbf{x}_\alpha - \mathbf{x}_{\mathrm{ref}}|\, /\,'
        r'\mathrm{RMS}\,|\boldsymbol{\Psi}_{\mathrm{ref}}|$',
        loc='left', fontsize=8,
    )
    ax_a.legend(fontsize=6, loc='upper left', frameon=False,
                handlelength=1.8)

    # -- (b) central-region P(k) ratio ---------------------------------------
    pk = data['pk']                                   # [seed, alpha, k]
    k = data['pk_k']
    ratio = pk[:, :-1, :] / pk[:, -1:, :]
    ratio_mean = ratio.mean(axis=0)
    ratio_std = ratio.std(axis=0)
    ax_b.axhline(1.0, color='k', lw=1, ls=':', zorder=1)
    for i_a in range(n_a - 1):
        ls = _LINESTYLES[i_a % len(_LINESTYLES)]
        ax_b.semilogx(k, ratio_mean[i_a], ls=ls, lw=1.4, zorder=3 + i_a)
        ax_b.fill_between(
            k, ratio_mean[i_a] - ratio_std[i_a],
            ratio_mean[i_a] + ratio_std[i_a],
            alpha=0.15, lw=0, zorder=2,
        )
    ax_b.set_xlabel(r'$k$ [$h\,\mathrm{Mpc}^{-1}$]')
    ax_b.set_title(
        r'$P_\alpha(k) / P_{\mathrm{ref}}(k)$ (central region)',
        loc='left', fontsize=8,
    )

    # -- (c) antipodal correlation excess ------------------------------------
    anti = data['anti_c']                             # [seed, alpha]
    excess = anti[:, :-1] - anti[:, -1:]
    ax_c.axhline(0.0, color='k', lw=1, ls=':', zorder=1)
    ax_c.errorbar(
        alphas[:-1], excess.mean(axis=0), yerr=excess.std(axis=0),
        fmt='o-', lw=1.4, ms=4, capsize=2, zorder=3,
    )
    ax_c.set_xlabel(r'$\alpha = L_{\mathrm{box}} / 2 R_{\mathrm{3D}}$')
    ax_c.set_title(
        r'antipodal correlation excess '
        r'$C_\alpha - C_{\mathrm{ref}}$',
        loc='left', fontsize=8,
    )

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
    ax_d.set_title('monopole diagnostics', loc='left', fontsize=8)
    ax_d.legend(fontsize=6, loc='best', frameon=False, handlelength=1.8)

    fig.tight_layout()
    if output:
        fig.savefig(output, bbox_inches='tight')
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
