#!/usr/bin/env python3
'''
Plot glass diagnostics beside the corresponding randomized controls.

For radial geometries, the four panels show:
(a) cumulative-aperture P(k) / P_shot against k / k_p
    (k_p = 2*pi / d_max): the glass must fall below the randomized control at k < k_p;
(b) radial density profile rho/rho_mean with mass-level interfaces
    marked;
(c) nearest-neighbour statistic u = d_NN / (m/rho)^{1/3} vs radius;
(d) residual-force statistic q = |F| d^2 / m vs radius.

The cubical figure uses three panels: power, nearest-neighbour distance,
and residual force.

Usage
-----
::

    python plot-diagnostics.py --glass diag_glass.npz --twin diag_twin.npz \
        --label cylindrical -o glass-quality.pdf
'''

from __future__ import annotations



import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from validation._common.plotting import atomic_savefig, setup_matplotlib
from validation._common.artifacts import load_archive

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _interfaces(d: dict) -> np.ndarray:
    '''Return the radii between consecutive mass levels.'''
    order = np.argsort(d['level_rmin'])
    rmax = d['level_rmax'][order]
    return rmax[:-1]


def _mark_interfaces(ax: plt.Axes, radii: np.ndarray) -> None:
    '''Mark shell interfaces with short ticks along the top edge.'''
    ax.vlines(
        radii, 0.96, 1.0, transform=ax.get_xaxis_transform(),
        color='0.55', lw=0.45, alpha=0.65,
    )


def plot_diagnostics(
    glass: dict, twin: dict, label: str, output: str | None,
) -> None:
    setup_matplotlib()
    geometry = str(glass['meta_geometry'])
    radial = geometry != 'cubical'
    if radial:
        fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0))
        ax_a, ax_b, ax_c, ax_d = axes.ravel()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35))
        ax_a, ax_b, ax_d = axes
        ax_c = None

    # -- (a) cumulative-aperture P(k)/shot ----------------------------------
    nz = len(glass['power_shot'])
    for iz in range(nz):
        kp = 2.0 * np.pi / glass['power_spacing'][iz]
        for d, ls in ((glass, '-'), (twin, '--')):
            k = np.asarray(d['power_k'][iz], dtype=np.float64)
            pk = np.asarray(d['power_pk'][iz], dtype=np.float64)
            shot = float(d['power_shot'][iz])
            if np.all(np.isnan(k)):
                continue
            lbl = None
            if radial and d is glass:
                lbl = rf'$r < {glass["power_rmax"][iz]:.0f}$'
            elif not radial:
                lbl = 'glass' if d is glass else 'randomized control'
            ax_a.loglog(k / kp, pk / shot, ls=ls, lw=1.2,
                        color=f'C{iz}', label=lbl)
    ax_a.axhline(1.0, color='k', lw=0.8, ls=':', zorder=1)
    ax_a.axvline(1.0, color='grey', lw=0.6, ls=':', zorder=1)
    ax_a.set_xlabel(r'$k / k_{\mathrm{p}}$')
    ax_a.set_ylabel(r'$P(k) / P_{\mathrm{shot}}$')
    ax_a.set_title('(a) Power relative to shot noise',
                   loc='left', fontsize=8)
    if radial:
        handles, labels = ax_a.get_legend_handles_labels()
        handles.extend([
            Line2D([0], [0], color='0.25', lw=1.2, ls='-', label='glass'),
            Line2D([0], [0], color='0.25', lw=1.2, ls='--',
                   label='randomized control'),
        ])
        labels.extend(['glass', 'randomized control'])
        ax_a.legend(
            handles, labels, fontsize=5.5, loc='lower right',
            frameon=False, ncol=2, handlelength=1.6,
        )
    else:
        ax_a.legend(fontsize=6, loc='lower right', frameon=False)

    if radial:
        r_g = 0.5 * (glass['prof_edges'][:-1] + glass['prof_edges'][1:])
        r_t = 0.5 * (twin['prof_edges'][:-1] + twin['prof_edges'][1:])
        ifaces = _interfaces(glass)

        # -- (b) density profile ---------------------------------------------
        ax_b.plot(r_g, glass['prof_rho'], '-', lw=1.2, label='glass')
        ax_b.plot(r_t, twin['prof_rho'], '--', lw=1.0, label='randomized control')
        _mark_interfaces(ax_b, ifaces)
        ax_b.axhline(1.0, color='k', lw=0.8, ls=':', zorder=1)
        ax_b.set_xlabel(r'$r$ [$\mathrm{Mpc}/h$]')
        ax_b.set_ylabel(r'$\varrho(r) / \bar{\varrho}$')
        ax_b.set_title('(b) Radial density', loc='left', fontsize=8)
        ax_b.legend(fontsize=6, loc='best', frameon=False)

        # -- (c) NN statistic profile ----------------------------------------
        ax_c.plot(r_g, glass['prof_u'], '-', lw=1.2, label='glass')
        ax_c.plot(r_t, twin['prof_u'], '--', lw=1.0, label='randomized control')
        _mark_interfaces(ax_c, ifaces)
        ax_c.set_xlabel(r'$r$ [$\mathrm{Mpc}/h$]')
        ax_c.set_ylabel(
            r'$\langle d_{\mathrm{NN}} / (m/\bar{\varrho})^{1/3} \rangle$'
        )
        ax_c.set_title('(c) Nearest-neighbour spacing',
                       loc='left', fontsize=8)
    else:
        centres = 0.5 * (glass['u_hist_edges'][:-1]
                         + glass['u_hist_edges'][1:])
        ax_b.plot(centres, glass['u_hist'], '-', lw=1.2, label='glass')
        ax_b.plot(centres, twin['u_hist'], '--', lw=1.0,
                  label='randomized control')
        ax_b.set_xlabel(r'$d_{\mathrm{NN}} / (m/\bar{\varrho})^{1/3}$')
        ax_b.set_ylabel('count')
        ax_b.set_title('(b) Nearest-neighbour spacing',
                       loc='left', fontsize=8)
        ax_b.legend(fontsize=6, loc='best', frameon=False)

    # -- (d) residual force ---------------------------------------------------
    has_force = bool(glass['meta_has_force']) and bool(twin['meta_has_force'])
    if has_force and radial and 'prof_q' in glass:
        ax_d.semilogy(r_g, glass['prof_q'], '-', lw=1.2, label='glass')
        ax_d.semilogy(r_t, twin['prof_q'], '--', lw=1.0,
                      label='randomized control')
        _mark_interfaces(ax_d, _interfaces(glass))
        ax_d.set_xlabel(r'$r$ [$\mathrm{Mpc}/h$]')
        ax_d.set_ylabel(r'$\langle |F| d^2 / m \rangle$ $(G=1)$')
        ax_d.set_title('(d) Residual force', loc='left', fontsize=8)
    elif has_force:
        labels = ['RMS', 'median', 'p99']
        x = np.arange(3)
        w = 0.35
        ax_d.bar(x - w / 2, glass['q_all'], w, label='glass')
        ax_d.bar(x + w / 2, twin['q_all'], w, label='randomized control')
        ax_d.set_yscale('log')
        ax_d.set_xticks(x, labels)
        ax_d.set_ylabel(r'$q = |F| d^2 / m$ $(G=1)$')
        ax_d.set_title('(c) Residual force', loc='left', fontsize=8)
        ax_d.legend(fontsize=6, loc='best', frameon=False)
    else:
        centres = 0.5 * (glass['u_hist_edges'][:-1]
                         + glass['u_hist_edges'][1:])
        ax_d.plot(centres, glass['u_hist'], '-', lw=1.2, label='glass')
        ax_d.plot(centres, twin['u_hist'], '--', lw=1.0,
                  label='randomized control')
        ax_d.set_xlabel(r'$d_{\mathrm{NN}} / (m/\bar{\varrho})^{1/3}$')
        ax_d.set_title('NN distance distribution (no force data)',
                       loc='left', fontsize=8)
        ax_d.legend(fontsize=6, loc='best', frameon=False)

    if label:
        fig.suptitle(label.capitalize(), fontsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96) if label else None)
    if output:
        atomic_savefig(fig, output, bbox_inches='tight')
        log.info('Figure saved to %s', output)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Plot glass diagnostics beside the corresponding randomized load.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--glass', type=str, required=True,
                   help='diagnose.py archive of the glass.')
    p.add_argument('--twin', type=str, required=True,
                   help='diagnose.py archive of the randomized control.')
    p.add_argument('--label', type=str, default='',
                   help='Figure title label (e.g. geometry).')
    p.add_argument('-o', '--output', type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    plot_diagnostics(
        load_archive(args.glass), load_archive(args.twin),
        args.label, args.output,
    )
