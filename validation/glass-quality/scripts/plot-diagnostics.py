#!/usr/bin/env python3
'''
Plot glass diagnostics beside the corresponding Poisson measurements.

The four panels show:
(a) zoned P(k) / P_shot against k / k_p (k_p = 2*pi / d_zone): the
    glass must fall below the twin's flat Poisson level at k < k_p;
(b) radial density profile rho/rho_mean with mass-level interfaces
    marked;
(c) nearest-neighbour statistic u = d_NN / (m/rho)^{1/3} vs radius;
(d) residual-force statistic q = |F| d^2 / m vs radius (only when the
    archives carry accelerations; otherwise the u histogram is shown).

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

from validation._common.plotting import atomic_savefig, setup_matplotlib
from validation._common.artifacts import load_archive

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _interfaces(d: dict) -> np.ndarray:
    '''Return the radii between consecutive mass levels.'''
    order = np.argsort(d['level_rmin'])
    rmax = d['level_rmax'][order]
    return rmax[:-1]


def plot_diagnostics(
    glass: dict, twin: dict, label: str, output: str | None,
) -> None:
    setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    geometry = str(glass['meta_geometry'])
    radial = geometry != 'cubical'

    # -- (a) zoned P(k)/shot -------------------------------------------------
    nz = len(glass['zone_shot'])
    for iz in range(nz):
        kp = 2.0 * np.pi / glass['zone_dbar'][iz]
        for d, ls in ((glass, '-'), (twin, '--')):
            k = np.asarray(d['zone_k'][iz], dtype=np.float64)
            pk = np.asarray(d['zone_pk'][iz], dtype=np.float64)
            shot = float(d['zone_shot'][iz])
            if np.all(np.isnan(k)):
                continue
            lbl = None
            if d is glass:
                if radial:
                    lbl = (rf'$r \in [{glass["zone_edges"][iz]:.0f},'
                           rf'{glass["zone_edges"][iz + 1]:.0f}]$')
                else:
                    lbl = 'glass'
            ax_a.loglog(k / kp, pk / shot, ls=ls, lw=1.2,
                        color=f'C{iz}', label=lbl)
    ax_a.axhline(1.0, color='k', lw=0.8, ls=':', zorder=1)
    ax_a.axvline(1.0, color='grey', lw=0.6, ls=':', zorder=1)
    ax_a.set_xlabel(r'$k / k_{\mathrm{p}}$')
    ax_a.set_title(r'$P(k) / P_{\mathrm{shot}}$ per zone '
                   r'(solid: glass, dashed: Poisson twin)',
                   loc='left', fontsize=8)
    ax_a.legend(fontsize=5, loc='lower right', frameon=False)

    if radial:
        r_g = 0.5 * (glass['prof_edges'][:-1] + glass['prof_edges'][1:])
        r_t = 0.5 * (twin['prof_edges'][:-1] + twin['prof_edges'][1:])
        ifaces = _interfaces(glass)

        # -- (b) density profile ---------------------------------------------
        ax_b.plot(r_g, glass['prof_rho'], '-', lw=1.2, label='glass')
        ax_b.plot(r_t, twin['prof_rho'], '--', lw=1.0, label='Poisson twin')
        for x in ifaces:
            ax_b.axvline(x, color='grey', lw=0.4, alpha=0.5, zorder=1)
        ax_b.axhline(1.0, color='k', lw=0.8, ls=':', zorder=1)
        ax_b.set_xlabel(r'$r$ [$\mathrm{Mpc}/h$]')
        ax_b.set_title(r'$\varrho(r) / \bar{\varrho}$ '
                       r'(interfaces marked)', loc='left', fontsize=8)
        ax_b.legend(fontsize=6, loc='best', frameon=False)

        # -- (c) NN statistic profile ----------------------------------------
        ax_c.plot(r_g, glass['prof_u'], '-', lw=1.2, label='glass')
        ax_c.plot(r_t, twin['prof_u'], '--', lw=1.0, label='Poisson twin')
        for x in ifaces:
            ax_c.axvline(x, color='grey', lw=0.4, alpha=0.5, zorder=1)
        ax_c.set_xlabel(r'$r$ [$\mathrm{Mpc}/h$]')
        ax_c.set_title(
            r'$\langle d_{\mathrm{NN}} / (m/\bar{\varrho})^{1/3} \rangle$',
            loc='left', fontsize=8,
        )
        ax_c.legend(fontsize=6, loc='best', frameon=False)
    else:
        for ax in (ax_b, ax_c):
            ax.set_axis_off()
        centres = 0.5 * (glass['u_hist_edges'][:-1]
                         + glass['u_hist_edges'][1:])
        ax_b.set_axis_on()
        ax_b.plot(centres, glass['u_hist'], '-', lw=1.2, label='glass')
        ax_b.plot(centres, twin['u_hist'], '--', lw=1.0,
                  label='Poisson twin')
        ax_b.set_xlabel(r'$d_{\mathrm{NN}} / (m/\bar{\varrho})^{1/3}$')
        ax_b.set_title('NN distance distribution', loc='left', fontsize=8)
        ax_b.legend(fontsize=6, loc='best', frameon=False)

    # -- (d) residual force ---------------------------------------------------
    has_force = bool(glass['meta_has_force']) and bool(twin['meta_has_force'])
    if has_force and radial and 'prof_q' in glass:
        ax_d.semilogy(r_g, glass['prof_q'], '-', lw=1.2, label='glass')
        ax_d.semilogy(r_t, twin['prof_q'], '--', lw=1.0,
                      label='Poisson twin')
        for x in _interfaces(glass):
            ax_d.axvline(x, color='grey', lw=0.4, alpha=0.5, zorder=1)
        ax_d.set_xlabel(r'$r$ [$\mathrm{Mpc}/h$]')
        ax_d.set_title(
            r'residual force $\langle |F| d^2 / m \rangle$ (G=1)',
            loc='left', fontsize=8,
        )
        ax_d.legend(fontsize=6, loc='best', frameon=False)
    elif has_force:
        labels = ['RMS', 'median', 'p99']
        x = np.arange(3)
        w = 0.35
        ax_d.bar(x - w / 2, glass['q_all'], w, label='glass')
        ax_d.bar(x + w / 2, twin['q_all'], w, label='Poisson twin')
        ax_d.set_yscale('log')
        ax_d.set_xticks(x, labels)
        ax_d.set_title(r'residual force $q = |F| d^2 / m$',
                       loc='left', fontsize=8)
        ax_d.legend(fontsize=6, loc='best', frameon=False)
    else:
        centres = 0.5 * (glass['u_hist_edges'][:-1]
                         + glass['u_hist_edges'][1:])
        ax_d.plot(centres, glass['u_hist'], '-', lw=1.2, label='glass')
        ax_d.plot(centres, twin['u_hist'], '--', lw=1.0,
                  label='Poisson twin')
        ax_d.set_xlabel(r'$d_{\mathrm{NN}} / (m/\bar{\varrho})^{1/3}$')
        ax_d.set_title('NN distance distribution (no force data)',
                       loc='left', fontsize=8)
        ax_d.legend(fontsize=6, loc='best', frameon=False)

    if label:
        fig.suptitle(label, fontsize=9)
    fig.tight_layout()
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
                   help='diagnose.py archive of the Poisson twin.')
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
