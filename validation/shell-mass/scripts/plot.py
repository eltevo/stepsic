#!/usr/bin/env python3
'''
Plot the particle mass distribution across radial shells for spherical
and cylindrical StePS geometries, comparing constant-omega (``omega``)
and constant-volume (``volume``) binning strategies.

Produces a vertical two-panel A&A single-column figure:

* **Top panel** - spherical geometry (R^3 via stereographic projection
  from S^3): particle mass as a function of radial distance from the
  centre.
* **Bottom panel** - cylindrical geometry (R^2 × R via stereographic
  projection from S^1 × R^2): particle mass as a function of radial
  distance from the central axis.

Each panel overlays both binning modes so the reader can immediately
see how the multiresolution mass profile differs between
constant-angle-step and constant-compact-volume strategies.

When ``--rcrit`` is given, the constant-resolution inner zone is
applied to both binning methods: particles inside the critical radius
all share the same mass, while the exterior follows the standard
multiresolution scaling.

Usage
-----
::

    python plot-shell-mass.py -o shell_mass.pdf
    python plot-shell-mass.py --D4D 75 --R3D 500 --nrbins 224 --nshell 12288
    python plot-shell-mass.py --rcrit 50 -o shell_mass_rcrit.pdf

All parameters have sensible defaults matching the Template-config.toml
values used in the stepsic paper.
'''

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stepsic.units import UNIT_V
from stepsic.geometry import (
    CylindricalBinner,
    CylindricalConstantVolume,
    CylindricalLinear,
    SphericalBinner,
    SphericalConstantVolume,
    SphericalLinear,
    _compute_rcrit_zones,
    shell_masses,
)
from validation import PLANCK2018, setup_matplotlib


AA_COL_WIDTH = 3.5

# Mass unit: 1 internal unit = 1e11 M_sun
MASS_TO_MSUN = 1e11


def _rho_mean(omega_m: float) -> float:
    r'''Compute :math:`\bar{\rho}` in internal units.

    .. math::
        \rho_\mathrm{crit} = \frac{3\,(100\,/\,U_V)^2}{8\pi}

    and :math:`\bar{\rho} = \Omega_m \, \rho_\mathrm{crit}`. This uses
    the h-unit convention where ``UNIT_V`` absorbs the ``h`` dependence,
    so :math:`\rho_\mathrm{crit}` is a universal constant of the unit
    system (see ``parameters._compute_derived_cosmo``).

    Parameters
    ----------
    omega_m : float
        Present-day matter density parameter.

    Returns
    -------
    float
        Mean matter density in internal (Mpc/h, 1e11 Msol) units.
    '''
    rho_crit = 3.0 * (100.0 / UNIT_V) ** 2 / (8.0 * np.pi)
    return omega_m * rho_crit


def _bin_centroids(
    binner: SphericalBinner | CylindricalBinner,
    n_bins: int,
) -> NDArray[np.float64]:
    '''Return the radial centroid of every bin.'''
    return np.array([binner.r_centroid(i) for i in range(n_bins)])


def _bin_edges(
    binner: SphericalBinner | CylindricalBinner,
    n_bins: int,
) -> NDArray[np.float64]:
    '''Return the n_bins+1 radial bin edges (starting from 0).'''
    return np.array([binner.r_limit(i) for i in range(n_bins + 1)])


def _make_spherical_binners(
    r_4d: float,
    n_bins: int,
    r_3d: float,
) -> dict[str, SphericalBinner]:
    '''Create both binning strategies for the spherical geometry.'''
    last_cell = n_bins * np.pi / (2 * np.arctan(r_3d / r_4d)) - n_bins
    return {
        'omega':  SphericalLinear(r_4d, n_bins, last_cell),
        'volume': SphericalConstantVolume(r_4d, n_bins, r_3d),
    }


def _make_cylindrical_binners(
    r_4d: float,
    n_bins: int,
    r_3d: float,
) -> dict[str, CylindricalBinner]:
    '''Create both binning strategies for the cylindrical geometry.'''
    last_cell = n_bins * np.pi / (2 * np.arctan(r_3d / r_4d)) - n_bins
    return {
        'omega':  CylindricalLinear(r_4d, n_bins, last_cell),
        'volume': CylindricalConstantVolume(r_4d, n_bins, r_3d),
    }


def _mass_profile(
    binner: SphericalBinner | CylindricalBinner,
    n_bins: int,
    n_per_shell: int,
    rho_mean: float,
    r_crit: float | None = None,
    Lz: float | None = None,
) -> NDArray[np.float64]:
    r'''Compute per-bin particle masses, optionally with an RCRIT zone.

    When ``r_crit`` is ``None``, this is a thin wrapper around
    :func:`shell_masses`. When set, it calls :func:`_compute_rcrit_zones`
    (the same code path used by the actual IC generator) to produce
    the two-zone mass profile.

    Parameters
    ----------
    binner : SphericalBinner or CylindricalBinner
        The radial binning strategy.
    n_bins : int
        Number of radial bins.
    n_per_shell : int
        Particles per shell.
    rho_mean : float
        Mean matter density in internal units.
    r_crit : float or None
        If set, defines the constant-resolution inner zone radius.
    Lz : float or None
        Cylinder height; required for cylindrical binners.

    Returns
    -------
    masses : ndarray of shape (n_bins,)
        Per-particle mass for each radial bin [internal units].
    '''
    if r_crit is None:
        return shell_masses(binner, n_bins, n_per_shell, rho_mean, Lz=Lz)

    zones = _compute_rcrit_zones(
        binner, n_bins, n_per_shell, rho_mean, r_crit, Lz=Lz,
    )
    masses = np.empty(n_bins, dtype=np.float64)
    masses[:zones.i_crit] = zones.mass_inside
    masses[zones.i_crit:] = zones.masses_outside
    return masses


def plot_shell_mass(
    *,
    D4D: float,
    R3D: float,
    nrbins: int,
    nshell: int,
    Lz: float,
    omega_m: float,
    r_crit: float | None,
    rcrit_modes: set[str],
    output: str,
) -> None:
    r'''
    Compute shell masses and produce the two-panel validation figure.

    Parameters
    ----------
    D4D : float
        Diameter of the compactification hypersphere [Mpc/h].
    R3D : float
        Simulation radius in the non-compact Euclidean space [Mpc/h].
    nrbins : int
        Number of radial bins.
    nshell : int
        Number of particles per shell.
    Lz : float
        Cylinder height (periodic z-dimension) [Mpc/h].
    omega_m : float
        Matter density parameter.
    r_crit : float or None
        If set, apply the constant-resolution inner zone to the
        binning methods listed in ``rcrit_modes``.
    rcrit_modes : set of str
        Which binning methods receive the RCRIT treatment when
        ``r_crit`` is not None. Subset of ``{'omega', 'volume'}``.
    output : str
        Output figure path.
    '''
    import matplotlib.pyplot as plt

    setup_matplotlib()

    r_4d = D4D / 2.0
    rho = _rho_mean(omega_m)

    # -- Build binners --------------------------------------------------------
    sph_binners = _make_spherical_binners(r_4d, nrbins, R3D)
    cyl_binners = _make_cylindrical_binners(r_4d, nrbins, R3D)

    # -- Compute masses -------------------------------------------------------
    # Each entry: (edges, masses) - always exactly two curves per panel.
    results: dict[str, dict[str, dict]] = {'sph': {}, 'cyl': {}}

    for label, binner in sph_binners.items():
        rc = r_crit if label in rcrit_modes else None
        masses = _mass_profile(binner, nrbins, nshell, rho, r_crit=rc)
        edges = _bin_edges(binner, nrbins)
        log.info(
            'Spherical %-6s: mass range [%.4e, %.4e] Msol, '
            'r range [%.2f, %.2f] Mpc/h  (RCRIT=%s)',
            label,
            masses[0] * MASS_TO_MSUN, masses[-1] * MASS_TO_MSUN,
            edges[1], edges[-1],
            f'{rc:.1f}' if rc is not None else 'off',
        )
        results['sph'][label] = {'masses': masses, 'edges': edges}

    for label, binner in cyl_binners.items():
        rc = r_crit if label in rcrit_modes else None
        masses = _mass_profile(
            binner, nrbins, nshell, rho, r_crit=rc, Lz=Lz,
        )
        edges = _bin_edges(binner, nrbins)
        log.info(
            'Cylindrical %-6s: mass range [%.4e, %.4e] Msol, '
            'r range [%.2f, %.2f] Mpc/h  (RCRIT=%s)',
            label,
            masses[0] * MASS_TO_MSUN, masses[-1] * MASS_TO_MSUN,
            edges[1], edges[-1],
            f'{rc:.1f}' if rc is not None else 'off',
        )
        results['cyl'][label] = {'masses': masses, 'edges': edges}

    # -- Plot -----------------------------------------------------------------
    fig, axes = plt.subplots(
        2, 1,
        figsize=(AA_COL_WIDTH, 2 * 1.1 + 0.35),
        sharex=True,
        gridspec_kw={'hspace': 0.15},
    )

    style = {
        'omega':  {'edgecolor': '#0072B2', 'linestyle': '-',
                   'label': r'constant $\Delta\omega$ binning'},
        'volume': {'edgecolor': '#D55E00', 'linestyle': '--',
                   'label': r'constant volume binning'},
    }

    # Collect all masses for a shared y-range across both panels.
    all_masses = np.concatenate([
        d['masses']
        for geom in results.values()
        for d in geom.values()
    ])

    for ax, geom_tag, title in zip(
        axes,
        ('sph', 'cyl'),
        (r'(a) Spherical',
         r'(b) Cylindrical'),
    ):
        for mode in ('omega', 'volume'):
            d = results[geom_tag][mode]
            ax.stairs(
                d['masses'],
                d['edges'],
                linewidth=1.0,
                fill=False,
                **style[mode],
            )
        ax.set_xlim(0, R3D)
        ax.set_yscale('log')
        # ax.set_ylim(all_masses.min() * 0.5, all_masses.max() * 2.0)
        ax.set_ylim(1e-1, 1e5)
        if geom_tag == 'sph':
            ax.set_title(
                r'$m_\mathrm{particle}$ [$10^{11}\,M_\odot$]',
                loc='left', fontsize=9
            )
        if geom_tag == 'cyl':
            ax.set_xlabel(r'$r$ [Mpc/$h$]')
        ax.text(
            0.03, 0.91, title,
            transform=ax.transAxes, fontsize=8,
            va='top', ha='left',
        )
        ax.legend(fontsize=7, loc='lower right', frameon=False)

    fig.tight_layout(h_pad=0.6)
    fig.savefig(output, dpi=300, bbox_inches='tight')
    log.info('Figure saved to %s', output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Plot particle mass across radial shells for spherical '
            'and cylindrical StePS geometries (omega vs. volume binning).'
        ),
    )
    parser.add_argument(
        '--D4D', type=float, default=75.0,
        help='Diameter of the compactification hypersphere [Mpc/h] '
             '(default: 75).',
    )
    parser.add_argument(
        '--R3D', type=float, default=500.0,
        help='Simulation radius in the Euclidean space [Mpc/h] '
             '(default: 500).',
    )
    parser.add_argument(
        '--nrbins', type=int, default=224,
        help='Number of radial bins (default: 224).',
    )
    parser.add_argument(
        '--nshell', type=int, default=12288,
        help='Particles per shell (default: 12288).',
    )
    parser.add_argument(
        '--Lz', type=float, default=200.0,
        help='Cylinder height (periodic z-dimension) [Mpc/h] '
             '(default: 200).',
    )
    parser.add_argument(
        '--omega-m', type=float, default=PLANCK2018['OMEGA_M'],
        help='Matter density parameter (default: Planck 2018).',
    )
    parser.add_argument(
        '--rcrit', type=float, default=None,
        help='Critical radius for the constant-resolution inner zone '
             '[Mpc/h]. Applied to the methods listed in --rcrit-modes.',
    )
    parser.add_argument(
        '--rcrit-modes', type=str, nargs='+', default=['omega'],
        choices=['omega', 'volume'],
        help='Which binning methods receive the RCRIT treatment '
             '(default: omega only).',
    )
    parser.add_argument(
        '-o', '--output', type=str,
        default='validation-shell-mass.pdf',
        help='Output figure path (default: validation-shell-mass.pdf).',
    )
    args = parser.parse_args()

    plot_shell_mass(
        D4D=args.D4D,
        R3D=args.R3D,
        nrbins=args.nrbins,
        nshell=args.nshell,
        Lz=args.Lz,
        omega_m=args.omega_m,
        r_crit=args.rcrit,
        rcrit_modes=set(args.rcrit_modes),
        output=args.output,
    )


if __name__ == '__main__':
    main()