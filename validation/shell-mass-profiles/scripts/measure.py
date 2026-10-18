#!/usr/bin/env python3
"""Measure shell mass profiles and save them to a NumPy archive."""

from __future__ import annotations

import argparse

import numpy as np
from numpy.typing import NDArray

from stepsic.geometry import (
    CylindricalBinner,
    CylindricalConstantVolume,
    CylindricalLinear,
    SphericalBinner,
    SphericalConstantVolume,
    SphericalLinear,
)
from stepsic.particles import _compute_rcrit_zones, shell_masses
from stepsic.units import UNIT_V
from validation._common.fields import VALIDATION_COSMOLOGY
from validation._common.evaluation import atomic_savez


def _rho_mean(omega_m: float) -> float:
    r'''Compute :math:`\bar{\rho}` in internal units.

    .. math::
        \rho_\mathrm{crit} = \frac{3\,(100\,/\,U_V)^2}{8\pi}

    and :math:`\bar{\rho} = \Omega_m \, \rho_\mathrm{crit}`. This uses
    the h-unit convention where ``UNIT_V`` absorbs the ``h`` dependence,
    so :math:`\rho_\mathrm{crit}` is a universal constant of the unit
    system (see ``parameters.compute_derived_cosmo``).

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


def measure(
    *,
    d4d_mpc_h: float,
    r3d_mpc_h: float,
    radial_bin_count: int,
    particles_per_shell: int,
    cylinder_length_mpc_h: float,
    omega_m: float,
    critical_radius_mpc_h: float | None,
    critical_modes: set[str],
    output: str,
) -> None:
    if d4d_mpc_h <= 0.0 or r3d_mpc_h <= 0.0:
        raise ValueError("D4D and R3D must be positive")
    if radial_bin_count <= 0 or particles_per_shell <= 0:
        raise ValueError("bin and shell counts must be positive")
    if cylinder_length_mpc_h <= 0.0:
        raise ValueError("cylinder length must be positive")
    r4d_mpc_h = d4d_mpc_h / 2.0
    rho_mean_internal = _rho_mean(omega_m)
    arrays: dict[str, np.ndarray] = {
        "meta_d4d_mpc_h": np.float64(d4d_mpc_h),
        "meta_r3d_mpc_h": np.float64(r3d_mpc_h),
        "meta_radial_bin_count": np.int64(radial_bin_count),
        "meta_particles_per_shell": np.int64(particles_per_shell),
        "meta_cylinder_length_mpc_h": np.float64(cylinder_length_mpc_h),
        "meta_omega_m": np.float64(omega_m),
        "meta_rho_mean_internal": np.float64(rho_mean_internal),
        "meta_critical_modes": np.asarray(sorted(critical_modes)),
    }
    if critical_radius_mpc_h is not None:
        arrays["meta_critical_radius_mpc_h"] = np.float64(critical_radius_mpc_h)
    binners = {
        "spherical": _make_spherical_binners(
            r4d_mpc_h, radial_bin_count, r3d_mpc_h
        ),
        "cylindrical": _make_cylindrical_binners(
            r4d_mpc_h, radial_bin_count, r3d_mpc_h
        ),
    }
    for geometry, geometry_binners in binners.items():
        for mode, binner in geometry_binners.items():
            rcrit = (
                critical_radius_mpc_h if mode in critical_modes else None
            )
            length = (
                cylinder_length_mpc_h
                if geometry == "cylindrical"
                else None
            )
            prefix = f"{geometry}_{mode}"
            arrays[f"{prefix}_edges_mpc_h"] = _bin_edges(
                binner, radial_bin_count
            )
            arrays[f"{prefix}_mass_internal"] = _mass_profile(
                binner,
                radial_bin_count,
                particles_per_shell,
                rho_mean_internal,
                r_crit=rcrit,
                Lz=length,
            )
    atomic_savez(output, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--D4D", type=float, default=75.0)
    parser.add_argument("--R3D", type=float, default=500.0)
    parser.add_argument("--nrbins", type=int, default=224)
    parser.add_argument("--nshell", type=int, default=12288)
    parser.add_argument("--Lz", type=float, default=200.0)
    parser.add_argument("--omega-m", type=float, default=VALIDATION_COSMOLOGY["OMEGA_M"])
    parser.add_argument("--rcrit", type=float)
    parser.add_argument(
        "--rcrit-modes",
        nargs="+",
        choices=("omega", "volume"),
        default=["omega"],
    )
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    measure(
        d4d_mpc_h=args.D4D,
        r3d_mpc_h=args.R3D,
        radial_bin_count=args.nrbins,
        particles_per_shell=args.nshell,
        cylinder_length_mpc_h=args.Lz,
        omega_m=args.omega_m,
        critical_radius_mpc_h=args.rcrit,
        critical_modes=set(args.rcrit_modes),
        output=args.output,
    )


if __name__ == "__main__":
    main()
