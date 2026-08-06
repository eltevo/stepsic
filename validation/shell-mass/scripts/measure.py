#!/usr/bin/env python3
"""Measure shell mass profiles into the campaign's numerical archive."""

from __future__ import annotations

import argparse

import numpy as np

from plot import (
    _bin_edges,
    _make_cylindrical_binners,
    _make_spherical_binners,
    _mass_profile,
    _rho_mean,
)
from validation import VALIDATION_COSMOLOGY
from validation._common.evaluation import atomic_savez


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
        "meta_critical_radius_mpc_h": np.float64(
            critical_radius_mpc_h
            if critical_radius_mpc_h is not None
            else np.nan
        ),
        "meta_critical_modes": np.asarray(sorted(critical_modes)),
    }
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
