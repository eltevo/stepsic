#!/usr/bin/env python3
"""Compare linear structure functions in padded periodic boxes with open space.

For a Gaussian field with the CAMB linear spectrum, smoothed by
F(k) = exp[-k^2 / (2 k_s^2)], the structure function

    S_AB(s) = E[(A(x+s) - A(x)) (B(x+s) - B(x))] = 2 [C_AB(0) - C_AB(s)]

of the density, the displacement and the tidal tensor is a sum over the
Fourier modes of the box. This script evaluates that sum on the stepsic
lattice (zero mode and Nyquist planes removed) for every padding factor
alpha = L_box / (2 R), and the corresponding integral for the target domain:
three-dimensional open space for a sphere, and continuous transverse modes
with discrete k_z = 2 pi n / L_z for a cylinder. Separations lie along x.

The figure shows the percentage deviation of each padded box from the target.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import jv

from validation._common.fields import init_cosmology
from validation._common.evaluation import atomic_savez
from validation._common.plotting import atomic_savefig, setup_matplotlib


GEOMETRIES = ('spherical', 'cylindrical')
COMPONENTS = np.array(
    ['density', 'psi_parallel', 'psi_perpendicular', 'T_parallel', 'T_cross'],
)
PANELS = (
    ('density', (0,)),
    (r'displacement ($\parallel$ solid, $\perp$ dashed)', (1, 2)),
    (r'tidal field ($xx$ solid, $xy$ dashed)', (3, 4)),
)
SEPARATION_FRACTIONS = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
# The filter is below exp(-100) beyond this multiple of its scale.
CUTOFF_FACTOR = 10.0
RADIAL_NODES = 512
ANGULAR_NODES = 192


def lattice_covariance(nvox, cell_mpc_h, separations_mpc_h, spectrum, filter_scale_h_mpc):
    """Covariance along x summed over the modes of a periodic lattice."""
    lengths = nvox * cell_mpc_h
    modes = [np.rint(np.fft.fftfreq(n) * n).astype(np.int64) for n in nvox]
    kx_axis, ky_axis, kz = (
        2.0 * np.pi * mode[2 * np.abs(mode) != n] / length
        for mode, n, length in zip(modes, nvox, lengths)
    )
    result = np.zeros((len(COMPONENTS), len(separations_mpc_h)))
    for kx in kx_axis:
        k2 = kx**2 + ky_axis[:, None]**2 + kz[None, :]**2
        zero_mode = k2 == 0.0
        k2[zero_mode] = 1.0
        weight = spectrum(np.sqrt(k2)) * np.exp(-k2 / filter_scale_h_mpc**2) / k2**2
        weight[zero_mode] = 0.0
        ky2 = ky_axis[:, None]**2
        factors = np.array([
            np.sum(weight * k2**2),
            np.sum(weight) * kx**2,
            np.sum(weight * ky2),
            np.sum(weight) * kx**4,
            np.sum(weight * ky2) * kx**2,
        ])
        result += factors[:, None] * np.cos(kx * separations_mpc_h)[None, :]
    return result / np.prod(lengths)


def sphere_continuum(separations_mpc_h, spectrum, filter_scale_h_mpc):
    """Covariance along x in three-dimensional open space."""
    k_node, k_weight = np.polynomial.legendre.leggauss(RADIAL_NODES)
    mu, mu_weight = np.polynomial.legendre.leggauss(ANGULAR_NODES)
    cutoff = CUTOFF_FACTOR * filter_scale_h_mpc
    k = (k_node + 1.0) * cutoff / 2.0
    k_weight = k_weight * cutoff / 2.0
    base = (
        (k_weight * k**2 * spectrum(k) * np.exp(-(k / filter_scale_h_mpc)**2))[:, None]
        * mu_weight[None, :] / (4.0 * np.pi**2)
    )
    mu2 = (mu**2)[None, :]
    inv_k2 = 1.0 / k[:, None]**2
    factors = np.array([
        np.ones((len(k), len(mu))),
        mu2 * inv_k2,
        (1.0 - mu2) / 2.0 * inv_k2,
        np.broadcast_to(mu2**2, (len(k), len(mu))),
        np.broadcast_to(mu2 * (1.0 - mu2) / 2.0, (len(k), len(mu))),
    ])
    phase = np.cos(k[:, None, None] * mu[None, :, None] * separations_mpc_h)
    return np.einsum('km,ckm,kms->cs', base, factors, phase)


def cylinder_continuum(separations_mpc_h, spectrum, filter_scale_h_mpc, length_mpc_h):
    """Covariance along x for open transverse axes and a periodic z axis."""
    node, node_weight = np.polynomial.legendre.leggauss(RADIAL_NODES)
    cutoff = CUTOFF_FACTOR * filter_scale_h_mpc
    kz_step = 2.0 * np.pi / length_mpc_h
    max_mode = int(cutoff / kz_step)
    result = np.zeros((len(COMPONENTS), len(separations_mpc_h)))
    for mode in range(-max_mode, max_mode + 1):
        kz = mode * kz_step
        q_max = np.sqrt(cutoff**2 - kz**2)
        q = ((node + 1.0) * q_max / 2.0)[:, None]
        k2 = q**2 + kz**2
        base = (
            (node_weight * q_max / 2.0)[:, None] * q * spectrum(np.sqrt(k2))
            * np.exp(-k2 / filter_scale_h_mpc**2)
            / (length_mpc_h * (2.0 * np.pi)**2)
        )
        argument = q * separations_mpc_h
        j0, j2, j4 = jv(0, argument), jv(2, argument), jv(4, argument)
        angular = np.pi * np.array([
            2.0 * j0,
            q**2 / k2**2 * (j0 - j2),
            q**2 / k2**2 * (j0 + j2),
            q**4 / k2**2 * (3.0 * j0 - 4.0 * j2 + j4) / 4.0,
            q**4 / k2**2 * (j0 - j4) / 4.0,
        ])
        result += np.sum(base * angular, axis=1)
    return result


def plot(alphas, residual_percent, output):
    setup_matplotlib()
    figure, axes = plt.subplots(2, 3, figsize=(7.07, 4.5), sharex=True)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i_geometry, row in enumerate(axes):
        for axis, (title, components) in zip(row, PANELS):
            for i_alpha, alpha in enumerate(alphas):
                for component, ls in zip(components, ('-', '--')):
                    axis.plot(
                        SEPARATION_FRACTIONS[1:],
                        residual_percent[i_geometry, i_alpha, component],
                        color=colors[i_alpha % len(colors)], lw=1.25, ls=ls,
                        marker='o', ms=2.2,
                        label=rf'$\alpha={alpha:g}$' if ls == '-' else None,
                    )
            axis.axhline(0.0, color='0.35', ls=':', lw=.8)
            axis.set_yscale('symlog', linthresh=.1, linscale=.8)
            axis.set_xticks(SEPARATION_FRACTIONS[1:])
            if i_geometry == 0:
                axis.set_title(title, fontsize=8)
            else:
                axis.set_xlabel(r'$s / R_{\mathrm{3D}}$')
        row[0].set_ylabel(
            ('Sphere' if i_geometry == 0 else 'Cylinder')
            + r': $S_{\mathrm{box}}/S_{\mathrm{open}}-1$ [\%]',
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc='lower center', ncol=len(alphas), fontsize=7,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, .06, 1.0, 1.0))
    atomic_savefig(figure, output)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nmesh', type=int, required=True,
                        help='Voxels across the domain diameter.')
    parser.add_argument('--radius', type=float, required=True, help='[Mpc/h]')
    parser.add_argument('--length', type=float, required=True,
                        help='Cylinder length [Mpc/h].')
    parser.add_argument('--redshift', type=float, required=True)
    parser.add_argument('--filter-scale', type=float, required=True,
                        help='Smoothing wavenumber k_s [h/Mpc].')
    parser.add_argument('--alphas', type=float, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figure', type=Path, required=True)
    args = parser.parse_args()

    alphas = np.array(args.alphas)
    cell_mpc_h = 2.0 * args.radius / args.nmesh
    separations = args.radius * SEPARATION_FRACTIONS
    growth = init_cosmology(
        [args.redshift], alphas[-1] * 2.0 * args.radius,
        kmax=CUTOFF_FACTOR * args.filter_scale,
    )[args.redshift]
    log_spectrum = CubicSpline(np.log(growth.kh_camb), np.log(growth.pk_input))

    def spectrum(k_h_mpc):
        return np.exp(log_spectrum(np.log(k_h_mpc)))

    continuum = np.array([
        sphere_continuum(separations, spectrum, args.filter_scale),
        cylinder_continuum(separations, spectrum, args.filter_scale, args.length),
    ])
    box = np.empty((len(GEOMETRIES), len(alphas), len(COMPONENTS), len(separations)))
    for i_geometry, geometry in enumerate(GEOMETRIES):
        for i_alpha, alpha in enumerate(alphas):
            n = round(alpha * args.nmesh)
            nz = n if geometry == 'spherical' else round(args.length / cell_mpc_h)
            box[i_geometry, i_alpha] = lattice_covariance(
                np.array([n, n, nz]), cell_mpc_h, separations, spectrum,
                args.filter_scale,
            )
            print(f'cosmological {geometry}: alpha={alpha:g}', flush=True)

    continuum_structure = 2.0 * (continuum[..., :1] - continuum)
    box_structure = 2.0 * (box[..., :1] - box)
    atomic_savez(
        args.output,
        meta_radius_mpc_h=np.float64(args.radius),
        meta_length_mpc_h=np.float64(args.length),
        meta_redshift=np.float64(args.redshift),
        meta_nmesh=np.int64(args.nmesh),
        meta_filter_scale_h_mpc=np.float64(args.filter_scale),
        geometries=np.array(GEOMETRIES),
        components=COMPONENTS,
        alphas=alphas,
        separations_mpc_h=separations,
        continuum_structure=continuum_structure,
        box_structure=box_structure,
    )
    plot(
        alphas,
        100.0 * (box_structure[..., 1:] / continuum_structure[:, None, :, 1:] - 1.0),
        args.figure,
    )


if __name__ == '__main__':
    main()
