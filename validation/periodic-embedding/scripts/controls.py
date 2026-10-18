#!/usr/bin/env python3
"""Compare lpt1 in a padded periodic box with an exact open-domain displacement.

The source is a pair of Gaussian density peaks of opposite sign, placed on the
x axis at +/- OFFSET_FRACTION * R with width WIDTH_FRACTION * R. Its total mass
vanishes, and the displacement that satisfies div(Psi) = -delta in open space
follows in closed form from the mass enclosed around each peak. For a sphere
the peaks are three-dimensional; for a cylinder they are uniform along z.

``stepsic.lpt.lpt1`` solves the same source on a periodic mesh of side
alpha * 2R. The difference from the open-domain displacement, sampled on
rings inside the domain, is the error caused by the periodic images.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfftn
from scipy.special import erf

from stepsic.lpt import lpt1
from validation._common.evaluation import atomic_savez
from validation._common.plotting import atomic_savefig, setup_matplotlib

OFFSET_FRACTION = 0.2
WIDTH_FRACTION = 0.1
SAMPLE_FRACTIONS = np.array([0.2, 0.6, 0.9])
SAMPLES_PER_RADIUS = 256


def open_displacement(positions, offset, sigma, geometry):
    """Displacement satisfying div(Psi) = -delta for the Gaussian pair."""
    dimension = 3 if geometry == 'spherical' else 2
    result = np.zeros_like(positions)
    for sign in (-1, 1):
        relative = positions[:, :dimension].copy()
        relative[:, 0] -= sign * offset
        r = np.linalg.norm(relative, axis=1)
        if dimension == 3:
            enclosed = np.sqrt(np.pi / 2) * sigma**3 * (
                erf(r / (np.sqrt(2) * sigma))
                - np.sqrt(2 / np.pi) * r / sigma * np.exp(-r * r / (2 * sigma**2))
            )
        else:
            enclosed = sigma**2 * -np.expm1(-r * r / (2 * sigma**2))
        result[:, :dimension] -= sign * relative * (enclosed / r**dimension)[:, None]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nmesh', type=int, required=True,
                   help='Voxels across the domain diameter.')
    p.add_argument('--geometry', choices=['spherical', 'cylindrical'], required=True)
    p.add_argument('--radius', type=float, required=True, help='Domain radius [Mpc/h].')
    p.add_argument('--length', type=float, required=True,
                   help='Cylinder length [Mpc/h] (cylindrical only).')
    p.add_argument('--alphas', type=float, nargs='+', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--figure', type=Path, required=True)
    a = p.parse_args()

    offset = OFFSET_FRACTION * a.radius
    sigma = WIDTH_FRACTION * a.radius
    meshes = np.array([a.nmesh // 2, a.nmesh])
    alphas = np.array(a.alphas)

    rng = np.random.default_rng(a.seed)
    directions = rng.normal(size=(len(SAMPLE_FRACTIONS) * SAMPLES_PER_RADIUS, 3))
    if a.geometry == 'cylindrical':
        directions[:, 2] = 0.0
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    ring = np.repeat(np.arange(len(SAMPLE_FRACTIONS)), SAMPLES_PER_RADIUS)
    positions = directions * (SAMPLE_FRACTIONS[ring] * a.radius)[:, None]
    exact = open_displacement(positions, offset, sigma, a.geometry)

    error = np.empty((len(meshes), len(alphas), len(SAMPLE_FRACTIONS)))
    for i_mesh, base in enumerate(meshes):
        cell = 2 * a.radius / base
        for i_alpha, alpha in enumerate(alphas):
            n = round(base * alpha)
            nz = n if a.geometry == 'spherical' else round(a.length / cell)
            nvox = np.array([n, n, nz])
            x, y, z = (
                (np.arange(count) + .5 - count / 2) * cell for count in nvox
            )
            delta = np.zeros(tuple(nvox))
            for sign in (-1, 1):
                radius2 = (x[:, None, None] - sign * offset)**2 + y[None, :, None]**2
                if a.geometry == 'spherical':
                    radius2 = radius2 + z[None, None, :]**2
                delta += sign * np.exp(-radius2 / (2 * sigma**2))
            displaced, _ = lpt1(positions, rfftn(delta), nvox, cell, 1., 1., method='cic')
            residual = displaced - positions - exact
            for i_ring in range(len(SAMPLE_FRACTIONS)):
                on_ring = ring == i_ring
                error[i_mesh, i_alpha, i_ring] = np.sqrt(
                    np.mean(residual[on_ring]**2) / np.mean(exact[on_ring]**2)
                )
            print(f'open-source {a.geometry}: base={base} padding={alpha:g}', flush=True)

    atomic_savez(
        a.output,
        meta_geometry=np.array(a.geometry),
        meta_radius_mpc_h=np.float64(a.radius),
        meshes=meshes,
        alphas=alphas,
        sample_fractions=SAMPLE_FRACTIONS,
        relative_source_error=error,
    )

    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.46, 3.2), constrained_layout=True)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i_mesh, (base, ls) in enumerate(zip(meshes, ('--', '-'))):
        for i_ring, fraction in enumerate(SAMPLE_FRACTIONS):
            ax.semilogy(
                alphas, error[i_mesh, :, i_ring], marker='.', ls=ls,
                color=colors[i_ring],
                label=rf'$N={base}$, $r={fraction:g}\,R$',
            )
    ax.set_xlabel(r'$\alpha = L_{\mathrm{box}} / 2 R_{\mathrm{3D}}$')
    ax.set_ylabel('relative RMS displacement error')
    fig.legend(fontsize=6, ncol=2, frameon=False, loc='outside upper center')
    atomic_savefig(fig, a.figure)
    plt.close(fig)


if __name__ == '__main__':
    main()
