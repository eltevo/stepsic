#!/usr/bin/env python3
"""Shared modes between the levels of production multiresolution fields.

The multiresolution scheme relies on a finer level containing every mode of
the adjacent coarser level with the same amplitude and phase. The difference
between the two levels' displacements at the same particles must then contain
only the modes that the finer level adds. The script evaluates each level of
the four-level hierarchies of the relative-motion campaign (padding alpha = 1)
at every particle of the relaxed spherical and cylindrical glasses, exactly as
the production mixing does, and compares the measured variance of each
adjacent-level difference with a mode-sum prediction:

* level j retains the modes below its Nyquist frequency;
* CIC interpolation at uniformly distributed positions transfers a mode's
  power by prod_i (2 + cos k_i Delta_j) / 3;
* the covariance of the shared modes is the position average of the product
  of the two levels' CIC transfers.

A control evaluates both levels at the particle position q, without the
production origin shift. The fields use 1LPT and fixed amplitudes.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from stepsic.lpt import lpt1
from stepsic.multiresolution import density_samples
from validation._common.evaluation import atomic_savez, load_npz

GEOMETRIES = ('spherical', 'cylindrical')
FIRST_SEED = 2000


def cic_transfer(k, cell, u):
    """Transfer of a mode exp(ikx) by CIC at fractional cell offset u."""
    return (1 - u) * np.exp(-1j * k * u * cell) + u * np.exp(1j * k * (1 - u) * cell)


def cross_transfer(k, fine, coarse, extent, shifted, samples=200_000):
    """Position average of the fine times conjugate coarse CIC transfers."""
    position = (np.arange(samples) + .5) / samples * extent
    if shifted:
        u_fine, u_coarse, phase = position / fine % 1, position / coarse % 1, 1.
    else:
        # Each level's first cell center lies at -L/2 + cell/2.
        u_fine = (position / fine - .5) % 1
        u_coarse = (position / coarse - .5) % 1
        phase = np.exp(1j * k * (coarse - fine) / 2)
    values = cic_transfer(k[:, None], fine, u_fine[None]) * np.conj(cic_transfer(k[:, None], coarse, u_coarse[None]))
    return phase * values.mean(axis=1)


def mode_axes(mesh, cell):
    index = np.rint(np.fft.fftfreq(mesh) * mesh).astype(int)
    return 2 * np.pi * index / (mesh * cell), abs(index) < mesh // 2


def mode_sum(mesh, cell, spline, factors):
    """(1/V) sum over retained modes of P k_a^2/k^4 prod_i factors_i, for a = x, y, z."""
    (kx, mx), (ky, my), (kz, mz) = (mode_axes(n, cell) for n in mesh)
    kx, ky, kz = kx[:, None, None], ky[None, :, None], kz[None, None, :]
    k2 = kx**2 + ky**2 + kz**2
    keep = (k2 > 0) & mx[:, None, None] & my[None, :, None] & mz[None, None, :]
    safe = np.where(k2 > 0, k2, 1.)
    weight = np.where(keep, np.exp(spline(.5 * np.log(safe))) / safe**2, 0.)
    weight = weight * np.real(factors[0][:, None, None] * factors[1][None, :, None] * factors[2][None, None, :])
    volume = np.prod(np.asarray(mesh) * cell)
    return np.array([np.sum(weight * kx**2), np.sum(weight * ky**2), np.sum(weight * kz**2)]) / volume


def predicted_difference(meshes, cells, spline):
    """Predicted per-component variance of each adjacent-level difference."""
    variance = [mode_sum(mesh, cell, spline, [(2 + np.cos(mode_axes(n, cell)[0] * cell)) / 3 for n in mesh])
                for mesh, cell in zip(meshes, cells)]
    result = {}
    for coarse in range(len(cells) - 1):
        fine = coarse + 1
        extent = meshes[coarse] * cells[coarse]
        for shifted in (True, False):
            factors = [cross_transfer(mode_axes(n, cells[coarse])[0], cells[fine], cells[coarse], length, shifted)
                       for n, length in zip(meshes[coarse], extent)]
            covariance = mode_sum(meshes[coarse], cells[coarse], spline, factors)
            result[coarse, shifted] = variance[fine] + variance[coarse] - 2 * covariance
    return result


def measure(inputs, geometry, seeds):
    q = inputs[f'{geometry}_q_mpc_h']
    meshes, cells = inputs[f'{geometry}_meshes'], inputs[f'{geometry}_cells_mpc_h']
    hierarchy = list(zip(meshes, cells))
    k, power = inputs['k_h_mpc'], inputs['power_mpc_h3']
    g1, _, aHf1, _ = inputs['growth']
    predicted = predicted_difference(meshes, cells, CubicSpline(np.log(k), np.log(power)))

    finest = min(cells)
    measured = {key: [] for key in predicted}
    for seed in seeds:
        fields = density_samples(k, power, hierarchy, dict(SEED=int(seed), DTYPE=np.float64, PAIRED=False, FIXED=True))
        level = {}
        for j, ((mesh, cell), delta_k) in enumerate(zip(hierarchy, fields)):
            # Production evaluates level j at q + (cell - finest)/2 (the origin shift).
            for shifted, shift in ((True, (cell - finest) / 2), (False, 0.)):
                x = q + shift
                displaced, _ = lpt1(x.copy(), delta_k, mesh, cell, g1=g1, aHf1=aHf1, method='cic')
                level[j, shifted] = displaced - x
        for coarse, shifted in predicted:
            difference = level[coarse + 1, shifted] - level[coarse, shifted]
            measured[coarse, shifted].append(np.mean(difference**2, axis=0))
        print(f'{geometry} seed {seed}: done', flush=True)

    result = {}
    for (coarse, shifted), prediction in predicted.items():
        values = np.array(measured[coarse, shifted]).sum(axis=1)
        mean, error = values.mean(), values.std(ddof=1) / np.sqrt(len(values))
        label = 'shifted' if shifted else 'unshifted'
        print(f'{geometry} cells {cells[coarse]:.2f} -> {cells[coarse + 1]:.2f} {label}: '
              f'rms difference predicted {np.sqrt(prediction.sum()):.5f}, measured {np.sqrt(mean):.5f} '
              f'(variance ratio {mean / prediction.sum():.4f} +/- {error / prediction.sum():.4f})', flush=True)
        result[f'{geometry}_pair{coarse}_{label}_predicted'] = prediction
        result[f'{geometry}_pair{coarse}_{label}_measured'] = np.array(measured[coarse, shifted])
    result[f'{geometry}_cells_mpc_h'] = cells
    result[f'{geometry}_meshes'] = meshes
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True,
                        help='inputs.npz written by relative-motion.py prepare')
    parser.add_argument('--seeds', type=int, required=True,
                        help=f'number of fixed-amplitude seeds, starting at {FIRST_SEED}')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs = load_npz(args.inputs)
    seeds = FIRST_SEED + np.arange(args.seeds)
    result = dict(seeds=seeds)
    for geometry in GEOMETRIES:
        result.update(measure(inputs, geometry, seeds))
    atomic_savez(args.output, **result)


if __name__ == '__main__':
    main()
