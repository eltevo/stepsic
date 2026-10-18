#!/usr/bin/env python3
"""Measure slab and cube-cut covariance against the represented Fourier modes."""

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

from stepsic.field import fourier_vectors
from stepsic.particles import create_grid
from validation._common.artifacts import load_archive
from validation._common.fields import generate_field, init_cosmology, run_lpt
from validation._common.evaluation import atomic_savez
from validation._common.plotting import atomic_savefig, setup_matplotlib
from validation._common.precision import simultaneous_band


def gaussian_window(nvox, cell_mpc_h, filter_scale_h_mpc):
    """Return exp(-k²/2k_s²) on the Hermitian mesh, with the Nyquist planes set to zero."""
    vectors = fourier_vectors(nvox, cell_mpc_h, hermitian=True)
    k2 = (vectors[0][:, None, None]**2 + vectors[1][None, :, None]**2
          + vectors[2][None, None, :]**2)
    window = np.exp(-k2/(2*filter_scale_h_mpc**2))
    window[nvox[0]//2, :, :] = 0
    window[:, nvox[1]//2, :] = 0
    window[:, :, -1] = 0
    return window


def mode_covariance(nvox, cell_mpc_h, k_h_mpc, power_mpc_h3, window, cut_cells=None):
    """Predict the variance of each displacement component from the mesh's Fourier modes.

    ``window`` multiplies the density amplitudes. The variance is taken about the sample mean. A box loses only its k = 0 mode that way. A cut of ``cut_cells`` consecutive z cells spanning the full x and y periods also loses part of every k_x = k_y = 0 mode, by the squared Fourier transform of the cut.
    """
    vectors = fourier_vectors(nvox, cell_mpc_h, hermitian=True)
    kx, ky, kz = vectors[0][:, None, None], vectors[1][None, :, None], vectors[2][None, None, :]
    k2 = kx*kx + ky*ky + kz*kz
    positive = k2 > 0
    spectrum = np.zeros_like(k2)
    spline = CubicSpline(np.log(k_h_mpc), np.log(power_mpc_h3))
    spectrum[positive] = np.exp(spline(np.log(np.sqrt(k2[positive]))))
    k2[0, 0, 0] = np.inf
    weight = np.full(len(vectors[2]), 2.)
    weight[0] = weight[-1] = 1
    variance = spectrum * window**2 * weight[None, None, :] / np.prod(nvox*cell_mpc_h)
    if cut_cells is not None:
        wz = np.abs(np.exp(1j * vectors[2][:, None] * cell_mpc_h * np.arange(cut_cells)).mean(axis=1))**2
        variance[0, 0, :] *= 1-wz
    return np.array([np.sum(variance*k*k/k2**2) for k in (kx, ky, kz)])


def plot_covariance(out, figure):
    """Plot the saved component variances and simultaneous sampling intervals."""
    setup_matplotlib()
    plt.rcParams.update({'font.size': 8, 'legend.fontsize': 7})
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.54, 4.8))
    fig.subplots_adjust(left=.14, right=.98, bottom=.18, top=.95, hspace=.57)
    meshes, aspects = out['meta_meshes'], out['meta_aspects']
    n = meshes[-1]
    measured, predicted = out[f'measured_{n}'], out[f'predicted_{n}']
    ratio = measured.mean(axis=0)[:, 0] / measured.mean(axis=0)[:, 1]
    for axis, (colour, marker) in enumerate(zip(
        ('#0072B2', '#D55E00', '#009E73'), ('o', 's', '^'),
    )):
        ax_a.plot(aspects, ratio[:, axis], color=colour, marker=marker,
                  ms=3.2, lw=.7, label=('$x$', '$y$', '$z$')[axis])
    # The two transverse predictions coincide by symmetry.
    for axis in (0, 2):
        ax_a.plot(aspects, predicted[:, 0, axis] / predicted[:, 1, axis],
                  color='k', ls='--', lw=.8,
                  label='Prediction' if axis == 0 else None)
    ax_a.axhline(1, color='0.65', ls=':', lw=.6, zorder=0)
    ax_a.set_title('Slab / cube-cut displacement variance', loc='left', fontsize=8)
    ax_a.text(.03, .06, '(a)', transform=ax_a.transAxes, va='bottom', ha='left')
    ax_a.set(xlabel=r'Aspect ratio $L/L_z$', xticks=aspects)
    handles, labels = ax_a.get_legend_handles_labels()
    ax_a.legend(handles[-1:] + handles[:-1], labels[-1:] + labels[:-1],
                loc='upper left', ncol=1, frameon=False, handlelength=1.8)

    positions = (8*np.arange(len(aspects))[:, None] + np.arange(6)).ravel()
    for index, n in enumerate(meshes):
        mean = 100*out[f'error_mean_{n}']
        low, high = 100*out[f'error_low_{n}'], 100*out[f'error_high_{n}']
        ax_b.errorbar(positions + .28*(index-(len(meshes)-1)/2), mean,
                      yerr=[mean-low, high-mean], fmt=('o', 's', '^')[index],
                      color=('#0072B2', '#D55E00', '#009E73')[index], ms=1.9,
                      elinewidth=.65, capsize=0, label=rf'$N={n}$')
    ax_b.axhspan(-1, 1, color='0.9', zorder=0)
    ax_b.axhline(0, color='0.55', lw=.5, zorder=0)
    ax_b.set_title(r'Measured / predicted variance $-\,1$ [\%]', loc='left', fontsize=8)
    ax_b.text(.03, .06, '(b)', transform=ax_b.transAxes, va='bottom', ha='left')
    ax_b.set_xticks(positions, ['$x$', '$y$', '$z$'] * (2*len(aspects)))
    ax_b.tick_params(axis='x', labelsize=6, length=2, top=False, pad=2)
    for group, aspect in enumerate(aspects):
        start = 8*group
        for offset, label in ((1, 'Slab'), (4, 'Cut')):
            ax_b.text(start+offset, -.13, label, ha='center', va='top',
                      transform=ax_b.get_xaxis_transform(), fontsize=6)
        ax_b.text(start+2.5, -.23, str(aspect), ha='center', va='top',
                  transform=ax_b.get_xaxis_transform(), fontsize=8)
        if group:
            ax_b.axvline(start-1.5, color='0.85', lw=.5, zorder=0)
    ax_b.set_xlabel(r'Aspect ratio $L/L_z$', labelpad=34)
    ax_b.legend(loc='upper left', ncol=1, frameon=False, handletextpad=.3)
    atomic_savefig(fig, figure)
    plt.close(fig)


def run(a):
    if a.nreal < 2:
        raise ValueError('at least two seeds are needed to form an interval')
    if any(n % (2*aspect) for n in a.nmesh for aspect in a.aspects):
        raise ValueError('every mesh must be divisible by twice every aspect ratio')
    growth = init_cosmology([a.z], a.Lcube, kmax=2*np.pi*max(a.nmesh)/a.Lcube)[a.z]
    out = dict(meta_meshes=np.array(a.nmesh), meta_aspects=np.array(a.aspects),
               meta_nreal=np.int64(a.nreal), meta_lbox_mpc_h=np.float64(a.Lcube),
               meta_filter_scale_h_mpc=np.float64(a.filter_scale),
               meta_ensemble=np.array('Gaussian'), meta_redshift=np.float64(a.z),
               meta_seed=np.int64(a.seed), meta_lpt_order=np.int64(1),
               meta_interval_family=np.array('all meshes, aspects, selections and components'),
               meta_numerical_accuracy=np.array('sampling intervals only; numerical bias bound not established'))
    start = time.monotonic()
    for n in a.nmesh:
        cell = a.Lcube/n
        cube = np.array([n]*3)
        pos = create_grid(cube, cell, return_coords=False)
        cube_window = gaussian_window(cube, cell, a.filter_scale)
        measured = np.empty((a.nreal, len(a.aspects), 2, 3))
        predicted = np.empty((len(a.aspects), 2, 3))
        windows = []
        slab_positions = []
        for i, aspect in enumerate(a.aspects):
            shape = np.array([n, n, n//aspect])
            windows.append(gaussian_window(shape, cell, a.filter_scale))
            predicted[i, 0] = mode_covariance(shape, cell, growth.kh_camb, growth.pk_input, windows[i])
            predicted[i, 1] = mode_covariance(cube, cell, growth.kh_camb, growth.pk_input, cube_window, n//aspect)
            slab_positions.append(create_grid(shape, cell, return_coords=False))
        for index in range(a.nreal):
            field = generate_field(cube, cell, growth, a.seed+index)*cube_window
            xp, _ = run_lpt(pos, field, cube, cell, growth, 1, 'cic', compensate=False)
            displacement = (xp-pos).reshape(n, n, n, 3)
            del field, xp
            for i, aspect in enumerate(a.aspects):
                nz = n//aspect
                lower = (n-nz)//2
                measured[index, i, 1] = displacement[:, :, lower:lower+nz].var(axis=(0,1,2))
                if aspect == 1:
                    measured[index, i, 0] = measured[index, i, 1]
                else:
                    shape = np.array([n, n, nz])
                    field = generate_field(shape, cell, growth, a.seed+index)*windows[i]
                    xp, _ = run_lpt(slab_positions[i], field, shape, cell, growth, 1, 'cic', compensate=False)
                    measured[index, i, 0] = (xp-slab_positions[i]).var(axis=0)
                    del field, xp
            print(f'covariance mesh={n} seed={index+1}/{a.nreal}', flush=True)
        out[f'measured_{n}'] = measured
        out[f'predicted_{n}'] = predicted
        errors = (measured/predicted-1).reshape(a.nreal, -1)
        mean, low, high = simultaneous_band(errors, np.zeros(errors.shape[1]), confidence=1-.05/len(a.nmesh))
        out[f'error_mean_{n}'], out[f'error_low_{n}'], out[f'error_high_{n}'] = mean, low, high
    atomic_savez(a.output, **out)
    print(f'covariance completed in {time.monotonic()-start:.1f} s; bands exclude numerical bias')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    r = commands.add_parser('run', help='measure and predict the variances')
    r.add_argument('--Lcube', type=float, required=True, help='cube side [Mpc/h]')
    r.add_argument('--aspects', type=int, nargs='+', required=True, help='ratios L/L_z')
    r.add_argument('--nmesh', type=int, nargs='+', required=True, help='cells along the cube side')
    r.add_argument('--nreal', type=int, required=True, help='independent seeds')
    r.add_argument('--filter-scale', type=float, required=True, help='Gaussian filter scale k_s [h/Mpc]')
    r.add_argument('--z', type=float, required=True, help='redshift')
    r.add_argument('--seed', type=int, required=True, help='first white-noise seed')
    r.add_argument('--output', type=Path, required=True)
    g = commands.add_parser('plot', help='plot a saved measurement')
    g.add_argument('--input', type=Path, required=True)
    g.add_argument('--figure', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'run':
        run(a)
    else:
        plot_covariance(load_archive(a.input), a.figure)


if __name__ == '__main__':
    main()
