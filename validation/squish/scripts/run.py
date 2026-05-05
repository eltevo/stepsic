#!/usr/bin/env python3
'''
Measure LPT P(k) recovery as a function of box aspect ratio.

Starting from a cubic box of side ``L_cube``, the z-dimension is
linearly decreased to ``L_z`` in ``N`` steps while the x and y
dimensions remain fixed. For each slab geometry
``L_cube × L_cube × L_z(i)``, a full LPT realisation is produced and
its particle P(k) is compared against the band-averaged theory
reference *for that specific box geometry*.

Results are written to an ``.npz`` archive that the companion
``validate_squish_plot.py`` plotting script can consume.

No matplotlib dependency - safe for headless HPC jobs.

Usage
-----
::

    python validate_squish_run.py \\
        --Lcube 1000 --Lz-min 200 --nsteps 5 \\
        --nmesh 256 --lpt 2 --z 31 --method cic \\
        --paired -o squish_data.npz
'''

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import logging
import time

import numpy as np
from scipy.interpolate import CubicSpline

from stepsic.field import create_grid, fourier_grid, wrap
from stepsic.pk import _bin_isotropic_modes, measure_pk

from validation import (
    ArrayF,
    ArrayI,
    GrowthData,
    generate_field,
    init_cosmology,
    run_lpt,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _generate_lz_steps(
    l_cube: float,
    lz_min: float,
    nsteps: int,
) -> ArrayF:
    '''Generate the sequence of L_z values from L_cube down to lz_min.

    Parameters
    ----------
    l_cube : float
        Starting (cubic) box side length [Mpc/h].
    lz_min : float
        Final (minimum) z-dimension [Mpc/h].
    nsteps : int
        Total number of steps (including the endpoints).

    Returns
    -------
    ndarray of shape (nsteps,)
        Linearly spaced L_z values from ``l_cube`` to ``lz_min``.
    '''
    if nsteps < 2:
        raise ValueError('nsteps must be >= 2 (need at least start and end).')
    if lz_min >= l_cube:
        raise ValueError(
            f'lz_min ({lz_min}) must be smaller than l_cube ({l_cube}).'
        )
    if lz_min <= 0:
        raise ValueError(f'lz_min must be positive, got {lz_min}.')
    return np.linspace(l_cube, lz_min, nsteps)


def _fixed_dk_mesh(
    dk: float,
    boxsize: ArrayF,
) -> tuple[ArrayI, ArrayF]:
    '''Compute mesh dimensions for a fixed physical cell size.

    Given a target cell size ``dk`` (held constant across all boxes in
    the validation suite), compute the number of voxels in each
    dimension by rounding to the nearest even integer. The effective
    mesh box ``nvox * dk`` may differ slightly from the requested
    ``boxsize`` - this is expected and documented.

    Parameters
    ----------
    dk : float
        Target physical cell size [Mpc/h], constant across all steps.
    boxsize : ndarray of shape (3,)
        Requested physical box dimensions [Mpc/h].

    Returns
    -------
    nvox : ndarray of int, shape (3,)
        Number of voxels per axis (guaranteed even).
    mesh_boxsize : ndarray of float, shape (3,)
        Effective periodic box: ``nvox * dk``.
    '''
    if np.any(boxsize < dk):
        raise ValueError(
            f'Cell size dk={dk:.6f} exceeds box dimension(s) {boxsize}.'
        )
    nvox_raw = np.rint(boxsize / dk).astype(int)
    nvox = nvox_raw + nvox_raw % 2  # round up to even
    mesh_boxsize = nvox.astype(np.float64) * dk
    return nvox, mesh_boxsize


def _band_average_reference_pk(
    kh_input: ArrayF,
    pk_input: ArrayF,
    nvox: ArrayI,
    dk: float,
    mesh_boxsize: ArrayF,
    *,
    kmin: float | None,
    kmax: float | None,
    dk_bin: float | None,
) -> tuple[ArrayF, ArrayF, ArrayI]:
    '''Band-average the input theory P(k) over the exact discrete
    Fourier modes for a given box geometry.
    '''
    _, kmod = fourier_grid(nvox, dk, hermitian=True)
    pk_spline = CubicSpline(
        np.log(kh_input), np.log(pk_input), extrapolate=False,
    )
    pk_grid = np.zeros_like(kmod)
    mask = kmod > 0.0
    pk_grid[mask] = np.exp(pk_spline(np.log(kmod[mask])))
    return _bin_isotropic_modes(
        pk_grid, kmod, nvox, mesh_boxsize, dk,
        kmin=kmin, kmax=kmax, dk_bin=dk_bin,
    )


def _run_single_lpt(
    seed_i: int,
    lpt_order: int,
    growth: GrowthData,
    nvox: ArrayI,
    dk: float,
    mesh_boxsize: ArrayF,
    pos_grid: ArrayF,
    method: str,
    *,
    kmax: float,
    paired: bool,
    counter_phase: bool,
) -> tuple[ArrayF, ArrayF, ArrayI]:
    r'''Generate one LPT realisation and measure its particle P(k).

    Passes the pre-computed ``nvox`` to :func:`stepsic.pk.measure_pk` to
    guarantee bin-edge consistency with the band-averaged reference in
    non-cubic boxes.
    '''
    delta_k = generate_field(
        nvox, dk, growth, seed_i, paired=paired, counter_phase=counter_phase,
    )

    compensate = method != 'cic'
    xpert, _ = run_lpt(
        pos_grid, delta_k, nvox, dk, growth,
        lpt_order, method, compensate=compensate,
    )
    pos_wrapped = wrap(xpert, mesh_boxsize)

    return measure_pk(
        pos_wrapped, mesh_boxsize, nvox=nvox,
        method='cic', deconvolve=True, interlace=True, subtract_shot=False,
        kmin=None, kmax=kmax, dk_bin=None,
    )


def run_squish_validation(
    l_cube: float,
    lz_min: float,
    nsteps: int,
    nmesh: int,
    lpt_order: int,
    redshift: float,
    method: str,
    seed: int,
    output: str,
    *,
    nreal: int = 1,
    paired: bool = False,
    kmax_fraction_nyquist: float = 0.8,
) -> None:
    r'''Run LPT P(k) recovery validation for a sequence of squished boxes.

    **Resolution strategy**: The ``nmesh`` parameter sets the number of
    cells along the z-axis of the *most squished* box (``L_z = lz_min``).
    This defines a fixed physical cell size ``dk = lz_min / nmesh`` that
    is reused for every box in the suite.

    Parameters
    ----------
    l_cube : float
        Cubic box side length [Mpc/h].
    lz_min : float
        Minimum z-dimension [Mpc/h].
    nsteps : int
        Number of L_z values (including endpoints).
    nmesh : int
        Grid resolution: cells along the z-axis of the most squished box.
    lpt_order : {1, 2}
        LPT order.
    redshift : float
        Target redshift for the ICs.
    method : str
        Mass-assignment scheme: ``'ngp'``, ``'cic'``, ``'tsc'``.
    seed : int
        Base RNG seed.
    output : str
        Path for the output ``.npz`` file.
    nreal : int
        Number of independent white-noise realisations to average.
    paired : bool
        If ``True``, use Angulo & Pontzen (2016) paired-fixed averaging.
    kmax_fraction_nyquist : float
        Maximum measured *k* as a fraction of the mesh Nyquist frequency.
    '''
    if not (0.0 < kmax_fraction_nyquist <= 1.0):
        raise ValueError('kmax_fraction_nyquist must lie in (0, 1].')
    if nreal < 1:
        raise ValueError('nreal must be >= 1.')
    if lpt_order not in (1, 2):
        raise ValueError(f'Unsupported LPT order: {lpt_order}.')

    lz_values = _generate_lz_steps(l_cube, lz_min, nsteps)

    # Fixed cell size from the most-squished box
    dk = lz_min / nmesh  # [Mpc/h]

    log.info(
        'Squish plan: L_cube = %.1f, L_z range = [%.1f, %.1f], '
        '%d steps, %dLPT, %s, z = %g, dk = %.6f (fixed)',
        l_cube, lz_values[0], lz_values[-1], nsteps,
        lpt_order, method.upper(), redshift, dk,
    )

    nvox_cubic, _ = _fixed_dk_mesh(dk, np.array([l_cube, l_cube, l_cube]))
    log.info(
        'Cubic box mesh: %dx%dx%d = %d cells (%.1f M particles)',
        nvox_cubic[0], nvox_cubic[1], nvox_cubic[2],
        int(np.prod(nvox_cubic)),
        int(np.prod(nvox_cubic)) / 1e6,
    )

    members_per_real = 2 if paired else 1
    n_total = nreal * members_per_real
    phase_members = (False, True) if paired else (False,)
    seeds = [seed + i for i in range(nreal)]

    k_ny = np.pi / dk
    kmax_camb = 1.05 * kmax_fraction_nyquist * k_ny

    growth_table = init_cosmology([redshift], l_cube, kmax=kmax_camb)
    growth = growth_table[redshift]

    # -- Metadata --
    results: dict[str, np.ndarray] = {}
    results['meta_l_cube'] = np.float64(l_cube)
    results['meta_lz_min'] = np.float64(lz_min)
    results['meta_nsteps'] = np.int64(nsteps)
    results['meta_lz_values'] = lz_values.astype(np.float64)
    results['meta_nmesh'] = np.int64(nmesh)
    results['meta_dk'] = np.float64(dk)
    results['meta_lpt_order'] = np.int64(lpt_order)
    results['meta_redshift'] = np.float64(redshift)
    results['meta_method'] = np.array(method)
    results['meta_seed'] = np.int64(seed)
    results['meta_nreal'] = np.int64(nreal)
    results['meta_paired'] = np.bool_(paired)
    results['meta_kmax_frac_ny'] = np.float64(kmax_fraction_nyquist)
    results['meta_n_total'] = np.int64(n_total)

    t_start = time.time()

    for i_step, lz in enumerate(lz_values):
        boxsize = np.array([l_cube, l_cube, lz], dtype=np.float64)
        nvox, mesh_boxsize = _fixed_dk_mesh(dk, boxsize)
        n_part = int(np.prod(nvox))
        kmax = kmax_fraction_nyquist * k_ny

        aspect = l_cube / lz
        tag = f'lz{lz:g}'

        log.info('%s', '=' * 72)
        log.info(
            '[%d/%d] L_z = %.1f Mpc/h (aspect %.2f:1) | '
            'mesh %dx%dx%d | dk = %.6f | N = %d | '
            'k_Ny = %.4f | k_max = %.4f',
            i_step + 1, nsteps, lz, aspect,
            nvox[0], nvox[1], nvox[2],
            dk, n_part, k_ny, kmax,
        )
        log.info('%s', '=' * 72)

        pos_grid, _ = create_grid(nvox, dk)

        k_ref, pk_ref, nmodes_ref = _band_average_reference_pk(
            growth.kh_camb, growth.pk_input, nvox, dk, mesh_boxsize,
            kmin=None, kmax=kmax, dk_bin=None,
        )
        results[f'ref_{tag}_k'] = k_ref
        results[f'ref_{tag}_pk'] = pk_ref
        results[f'ref_{tag}_nmodes'] = nmodes_ref

        pk_lpt_accum = np.zeros_like(pk_ref)
        t0 = time.time()

        for i_real, seed_i in enumerate(seeds, start=1):
            for counter_phase in phase_members:
                member_tag = 'counter' if counter_phase else 'primary'
                log.info(
                    '  %dLPT %s L_z=%.1f: seed=%d, member=%s [%d/%d]',
                    lpt_order, method.upper(), lz,
                    seed_i, member_tag, i_real, nreal,
                )
                k_p, pk_p, _ = _run_single_lpt(
                    seed_i, lpt_order, growth,
                    nvox, dk, mesh_boxsize, pos_grid,
                    method,
                    kmax=kmax, paired=paired,
                    counter_phase=counter_phase,
                )
                if not (np.array_equal(k_ref.shape, k_p.shape)
                        and np.allclose(k_ref, k_p)):
                    raise RuntimeError(
                        f'Particle bins do not match reference for '
                        f'L_z={lz}. '
                        f'ref: n={k_ref.size}, k=[{k_ref[0]:.6e}, '
                        f'{k_ref[-1]:.6e}]; '
                        f'part: n={k_p.size}, k=[{k_p[0]:.6e}, '
                        f'{k_p[-1]:.6e}]'
                    )
                pk_lpt_accum += pk_p

        dt = time.time() - t0
        results[f'lpt_{tag}_k'] = k_ref
        results[f'lpt_{tag}_pk'] = pk_lpt_accum / n_total
        results[f'lpt_{tag}_nmodes'] = nmodes_ref

        log.info(
            '  L_z=%.1f done: %.1f s (%d evaluations)',
            lz, dt, n_total,
        )

    dt_total = time.time() - t_start
    np.savez(output, **results)
    log.info(
        'Results written to %s (%d arrays, %.1f s total)',
        output, len(results), dt_total,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run stepsic P(k) recovery validation for squished '
            '(slab) box geometries. Save results to an .npz archive.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--Lcube', type=float, default=1000.0,
        help='Cubic box side length [Mpc/h].',
    )
    parser.add_argument(
        '--Lz-min', type=float, default=200.0, dest='lz_min',
        help='Minimum z-dimension [Mpc/h].',
    )
    parser.add_argument(
        '--nsteps', type=int, default=5,
        help='Number of L_z values (linearly spaced, including endpoints).',
    )
    parser.add_argument(
        '--nmesh', type=int, default=256,
        help='Grid resolution: cells along the z-axis of the most squished '
             'box.',
    )
    parser.add_argument(
        '--lpt', type=int, default=2, choices=[1, 2],
        help='LPT order.',
    )
    parser.add_argument(
        '--z', type=float, default=31.0, dest='redshift',
        help='Target redshift.',
    )
    parser.add_argument(
        '--method', type=str, default='cic',
        choices=['ngp', 'cic', 'tsc'],
        help='Mass-assignment scheme for LPT interpolation.',
    )
    parser.add_argument(
        '--seed', type=int, default=137,
        help='Base RNG seed.',
    )
    parser.add_argument(
        '--nreal', type=int, default=1,
        help='Number of independent realisations to average.',
    )
    parser.add_argument(
        '--paired', action='store_true',
        help='Enable Angulo & Pontzen (2016) paired-fixed averaging.',
    )
    parser.add_argument(
        '--kmax-frac-ny', type=float, default=0.8,
        dest='kmax_fraction_nyquist',
        help='Maximum measured k as a fraction of the mesh Nyquist frequency.',
    )
    parser.add_argument(
        '-o', '--output', type=str, default='squish_data.npz',
        help='Output .npz file path.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_squish_validation(
        l_cube=args.Lcube,
        lz_min=args.lz_min,
        nsteps=args.nsteps,
        nmesh=args.nmesh,
        lpt_order=args.lpt,
        redshift=args.redshift,
        method=args.method,
        seed=args.seed,
        output=args.output,
        nreal=args.nreal,
        paired=args.paired,
        kmax_fraction_nyquist=args.kmax_fraction_nyquist,
    )