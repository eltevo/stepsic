#!/usr/bin/env python3
'''
Compute validation data for stepsic P(k) recovery.

Generates LPT-displaced particle P(k) measurements for one or more
combinations of mesh resolution, LPT order, target redshift, and
mass-assignment scheme. Results are serialised to a ``.npz`` archive
that a companion plotting script can consume.

Also records direct-field P(k) measurements for debugging
(``--debug-field``); these are skipped by default in paired-fixed mode
since they are trivially unity.

No matplotlib dependency - safe for headless HPC jobs.

Usage
-----
::

    # Panel A: resolution dependence (64, 128, 256 at z=31, 2LPT, CIC)
    python validate_grid_run.py --nmesh 64 128 256 --lpt 2 --z 31 \\
        --method cic --paired -o panel_a.npz

    # Panel B: redshift dependence (z=63, 31, 15 at 128^3, 2LPT, CIC)
    python validate_grid_run.py --nmesh 128 --lpt 2 --z 63 31 15 \\
        --method cic --paired -o panel_b.npz

    # Panel C: 1LPT vs 2LPT stress test (z=15, 128^3, CIC)
    python validate_grid_run.py --nmesh 128 --lpt 1 2 --z 15 \\
        --method cic --paired -o panel_c.npz

    # Panel D: MAS comparison (NGP, CIC, TSC at 256^3, z=31, 2LPT)
    python validate_grid_run.py --nmesh 256 --lpt 2 --z 31 \\
        --method ngp cic tsc --paired -o panel_d.npz
'''

from __future__ import annotations

from pathlib import Path


import argparse
import logging
import time
from typing import Iterable

import numpy as np

from validation._common.evaluation import atomic_savez
from scipy.interpolate import CubicSpline

from stepsic.field import (
    create_grid,
    cubic_voxels,
    fourier_kmod,
    wrap,
)
from stepsic.pk import (
    _bin_isotropic_modes,
    measure_pk,
    measure_pk_from_delta_k,
)

from validation import (
    ArrayF,
    ArrayI,
    GrowthData,
    generate_field,
    init_cosmology,
    run_lpt,
    z_tag,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _iter_lpt_orders(lpt_orders: Iterable[int]) -> list[int]:
    '''Validate and normalize the requested LPT orders.'''
    orders = sorted(set(int(order) for order in lpt_orders))
    invalid = [order for order in orders if order not in (1, 2)]
    if invalid:
        raise ValueError(f'Unsupported LPT orders requested: {invalid}')
    return orders


def _band_average_reference_pk(
    kh_input: ArrayF,
    pk_input: ArrayF,
    nvox: ArrayI,
    dk: float,
    boxsize: ArrayF,
    *,
    kmin: float | None,
    kmax: float | None,
    dk_bin: float | None,
) -> tuple[ArrayF, ArrayF, ArrayI]:
    '''Band-average the input theory P(k) over the exact discrete Fourier modes.'''
    kmod = fourier_kmod(nvox, dk, hermitian=True)
    pk_spline = CubicSpline(
        np.log(kh_input), np.log(pk_input), extrapolate=False,
    )
    pk_grid = np.zeros_like(kmod)
    mask = kmod > 0.0
    pk_grid[mask] = np.exp(pk_spline(np.log(kmod[mask])))
    return _bin_isotropic_modes(
        pk_grid, kmod, nvox, boxsize, dk,
        kmin=kmin, kmax=kmax, dk_bin=dk_bin,
    )


def _run_single_field(
    seed_i: int,
    growth: GrowthData,
    nvox: ArrayI,
    dk: float,
    boxsize: ArrayF,
    *,
    kmax: float,
    paired: bool,
    counter_phase: bool,
) -> tuple[ArrayF, ArrayF, ArrayI]:
    r'''Generate one δ_k realisation and measure its direct Fourier P(k).'''
    delta_k = generate_field(
        nvox, dk, growth, seed_i, paired=paired, counter_phase=counter_phase,
    )
    return measure_pk_from_delta_k(
        delta_k, boxsize, nvox, kmin=None, kmax=kmax, dk_bin=None,
    )


def _run_single_lpt(
    seed_i: int,
    lpt_order: int,
    growth: GrowthData,
    nvox: ArrayI,
    dk: float,
    boxsize: ArrayF,
    pos_grid: ArrayF,
    compensate: bool,
    method: str,
    nmesh: int,
    *,
    kmax: float,
    paired: bool,
    counter_phase: bool,
) -> tuple[ArrayF, ArrayF, ArrayI]:
    r'''Generate one LPT realisation and measure its particle P(k).'''
    delta_k = generate_field(
        nvox, dk, growth, seed_i, paired=paired, counter_phase=counter_phase,
    )

    compensate = method != 'cic'
    xpert, _ = run_lpt(
        pos_grid, delta_k, nvox, dk, growth,
        lpt_order, method, compensate=compensate,
    )
    pos_wrapped = wrap(xpert, boxsize)

    # Measurement uses a fixed MAS (CIC + deconvolution + interlacing)
    # independent of the LPT interpolation method under test.
    return measure_pk(
        pos_wrapped, boxsize, nmesh,
        method='cic', deconvolve=True, interlace=True, subtract_shot=False,
        kmin=None, kmax=kmax, dk_bin=None,
    )


# ---------------------------------------------------------------------------
# Archive key convention
# ---------------------------------------------------------------------------
#
#   ref_{method}_{nmesh}_z{z}_k        - band-averaged reference k
#   ref_{method}_{nmesh}_z{z}_pk       - band-averaged reference P(k)
#   ref_{method}_{nmesh}_z{z}_nmodes   - independent Fourier modes
#
#   lpt{order}_{method}_{nmesh}_z{z}_k
#   lpt{order}_{method}_{nmesh}_z{z}_pk
#   lpt{order}_{method}_{nmesh}_z{z}_nmodes
#
#   field_{method}_{nmesh}_z{z}_*      (only with --debug-field)
#
# Scalar metadata live under meta_* keys.


def run_validation(
    nmesh_list: list[int],
    lpt_orders: list[int],
    lbox_scalar: float,
    redshifts: list[float],
    methods: list[str],
    seed: int,
    output: str,
    *,
    nreal: int = 1,
    paired: bool = False,
    compensate: bool = True,
    kmax_fraction_nyquist: float = 1.0,
    debug_field: bool = False,
) -> None:
    r'''Run LPT P(k) validation across all requested parameter combinations.

    The outer loop order is:  redshift -> nmesh -> method -> lpt_order.
    Cosmology (CAMB + growth factors) is computed once; grid setup
    (``create_grid``) once per nmesh; LPT + P(k) once per
    (method, lpt_order) combination.

    Parameters
    ----------
    nmesh_list : list of int
        Grid resolutions to validate.
    lpt_orders : list of int
        LPT orders to run (1 and/or 2).
    lbox_scalar : float
        Cubic box side length [Mpc/h].
    redshifts : list of float
        Target redshift(s) for the ICs.
    methods : list of str
        Mass-assignment scheme(s): ``'ngp'``, ``'cic'``, ``'tsc'``.
    seed : int
        Base RNG seed.
    output : str
        Path for the output ``.npz`` file.
    nreal : int
        Number of independent white-noise realisations to average.
    paired : bool
        If ``True``, expand each realisation into an Angulo & Pontzen
        (2016) paired-fixed pair.
    compensate : bool
        If ``True``, apply the standard LPT compensation kernel.
    kmax_fraction_nyquist : float
        Maximum measured `k` as a fraction of the mesh Nyquist frequency.
    debug_field : bool
        If ``True``, also measure the direct Fourier-field P(k).
    '''
    if not (0.0 < kmax_fraction_nyquist <= 1.0):
        raise ValueError('kmax_fraction_nyquist must lie in (0, 1].')
    if nreal < 1:
        raise ValueError('nreal must be >= 1.')

    boxsize = np.array([lbox_scalar] * 3, dtype=np.float64)
    lpt_orders = _iter_lpt_orders(lpt_orders)

    # Determine the maximum k any mesh will need, so CAMB covers it.
    # The P(k) bins extend up to kmax + k_fund (the upper edge of the last bin),
    # so kmax_camb must reach there - a flat 5% buffer is too small for coarse
    # meshes where k_fund is a significant fraction of k_ny.
    k_fund_box = 2.0 * np.pi / np.max(boxsize)
    max_requested_k = 0.0
    for nmesh in nmesh_list:
        _, dk_tmp = cubic_voxels(nmesh, boxsize)
        k_ny_tmp = np.pi / dk_tmp
        max_requested_k = max(max_requested_k, kmax_fraction_nyquist * k_ny_tmp)
    kmax_camb = max_requested_k + k_fund_box

    members_per_real = 2 if paired else 1
    n_total = nreal * members_per_real
    phase_members = (False, True) if paired else (False,)
    seeds = [seed + i for i in range(nreal)]

    growth_table = init_cosmology(redshifts, lbox_scalar, kmax=kmax_camb)

    results: dict[str, np.ndarray] = {}
    results['meta_nmesh_list'] = np.asarray(nmesh_list, dtype=np.int64)
    results['meta_lpt_orders'] = np.asarray(lpt_orders, dtype=np.int64)
    results['meta_redshifts'] = np.asarray(redshifts, dtype=np.float64)
    results['meta_methods'] = np.array(methods)
    results['meta_lbox'] = np.float64(lbox_scalar)
    results['meta_seed'] = np.int64(seed)
    results['meta_nreal'] = np.int64(nreal)
    results['meta_paired'] = np.bool_(paired)
    results['meta_kmax_frac_ny'] = np.float64(kmax_fraction_nyquist)
    results['meta_n_total'] = np.int64(n_total)

    n_combos = len(redshifts) * len(nmesh_list) * len(methods) * len(lpt_orders)
    log.info(
        'Validation sweep: %d redshift(s) | %d mesh(es) | %d method(s) '
        '| %d LPT order(s) = %d combinations, %d evaluations each',
        len(redshifts), len(nmesh_list), len(methods), len(lpt_orders),
        n_combos, n_total,
    )

    combo_idx = 0
    t_start = time.time()

    for z in redshifts:
        growth = growth_table[z]
        zk = z_tag(z)

        for nmesh in nmesh_list:
            nvox, dk = cubic_voxels(nmesh, boxsize)
            n_part = int(np.prod(nvox))
            k_ny = np.pi / dk
            kmax = kmax_fraction_nyquist * k_ny

            log.info('%s', '=' * 72)
            log.info(
                'z = %g | Mesh %d^3 (%dx%dx%d) | dk = %.6f Mpc/h | '
                'N = %d | k_Ny = %.4f h/Mpc | k_max = %.4f h/Mpc',
                z, nmesh, nvox[0], nvox[1], nvox[2], dk, n_part, k_ny, kmax,
            )
            log.info('%s', '=' * 72)

            pos_grid, _ = create_grid(nvox, dk)

            for method in methods:
                # Band-averaged reference
                ref_prefix = f'ref_{method}_{nmesh}_{zk}'
                if f'{ref_prefix}_k' not in results:
                    k_ref, pk_ref, nmodes_ref = _band_average_reference_pk(
                        growth.kh_camb, growth.pk_input, nvox, dk, boxsize,
                        kmin=None, kmax=kmax, dk_bin=None,
                    )
                    results[f'{ref_prefix}_k'] = k_ref
                    results[f'{ref_prefix}_pk'] = pk_ref
                    results[f'{ref_prefix}_nmodes'] = nmodes_ref
                else:
                    k_ref = results[f'{ref_prefix}_k']
                    pk_ref = results[f'{ref_prefix}_pk']
                    nmodes_ref = results[f'{ref_prefix}_nmodes']

                # -- Optional direct-field P(k) (debug only) --
                run_field = debug_field and not paired
                if run_field:
                    pk_field_accum = np.zeros_like(pk_ref)
                    for i_real, seed_i in enumerate(seeds, start=1):
                        for counter_phase in phase_members:
                            tag = 'counter' if counter_phase else 'primary'
                            log.info(
                                '  Direct field [%s]: seed=%d, member=%s '
                                '[%d/%d]',
                                method, seed_i, tag, i_real, nreal,
                            )
                            k_f, pk_f, _ = _run_single_field(
                                seed_i, growth, nvox, dk, boxsize,
                                kmax=kmax, paired=paired,
                                counter_phase=counter_phase,
                            )
                            if not (np.array_equal(k_ref.shape, k_f.shape)
                                    and np.allclose(k_ref, k_f)):
                                raise RuntimeError(
                                    'Direct-field and reference binning '
                                    'are inconsistent.'
                                )
                            pk_field_accum += pk_f

                    fld_prefix = f'field_{method}_{nmesh}_{zk}'
                    results[f'{fld_prefix}_k'] = k_ref
                    results[f'{fld_prefix}_pk'] = pk_field_accum / n_total
                    results[f'{fld_prefix}_nmodes'] = nmodes_ref

                # -- LPT particle P(k) --
                for lpt_order in lpt_orders:
                    combo_idx += 1
                    pk_lpt_accum = np.zeros_like(pk_ref)
                    t0 = time.time()

                    for i_real, seed_i in enumerate(seeds, start=1):
                        for counter_phase in phase_members:
                            tag = 'counter' if counter_phase else 'primary'
                            log.info(
                                '  [%d/%d] %dLPT %d^3 %s z=%g: seed=%d, '
                                'member=%s [%d/%d]',
                                combo_idx, n_combos,
                                lpt_order, nmesh, method.upper(), z,
                                seed_i, tag, i_real, nreal,
                            )
                            k_p, pk_p, _ = _run_single_lpt(
                                seed_i, lpt_order, growth,
                                nvox, dk, boxsize, pos_grid,
                                compensate, method, nmesh,
                                kmax=kmax, paired=paired,
                                counter_phase=counter_phase,
                            )
                            if not (np.array_equal(k_ref.shape, k_p.shape)
                                    and np.allclose(k_ref, k_p)):
                                raise RuntimeError(
                                    'Particle bins do not match reference '
                                    f'for {lpt_order}LPT {nmesh}^3 '
                                    f'{method} z={z}.'
                                )
                            pk_lpt_accum += pk_p

                    dt = time.time() - t0
                    lpt_prefix = f'lpt{lpt_order}_{method}_{nmesh}_{zk}'
                    results[f'{lpt_prefix}_k'] = k_ref
                    results[f'{lpt_prefix}_pk'] = pk_lpt_accum / n_total
                    results[f'{lpt_prefix}_nmodes'] = nmodes_ref

                    log.info(
                        '  %dLPT %d^3 %s z=%g: %.1f s (%d evaluations)',
                        lpt_order, nmesh, method.upper(), z, dt, n_total,
                    )

    dt_total = time.time() - t_start
    atomic_savez(output, **results)
    log.info(
        'Results written to %s (%d arrays, %.1f s total)',
        output, len(results), dt_total,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run stepsic P(k) validation across multiple resolutions, '
            'redshifts, LPT orders, and MAS methods. Save results to '
            'an .npz archive for plotting.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--nmesh', type=int, nargs='+', default=[128],
        help='Mesh resolution(s) to test.',
    )
    parser.add_argument(
        '--lpt', type=int, nargs='+', default=[1, 2],
        help='LPT order(s) to test. Allowed: 1, 2.',
    )
    parser.add_argument(
        '--Lbox', type=float, default=500.0,
        help='Box side length [Mpc/h].',
    )
    parser.add_argument(
        '--z', type=float, nargs='+', default=[31.0], dest='redshifts',
        help='Target redshift(s).',
    )
    parser.add_argument(
        '--method', type=str, nargs='+', default=['cic'],
        choices=['ngp', 'cic', 'tsc'],
        help='Mass-assignment scheme(s).',
    )
    parser.add_argument(
        '--seed', type=int, default=137,
        help='Base RNG seed.',
    )
    parser.add_argument(
        '--nreal', type=int, default=1,
        help='Number of independent realisations to average. '
             'Seeds: [seed, seed+1, ..., seed+nreal-1].',
    )
    parser.add_argument(
        '--paired', action='store_true',
        help='Enable Angulo & Pontzen (2016) paired-fixed averaging.',
    )
    parser.add_argument(
        '--compensate', action='store_true',
        help='Apply the standard LPT compensation kernel.',
    )
    parser.add_argument(
        '--kmax-frac-ny', type=float, default=0.8,
        dest='kmax_fraction_nyquist',
        help='Maximum measured k as a fraction of the mesh Nyquist frequency.',
    )
    parser.add_argument(
        '--debug-field', action='store_true', dest='debug_field',
        help='Also measure direct Fourier-field P(k) (skipped in '
             'paired mode; useful for debugging).',
    )
    parser.add_argument(
        '-o', '--output', type=str, default='validation_data.npz',
        help='Output .npz file path.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_validation(
        nmesh_list=args.nmesh,
        lpt_orders=args.lpt,
        lbox_scalar=args.Lbox,
        redshifts=args.redshifts,
        methods=args.method,
        seed=args.seed,
        output=args.output,
        nreal=args.nreal,
        paired=args.paired,
        compensate=args.compensate,
        kmax_fraction_nyquist=args.kmax_fraction_nyquist,
        debug_field=args.debug_field,
    )