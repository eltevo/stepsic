#!/usr/bin/env python3
'''
Measure how well particle power recovers the input spectrum.

The particle load is a periodic glass, tiled to fill the box and displaced with LPT. The script varies the Fourier mesh, LPT order, target redshift, and interpolation method, always with the interpolation window compensated, then writes the measurements to a NumPy archive. The power spectrum is measured on a mesh as fine as the load, whatever the Fourier mesh.

``--debug-field`` also records power directly from the Fourier field. Paired-fixed runs omit this by default because the ratio is identically one.

Usage
-----
::

    # Panel A: three Fourier meshes on a 256^3 glass tiled four times per side
    python run.py --glass glass.hdf5 --tiles 4 --nmesh 128 256 512 \\
        --lpt 2 --z 31 --method cic --paired -o panel_a.npz

    # Panel D: interpolation methods
    python run.py --glass glass.hdf5 --tiles 4 --nmesh 512 \\
        --lpt 2 --z 31 --method ngp cic tsc --paired -o panel_d.npz
'''

from __future__ import annotations

from pathlib import Path


import argparse
import logging
import time
from typing import Iterable

import numpy as np

from validation._common.evaluation import atomic_savez

from stepsic.field import cubic_voxels
from stepsic.pk import measure_pk_from_delta_k

from validation._common.fields import (
    ArrayF,
    ArrayI,
    GrowthData,
    InterlacedPowerMeter,
    band_average_reference_pk,
    generate_field,
    init_cosmology,
    iter_tiled_load,
    load_glass_tile,
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
#   load_{nmesh}_k, load_{nmesh}_pk    - P(k) of the undisplaced load in the
#                                        bins of Fourier mesh {nmesh}
#
#   field_{method}_{nmesh}_z{z}_*      (only with --debug-field)
#
# Scalar metadata live under meta_* keys. {nmesh} is the Fourier mesh.


def run_validation(
    nmesh_list: list[int],
    lpt_orders: list[int],
    lbox_scalar: float,
    redshifts: list[float],
    methods: list[str],
    seed: int,
    output: str,
    *,
    glass: str,
    tiles: int,
    tiles_per_chunk: int = 4,
    nreal: int = 1,
    paired: bool = False,
    kmax_fraction_nyquist: float = 1.0,
    debug_field: bool = False,
) -> None:
    r'''Run LPT P(k) validation across all requested parameter combinations.

    The outer loop order is:  redshift -> nmesh -> method -> lpt_order.
    Cosmology (CAMB + growth factors) is computed once; LPT + P(k) once per
    (method, lpt_order) combination. The load is displaced and deposited
    ``tiles_per_chunk`` tiles at a time.

    Parameters
    ----------
    nmesh_list : list of int
        Fourier mesh resolutions to validate.
    lpt_orders : list of int
        LPT orders to run (1 and/or 2).
    lbox_scalar : float
        Cubic box side length [Mpc/h].
    redshifts : list of float
        Target redshift(s) for the ICs.
    methods : list of str
        Interpolation scheme(s) of the LPT step: ``'ngp'``, ``'cic'``, ``'tsc'``.
    seed : int
        Base RNG seed.
    output : str
        Path for the output ``.npz`` file.
    glass : str
        Periodic cubic glass snapshot (Gadget HDF5).
    tiles : int
        Number of copies of the glass along each side of the box.
    tiles_per_chunk : int
        Number of glass copies held in memory at once.
    nreal : int
        Number of independent white-noise realisations to average.
    paired : bool
        If ``True``, expand each realisation into an Angulo & Pontzen
        (2016) paired-fixed pair.
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

    tile_side = lbox_scalar / tiles
    tile = load_glass_tile(glass, tile_side)
    n_load = round(len(tile) ** (1.0 / 3.0)) * tiles
    if n_load < max(nmesh_list):
        raise ValueError(
            f'The load has {n_load} particles per side, fewer than the '
            f'finest Fourier mesh ({max(nmesh_list)}).'
        )
    # The spectrum is measured on a mesh as fine as the load.
    nvox_meas, dk_meas = cubic_voxels(n_load, boxsize)

    def tiled_load():
        return iter_tiled_load(tile, tile_side, tiles, tiles_per_chunk)

    # The field takes P(k) at every mesh mode, out to the corner at sqrt(3) k_Ny
    # of the finest mesh.
    k_ny_max = max(np.pi / cubic_voxels(nmesh, boxsize)[1] for nmesh in nmesh_list)
    kmax_camb = 1.05 * np.sqrt(3.0) * k_ny_max

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
    results['meta_glass'] = np.array(str(glass))
    results['meta_tiles'] = np.int64(tiles)
    results['meta_load_per_side'] = np.int64(n_load)
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
            k_ny = np.pi / dk
            kmax = kmax_fraction_nyquist * k_ny

            log.info('%s', '=' * 72)
            log.info(
                'z = %g | Mesh %d^3 (%dx%dx%d) | dk = %.6f Mpc/h | '
                'load = %d^3 | k_Ny = %.4f h/Mpc | k_max = %.4f h/Mpc',
                z, nmesh, nvox[0], nvox[1], nvox[2], dk, n_load, k_ny, kmax,
            )
            log.info('%s', '=' * 72)

            if f'load_{nmesh}_k' not in results:
                meter = InterlacedPowerMeter(nvox_meas, boxsize)
                for chunk in tiled_load():
                    meter.add(chunk)
                k_load, pk_load, _ = meter.power(kmax=kmax)
                results[f'load_{nmesh}_k'] = k_load
                results[f'load_{nmesh}_pk'] = pk_load

            for method in methods:
                # Band-averaged reference
                ref_prefix = f'ref_{method}_{nmesh}_{zk}'
                if f'{ref_prefix}_k' not in results:
                    k_ref, pk_ref, nmodes_ref = band_average_reference_pk(
                        growth.kh_camb, growth.pk_input,
                        nvox_meas, dk_meas, boxsize,
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
                            delta_k = generate_field(
                                nvox, dk, growth, seed_i, paired=paired,
                                counter_phase=counter_phase,
                            )
                            meter = InterlacedPowerMeter(nvox_meas, boxsize)
                            for chunk in tiled_load():
                                xpert, _ = run_lpt(
                                    chunk, delta_k, nvox, dk, growth,
                                    lpt_order, method, compensate=True,
                                )
                                meter.add(xpert)
                            del delta_k
                            k_p, pk_p, _ = meter.power(kmax=kmax)
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
            'Measure particle power at several resolutions, redshifts, LPT orders, and mass-assignment methods, then save a NumPy archive.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--nmesh', type=int, nargs='+', default=[128],
        help='Fourier mesh resolution(s) to test.',
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
        help='Interpolation scheme(s) of the LPT step.',
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
        '--glass', type=str, required=True,
        help='Periodic cubic glass snapshot (Gadget HDF5).',
    )
    parser.add_argument(
        '--tiles', type=int, default=1,
        help='Copies of the glass along each side of the box.',
    )
    parser.add_argument(
        '--tiles-per-chunk', type=int, default=4, dest='tiles_per_chunk',
        help='Glass copies held in memory at once.',
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
        glass=args.glass,
        tiles=args.tiles,
        tiles_per_chunk=args.tiles_per_chunk,
        kmax_fraction_nyquist=args.kmax_fraction_nyquist,
        debug_field=args.debug_field,
    )
