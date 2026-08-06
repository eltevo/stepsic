#!/usr/bin/env python3
'''
Compute displacement and velocity field statistics for stepsic
validation.

Generates LPT-displaced particle positions and velocities for a single
(Lx, Ly, Lz, nmesh, z, lpt_order, method) configuration.

Serialises pre-binned histograms to a ``.npz`` archive that the
companion ``validate_fields_plot.py`` script can consume.

Usage
-----
::

    python validate_fields_run.py \\
        --Lbox 500 500 500 --nmesh 128 --lpt 2 --z 31 \\
        --method cic --seed 137 --nbins 120 \\
        -o fields_data.npz
'''

from __future__ import annotations

from pathlib import Path


import argparse
import logging
import shutil
import tempfile
import time

import numpy as np

from validation._common.evaluation import atomic_savez

from stepsic.field import create_grid, cubic_voxels

from validation import (
    ArrayF,
    GrowthData,
    generate_field,
    histogram,
    init_cosmology,
    parse_boxsize,
    run_lpt,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def run_fields(
    lbox: list[float],
    nmesh: int,
    lpt_order: int,
    redshift: float,
    method: str,
    seed: int,
    output: str,
    *,
    nbins: int = 120,
    nreal: int = 1,
    paired: bool = False,
) -> None:
    r'''
    Generate LPT displacement/velocity fields and serialise histograms.

    When ``nreal > 1`` or ``paired=True``, multiple realisations are
    generated and their histograms are accumulated into common bins
    (determined by the first evaluation) then averaged. Each
    realisation draws independent phases from the same :math:`P(k)`,
    so the histogram shape is stable and averaging suppresses
    bin-to-bin cosmic variance.

    Parameters
    ----------
    lbox : list of float
        Box dimensions ``[Lx, Ly, Lz]`` in [Mpc/h].
    nmesh : int
        Number of grid cells along the shortest box dimension.
    lpt_order : {1, 2}
        LPT order.
    redshift : float
        Target redshift for the ICs.
    method : str
        Interpolation method for LPT: ``'ngp'``, ``'cic'``, ``'tsc'``.
    seed : int
        Base white-noise RNG seed.
    output : str
        Path for the output ``.npz`` file.
    nbins : int
        Number of histogram bins.
    nreal : int
        Number of independent realisations to average.
        Seeds: ``[seed, seed+1, ..., seed+nreal-1]``.
    paired : bool
        If ``True``, enable Angulo & Pontzen (2016) paired-fixed mode.
        Each realisation produces two evaluations (primary +
        counter-phased).
    '''
    if nreal < 1:
        raise ValueError(f'nreal must be >= 1, got {nreal}.')
    boxsize = parse_boxsize(lbox)
    log.info('Box: %.1f x %.1f x %.1f Mpc/h', *boxsize)

    # -- Grid setup --
    nvox, dk = cubic_voxels(nmesh, boxsize)
    n_part = int(np.prod(nvox))
    k_ny = np.pi / dk

    log.info(
        'Mesh: %dx%dx%d = %d particles | dk = %.6f Mpc/h | '
        'k_Ny = %.4f h/Mpc',
        nvox[0], nvox[1], nvox[2], n_part, dk, k_ny,
    )

    pos_grid, _ = create_grid(nvox, dk)

    # -- Cosmology --
    lbox_max = float(np.max(boxsize))
    kmax_camb = 1.05 * k_ny
    growth_table = init_cosmology(
        [redshift], lbox_max, kmax=kmax_camb,
    )
    growth = growth_table[redshift]

    # -- Realisation loop setup --
    paired_states = (False, True) if paired else (False,)
    seeds = [seed + i for i in range(nreal)]
    members_per_real = len(paired_states)
    n_total = nreal * members_per_real

    log.info(
        'Workload: %d realisation(s) * %d member(s) = %d evaluations',
        nreal, members_per_real, n_total,
    )

    # -- Temporary sample storage and global-range bookkeeping --
    # We must not let the first realisation define the histogram binning,
    # because that can clip later realisations and produce inconsistent
    # tails in the averaged histograms. Instead, we serialise every
    # evaluation, collect the global ranges, then build one common binning
    # and histogram everything in a second pass.
    hist_names = ['disp_x', 'disp_y', 'disp_z', 'disp_tot', 'vel_tot']
    hist_groups = {
        'disp_components': ['disp_x', 'disp_y', 'disp_z'],
        'disp_tot': ['disp_tot'],
        'vel_tot': ['vel_tot'],
    }
    group_min: dict[str, float] = {name: np.inf for name in hist_groups}
    group_max: dict[str, float] = {name: -np.inf for name in hist_groups}

    stat_disp_mean_sum = 0.0
    stat_disp_max = 0.0
    stat_vel_mean_sum = 0.0
    stat_vel_max = 0.0

    t0 = time.time()
    eval_idx = 0
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(
        tempfile.mkdtemp(prefix='validate_fields_', dir=output_path.parent)
    )
    sample_paths: list[Path] = []

    for i_real, seed_i in enumerate(seeds, start=1):
        for counter_phase in paired_states:
            eval_idx += 1
            member_tag = 'counter' if counter_phase else 'primary'
            log.info(
                '  [%d/%d] %dLPT %s: seed=%d, member=%s',
                eval_idx, n_total,
                lpt_order, method.upper(), seed_i, member_tag,
            )

            delta_k = generate_field(
                nvox, dk, growth, seed_i,
                paired=paired, counter_phase=counter_phase,
            )

            log.info(
                '    Running %dLPT (%s, compensate=%s)...',
                lpt_order, method, method == 'tsc',
            )
            t_lpt = time.time()
            xpert, vpert = run_lpt(
                pos_grid, delta_k, nvox, dk, growth,
                lpt_order, method,
            )
            log.info('    LPT done in %.1f s', time.time() - t_lpt)

            disp_kpc = (xpert - pos_grid) * 1e3  # [kpc/h]
            samples = {
                'disp_x': np.abs(disp_kpc[:, 0]),
                'disp_y': np.abs(disp_kpc[:, 1]),
                'disp_z': np.abs(disp_kpc[:, 2]),
                'disp_tot': np.linalg.norm(disp_kpc, axis=1),
                'vel_tot': np.linalg.norm(vpert, axis=1),
            }
            dx = disp_kpc[:, 0]
            dy = disp_kpc[:, 1]
            log.info('mean(|dx|)/mean(|dy|) = %.8f', np.mean(np.abs(dx)) / np.mean(np.abs(dy)))
            log.info('std(dx)/std(dy)       = %.8f', np.std(dx) / np.std(dy))

            t_store = time.time()
            sample_path = temp_dir / f'eval_{eval_idx:04d}.npz'
            np.savez(sample_path, **samples)
            sample_paths.append(sample_path)
            log.info('    Samples written in %.1f s', time.time() - t_store)

            for group_name, member_names in hist_groups.items():
                member_min = min(float(np.min(samples[name])) for name in member_names)
                member_max = max(float(np.max(samples[name])) for name in member_names)
                group_min[group_name] = min(group_min[group_name], member_min)
                group_max[group_name] = max(group_max[group_name], member_max)

            stat_disp_mean_sum += float(np.mean(samples['disp_tot']))
            stat_disp_max = max(stat_disp_max, float(np.max(samples['disp_tot'])))
            stat_vel_mean_sum += float(np.mean(samples['vel_tot']))
            stat_vel_max = max(stat_vel_max, float(np.max(samples['vel_tot'])))

            log.info(
                '    |Psi|: mean=%.2f, max=%.2f kpc/h',
                np.mean(samples['disp_tot']), np.max(samples['disp_tot']),
            )
            log.info(
                '    |v|: mean=%.2f, max=%.2f km/s',
                np.mean(samples['vel_tot']), np.max(samples['vel_tot']),
            )

    group_edges: dict[str, ArrayF] = {}
    edges: dict[str, ArrayF] = {}
    accum: dict[str, ArrayF] = {}
    for group_name, member_names in hist_groups.items():
        lo = group_min[group_name]
        hi = group_max[group_name]
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(
                f'Invalid histogram range for {group_name}: [{lo}, {hi}].'
            )
        if hi <= lo:
            hi = np.nextafter(lo, np.inf)
        group_edges[group_name] = np.linspace(lo, hi, nbins + 1, dtype=np.float64)
        for name in member_names:
            edges[name] = group_edges[group_name]
            accum[name] = np.zeros(nbins, dtype=np.float64)

    for sample_path in sample_paths:
        with np.load(sample_path) as sample_data:
            for name in hist_names:
                counts, _ = np.histogram(sample_data[name], bins=edges[name])
                accum[name] += counts.astype(np.float64)
        sample_path.unlink()

    for name in hist_names:
        accum[name] /= n_total

    # for name in hist_names:
    #     bin_widths = np.diff(edges[name])
    #     accum[name] /= (n_part * bin_widths)

    stat_disp_counts = accum['disp_tot']
    stat_vel_counts = accum['vel_tot']
    stat_disp_centres = 0.5 * (edges['disp_tot'][:-1] + edges['disp_tot'][1:])
    stat_vel_centres = 0.5 * (edges['vel_tot'][:-1] + edges['vel_tot'][1:])

    disp_cdf = np.cumsum(stat_disp_counts)
    vel_cdf = np.cumsum(stat_vel_counts)
    disp_half = 0.5 * disp_cdf[-1]
    vel_half = 0.5 * vel_cdf[-1]
    stat_disp_median = float(stat_disp_centres[np.searchsorted(disp_cdf, disp_half)])
    stat_vel_median = float(stat_vel_centres[np.searchsorted(vel_cdf, vel_half)])

    temp_dir.rmdir()

    # -- Serialise --
    results: dict[str, np.ndarray] = {}

    # Metadata
    results['meta_lbox'] = boxsize
    results['meta_nmesh'] = np.int64(nmesh)
    results['meta_nvox'] = nvox.astype(np.int64)
    results['meta_dk'] = np.float64(dk)
    results['meta_k_ny'] = np.float64(k_ny)
    results['meta_lpt_order'] = np.int64(lpt_order)
    results['meta_redshift'] = np.float64(redshift)
    results['meta_method'] = np.array(method)
    results['meta_seed'] = np.int64(seed)
    results['meta_npart'] = np.int64(n_part)
    results['meta_nbins'] = np.int64(nbins)
    results['meta_d1'] = np.float64(growth.d1)
    results['meta_compensate'] = np.bool_(method == 'tsc')
    results['meta_nreal'] = np.int64(nreal)
    results['meta_paired'] = np.bool_(paired)
    results['meta_n_total'] = np.int64(n_total)

    # Histograms - centres and widths from fixed edges, averaged counts
    for name in hist_names:
        e = edges[name]
        results[f'hist_{name}_centres'] = 0.5 * (e[:-1] + e[1:])
        results[f'hist_{name}_counts'] = accum[name]
        results[f'hist_{name}_widths'] = np.diff(e)

    # Summary scalars (ensemble-averaged where applicable)
    results['stat_disp_mean'] = np.float64(stat_disp_mean_sum / n_total)
    results['stat_disp_median'] = np.float64(stat_disp_median)
    results['stat_disp_max'] = np.float64(stat_disp_max)
    results['stat_vel_mean'] = np.float64(stat_vel_mean_sum / n_total)
    results['stat_vel_median'] = np.float64(stat_vel_median)
    results['stat_vel_max'] = np.float64(stat_vel_max)

    atomic_savez(output, **results)
    log.info(
        'Results written to %s (%d arrays, %d evaluations, %.1f s total)',
        output, len(results), n_total, time.time() - t0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Generate LPT displacement/velocity fields and serialise '
            'histogram data to an .npz archive for plotting.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--Lbox', type=float, nargs=3, default=[500.0, 500.0, 500.0],
        help='Box dimensions [Lx, Ly, Lz] in Mpc/h.',
    )
    parser.add_argument(
        '--nmesh', type=int, default=128,
        help='Grid resolution: cells along the shortest box dimension.',
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
        help='Interpolation method for LPT.',
    )
    parser.add_argument(
        '--seed', type=int, default=137,
        help='White-noise RNG seed.',
    )
    parser.add_argument(
        '--nbins', type=int, default=120,
        help='Number of histogram bins.',
    )
    parser.add_argument(
        '--nreal', type=int, default=1,
        help='Number of independent realisations to average. '
             'Seeds: [seed, seed+1, ..., seed+nreal-1].',
    )
    parser.add_argument(
        '--paired', action='store_true',
        help='Enable Angulo & Pontzen (2016) paired-fixed mode.',
    )
    parser.add_argument(
        '-o', '--output', type=str, default='fields_data.npz',
        help='Output .npz file path.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_fields(
        lbox=args.Lbox,
        nmesh=args.nmesh,
        lpt_order=args.lpt,
        redshift=args.redshift,
        method=args.method,
        seed=args.seed,
        output=args.output,
        nbins=args.nbins,
        nreal=args.nreal,
        paired=args.paired,
    )