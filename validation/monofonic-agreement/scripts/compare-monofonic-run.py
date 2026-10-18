#!/usr/bin/env python3
'''
Compare initial conditions produced by stepsic and monofonIC.

The script measures both power spectra with the same interlaced CIC deposition and compares matched particles' displacements and velocities.

It writes the measurements to a NumPy archive for the plotting script.

Usage
-----
::

    python compare-monofonic-run.py \\
        --stepsic-ic   stepsic_output/ic.hdf5 \\
        --monofonic-ic monofonic_output/monofonic_ics.hdf5 \\
        --lbox 1000 --nmesh 512 --redshift 31 \\
        --method cic --lpt-order 2 \\
        -o output/monofonic_comparison.npz
'''

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from validation._common.evaluation import atomic_savez

from stepsic.field import cubic_voxels
from stepsic.pk import measure_pk

from particles import (
    ArrayF,
    lagrangian_lattice_from_id,
    read_gadget_hdf5,
    sort_by_id,
    wrap_periodic,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Per-particle residuals
# ---------------------------------------------------------------------------

def _per_particle_residuals(
    field_a: ArrayF,
    field_b: ArrayF,
) -> tuple[ArrayF, ArrayF]:
    '''
    Compute absolute and relative per-particle residuals.

    Parameters
    ----------
    field_a, field_b : ndarray of shape (N, 3)
        Per-particle vector fields (e.g. displacements or velocities)
        from the two codes.

    Returns
    -------
    abs_residual : ndarray of shape (N,)
        |field_a - field_b| per particle.
    rel_residual : ndarray of shape (N,)
        |field_a - field_b| / |field_b| per particle.
        Particles with |field_b| < eps are masked out (set to NaN).
    '''
    diff = field_a - field_b
    abs_residual = np.linalg.norm(diff, axis=1)
    mag_b = np.linalg.norm(field_b, axis=1)
    rel_residual = np.full_like(abs_residual, np.nan)
    mask = mag_b > 1e-15
    rel_residual[mask] = abs_residual[mask] / mag_b[mask]
    return abs_residual, rel_residual


# ---------------------------------------------------------------------------
#  Histogramming
# ---------------------------------------------------------------------------

def _histogram(
    values: ArrayF,
    nbins: int = 100,
    *,
    drop_nan: bool = True,
) -> tuple[ArrayF, ArrayF, ArrayF]:
    '''Compute histogram with fixed bin edges, return centres, counts, widths.'''
    if drop_nan:
        values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros(nbins), np.zeros(nbins), np.ones(nbins)
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi <= lo:
        hi = np.nextafter(lo, np.inf)
    edges = np.linspace(lo, hi, nbins + 1)
    counts, _ = np.histogram(values, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    return centres, counts.astype(np.float64), widths


# ---------------------------------------------------------------------------
#  Main comparison
# ---------------------------------------------------------------------------

def run_comparison(
    stepsic_ic: str,
    monofonic_ic: str,
    lbox: float,
    nmesh: int,
    redshift: float,
    method: str,
    lpt_order: int,
    output: str,
    *,
    stepsic_offset: float = 0.5,
    monofonic_offset: float = 0.0,
    nbins_hist: int = 200,
    kmax_frac: float = 0.8,
) -> None:
    '''
    Full comparison pipeline: P(k) + displacement + velocity residuals.

    Parameters
    ----------
    stepsic_ic : str
        Path to the stepsic IC (Gadget HDF5).
    monofonic_ic : str
        Path to the monofonIC IC (Gadget HDF5).
    lbox : float
        Cubic box side length [Mpc/h].
    nmesh : int
        Grid resolution (particles = nmesh^3).
    redshift : float
        IC redshift.
    method : str
        MAS method for P(k) measurement.
    lpt_order : int
        LPT order (1 or 2, metadata only).
    output : str
        Output .npz archive path.
    nbins_hist : int
        Number of bins for residual histograms.
    kmax_frac : float
        Maximum k as fraction of Nyquist for P(k) comparison.
    '''
    t0 = time.time()
    boxsize = np.array([lbox, lbox, lbox], dtype=np.float64)
    results: dict[str, np.ndarray] = {}

    # -- Read both IC files --
    log.info('Reading stepsic IC: %s', stepsic_ic)
    pos_s, vel_s, ids_s = read_gadget_hdf5(stepsic_ic)
    log.info('  N = %d particles', len(ids_s))

    log.info('Reading monofonIC IC: %s', monofonic_ic)
    pos_m, vel_m, ids_m = read_gadget_hdf5(monofonic_ic)
    log.info('  N = %d particles', len(ids_m))

    if len(ids_s) != len(ids_m):
        raise ValueError(
            f'Particle count mismatch: stepsic has {len(ids_s)}, '
            f'monofonIC has {len(ids_m)}.'
        )

    # -- Sort by ID for consistent matching --
    # Both codes should assign IDs 0..N-1 in grid order, but the
    # grid traversal order (C vs Fortran, x-fast vs z-fast) may differ.
    log.info('Sorting by particle ID...')
    pos_s, vel_s, ids_s = sort_by_id(pos_s, vel_s, ids_s)
    pos_m, vel_m, ids_m = sort_by_id(pos_m, vel_m, ids_m)

    if not np.array_equal(ids_s, ids_m):
        raise ValueError('Particle identities must match exactly for field comparison')
    if not np.array_equal(ids_s, np.arange(nmesh**3)):
        raise ValueError('Particle identities must be the complete zero-based lattice')

    n_part = len(ids_s)

    # -- Reconstruct each code's Lagrangian lattice from particle IDs --
    # Both codes use id = (ix * N + iy) * N + iz, but place particles at
    # different points within the cell: stepsic uses cell centres
    # (offset 0.5), monofonIC's "sc" load uses cell corners (offset 0.0).
    # Using a single shared grid for both would alias the half-cell
    # convention difference into a (sqrt(3)/2) * dx offset that swamps
    # the actual LPT residual.
    nvox, dk = cubic_voxels(nmesh, np.array([lbox] * 3, dtype=np.float64))
    log.info(
        'Reconstructing Lagrangian lattices: %d^3, dx=%g Mpc/h '
        '(stepsic offset=%g, monofonIC offset=%g cells)',
        nmesh, dk, stepsic_offset, monofonic_offset,
    )
    lattice_s = lagrangian_lattice_from_id(
        ids_s, nmesh, dk, cell_offset=stepsic_offset,
    )
    lattice_m = lagrangian_lattice_from_id(
        ids_m, nmesh, dk, cell_offset=monofonic_offset,
    )

    # -- Unit convention check --
    pos_range_s = np.ptp(pos_s, axis=0)
    pos_range_m = np.ptp(pos_m, axis=0)
    log.info('  stepsic  position range: [%.2f, %.2f, %.2f]', *pos_range_s)
    log.info('  monofonIC position range: [%.2f, %.2f, %.2f]', *pos_range_m)

    for label, rng in [('stepsic', pos_range_s), ('monofonIC', pos_range_m)]:
        if np.any(rng < 0.5 * lbox) or np.any(rng > 1.5 * lbox):
            log.warning(
                '%s position range looks suspicious relative to Lbox=%g. '
                'Check unit conventions (Mpc vs Mpc/h).', label, lbox,
            )

    # -- Displacements --
    # Ψ = pos - lattice, wrapped to [-L/2, +L/2] for periodicity.
    log.info('Computing displacements...')
    disp_s = wrap_periodic(pos_s - lattice_s, lbox)
    disp_m = wrap_periodic(pos_m - lattice_m, lbox)
    del lattice_s, lattice_m

    log.info(
        '  median |Ψ| stepsic = %.4e, monofonIC = %.4e Mpc/h',
        np.median(np.linalg.norm(disp_s, axis=1)),
        np.median(np.linalg.norm(disp_m, axis=1)),
    )

    abs_res_disp, rel_res_disp = _per_particle_residuals(disp_s, disp_m)
    abs_res_disp_kpc = abs_res_disp * 1e3  # [kpc/h]

    log.info(
        'Displacement residuals: median(|Δ|) = %.4e kpc/h, '
        'max(|Δ|) = %.4e kpc/h',
        np.nanmedian(abs_res_disp_kpc), np.nanmax(abs_res_disp_kpc),
    )
    log.info(
        'Relative residuals: median = %.4e, 99th pctl = %.4e',
        np.nanmedian(rel_res_disp), np.nanpercentile(rel_res_disp, 99),
    )

    # -- Velocities --
    log.info('Computing velocity residuals...')
    abs_res_vel, rel_res_vel = _per_particle_residuals(vel_s, vel_m)

    log.info(
        'Velocity residuals: median(|Δv|) = %.4e km/s, '
        'max(|Δv|) = %.4e km/s',
        np.nanmedian(abs_res_vel), np.nanmax(abs_res_vel),
    )
    log.info(
        'Relative residuals: median = %.4e, 99th pctl = %.4e',
        np.nanmedian(rel_res_vel), np.nanpercentile(rel_res_vel, 99),
    )

    # -- Histograms --
    log.info('Binning residual histograms (%d bins)...', nbins_hist)

    h_disp_abs_c, h_disp_abs_n, h_disp_abs_w = _histogram(
        abs_res_disp_kpc, nbins_hist,
    )
    h_disp_rel_c, h_disp_rel_n, h_disp_rel_w = _histogram(
        rel_res_disp, nbins_hist,
    )
    h_vel_abs_c, h_vel_abs_n, h_vel_abs_w = _histogram(
        abs_res_vel, nbins_hist,
    )
    h_vel_rel_c, h_vel_rel_n, h_vel_rel_w = _histogram(
        rel_res_vel, nbins_hist,
    )

    # -- Displacement magnitude histograms for direct overlay --
    # Magnitudes in kpc/h for better readability at high z.
    mag_disp_s = np.linalg.norm(disp_s, axis=1) * 1e3
    mag_disp_m = np.linalg.norm(disp_m, axis=1) * 1e3
    lo = min(mag_disp_s.min(), mag_disp_m.min())
    hi = max(mag_disp_s.max(), mag_disp_m.max())
    disp_edges = np.linspace(lo, hi if hi > lo else lo + 1, nbins_hist + 1)
    h_disp_s, _ = np.histogram(mag_disp_s, bins=disp_edges)
    h_disp_m, _ = np.histogram(mag_disp_m, bins=disp_edges)
    disp_centres = 0.5 * (disp_edges[:-1] + disp_edges[1:])

    mag_vel_s = np.linalg.norm(vel_s, axis=1)
    mag_vel_m = np.linalg.norm(vel_m, axis=1)
    lo_v = min(mag_vel_s.min(), mag_vel_m.min())
    hi_v = max(mag_vel_s.max(), mag_vel_m.max())
    vel_edges = np.linspace(lo_v, hi_v if hi_v > lo_v else lo_v + 1, nbins_hist + 1)
    h_vel_s, _ = np.histogram(mag_vel_s, bins=vel_edges)
    h_vel_m, _ = np.histogram(mag_vel_m, bins=vel_edges)
    vel_centres = 0.5 * (vel_edges[:-1] + vel_edges[1:])

    # -- Serialize --
    results['meta_lbox'] = np.float64(lbox)
    results['meta_nmesh'] = np.int64(nmesh)
    results['meta_redshift'] = np.float64(redshift)
    results['meta_method'] = np.array(method)
    results['meta_lpt_order'] = np.int64(lpt_order)
    results['meta_n_part'] = np.int64(n_part)
    results['meta_kmax_frac'] = np.float64(kmax_frac)

    # Displacement residuals
    results['hist_disp_abs_centres'] = h_disp_abs_c
    results['hist_disp_abs_counts'] = h_disp_abs_n
    results['hist_disp_abs_widths'] = h_disp_abs_w
    results['hist_disp_rel_centres'] = h_disp_rel_c
    results['hist_disp_rel_counts'] = h_disp_rel_n
    results['hist_disp_rel_widths'] = h_disp_rel_w
    results['stat_disp_abs_median'] = np.float64(np.nanmedian(abs_res_disp_kpc))
    results['stat_disp_abs_max'] = np.float64(np.nanmax(abs_res_disp_kpc))
    results['stat_disp_rel_median'] = np.float64(np.nanmedian(rel_res_disp))
    results['stat_disp_rel_p99'] = np.float64(np.nanpercentile(rel_res_disp, 99))

    # Displacement magnitude overlay histograms
    results['hist_disp_mag_centres'] = disp_centres
    results['hist_disp_mag_stepsic'] = h_disp_s.astype(np.float64)
    results['hist_disp_mag_monofonic'] = h_disp_m.astype(np.float64)

    # Velocity residuals
    results['hist_vel_abs_centres'] = h_vel_abs_c
    results['hist_vel_abs_counts'] = h_vel_abs_n
    results['hist_vel_abs_widths'] = h_vel_abs_w
    results['hist_vel_rel_centres'] = h_vel_rel_c
    results['hist_vel_rel_counts'] = h_vel_rel_n
    results['hist_vel_rel_widths'] = h_vel_rel_w
    results['stat_vel_abs_median'] = np.float64(np.nanmedian(abs_res_vel))
    results['stat_vel_abs_max'] = np.float64(np.nanmax(abs_res_vel))
    results['stat_vel_rel_median'] = np.float64(np.nanmedian(rel_res_vel))
    results['stat_vel_rel_p99'] = np.float64(np.nanpercentile(rel_res_vel, 99))

    # Velocity magnitude overlay histograms
    results['hist_vel_mag_centres'] = vel_centres
    results['hist_vel_mag_stepsic'] = h_vel_s.astype(np.float64)
    results['hist_vel_mag_monofonic'] = h_vel_m.astype(np.float64)

    # P(k) is measured last, when only the particle positions remain in memory.
    del ids_s, ids_m, disp_s, disp_m, vel_s, vel_m
    del mag_disp_s, mag_disp_m, mag_vel_s, mag_vel_m
    del abs_res_disp, abs_res_disp_kpc, rel_res_disp, abs_res_vel, rel_res_vel

    # -- P(k) --
    log.info('Measuring P(k) from stepsic IC...')
    nvox, dk = cubic_voxels(nmesh, boxsize)
    k_ny = np.pi / dk
    kmax = kmax_frac * k_ny

    k_s, pk_s, nm_s = measure_pk(
        pos_s, boxsize, nmesh,
        method='cic', deconvolve=True, interlace=True,
        subtract_shot=False, kmin=None, kmax=kmax, dk_bin=None,
    )

    log.info('Measuring P(k) from monofonIC IC...')
    k_m, pk_m, nm_m = measure_pk(
        pos_m, boxsize, nmesh,
        method='cic', deconvolve=True, interlace=True,
        subtract_shot=False, kmin=None, kmax=kmax, dk_bin=None,
    )

    # Verify k-binning is identical.
    if not np.allclose(k_s, k_m):
        raise RuntimeError('k-bin centres differ between the two P(k) measurements.')

    pk_ratio = pk_s / pk_m
    log.info(
        'P(k) ratio: median = %.6f, min = %.6f, max = %.6f',
        np.median(pk_ratio), np.min(pk_ratio), np.max(pk_ratio),
    )

    results['meta_k_ny'] = np.float64(k_ny)
    results['pk_k'] = k_s
    results['pk_stepsic'] = pk_s
    results['pk_monofonic'] = pk_m
    results['pk_ratio'] = pk_ratio
    results['pk_nmodes'] = nm_s

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    atomic_savez(output, **results)
    dt = time.time() - t0
    log.info('Comparison written to %s (%d arrays, %.1f s)', output, len(results), dt)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Compare stepsic and monofonIC ICs: P(k), Ψ, v.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--stepsic-ic', type=str, required=True,
                   help='Path to stepsic IC (HDF5).')
    p.add_argument('--monofonic-ic', type=str, required=True,
                   help='Path to monofonIC IC (HDF5).')
    p.add_argument('--lbox', type=float, required=True,
                   help='Cubic box side length [Mpc/h].')
    p.add_argument('--nmesh', type=int, required=True,
                   help='Grid resolution per dimension.')
    p.add_argument('--redshift', type=float, required=True,
                   help='IC redshift (metadata).')
    p.add_argument('--method', type=str, default='cic',
                   choices=['ngp', 'cic', 'tsc'],
                   help='MAS method for P(k) measurement.')
    p.add_argument('--lpt-order', type=int, default=2,
                   choices=[1, 2],
                   help='LPT order (metadata only).')
    p.add_argument('--stepsic-offset', type=float, default=0.5,
                   help='Stepsic lattice offset (cells). 0.5 = cell-centred.')
    p.add_argument('--monofonic-offset', type=float, default=0.0,
                   help='monofonIC lattice offset (cells). 0.0 = "sc" corner load.')
    p.add_argument('--nbins', type=int, default=60,
                   help='Number of histogram bins.')
    p.add_argument('--kmax-frac', type=float, default=0.8,
                   help='Max k as fraction of Nyquist.')
    p.add_argument('-o', '--output', type=str, required=True,
                   help='Output .npz archive path.')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_comparison(
        stepsic_ic=args.stepsic_ic,
        monofonic_ic=args.monofonic_ic,
        lbox=args.lbox,
        nmesh=args.nmesh,
        redshift=args.redshift,
        method=args.method,
        lpt_order=args.lpt_order,
        output=args.output,
        stepsic_offset=args.stepsic_offset,
        monofonic_offset=args.monofonic_offset,
        nbins_hist=args.nbins,
        kmax_frac=args.kmax_frac,
    )
