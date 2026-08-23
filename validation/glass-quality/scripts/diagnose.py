#!/usr/bin/env python3
'''
Measure the power, spacing, density, and residual forces of a glass.

For spherical and cylindrical glasses, particles are grouped into radial zones with equal particle counts. Cubical glasses use the full periodic box.

The archive contains mass-weighted power relative to shot noise, nearest-neighbour distance relative to local spacing, radial density, and mass-level boundaries. If the snapshot contains accelerations, it also contains ``q = |F| d^2 / m`` in internal units with ``G = 1``.

For a tiled glass, ``--tile-lz`` restricts the measurement to the first repeated z section.

Run the same measurement on the Poisson twin to compare the relaxed glass with random particle positions.

Usage
-----
::

    python diagnose.py glass.hdf5 -o diag.npz --geometry cylindrical \
        --r3d 500 --lz 1000
'''

from __future__ import annotations



import argparse
import logging

import h5py
import numpy as np
from scipy.spatial import cKDTree

from stepsic.pk import measure_pk
from stepsic.field import wrap
from validation._common.evaluation import atomic_savez

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _pad_rows(rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return finite numeric rows padded with NaN and their true lengths."""
    lengths = np.asarray([len(row) for row in rows], dtype=np.int64)
    width = int(lengths.max(initial=0))
    padded = np.full((len(rows), width), np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        padded[index, : lengths[index]] = np.asarray(row, dtype=np.float64)
    return padded, lengths


def load_snapshot(path: str) -> tuple[np.ndarray, np.ndarray,
                                      np.ndarray | None, float | None]:
    '''Return (pos, mass, accel_or_None, boxsize_or_None).'''
    with h5py.File(path, 'r') as f:
        g = f['PartType1']
        pos = g['Coordinates'][:].astype(np.float64)
        mass = g['Masses'][:].astype(np.float64)
        accel = None
        if 'Accelerations' in g:
            accel = g['Accelerations'][:].astype(np.float64)
        boxsize = None
        if 'Header' in f and 'BoxSize' in f['Header'].attrs:
            boxsize = float(f['Header'].attrs['BoxSize'])
    return pos, mass, accel, boxsize


def _radius(pos: np.ndarray, geometry: str) -> np.ndarray:
    if geometry == 'spherical':
        return np.linalg.norm(pos, axis=1)
    return np.hypot(pos[:, 0], pos[:, 1])


def zone_pk_cubes(
    pos: np.ndarray,
    mass: np.ndarray,
    r: np.ndarray,
    r0: float,
    r1: float,
    z_range: tuple[float, float] | None,
    geometry: str,
    ncubes: int,
    pk_nmesh: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    '''Average mass-weighted P(k) over cubes inscribed in a radial zone.

    Returns (k, pk_mean, shot_mean, cube_side). The cubes are treated
    as periodic for the FFT; that is exact only for the cubical case,
    but the sub-Poisson suppression signature lives at wavelengths well
    below the cube side where the window leakage is negligible.
    '''
    safety = 0.95
    if r0 <= 0.0:
        # Innermost zone: one cube inscribed in the r1-sphere/cylinder.
        dim_factor = np.sqrt(3.0) if geometry == 'spherical' else np.sqrt(2.0)
        side = safety * 2.0 * r1 / dim_factor
        centres = [np.zeros(3)]
    else:
        r_mid = 0.5 * (r0 + r1)
        dim_factor = np.sqrt(3.0) if geometry == 'spherical' else np.sqrt(2.0)
        side = safety * (r1 - r0) / dim_factor
        centres = []
        for _ in range(ncubes):
            if geometry == 'spherical':
                v = rng.normal(size=3)
                v /= np.linalg.norm(v)
                centres.append(r_mid * v)
            else:
                phi = rng.uniform(0.0, 2.0 * np.pi)
                centres.append(np.array([r_mid * np.cos(phi),
                                         r_mid * np.sin(phi), 0.0]))
    if z_range is not None:
        z_lo, z_hi = z_range
        side = min(side, safety * (z_hi - z_lo))
        z_mid = 0.5 * (z_lo + z_hi)
        for c in centres:
            c[2] = z_mid

    boxsize = np.array([side, side, side])
    k_ref = None
    pk_acc = None
    shot_acc = 0.0
    n_used = 0
    for c in centres:
        lo = np.asarray(c) - side / 2.0
        sel = np.all((pos >= lo) & (pos < lo + side), axis=1)
        n_sel = int(sel.sum())
        if n_sel < 32:
            continue
        m_sel = mass[sel]
        k, pk, _ = measure_pk(
            pos[sel] - lo, boxsize, nmesh=pk_nmesh, mass=m_sel,
            method='cic', deconvolve=True, interlace=True,
            subtract_shot=False,
        )
        shot = side**3 * float(np.sum(m_sel**2)) / float(np.sum(m_sel))**2
        if pk_acc is None:
            k_ref, pk_acc = k, np.zeros_like(pk)
        pk_acc += pk
        shot_acc += shot
        n_used += 1
    if n_used == 0:
        nan = np.full(1, np.nan)
        return nan, nan, np.nan, side
    return k_ref, pk_acc / n_used, shot_acc / n_used, side


def diagnose(
    pos: np.ndarray,
    mass: np.ndarray,
    accel: np.ndarray | None,
    geometry: str,
    boxsize: float | None,
    nzones: int,
    ncubes: int,
    pk_nmesh: int,
    nprof: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(pos)
    m_tot = float(mass.sum())

    # -- mean density and domain --------------------------------------------
    if geometry == 'cubical':
        lo, hi = pos.min(axis=0), pos.max(axis=0)
        vol = float(np.prod(hi - lo)) if boxsize is None else boxsize**3
        r = None
        r_max = None
    else:
        r = _radius(pos, geometry)
        r_max = float(r.max())
        if geometry == 'spherical':
            vol = 4.0 / 3.0 * np.pi * r_max**3
        else:
            lz = float(pos[:, 2].max() - pos[:, 2].min())
            vol = np.pi * r_max**2 * lz
    rho_mean = m_tot / vol
    d_exp = np.cbrt(mass / rho_mean)

    results: dict[str, np.ndarray] = {
        'meta_geometry': np.array(geometry),
        'meta_n_part': np.int64(n),
        'meta_rho_mean': np.float64(rho_mean),
        'meta_volume': np.float64(vol),
    }

    # -- nearest-neighbour statistic ----------------------------------------
    if geometry == 'cubical' and boxsize is not None:
        pos_w = wrap(pos.copy(), np.array([boxsize] * 3))
        tree = cKDTree(pos_w, boxsize=boxsize)
        d_nn = tree.query(pos_w, k=2)[0][:, 1]
    else:
        tree = cKDTree(pos)
        d_nn = tree.query(pos, k=2)[0][:, 1]
    u = d_nn / d_exp

    levels = np.unique(mass)
    lvl_stats = np.zeros((len(levels), 6))
    for i, m in enumerate(levels):
        sel = mass == m
        r_sel = r[sel] if r is not None else np.zeros(int(sel.sum()))
        lvl_stats[i] = (
            m, int(sel.sum()),
            float(r_sel.min()), float(r_sel.max()),
            float(u[sel].mean()), float(u[sel].std()),
        )
    results['level_mass'] = lvl_stats[:, 0]
    results['level_count'] = lvl_stats[:, 1]
    results['level_rmin'] = lvl_stats[:, 2]
    results['level_rmax'] = lvl_stats[:, 3]
    results['level_u_mean'] = lvl_stats[:, 4]
    results['level_u_std'] = lvl_stats[:, 5]

    u_bins = np.linspace(0.0, 2.5, 101)
    results['u_hist_edges'] = u_bins
    results['u_hist'] = np.histogram(u, bins=u_bins)[0].astype(np.float64)
    results['u_mean'] = np.float64(u.mean())
    results['u_std'] = np.float64(u.std())

    # -- residual-force statistic -------------------------------------------
    has_force = accel is not None
    results['meta_has_force'] = np.bool_(has_force)
    if has_force:
        q = np.linalg.norm(accel, axis=1) * d_exp**2 / mass
        results['q_all'] = np.array([
            np.sqrt(np.mean(q**2)), np.median(q), np.percentile(q, 99),
        ])

    # -- radial structure (non-cubical only) --------------------------------
    if geometry != 'cubical':
        prof_edges = np.linspace(0.0, r_max, nprof + 1)
        idx = np.clip(
            np.searchsorted(prof_edges, r, side='right') - 1, 0, nprof - 1,
        )
        m_bin = np.bincount(idx, weights=mass, minlength=nprof)
        if geometry == 'spherical':
            vols = 4.0 / 3.0 * np.pi * np.diff(prof_edges**3)
        else:
            vols = np.pi * np.diff(prof_edges**2) * lz
        results['prof_edges'] = prof_edges
        results['prof_rho'] = m_bin / vols / rho_mean
        sum_u = np.bincount(idx, weights=u, minlength=nprof)
        cnt = np.bincount(idx, minlength=nprof).astype(np.float64)
        with np.errstate(invalid='ignore'):
            results['prof_u'] = np.where(cnt > 0, sum_u / cnt, np.nan)
        if has_force:
            sum_q = np.bincount(idx, weights=q, minlength=nprof)
            with np.errstate(invalid='ignore'):
                results['prof_q'] = np.where(cnt > 0, sum_q / cnt, np.nan)

        # zones by particle-count quantiles of r
        zone_edges = np.quantile(r, np.linspace(0.0, 1.0, nzones + 1))
        zone_edges[0], zone_edges[-1] = 0.0, r_max
        results['zone_edges'] = zone_edges
        z_range = None
        if geometry == 'cylindrical':
            z_range = (float(pos[:, 2].min()), float(pos[:, 2].max()))

        zone_k, zone_pk, zone_shot, zone_side, zone_dbar, zone_q = \
            [], [], [], [], [], []
        for iz in range(nzones):
            r0, r1 = zone_edges[iz], zone_edges[iz + 1]
            in_z = (r >= r0) & (r < r1) if iz < nzones - 1 else (r >= r0)
            k, pk, shot, side = zone_pk_cubes(
                pos, mass, r, r0, r1, z_range, geometry,
                ncubes, pk_nmesh, rng,
            )
            zone_k.append(k)
            zone_pk.append(pk)
            zone_shot.append(shot)
            zone_side.append(side)
            zone_dbar.append(float(np.mean(d_exp[in_z])))
            if has_force:
                zone_q.append([
                    np.sqrt(np.mean(q[in_z]**2)), np.median(q[in_z]),
                    np.percentile(q[in_z], 99),
                ])
        # k-vectors can differ per zone; NaN padding keeps the archive typed.
        results['zone_k'], results['zone_count'] = _pad_rows(zone_k)
        results['zone_pk'], pk_count = _pad_rows(zone_pk)
        if not np.array_equal(results['zone_count'], pk_count):
            raise RuntimeError("zone k and power rows have different lengths")
        results['zone_shot'] = np.array(zone_shot, dtype=np.float64)
        results['zone_side'] = np.array(zone_side, dtype=np.float64)
        results['zone_dbar'] = np.array(zone_dbar, dtype=np.float64)
        if has_force:
            results['zone_q'] = np.array(zone_q, dtype=np.float64)
    else:
        # Cubical: one periodic full-box measurement.
        b = boxsize if boxsize is not None else float((hi - lo).max())
        pos_w = wrap(pos - pos.min(axis=0), np.array([b] * 3))
        k, pk, _ = measure_pk(
            pos_w, np.array([b] * 3), nmesh=pk_nmesh, mass=mass,
            method='cic', deconvolve=True, interlace=True,
            subtract_shot=False,
        )
        shot = b**3 * float(np.sum(mass**2)) / m_tot**2
        results['zone_k'] = np.asarray([k], dtype=np.float64)
        results['zone_pk'] = np.asarray([pk], dtype=np.float64)
        results['zone_count'] = np.array([len(k)], dtype=np.int64)
        results['zone_shot'] = np.array([shot])
        results['zone_side'] = np.array([b])
        results['zone_dbar'] = np.array([float(np.mean(d_exp))])
        results['zone_edges'] = np.array([0.0, b])
        if has_force:
            results['zone_q'] = np.array([[
                np.sqrt(np.mean(q**2)), np.median(q), np.percentile(q, 99),
            ]])

    return results


def main() -> None:
    p = argparse.ArgumentParser(
        description='Measure the power, spacing, density, and residual forces of a glass.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('snapshot', type=str)
    p.add_argument('-o', '--output', type=str, required=True)
    p.add_argument('--geometry', type=str, required=True,
                   choices=['spherical', 'cylindrical', 'cubical'])
    p.add_argument('--nzones', type=int, default=4)
    p.add_argument('--ncubes', type=int, default=6,
                   help='Sample cubes per zone for the zoned P(k).')
    p.add_argument('--pk-nmesh', type=int, default=64)
    p.add_argument('--nprof', type=int, default=100,
                   help='Fine radial bins for the profiles.')
    p.add_argument('--seed', type=int, default=0,
                   help='RNG seed for cube placement.')
    p.add_argument('--tile-lz', type=float, default=None,
                   help='If the snapshot is a tiled torus (rescale.py), '
                        'keep only particles with z < this before '
                        'diagnosing (one tile).')
    args = p.parse_args()

    pos, mass, accel, boxsize = load_snapshot(args.snapshot)
    if args.tile_lz is not None:
        z = np.mod(pos[:, 2], boxsize) if boxsize else pos[:, 2]
        sel = z < args.tile_lz
        pos, mass = pos[sel], mass[sel]
        if accel is not None:
            accel = accel[sel]
        boxsize = None  # box is anisotropic; extents are used
        log.info('Tile cut z < %g: %d particles kept.',
                 args.tile_lz, len(pos))

    log.info('Diagnosing %s: %d particles, %d mass levels, forces: %s',
             args.snapshot, len(pos), len(np.unique(mass)),
             accel is not None)
    results = diagnose(
        pos, mass, accel, args.geometry, boxsize,
        args.nzones, args.ncubes, args.pk_nmesh, args.nprof, args.seed,
    )
    atomic_savez(args.output, **results)
    log.info('Diagnostics written to %s (%d arrays).',
             args.output, len(results))


if __name__ == '__main__':
    main()
