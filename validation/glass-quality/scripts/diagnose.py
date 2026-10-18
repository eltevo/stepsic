#!/usr/bin/env python3
'''
Measure the power, spacing, density, and residual forces of a glass.

For spherical and cylindrical glasses, power is measured in cumulative central apertures. A spherical aperture uses an inscribed cube; a cylindrical aperture uses an inscribed square cross-section and the full periodic axis. Cubical glasses use the full periodic box.

The archive contains mass-weighted power relative to shot noise, nearest-neighbour distance relative to local spacing, radial density, and mass-level boundaries. If the snapshot contains accelerations, it also contains ``q = |F| d^2 / m`` in internal units with ``G = 1``.

For a tiled glass, ``--tile-lz`` restricts the measurement to the first repeated z section.

Run the same measurement on the Poisson twin to compare the relaxed glass with random particle positions.

Usage
-----
::

    python diagnose.py glass.hdf5 -o diag.npz --geometry cylindrical \
        --radius-mpc-h 500 --length-mpc-h 1000
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


def radial_power_apertures(
    pos: np.ndarray,
    mass: np.ndarray,
    d_exp: np.ndarray,
    radius_mpc_h: float,
    length_mpc_h: float | None,
    geometry: str,
    napertures: int,
    pk_nmesh: int,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    '''Measure power in cumulative central apertures.

    The Cartesian FFT boxes are fully contained by the open radial domain.
    Cylindrical boxes retain the complete periodic z-axis. The 0.95 factor
    keeps their corners away from the relaxed load's outer boundary. The FFT
    treats each cutout as periodic; the matched twin controls that window, and
    the evaluator uses only scales below the coarsest particle spacing.
    '''
    aperture_rmax = radius_mpc_h * (
        np.arange(1, napertures + 1, dtype=np.float64) / napertures
    )
    positions = np.asarray(pos, dtype=np.float64).copy()
    if geometry == 'cylindrical':
        assert length_mpc_h is not None
        positions[:, 2] = np.mod(positions[:, 2], length_mpc_h)

    power_k: list[np.ndarray] = []
    power_pk: list[np.ndarray] = []
    power_shot: list[float] = []
    power_spacing: list[float] = []
    power_boxsize: list[np.ndarray] = []
    safety = 0.95
    dimension_factor = np.sqrt(3.0 if geometry == 'spherical' else 2.0)
    for outer_radius_mpc_h in aperture_rmax:
        transverse_size_mpc_h = (
            safety * 2.0 * outer_radius_mpc_h / dimension_factor
        )
        if geometry == 'spherical':
            boxsize_mpc_h = np.full(3, transverse_size_mpc_h)
            origin_mpc_h = -0.5 * boxsize_mpc_h
        else:
            boxsize_mpc_h = np.array([
                transverse_size_mpc_h,
                transverse_size_mpc_h,
                length_mpc_h,
            ])
            origin_mpc_h = np.array([
                -0.5 * transverse_size_mpc_h,
                -0.5 * transverse_size_mpc_h,
                0.0,
            ])
        selected = np.all(
            (positions >= origin_mpc_h)
            & (positions < origin_mpc_h + boxsize_mpc_h),
            axis=1,
        )
        particle_count = int(np.count_nonzero(selected))
        if particle_count < 32:
            raise ValueError(
                f'power aperture r < {outer_radius_mpc_h:g} Mpc/h '
                f'contains {particle_count} particles; at least 32 are required'
            )
        m_sel = mass[selected]
        k, pk, _ = measure_pk(
            positions[selected] - origin_mpc_h,
            boxsize_mpc_h,
            nmesh=pk_nmesh,
            mass=m_sel,
            method='cic', deconvolve=True, interlace=True,
            subtract_shot=False,
        )
        selected_mass = float(np.sum(m_sel, dtype=np.float64))
        shot = (
            float(np.prod(boxsize_mpc_h, dtype=np.float64))
            * float(np.sum(m_sel**2, dtype=np.float64))
            / selected_mass**2
        )
        power_k.append(k)
        power_pk.append(pk)
        power_shot.append(shot)
        power_spacing.append(float(np.max(d_exp[selected])))
        power_boxsize.append(boxsize_mpc_h)
    return (
        power_k,
        power_pk,
        np.asarray(power_shot),
        np.asarray(power_spacing),
        aperture_rmax,
        np.asarray(power_boxsize),
    )


def diagnose(
    pos: np.ndarray,
    mass: np.ndarray,
    accel: np.ndarray | None,
    geometry: str,
    boxsize: float | None,
    nzones: int,
    pk_nmesh: int,
    nprof: int,
    *,
    radius_mpc_h: float | None = None,
    length_mpc_h: float | None = None,
) -> dict[str, np.ndarray]:
    pos = np.asarray(pos, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1:] != (3,) or len(pos) < 2:
        raise ValueError('positions must have shape (n, 3) with n >= 2')
    if (
        mass.shape != (len(pos),)
        or not np.all(np.isfinite(mass))
        or np.any(mass <= 0.0)
    ):
        raise ValueError(
            'masses must be finite and positive with one value per particle'
        )
    if not np.all(np.isfinite(pos)):
        raise ValueError('positions must be finite')
    if accel is not None and (np.asarray(accel).shape != pos.shape or not np.all(np.isfinite(accel))):
        raise ValueError('accelerations must be finite and match the position array')
    if geometry not in ('cubical', 'spherical', 'cylindrical'):
        raise ValueError('geometry must be cubical, spherical, or cylindrical')
    if min(nzones, pk_nmesh, nprof) < 1:
        raise ValueError('zone, power-mesh, and profile counts must be positive')
    if geometry != 'cubical' and (
        radius_mpc_h is None
        or not np.isfinite(radius_mpc_h)
        or radius_mpc_h <= 0.0
    ):
        raise ValueError('radial geometries require a finite positive radius')
    if geometry == 'cylindrical' and (
        length_mpc_h is None
        or not np.isfinite(length_mpc_h)
        or length_mpc_h <= 0.0
    ):
        raise ValueError('cylindrical geometry requires a finite positive length')
    n = len(pos)
    m_tot = float(np.sum(mass, dtype=np.float64))

    # -- mean density and domain --------------------------------------------
    if geometry == 'cubical':
        lo, hi = pos.min(axis=0), pos.max(axis=0)
        if boxsize is not None and (not np.isfinite(boxsize) or boxsize <= 0.0):
            raise ValueError('cubical box size must be finite and positive')
        vol = float(np.prod(hi - lo)) if boxsize is None else boxsize**3
        r = None
        r_max = None
    else:
        r = _radius(pos, geometry)
        assert radius_mpc_h is not None
        r_max = radius_mpc_h
        radial_tolerance_mpc_h = 8.0 * np.finfo(np.float32).eps * r_max
        if np.any(r > r_max + radial_tolerance_mpc_h):
            raise ValueError('particles lie outside the configured radial domain')
        if geometry == 'spherical':
            vol = 4.0 / 3.0 * np.pi * r_max**3
        else:
            assert length_mpc_h is not None
            lz = length_mpc_h
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
    elif geometry == 'cylindrical':
        copies = np.concatenate([pos + [0, 0, offset] for offset in (-lz, 0., lz)])
        tree = cKDTree(copies)
        d_nn = tree.query(pos, k=2)[0][:, 1]
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
        results['prof_count'] = cnt
        with np.errstate(invalid='ignore'):
            results['prof_u'] = np.where(cnt > 0, sum_u / cnt, np.nan)
        if has_force:
            sum_q = np.bincount(idx, weights=q, minlength=nprof)
            with np.errstate(invalid='ignore'):
                results['prof_q'] = np.where(cnt > 0, sum_q / cnt, np.nan)

        power = radial_power_apertures(
            pos,
            mass,
            d_exp,
            r_max,
            length_mpc_h,
            geometry,
            nzones,
            pk_nmesh,
        )
        (
            power_k,
            power_pk,
            power_shot,
            power_spacing,
            power_rmax,
            power_boxsize,
        ) = power
        results['power_k'], results['power_count'] = _pad_rows(power_k)
        results['power_pk'], pk_count = _pad_rows(power_pk)
        if not np.array_equal(results['power_count'], pk_count):
            raise RuntimeError('power k and spectrum rows have different lengths')
        results['power_shot'] = power_shot
        results['power_spacing'] = power_spacing
        results['power_rmax'] = power_rmax
        results['power_boxsize'] = power_boxsize

        zone_edges = np.linspace(0.0, r_max, nzones + 1)
        results['zone_edges'] = zone_edges
        zone_q = []
        for iz in range(nzones):
            r0, r1 = zone_edges[iz], zone_edges[iz + 1]
            in_z = (r >= r0) & (r < r1) if iz < nzones - 1 else (r >= r0)
            if not np.any(in_z):
                raise ValueError('physical force zone contains no particles')
            if has_force:
                zone_q.append([
                    np.sqrt(np.mean(q[in_z]**2)), np.median(q[in_z]),
                    np.percentile(q[in_z], 99),
                ])
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
        results['power_k'] = np.asarray([k], dtype=np.float64)
        results['power_pk'] = np.asarray([pk], dtype=np.float64)
        results['power_count'] = np.array([len(k)], dtype=np.int64)
        results['power_shot'] = np.array([shot])
        results['power_spacing'] = np.array([float(np.max(d_exp))])
        results['power_rmax'] = np.array([b])
        results['power_boxsize'] = np.asarray([[b, b, b]])
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
    p.add_argument('--pk-nmesh', type=int, default=64)
    p.add_argument('--nprof', type=int, default=100,
                   help='Fine radial bins for the profiles.')
    p.add_argument('--radius-mpc-h', type=float,
                   help='Configured sphere or cylinder radius [Mpc/h].')
    p.add_argument('--length-mpc-h', type=float,
                   help='Configured periodic cylinder length [Mpc/h].')
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
        args.nzones, args.pk_nmesh, args.nprof,
        radius_mpc_h=args.radius_mpc_h,
        length_mpc_h=args.length_mpc_h,
    )
    atomic_savez(args.output, **results)
    log.info('Diagnostics written to %s (%d arrays).',
             args.output, len(results))


if __name__ == '__main__':
    main()
