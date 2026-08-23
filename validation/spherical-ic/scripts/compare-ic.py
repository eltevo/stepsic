#!/usr/bin/env python3
'''
Compare embedded initial conditions with a periodic cube.

Both inputs must use the same box, mesh, seed, cosmology, and redshift so they share every Gaussian mode. The central comparison then measures differences caused by particle geometry, multiresolution mixing, and interpolation away from grid points:

* **velocity difference** - both particle sets are deposited with CIC onto the same central grid. The archive records the RMS difference, its radial profile, and its value relative to the periodic grid. At 1LPT, velocity is proportional to displacement;
* **core P(k)** - mass-weighted power inside the core radius, using the same spherical or cylindrical window for both inputs;
* **velocity variance** - ``var(v)`` along each axis in the core. For cylindrical geometry, z is the periodic axis.

Usage
-----
::

    python compare-ic.py --geom-ic sphere/ic.hdf5 --ref-ic cube/ic.hdf5 \
        --geometry spherical --lbox 1000 --rcore 250 -o compare.npz
'''

from __future__ import annotations



import argparse
import logging

import h5py
import numpy as np

from validation._common.evaluation import atomic_savez

from stepsic.interpolation import deposit_field
from stepsic.pk import measure_pk
from stepsic.field import wrap

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_ic(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, 'r') as f:
        g = f['PartType1']
        pos = g['Coordinates'][:].astype(np.float64)
        vel = g['Velocities'][:].astype(np.float64)
        mass = g['Masses'][:].astype(np.float64)
    return pos, vel, mass


def unwrap_centred(pos: np.ndarray, sizes: np.ndarray,
                   wrapped: tuple[bool, bool, bool]) -> np.ndarray:
    '''Convert wrapped output coordinates back to centred coordinates.

    ``stepsic.field.wrap`` stores periodic-axis coordinates as
    ``mod(x + L/2, L)`` (corner-origin); the physical centred
    coordinate is therefore ``stored - L/2``. Non-periodic axes are already centred.
    '''
    out = pos.copy()
    for i in range(3):
        if wrapped[i]:
            out[:, i] -= sizes[i] / 2.0
    return out


def _core_radius_coord(pos: np.ndarray, geometry: str) -> np.ndarray:
    if geometry == 'spherical':
        return np.linalg.norm(pos, axis=1)
    return np.hypot(pos[:, 0], pos[:, 1])


def compare(
    geom_ic: str,
    ref_ic: str,
    geometry: str,
    lbox: float,
    rcore: float,
    lz: float | None,
    ncmp: int,
    pk_nmesh: int,
    nprof: int,
    output: str,
) -> None:
    pos_g, vel_g, mass_g = load_ic(geom_ic)
    pos_r, vel_r, mass_r = load_ic(ref_ic)
    log.info('Geometry IC: %d particles; reference IC: %d particles',
             len(pos_g), len(pos_r))

    sizes = np.array([lbox, lbox, lz if lz is not None else lbox])
    # Reference cube: PERIODIC=[1,1,1], all axes wrapped. Geometry IC:
    # spherical has no periodic axis; cylindrical wraps z only.
    pos_r = unwrap_centred(pos_r, sizes, (True, True, True))
    if geometry == 'cylindrical':
        pos_g = unwrap_centred(pos_g, sizes, (False, False, True))

    r_g = _core_radius_coord(pos_g, geometry)
    r_r = _core_radius_coord(pos_r, geometry)
    in_g = r_g < rcore
    in_r = r_r < rcore

    # -- common comparison volume -------------------------------------------
    if geometry == 'spherical':
        lo = np.array([-rcore] * 3)
        boxsize = np.array([2.0 * rcore] * 3)
    else:
        if lz is None:
            raise ValueError('cylindrical comparison requires --Lz')
        lo = np.array([-rcore, -rcore, -lz / 2.0])
        boxsize = np.array([2.0 * rcore, 2.0 * rcore, lz])
    nvox = np.array([ncmp, ncmp, ncmp], dtype=np.int64)
    if geometry == 'cylindrical':
        # keep cells cubic in the transverse plane; z cells span lz/ncmp_z
        ncmp_z = max(2, int(round(ncmp * lz / (2.0 * rcore))))
        nvox = np.array([ncmp, ncmp, ncmp_z], dtype=np.int64)

    def velocity_field(pos, vel, mass, sel):
        vals = np.column_stack([
            mass[sel],
            mass[sel, None] * vel[sel],
        ])
        grid = deposit_field(
            pos[sel], nvox, boxsize, values=vals,
            periodic=False, method='cic', origin=lo,
        )
        m_cell = grid[0]
        with np.errstate(invalid='ignore', divide='ignore'):
            v_cell = np.where(m_cell > 0.0, grid[1:] / m_cell, np.nan)
        return m_cell, v_cell

    m_cell_g, v_cell_g = velocity_field(pos_g, vel_g, mass_g, in_g)
    m_cell_r, v_cell_r = velocity_field(pos_r, vel_r, mass_r, in_r)

    # Only cells well inside the window and populated in both fields.
    cell = boxsize / nvox
    axes = [lo[i] + (np.arange(nvox[i]) + 0.5) * cell[i] for i in range(3)]
    cx, cy, cz = np.meshgrid(*axes, indexing='ij')
    if geometry == 'spherical':
        r_cell = np.sqrt(cx**2 + cy**2 + cz**2)
    else:
        r_cell = np.hypot(cx, cy)
    good = (
        (r_cell < 0.9 * rcore)
        & (m_cell_g > 0.0) & (m_cell_r > 0.0)
        & np.all(np.isfinite(v_cell_g), axis=0)
        & np.all(np.isfinite(v_cell_r), axis=0)
    )
    n_good = int(good.sum())
    if n_good == 0:
        raise RuntimeError('No commonly populated cells in the core region.')

    dv = np.linalg.norm(v_cell_g - v_cell_r, axis=0)
    v_ref_mag = np.linalg.norm(v_cell_r, axis=0)
    rms_ref = float(np.sqrt(np.mean(v_ref_mag[good] ** 2)))
    rms_dv = float(np.sqrt(np.mean(dv[good] ** 2)))
    log.info('Velocity-field residual: RMS %.6g of reference RMS %.6g '
             '(relative %.4e, %d cells)',
             rms_dv, rms_ref, rms_dv / rms_ref, n_good)

    prof_edges = np.linspace(0.0, 0.9 * rcore, nprof + 1)
    prof_rel = np.full(nprof, np.nan)
    for i in range(nprof):
        m = good & (r_cell >= prof_edges[i]) & (r_cell < prof_edges[i + 1])
        if np.any(m):
            prof_rel[i] = np.sqrt(np.mean(dv[m] ** 2)) / rms_ref

    # -- windowed core P(k) --------------------------------------------------
    def core_pk(pos, mass, sel):
        return measure_pk(
            wrap(pos[sel] - lo, boxsize), boxsize, nvox=_pk_mesh(boxsize),
            mass=mass[sel], method='cic', deconvolve=True, interlace=True,
            subtract_shot=False,
        )

    def _pk_mesh(bs):
        dk = float(np.min(bs)) / pk_nmesh
        nv = np.rint(bs / dk).astype(np.int64)
        return nv + nv % 2

    k_g, pk_g, _ = core_pk(pos_g, mass_g, in_g)
    k_r, pk_r, _ = core_pk(pos_r, mass_r, in_r)
    if not np.allclose(k_g, k_r):
        raise RuntimeError('P(k) bins differ between geometry and reference.')

    # -- component variances -------------------------------------------------
    var_g = vel_g[in_g].var(axis=0)
    var_r = vel_r[in_r].var(axis=0)

    atomic_savez(
        output,
        meta_geometry=np.array(geometry),
        meta_lbox=np.float64(lbox),
        meta_rcore=np.float64(rcore),
        meta_lz=np.float64(lz if lz is not None else 0.0),
        meta_n_geom=np.int64(len(pos_g)),
        meta_n_ref=np.int64(len(pos_r)),
        meta_n_geom_core=np.int64(np.count_nonzero(in_g)),
        meta_n_ref_core=np.int64(np.count_nonzero(in_r)),
        meta_n_cells=np.int64(n_good),
        rms_dv=np.float64(rms_dv),
        rms_ref=np.float64(rms_ref),
        prof_edges=prof_edges,
        prof_rel=prof_rel,
        pk_k=k_g,
        pk_geom=pk_g,
        pk_ref=pk_r,
        var_geom=var_g,
        var_ref=var_r,
    )
    log.info('Comparison written to %s', output)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Compare embedded initial conditions with a periodic cube.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--geom-ic', type=str, required=True)
    p.add_argument('--ref-ic', type=str, required=True)
    p.add_argument('--geometry', type=str, default='spherical',
                   choices=['spherical', 'cylindrical'])
    p.add_argument('--lbox', type=float, required=True,
                   help='Shared Fourier-box side [Mpc/h].')
    p.add_argument('--rcore', type=float, required=True,
                   help='Core comparison radius [Mpc/h].')
    p.add_argument('--Lz', type=float, default=None, dest='lz',
                   help='Cylinder length (cylindrical only).')
    p.add_argument('--ncmp', type=int, default=32,
                   help='Comparison-grid cells across the core cube.')
    p.add_argument('--pk-nmesh', type=int, default=64)
    p.add_argument('--nprof', type=int, default=10)
    p.add_argument('-o', '--output', type=str, required=True)
    return p.parse_args()


if __name__ == '__main__':
    a = parse_args()
    compare(
        geom_ic=a.geom_ic, ref_ic=a.ref_ic, geometry=a.geometry,
        lbox=a.lbox, rcore=a.rcore, lz=a.lz, ncmp=a.ncmp,
        pk_nmesh=a.pk_nmesh, nprof=a.nprof, output=a.output,
    )
