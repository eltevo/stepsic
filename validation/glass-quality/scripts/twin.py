#!/usr/bin/env python3
'''
Randomize an intended uniform-density mass distribution or angles at fixed radii.

Mass-band volumes follow their fraction of the total mass in the declared domain, and the bands stack outward in the order of each level's median glass radius. Cylindrical z is drawn from [0, L), the frame of stepsic's cylindrical loads. Angular randomization preserves the measured radial structure.

The output keeps the input HDF5 layout so the same StePS force calculation can read it.

Usage
-----
::

    python twin.py glass.hdf5 twin.hdf5 --geometry spherical --radius-mpc-h 500 --seed 12345
'''

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _radius(pos: np.ndarray, geometry: str) -> np.ndarray:
    if geometry == 'spherical':
        return np.linalg.norm(pos, axis=1)
    if geometry == 'cylindrical':
        return np.hypot(pos[:, 0], pos[:, 1])
    raise ValueError(f'No radial coordinate for geometry {geometry}.')


def make_twin(pos: np.ndarray, mass: np.ndarray, geometry: str,
              seed: int, *, radius_mpc_h=None, length_mpc_h=None,
              boxsize_mpc_h=None, angular_only=False) -> np.ndarray:
    """Randomize an intended uniform-density load, or angles at fixed radii."""
    pos, mass = np.asarray(pos, dtype=float), np.asarray(mass, dtype=float)
    if pos.ndim != 2 or pos.shape[1:] != (3,) or len(pos) == 0 or mass.shape != (len(pos),):
        raise ValueError('positions and masses must be non-empty peers')
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError('positions and positive masses must be finite')
    rng = np.random.default_rng(seed)
    if geometry == 'cubical':
        if boxsize_mpc_h is None or not np.isfinite(boxsize_mpc_h) or boxsize_mpc_h <= 0:
            raise ValueError('cubical controls require a finite positive box size')
        if angular_only:
            raise ValueError('angular randomization requires a radial geometry')
        return rng.uniform(0, boxsize_mpc_h, size=pos.shape)
    r = _radius(pos, geometry)
    if radius_mpc_h is None or not np.isfinite(radius_mpc_h) or radius_mpc_h <= 0:
        raise ValueError('radial controls require a finite positive intended radius')
    if geometry == 'cylindrical' and (length_mpc_h is None or not np.isfinite(length_mpc_h) or length_mpc_h <= 0):
        raise ValueError('cylindrical controls require a finite positive intended length')
    levels, counts = np.unique(mass, return_counts=True)
    dim = 3 if geometry == 'spherical' else 2
    # Bands stack outward in the glass's radial order of the mass levels.
    order = np.argsort([np.median(r[mass == m]) for m in levels])
    levels, counts = levels[order], counts[order]
    # Uniform target density fixes each mass band's volume from its total mass.
    volume_fraction = np.r_[0., np.cumsum(levels*counts)/np.sum(mass)]
    new_pos = np.empty_like(pos)
    for i,m in enumerate(levels):
        sel = mass == m
        n = int(sel.sum())
        radii = r[sel] if angular_only else radius_mpc_h*rng.uniform(volume_fraction[i],volume_fraction[i+1],n)**(1/dim)
        if geometry == 'spherical':
            directions = rng.normal(size=(n,3))
            directions /= np.linalg.norm(directions,axis=1)[:,None]
            new_pos[sel] = radii[:,None]*directions
        else:
            angle = rng.uniform(0,2*np.pi,n)
            new_pos[sel,0] = radii*np.cos(angle)
            new_pos[sel,1] = radii*np.sin(angle)
            new_pos[sel,2] = pos[sel,2] if angular_only else rng.uniform(0,length_mpc_h,n)
    return new_pos


def main() -> None:
    p = argparse.ArgumentParser(
        description='Randomize a glass while preserving its masses and radial bands.')
    p.add_argument('input', type=str)
    p.add_argument('output', type=str)
    p.add_argument('--geometry', type=str, required=True,
                   choices=['spherical', 'cylindrical', 'cubical'])
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--radius-mpc-h', type=float)
    p.add_argument('--length-mpc-h', type=float)
    p.add_argument('--boxsize-mpc-h', type=float)
    p.add_argument('--angular-only', action='store_true')
    args = p.parse_args()

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        temporary = Path(staging) / 'twin.hdf5'
        with h5py.File(args.input, 'r') as src, \
                h5py.File(temporary, 'w') as dst:
            pos = src['PartType1/Coordinates'][:]
            mass = src['PartType1/Masses'][:]
            new_pos = make_twin(pos.astype(np.float64), mass, args.geometry,
                                args.seed, radius_mpc_h=args.radius_mpc_h,
                                length_mpc_h=args.length_mpc_h, boxsize_mpc_h=args.boxsize_mpc_h,
                                angular_only=args.angular_only)
            for name, grp in src.items():
                if name == 'PartType1':
                    g = dst.create_group('PartType1')
                    for dname, dset in grp.items():
                        if dname == 'Coordinates':
                            g.create_dataset(dname, data=new_pos.astype(dset.dtype))
                        else:
                            g.create_dataset(dname, data=dset[:])
                    for k, v in grp.attrs.items():
                        g.attrs[k] = v
                else:
                    src.copy(name, dst)
            for k, v in src.attrs.items():
                dst.attrs[k] = v

        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)

    log.info('Poisson twin written to %s (%d particles, %s).',
             args.output, len(mass), args.geometry)


if __name__ == '__main__':
    main()
