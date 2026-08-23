#!/usr/bin/env python3
'''
Randomize particle positions while preserving a glass's masses and radial bands.

Positions are drawn uniformly within each mass level's radial extent: an annulus for cylindrical loads, a shell for spherical loads, and the whole box for cubical loads. The randomized snapshot provides a Poisson comparison for the glass diagnostics.

The output keeps the input HDF5 layout so the same StePS force calculation can read it.

Usage
-----
::

    python twin.py glass.hdf5 twin.hdf5 --geometry spherical --seed 12345
'''

from __future__ import annotations

import argparse
import logging

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
              seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    new_pos = np.empty_like(pos)

    if geometry == 'cubical':
        lo = pos.min(axis=0)
        hi = pos.max(axis=0)
        new_pos[:] = rng.uniform(lo, hi, size=pos.shape)
        return new_pos

    r = _radius(pos, geometry)
    z = pos[:, 2]
    dim = 3 if geometry == 'spherical' else 2
    for m in np.unique(mass):
        sel = mass == m
        n = int(sel.sum())
        r0, r1 = float(r[sel].min()), float(r[sel].max())
        # Uniform in the annulus/shell volume: uniform in r^dim.
        u = rng.uniform(size=n)
        radii = (u * (r1**dim - r0**dim) + r0**dim) ** (1.0 / dim)
        if geometry == 'spherical':
            v = rng.normal(size=(n, 3))
            v /= np.linalg.norm(v, axis=1)[:, None]
            new_pos[sel] = radii[:, None] * v
        else:
            phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
            new_pos[sel, 0] = radii * np.cos(phi)
            new_pos[sel, 1] = radii * np.sin(phi)
            new_pos[sel, 2] = rng.uniform(z.min(), z.max(), size=n)
    return new_pos


def main() -> None:
    p = argparse.ArgumentParser(
        description='Randomize a glass while preserving its masses and radial bands.')
    p.add_argument('input', type=str)
    p.add_argument('output', type=str)
    p.add_argument('--geometry', type=str, required=True,
                   choices=['spherical', 'cylindrical', 'cubical'])
    p.add_argument('--seed', type=int, default=12345)
    args = p.parse_args()

    with h5py.File(args.input, 'r') as src, \
            h5py.File(args.output, 'w') as dst:
        pos = src['PartType1/Coordinates'][:]
        mass = src['PartType1/Masses'][:]
        new_pos = make_twin(pos.astype(np.float64), mass, args.geometry,
                            args.seed)
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

    log.info('Poisson twin written to %s (%d particles, %s).',
             args.output, len(mass), args.geometry)


if __name__ == '__main__':
    main()
