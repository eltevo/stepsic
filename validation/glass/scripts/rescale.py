#!/usr/bin/env python3
'''
Anisotropically rescale a cubic toroidal glass and tile it back to a cube.

Linear rescaling reuses a toroidal glass in boxes of a different aspect
ratio, but a force-free glass does not remain force-free after the rescale.
This tool prepares configurations for measuring that residual with StePS.

StePS's periodic mode only supports cubic boxes, so the anisotropic
torus is measured through an exact reduction: compressing the z
coordinates by 1/s (positions ``(x, y, z/s)`` in the box
``[L, L, L/s]``) and then stacking ``s`` copies along z reproduces, in
the ``L^3`` periodic cube, exactly the force field of the ``[L, L,
L/s]`` periodic torus (the torus IS the cube restricted to s-fold
z-periodic configurations). Forces read off any one tile are the torus
forces. ``s = 1`` passes the input through (control configuration).

Velocities are zeroed; particle IDs are re-enumerated; the Header
particle counts are updated for the tiled total.

Usage
-----
::

    python rescale.py glass.hdf5 rescaled.hdf5 --aspect 2
'''

from __future__ import annotations

import argparse
import atexit
import logging
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def rescale_and_tile(pos: np.ndarray, lbox: float, aspect: int) -> np.ndarray:
    '''Compress z by 1/aspect and stack `aspect` copies along z.'''
    z = np.mod(pos[:, 2], lbox) / aspect
    tiles = []
    for k in range(aspect):
        t = pos.copy()
        t[:, 2] = z + k * lbox / aspect
        tiles.append(t)
    return np.concatenate(tiles, axis=0)


def main() -> None:
    p = argparse.ArgumentParser(
        description='Aspect-rescale + tile a cubic toroidal glass.')
    p.add_argument('input', type=str)
    p.add_argument('output', type=str)
    p.add_argument('--aspect', type=int, required=True,
                   help='Integer aspect ratio s (z compressed by 1/s); '
                        '1 = pass-through control.')
    args = p.parse_args()
    if args.aspect < 1:
        raise ValueError('aspect must be >= 1.')
    s = args.aspect

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-", dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)

    def cleanup_temporary() -> None:
        temporary.unlink(missing_ok=True)

    atexit.register(cleanup_temporary)
    with h5py.File(args.input, 'r') as src, \
            h5py.File(temporary, 'w') as dst:
        pos = src['PartType1/Coordinates'][:].astype(np.float64)
        mass = src['PartType1/Masses'][:]
        lbox = float(src['Header'].attrs['BoxSize'])
        # Positions may be centred ([-L/2, L/2)) or wrapped ([0, L));
        # normalize to [0, L) before compressing so tiling is exact.
        if pos.min() < 0.0:
            pos += lbox / 2.0
        new_pos = rescale_and_tile(pos, lbox, s)
        n_new = len(new_pos)
        # Conserve the mean density of the physical L x L x L/s torus.
        new_mass = np.tile(mass / s, s)

        g = dst.create_group('PartType1')
        src_g = src['PartType1']
        g.create_dataset(
            'Coordinates',
            data=new_pos.astype(src_g['Coordinates'].dtype),
        )
        g.create_dataset('Masses', data=new_mass)
        g.create_dataset(
            'Velocities',
            data=np.zeros((n_new, 3),
                          dtype=src_g['Coordinates'].dtype),
        )
        g.create_dataset('ParticleIDs',
                         data=np.arange(n_new, dtype=np.uint64))

        hdr = dst.create_group('Header')
        for k, v in src['Header'].attrs.items():
            hdr.attrs[k] = v
        npart = np.array(src['Header'].attrs['NumPart_ThisFile']).copy()
        npart[1] = n_new
        hdr.attrs['NumPart_ThisFile'] = npart
        if 'NumPart_Total' in hdr.attrs:
            total = np.array(hdr.attrs['NumPart_Total']).copy()
            total[1] = n_new
            hdr.attrs['NumPart_Total'] = total

    file_descriptor = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)
    os.replace(temporary, output)
    directory_descriptor = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)

    log.info(
        'Aspect %d:1 torus prepared as tiled cube: %d particles -> %s',
        s, n_new, output,
    )


if __name__ == '__main__':
    main()
