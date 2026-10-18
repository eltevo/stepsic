'''Read periodic initial conditions and reconstruct their Lagrangian lattice.'''

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from stepsic.io import CosmoIO

ArrayF: TypeAlias = NDArray[np.float64]


def read_gadget_hdf5(
    path: str,
    part_type: int = 1,
) -> tuple[ArrayF, ArrayF, ArrayF]:
    '''
    Read positions, velocities, and IDs from a Gadget HDF5 IC file.

    A snapshot split over several numbered files is read as a whole from
    the path of any one of them.

    Returns
    -------
    pos : ndarray of shape (N, 3)
        Particle positions.
    vel : ndarray of shape (N, 3)
        Particle velocities.
    ids : ndarray of shape (N,)
        Particle IDs.
    '''
    ids, pos, vel = CosmoIO.load_snapshot(
        Path(path),
        'ParticleIDs',
        'Coordinates',
        'Velocities',
        part_type=part_type,
    )
    pos = pos.astype(np.float64)
    vel = vel.astype(np.float64)
    ids = ids.astype(np.int64)
    return pos, vel, ids


def sort_by_id(
    pos: ArrayF, vel: ArrayF, ids: ArrayF,
) -> tuple[ArrayF, ArrayF, ArrayF]:
    '''Sort arrays by particle ID so the same particles can be compared.'''
    order = np.argsort(ids)
    return pos[order], vel[order], ids[order]


def lagrangian_lattice_from_id(
    ids: NDArray[np.int64],
    nmesh: int,
    dx: float,
    *,
    cell_offset: float,
) -> ArrayF:
    '''
    Reconstruct each particle's unperturbed Lagrangian position from its ID.

    Both stepsic and monofonIC index particles as
    ``id = (ix * N + iy) * N + iz`` (C-order, z-fastest), but they place
    them at different points within the cell:

    - ``cell_offset = 0.5``: cell-centered, at ``(i + 0.5) * dx`` (stepsic).
    - ``cell_offset = 0.0``: cell-corner, at ``i * dx`` (monofonIC ``sc``).

    Returns positions in the [0, L) convention used by the Gadget output.
    '''
    ix = ids // (nmesh * nmesh)
    iy = (ids // nmesh) % nmesh
    iz = ids % nmesh
    grid = np.stack([ix, iy, iz], axis=1).astype(np.float64)
    return (grid + cell_offset) * dx


def wrap_periodic(d: ArrayF, lbox: float) -> ArrayF:
    '''Wrap a displacement vector to [-L/2, +L/2] (periodic BCs).'''
    return d - np.round(d / lbox) * lbox
