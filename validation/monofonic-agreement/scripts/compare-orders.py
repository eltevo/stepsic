#!/usr/bin/env python3
'''
Compare the first-order fields and the second-order corrections of stepsic
and monofonIC at matched particle IDs.

The second-order correction of each code is the difference between its 2LPT
and 1LPT snapshots from the same white noise. The script reports the relative
RMS residual of every field and, per wavenumber bin, the amplitude ratio and
the cross-correlation between the two codes.

The campaign step isolated_orders runs it on the snapshots of the 1LPT and
2LPT runs of one mesh size.

Usage
-----
::

    python compare-orders.py \\
        --stepsic-one   runs/custom/<lpt1>/cache/stepsic/ic.hdf5 \\
        --stepsic-two   runs/custom/<lpt2>/cache/stepsic/ic.hdf5 \\
        --monofonic-one runs/custom/<lpt1>/cache/monofonic/monofonic_ics.hdf5 \\
        --monofonic-two runs/custom/<lpt2>/cache/monofonic/monofonic_ics.hdf5 \\
        --nmesh 64 --box-mpc-h 1000 --redshift 31 \\
        --output isolated-orders.npz
'''

import argparse
from pathlib import Path

import numpy as np
from scipy.fft import rfftn

from validation._common.evaluation import archive_provenance, atomic_savez

from particles import (
    lagrangian_lattice_from_id,
    read_gadget_hdf5,
    sort_by_id,
    wrap_periodic,
)


def read(path, nmesh):
    '''Return positions and velocities ordered by lattice index.'''
    pos, vel, ids = sort_by_id(*read_gadget_hdf5(path))
    if not np.array_equal(ids, np.arange(nmesh**3)):
        raise ValueError(
            f'{path}: particle IDs must be the complete zero-based lattice'
        )
    return pos, vel


def mode_statistics(first, second, mesh, box):
    '''Bin the amplitude ratio and cross-correlation of two vector fields.'''
    fields = [
        rfftn(field.reshape(mesh, mesh, mesh, 3), axes=(0, 1, 2))
        for field in (first, second)
    ]
    kxy = 2 * np.pi * np.fft.fftfreq(mesh, box / mesh)
    kz = 2 * np.pi * np.fft.rfftfreq(mesh, box / mesh)
    kmag = np.sqrt(
        kxy[:, None, None]**2 + kxy[None, :, None]**2 + kz[None, None, :]**2
    )
    bins = (kmag / (2 * np.pi / box)).astype(int)
    selected = (kmag > 0) & (kmag < np.pi * mesh / box)
    # The half-spectrum holds one of each conjugate pair except on the
    # kz = 0 and Nyquist planes.
    weights = np.broadcast_to(
        np.r_[1., np.full(mesh // 2 - 1, 2.), 1.], kmag.shape,
    )
    counts = np.bincount(bins[selected], weights=weights[selected])
    keep = counts > 0

    def mean(value):
        total = np.bincount(
            bins[selected], weights=weights[selected] * value[selected],
        )
        return total[keep] / counts[keep]

    auto = [mean(np.sum(abs(field)**2, axis=-1)) for field in fields]
    cross = mean(np.sum((fields[0] * fields[1].conj()).real, axis=-1))
    if np.any(auto[0] <= 0) or np.any(auto[1] <= 0):
        raise ValueError('a field has no power in a wavenumber bin')
    return (
        mean(kmag),
        np.sqrt(auto[0] / auto[1]),
        cross / np.sqrt(auto[0] * auto[1]),
    )


def main():
    parser = argparse.ArgumentParser(
        description='Compare stepsic and monofonIC order by order.',
    )
    for name in ('stepsic-one', 'stepsic-two', 'monofonic-one', 'monofonic-two'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--nmesh', type=int, required=True)
    parser.add_argument('--box-mpc-h', type=float, required=True)
    parser.add_argument('--redshift', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    box, mesh = args.box_mpc_h, args.nmesh

    linear, quadratic, provenance = [], [], {}
    offsets = {'stepsic': 0.5, 'monofonic': 0.0}
    for code, offset in offsets.items():
        paths = [getattr(args, f'{code}_{order}') for order in ('one', 'two')]
        first, second = [read(path, mesh) for path in paths]
        lattice = lagrangian_lattice_from_id(
            np.arange(mesh**3), mesh, box / mesh, cell_offset=offset,
        )
        linear.append([wrap_periodic(first[0] - lattice, box), first[1]])
        quadratic.append([
            wrap_periodic(second[0] - first[0], box), second[1] - first[1],
        ])
        provenance[code] = np.array([
            archive_provenance(path)['archive_sha256'] for path in paths
        ])

    arrays = dict(
        meta_nmesh=np.int64(mesh),
        meta_box_mpc_h=np.float64(box),
        meta_redshift=np.float64(args.redshift),
        meta_origin_offsets_cells=np.array(list(offsets.values())),
        meta_origins=np.array(
            'same white-noise lattice indices; '
            'coordinate origins differ by half a cell'
        ),
        meta_normalization=np.array('raw amplitudes; no fitted correction'),
        meta_limitations=np.array(
            'de-aliasing, transfer normalization and numerical convergence '
            'require separate controls'
        ),
        meta_implementation_sha256=np.array(
            archive_provenance(__file__)['archive_sha256']
        ),
        **provenance,
    )
    for name, fields in (('first', linear), ('second', quadratic)):
        for j, field_name in enumerate(('displacement', 'velocity')):
            key = f'{name}_{field_name}'
            first, second = fields[0][j], fields[1][j]
            k, transfer, cross = mode_statistics(first, second, mesh, box)
            residual = np.sqrt(np.mean((first - second)**2) / np.mean(second**2))
            arrays[f'{key}_rms_residual'] = np.float64(residual)
            arrays[f'{key}_transfer'] = transfer
            arrays[f'{key}_cross_correlation'] = cross
            print(f'{key}: raw RMS residual {residual:.8g}')
    arrays['k_h_mpc'] = k
    atomic_savez(args.output, **arrays)


if __name__ == '__main__':
    main()
