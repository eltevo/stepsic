#!/usr/bin/env python3
'''
Compare a native stepsic slab with an equally sized region cut from an isotropic cube.

The comparison uses:

* a **native slab** realization ``L x L x L/s`` (its own periodic mode
  lattice, so k_z is quantized at 2*pi/(L/s): supra-slab z-modes are
  absent by construction);
* a **cube cut**: the central slab region of an ``L^3`` periodic
  realization at the same cell size. It retains the cube's long modes.

Each random field contributes per-component displacement and velocity
variances for the native slab and the cube cut.

The two boxes have different Fourier modes, so the same seed does not create identical fields. Repeating the comparison with ``--nreal`` measures the resulting variation; paired-fixed fields can reduce it.

Usage
-----
::

    python run.py --Lcube 1000 --aspect 10 --nmesh 256 --lpt 2 --z 31 \
        --nreal 5 -o slab_data.npz
'''

from __future__ import annotations



import argparse
import logging
import time

import numpy as np

from validation._common.evaluation import atomic_savez

from stepsic.particles import create_grid

from validation._common.fields import (
    generate_field, init_cosmology, run_lpt,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def run_slab_validation(
    l_cube: float,
    aspect: int,
    nmesh: int,
    lpt_order: int,
    redshift: float,
    seed: int,
    nreal: int,
    paired: bool,
    output: str,
) -> None:
    if nreal < 2:
        raise ValueError('nreal must be >= 2 to estimate sampling uncertainty.')
    if nmesh % (2 * aspect) != 0:
        raise ValueError(
            f'nmesh={nmesh} must be divisible by 2*aspect={2 * aspect} '
            f'so the slab mesh is even.'
        )
    delta = l_cube / nmesh
    lz = l_cube / aspect
    nz = nmesh // aspect
    nvox_cube = np.array([nmesh, nmesh, nmesh], dtype=np.int64)
    nvox_slab = np.array([nmesh, nmesh, nz], dtype=np.int64)
    k_ny = np.pi / delta
    # The field takes P(k) at every mesh mode, out to the corner at sqrt(3) k_Ny.
    growth = init_cosmology(
        [redshift], l_cube, kmax=1.05 * np.sqrt(3.0) * k_ny)[redshift]

    pos_slab = create_grid(nvox_slab, delta, return_coords=False)
    pos_cube = create_grid(nvox_cube, delta, return_coords=False)
    cut = np.abs(pos_cube[:, 2]) < lz / 2.0
    pos_cut = pos_cube[cut]

    log.info(
        'Slab fairness: L=%g, aspect %d:1 (Lz=%g), delta=%g, %dLPT, z=%g, '
        '%d realization(s), paired=%s',
        l_cube, aspect, lz, delta, lpt_order, redshift, nreal, paired,
    )

    members = [(s, cp) for s in range(nreal)
               for cp in ((False, True) if paired else (False,))]
    n_mem = len(members)

    res: dict[str, list] = {k: [] for k in (
        'var_disp_slab', 'var_disp_cut',
        'var_vel_slab', 'var_vel_cut',
    )}
    t0 = time.time()
    for i_m, (i_s, counter) in enumerate(members):
        seed_i = seed + i_s
        log.info('[%d/%d] seed=%d counter=%s', i_m + 1, n_mem, seed_i, counter)

        # Native slab realization.
        dk_slab = generate_field(
            nvox_slab, delta, growth, seed_i,
            paired=paired, counter_phase=counter,
        )
        x_s, v_s = run_lpt(pos_slab, dk_slab, nvox_slab, delta, growth,
                           lpt_order, 'cic', compensate=False)
        del dk_slab
        psi_s = x_s - pos_slab
        res['var_disp_slab'].append(
            np.var(psi_s, axis=0, dtype=np.float64)
        )
        res['var_vel_slab'].append(
            np.var(v_s, axis=0, dtype=np.float64)
        )

        # Periodic cube realization, same seed (distinct mode lattice).
        dk_cube = generate_field(
            nvox_cube, delta, growth, seed_i,
            paired=paired, counter_phase=counter,
        )
        x_c, v_c = run_lpt(pos_cube, dk_cube, nvox_cube, delta, growth,
                           lpt_order, 'cic', compensate=False)
        del dk_cube
        psi_cut = x_c[cut] - pos_cut
        res['var_disp_cut'].append(
            np.var(psi_cut, axis=0, dtype=np.float64)
        )
        res['var_vel_cut'].append(
            np.var(v_c[cut], axis=0, dtype=np.float64)
        )
        del x_s, v_s, x_c, v_c, psi_s, psi_cut

    out: dict[str, np.ndarray] = {
        'meta_l_cube': np.float64(l_cube),
        'meta_aspect': np.int64(aspect),
        'meta_lz': np.float64(lz),
        'meta_nmesh': np.int64(nmesh),
        'meta_delta': np.float64(delta),
        'meta_lpt_order': np.int64(lpt_order),
        'meta_redshift': np.float64(redshift),
        'meta_seed': np.int64(seed),
        'meta_nreal': np.int64(nreal),
        'meta_paired': np.bool_(paired),
        'meta_n_total': np.int64(n_mem),
    }
    for key in ('var_disp_slab', 'var_disp_cut',
                'var_vel_slab', 'var_vel_cut'):
        out[key] = np.asarray(res[key])
    atomic_savez(output, **out)
    log.info('Results written to %s (%.1f s).', output, time.time() - t0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Compare a native slab with an equally sized region cut from a cube.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--Lcube', type=float, default=1000.0)
    p.add_argument('--aspect', type=int, default=10,
                   help='Slab aspect ratio s (Lz = Lcube/s).')
    p.add_argument('--nmesh', type=int, default=256,
                   help='Cells along the cube side (fixed cell size).')
    p.add_argument('--lpt', type=int, default=2, choices=[1, 2])
    p.add_argument('--z', type=float, default=31.0, dest='redshift')
    p.add_argument('--seed', type=int, default=137)
    p.add_argument('--nreal', type=int, default=5)
    p.add_argument('--paired', action='store_true')
    p.add_argument('-o', '--output', type=str, default='slab_data.npz')
    return p.parse_args()


if __name__ == '__main__':
    a = parse_args()
    run_slab_validation(
        l_cube=a.Lcube, aspect=a.aspect, nmesh=a.nmesh, lpt_order=a.lpt,
        redshift=a.redshift, seed=a.seed, nreal=a.nreal, paired=a.paired,
        output=a.output,
    )
