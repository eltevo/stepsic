#!/usr/bin/env python3
'''
Compare a native stepsic slab with an equally sized region cut from an isotropic cube.

Recovering the input ``P(k)`` on the slab's own Fourier modes cannot show which longer z modes are missing. This comparison instead uses:

* a **native slab** realization ``L x L x L/s`` (its own periodic mode
  lattice, so k_z is quantized at 2*pi/(L/s): supra-slab z-modes are
  absent by construction);
* a **cube cut**: the central slab region of an ``L^3`` periodic
  realization at the same cell size. Its density field contains the
  full isotropic mode content, folded into the slab window.

Both fields use the same estimator. Particles are weighted by a Tukey window across the slab, deposited on the same mesh, and normalized by the corresponding undisplaced load. This removes the window profile, leaving the difference in available Fourier modes. Each random field contributes:

* isotropic windowed P(k) for native and cut;
* the same split into line-of-sight (mu = |k_z|/k > MU_SPLIT) and
  transverse (mu < MU_SPLIT) mode sets;
* per-component displacement variances, showing the suppression of ``var(Psi_z)`` when the native slab lacks long z modes.

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
import scipy.fft

from stepsic.field import create_grid, fourier_vectors, wrap
from stepsic.interpolation import deposit_field

from validation._common.cosmology_fields import (
    ArrayF,
    generate_field,
    init_cosmology,
    run_lpt,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def tukey_window(z: ArrayF, lz: float, taper_frac: float) -> ArrayF:
    '''Tukey (cosine-tapered) window over z in [-lz/2, lz/2].'''
    u = np.abs(z) / (lz / 2.0)          # 0 at centre, 1 at the edge
    flat = 1.0 - taper_frac
    w = np.ones_like(u)
    edge = u > flat
    w[edge] = 0.5 * (1.0 + np.cos(np.pi * (u[edge] - flat) / taper_frac))
    w[u >= 1.0] = 0.0
    return w


def windowed_pk(
    pos: ArrayF,
    pos_lagr: ArrayF,
    weights: ArrayF,
    boxsize: ArrayF,
    nvox: np.ndarray,
    mu_split: float,
    kmax: float,
) -> tuple[ArrayF, ArrayF, ArrayF, ArrayF]:
    '''Windowed P(k) with the window profile cancelled exactly.

    delta_W(x) = n_W(x) / nbar_W(x) - 1, where n_W deposits the
    displaced particles and nbar_W the same weights at the Lagrangian
    positions. Both fields are treated with the identical procedure, so
    ratios between them are window-independent. Returns
    (k_centres, pk_iso, pk_transverse, pk_los).
    '''
    n_w = deposit_field(
        wrap(pos, boxsize), nvox, boxsize, values=weights,
        periodic=True, method='cic', origin=-boxsize / 2.0,
    )
    nbar_w = deposit_field(
        wrap(pos_lagr, boxsize), nvox, boxsize, values=weights,
        periodic=True, method='cic', origin=-boxsize / 2.0,
    )
    with np.errstate(invalid='ignore', divide='ignore'):
        delta = np.where(nbar_w > 0.0, n_w / nbar_w - 1.0, 0.0)

    vcell = float(np.prod(boxsize / nvox))
    volume = float(np.prod(boxsize))
    # Effective-volume normalization: nbar_w traces the window profile,
    # whose flat region is its maximum, so nbar_w/max approximates W(z).
    w2_mean = float(np.mean((nbar_w / nbar_w.max()) ** 2))
    delta_k = scipy.fft.rfftn(delta, workers=-1) * vcell
    power = np.abs(delta_k) ** 2 / (volume * w2_mean)

    dk = float(np.min(boxsize)) / nvox[np.argmin(boxsize)]
    kx, ky, kz = fourier_vectors(nvox, dk, hermitian=True)
    kmod = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2
                   + kz[None, None, :] ** 2)
    with np.errstate(invalid='ignore', divide='ignore'):
        mu = np.where(kmod > 0.0, np.abs(kz[None, None, :]) / kmod, 0.0)

    # rfftn multiplicity: kz>0 planes represent +/-kz.
    mult = np.ones_like(power)
    mult[:, :, 1:] = 2.0
    if nvox[2] % 2 == 0:
        mult[:, :, -1] = 1.0

    k_f = 2.0 * np.pi / float(np.max(boxsize))
    edges = np.arange(k_f, kmax, k_f)
    k_c = np.zeros(len(edges) - 1)
    out = []
    for sel_mask in (
        kmod > 0.0,
        (kmod > 0.0) & (mu < mu_split),
        (kmod > 0.0) & (mu >= mu_split),
    ):
        pk_b = np.full(len(edges) - 1, np.nan)
        for i in range(len(edges) - 1):
            m = sel_mask & (kmod >= edges[i]) & (kmod < edges[i + 1])
            if np.any(m):
                pk_b[i] = np.average(power[m], weights=mult[m])
                if k_c[i] == 0.0:
                    k_c[i] = np.average(kmod[m], weights=mult[m])
        out.append(pk_b)
    return k_c, out[0], out[1], out[2]


def run_slab_validation(
    l_cube: float,
    aspect: int,
    nmesh: int,
    lpt_order: int,
    redshift: float,
    seed: int,
    nreal: int,
    paired: bool,
    taper_frac: float,
    mu_split: float,
    kmax_frac_ny: float,
    output: str,
) -> None:
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
    box_cube = np.array([l_cube] * 3)
    box_slab = np.array([l_cube, l_cube, lz])

    k_ny = np.pi / delta
    kmax = kmax_frac_ny * k_ny
    growth = init_cosmology([redshift], l_cube, kmax=1.05 * kmax)[redshift]

    pos_slab = create_grid(nvox_slab, delta, return_coords=False)
    pos_cube = create_grid(nvox_cube, delta, return_coords=False)
    cut = np.abs(pos_cube[:, 2]) < lz / 2.0
    pos_cut = pos_cube[cut]
    w_slab = tukey_window(pos_slab[:, 2], lz, taper_frac)
    w_cut = tukey_window(pos_cut[:, 2], lz, taper_frac)

    log.info(
        'Slab fairness: L=%g, aspect %d:1 (Lz=%g), delta=%g, %dLPT, z=%g, '
        '%d realization(s), paired=%s',
        l_cube, aspect, lz, delta, lpt_order, redshift, nreal, paired,
    )

    members = [(s, cp) for s in range(nreal)
               for cp in ((False, True) if paired else (False,))]
    n_mem = len(members)

    pk_shape = None
    res: dict[str, list] = {k: [] for k in (
        'pk_k', 'pk_slab', 'pk_slab_t', 'pk_slab_l',
        'pk_cut', 'pk_cut_t', 'pk_cut_l',
        'var_slab', 'var_cut',
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
        x_s, _ = run_lpt(pos_slab, dk_slab, nvox_slab, delta, growth,
                         lpt_order, 'cic', compensate=False)
        del dk_slab
        psi_s = x_s - pos_slab
        res['var_slab'].append(psi_s.var(axis=0))

        # Periodic cube realization, same seed (distinct mode lattice).
        dk_cube = generate_field(
            nvox_cube, delta, growth, seed_i,
            paired=paired, counter_phase=counter,
        )
        x_c, _ = run_lpt(pos_cube, dk_cube, nvox_cube, delta, growth,
                         lpt_order, 'cic', compensate=False)
        del dk_cube
        psi_cut = x_c[cut] - pos_cut
        res['var_cut'].append(psi_cut.var(axis=0))

        k_c, p, pt, pl = windowed_pk(
            x_s, pos_slab, w_slab, box_slab, nvox_slab, mu_split, kmax,
        )
        res['pk_slab'].append(p)
        res['pk_slab_t'].append(pt)
        res['pk_slab_l'].append(pl)
        # The cube cut is deposited into the same slab box; x/y are
        # genuinely periodic, and the Tukey weights kill the z edges.
        k_c2, p, pt, pl = windowed_pk(
            x_c[cut], pos_cut, w_cut, box_slab, nvox_slab, mu_split, kmax,
        )
        res['pk_cut'].append(p)
        res['pk_cut_t'].append(pt)
        res['pk_cut_l'].append(pl)
        if pk_shape is None:
            pk_shape = k_c.shape
            res['pk_k'] = k_c
        del x_s, x_c, psi_s, psi_cut

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
        'meta_taper_frac': np.float64(taper_frac),
        'meta_mu_split': np.float64(mu_split),
        'pk_k': res['pk_k'],
    }
    for key in ('pk_slab', 'pk_slab_t', 'pk_slab_l',
                'pk_cut', 'pk_cut_t', 'pk_cut_l',
                'var_slab', 'var_cut'):
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
    p.add_argument('--taper-frac', type=float, default=0.25,
                   help='Tukey taper fraction of the half-thickness.')
    p.add_argument('--mu-split', type=float, default=0.5,
                   help='|k_z|/k boundary between transverse and LOS bins.')
    p.add_argument('--kmax-frac-ny', type=float, default=0.8)
    p.add_argument('-o', '--output', type=str, default='slab_data.npz')
    return p.parse_args()


if __name__ == '__main__':
    a = parse_args()
    run_slab_validation(
        l_cube=a.Lcube, aspect=a.aspect, nmesh=a.nmesh, lpt_order=a.lpt,
        redshift=a.redshift, seed=a.seed, nreal=a.nreal, paired=a.paired,
        taper_frac=a.taper_frac, mu_split=a.mu_split,
        kmax_frac_ny=a.kmax_frac_ny, output=a.output,
    )
