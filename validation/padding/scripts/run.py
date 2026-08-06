#!/usr/bin/env python3
'''
Quantify periodic-embedding error as a function of bounding-box padding.

stepsic builds every geometry's Gaussian field and LPT solution on a
periodic cuboid Fourier box; non-periodic domains (sphere, cylinder) are
simply embedded in that box. This sweep measures the residual periodicity
and boundary contamination as a function of the padding ratio

    alpha = LBOX / (2 * R_3D)

on the non-periodic axes, holding the voxel size and the *physical*
stochastic source fixed.

Shared-realization construction
-------------------------------
``stepsic.field.white_noise`` keys each Fourier mode's random draw to its
integer mode triple, so calling it with the same seed on boxes of
different physical size yields *stretched* realizations, not shared ones.
To isolate embedding systematics from realization scatter, white noise is
therefore drawn once per seed on the largest (reference) box, transformed
to real space - where it is an iid unit-Gaussian per voxel - and the
centred voxel-aligned sub-cube is cropped for each smaller alpha. A crop
of iid noise is itself exactly valid white noise, and the stochastic
source inside the domain is bit-for-bit identical across alpha. Two
effects then separate every smaller box from the reference:

* mode truncation - wavelengths longer than the small box do not exist;
* image contamination - the periodic images of the domain sit closer.

Both are genuine embedding error, so per-particle differences against the
largest-alpha box measure periodic-embedding error. The reference box is itself periodic, so alpha_max must be
generous (default 2.0); the measured error is the error *relative to*
that best available embedding.

Metrics (per alpha, per seed)
-----------------------------
* RMS and 99th-percentile displacement/velocity difference against the
  reference, in radial shells of the domain radius;
* normalized antipodal boundary-shell displacement correlation (the
  direct periodic-image signature), for every alpha including the
  reference;
* mass-weighted P(k) of the central region (common spherical/cylindrical
  window, identical particles, so the ratio to the reference is window-
  independent);
* monopole diagnostics after displacement: mass fraction remaining
  inside the domain, mean radial displacement of the boundary shell,
  mean |Psi_r| at the boundary and the implied surface-flux estimate
  S * <|Psi_r|> / V.

Results are written to an ``.npz`` archive consumed by ``plot.py``.

Usage
-----
::

    python run.py --geometry spherical --R3d 500 --ngrid0 128 \
        --alphas 1.0 1.125 1.25 1.5 2.0 --nseeds 10 -o padding.npz
'''

from __future__ import annotations



import argparse
import logging
import time

import numpy as np

from validation._common.evaluation import atomic_savez
import scipy.fft
from scipy.spatial import cKDTree

from stepsic.field import white_noise, generate_delta_k, wrap
from stepsic.geometry import create_shell_particles
from stepsic.pk import measure_pk
from stepsic.units import UNIT_V

from validation import (
    ArrayF,
    ArrayI,
    GrowthData,
    init_cosmology,
    run_lpt,
    VALIDATION_COSMOLOGY,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Internal-unit critical density for h-independent coordinates; matches
# stepsic/parameters.py ("h^2 as we use [U/h] internally").
RHO_CRIT = 3.0 * (100.0 / UNIT_V) ** 2 / (8.0 * np.pi)


def _validate_alphas(alphas: list[float], n0: int, nz: int | None) -> ArrayI:
    '''Map padding ratios to even voxel counts on the padded axes.

    Every ``alpha * n0`` must round to an even integer so that the
    centred real-space crop is voxel-aligned with the reference lattice
    (both boxes centred at the origin with cell centres at
    ``(i - N/2 + 0.5) * delta``).
    '''
    if len(alphas) < 2:
        raise ValueError('Need at least two alphas (last one is the reference).')
    if not np.all(np.diff(alphas) > 0):
        raise ValueError(f'alphas must be strictly increasing, got {alphas}.')
    n_alpha = []
    for a in alphas:
        n = a * n0
        if abs(n - round(n)) > 1e-9 or round(n) % 2 != 0:
            raise ValueError(
                f'alpha={a} * ngrid0={n0} = {n} is not an even integer; '
                f'choose alphas that keep the crop voxel-aligned.'
            )
        n_alpha.append(round(n))
    if nz is not None and nz % 2 != 0:
        raise ValueError(f'LZ voxel count {nz} must be even.')
    return np.asarray(n_alpha, dtype=np.int64)


def _build_load(
    geometry: str,
    r_3d: float,
    d_4d: float,
    rcrit: float,
    bin_mode: str,
    nrbins: int,
    nshell: int,
    lz: float,
    omega_m: float,
    load_seed: int,
) -> tuple[ArrayF, ArrayF]:
    '''Create the alpha-independent shell particle load (production path).'''
    params: dict = {
        'GEOMETRY': geometry,
        'BIN_MODE': bin_mode,
        'D_4D': d_4d,
        'NRBINS': nrbins,
        'R_3D': r_3d,
        'NSHELL': nshell,
        'SEED': load_seed,
        'RHO_MEAN': omega_m * RHO_CRIT,
        'OMEGA_M': omega_m,
        # Only min(LBOX) (= L_z) is read for cylindrical mass assignment.
        'LBOX': np.array([2.0 * r_3d, 2.0 * r_3d, lz]),
    }
    if rcrit > 0.0:
        params['RCRIT'] = rcrit
    pos, mass = create_shell_particles(params)
    return pos, mass


def _crop_white_noise(w_real: ArrayF, nvox: ArrayI) -> np.ndarray:
    '''Centred voxel-aligned crop of real-space white noise -> W(k).

    Returns the half-complex Fourier field in the same convention as
    :func:`stepsic.field.white_noise` (DC and Nyquist planes zeroed).
    '''
    off = [(nb - int(ns)) // 2 for nb, ns in zip(w_real.shape, nvox)]
    sl = tuple(slice(o, o + int(ns)) for o, ns in zip(off, nvox))
    w_k = scipy.fft.rfftn(w_real[sl], axes=(0, 1, 2), workers=-1)
    nx, ny, nz = (int(n) for n in nvox)
    w_k[nx // 2, :, :] = 0.0
    w_k[:, ny // 2, :] = 0.0
    w_k[:, :, nz // 2] = 0.0
    w_k[0, 0, 0] = 0.0
    return w_k


def _domain_radius(pos: ArrayF, geometry: str) -> ArrayF:
    '''Radius relevant to the open boundary: 3D for sphere, transverse
    (x-y) for cylinder (the z axis is genuinely periodic).'''
    if geometry == 'spherical':
        return np.linalg.norm(pos, axis=1)
    return np.hypot(pos[:, 0], pos[:, 1])


def _antipodal_pairs(
    pos: ArrayF,
    r_load: ArrayF,
    r_3d: float,
    geometry: str,
    boundary_frac: float,
) -> tuple[ArrayI, ArrayI]:
    '''Index pairs (i, j) of boundary-shell particles with x_j ~ -x_i
    (sphere) or (x_j, y_j) ~ (-x_i, -y_i) at similar z (cylinder).

    Pairs farther apart than twice the shell's median nearest-neighbour
    spacing are dropped: they would dilute the antipodal estimator with
    uncorrelated bulk separations.
    '''
    sel = np.flatnonzero(r_load > (1.0 - boundary_frac) * r_3d)
    p = pos[sel]
    target = -p.copy()
    if geometry == 'cylindrical':
        target[:, 2] = p[:, 2]
    tree = cKDTree(p)
    nn_d, _ = tree.query(p, k=2)
    d_max = 2.0 * np.median(nn_d[:, 1])
    d, j = tree.query(target)
    valid = d < d_max
    return sel[valid], sel[j[valid]]


def _core_selection(
    pos: ArrayF,
    r_load: ArrayF,
    r_3d: float,
    lz: float,
    geometry: str,
    core_frac: float,
) -> tuple[np.ndarray, ArrayF, ArrayF]:
    '''Boolean mask, box lower corner, and box size for the central-region
    P(k) measurement.'''
    r_core = core_frac * r_3d
    mask = r_load < r_core
    if geometry == 'spherical':
        lo = np.array([-r_core, -r_core, -r_core])
        boxsize = np.array([2.0 * r_core, 2.0 * r_core, 2.0 * r_core])
    else:
        z = pos[:, 2]
        lo = np.array([-r_core, -r_core, float(z.min())])
        boxsize = np.array([2.0 * r_core, 2.0 * r_core, lz])
    return mask, lo, boxsize


def _pk_mesh(boxsize: ArrayF, nmesh: int) -> ArrayI:
    '''Even cubic-voxel mesh for the core P(k) box.'''
    dk = float(np.min(boxsize)) / nmesh
    nvox_raw = np.rint(boxsize / dk).astype(np.int64)
    nvox = nvox_raw + nvox_raw % 2
    if not np.allclose(nvox * dk, boxsize, rtol=1e-9):
        raise ValueError(
            f'Core P(k) box {boxsize} is not commensurate with cubic cells '
            f'at nmesh={nmesh}; adjust CORE_FRAC / LZ / PK_NMESH.'
        )
    return nvox


def run_padding_sweep(
    geometry: str,
    r_3d: float,
    d_4d: float,
    rcrit: float,
    bin_mode: str,
    nrbins: int,
    nshell: int,
    lz: float,
    ngrid0: int,
    alphas: list[float],
    lpt_order: int,
    redshift: float,
    seed: int,
    nseeds: int,
    load_seed: int,
    boundary_frac: float,
    core_frac: float,
    pk_nmesh: int,
    nprof: int,
    output: str,
) -> None:
    if geometry not in ('spherical', 'cylindrical'):
        raise ValueError(f'Unsupported geometry: {geometry}.')
    if geometry == 'cylindrical' and lz > 2.0 * r_3d:
        # create_shell_particles reads L_z as min(LBOX).
        raise ValueError(f'Cylindrical LZ={lz} must be <= 2*R_3D={2 * r_3d}.')

    delta = 2.0 * r_3d / ngrid0  # fixed cell size on every axis [Mpc/h]
    nz = None
    if geometry == 'cylindrical':
        nz_f = lz / delta
        if abs(nz_f - round(nz_f)) > 1e-9:
            raise ValueError(
                f'LZ={lz} is not an integer number of cells (delta={delta}).'
            )
        nz = round(nz_f)
    n_alpha = _validate_alphas(alphas, ngrid0, nz)
    alphas_arr = np.asarray(alphas, dtype=np.float64)
    n_max = int(n_alpha[-1])

    def nvox_of(n_pad: int) -> ArrayI:
        if geometry == 'spherical':
            return np.array([n_pad, n_pad, n_pad], dtype=np.int64)
        return np.array([n_pad, n_pad, nz], dtype=np.int64)

    nvox_ref = nvox_of(n_max)
    log.info(
        'Padding sweep: %s R_3D=%g, delta=%g, alphas=%s -> nvox %s..%s, '
        '%dLPT, z=%g, %d seeds',
        geometry, r_3d, delta, alphas, nvox_of(int(n_alpha[0])).tolist(),
        nvox_ref.tolist(), lpt_order, redshift, nseeds,
    )

    # -- alpha-independent particle load (production shell path) ------------
    pos_load, mass = _build_load(
        geometry, r_3d, d_4d, rcrit, bin_mode, nrbins, nshell, lz,
        VALIDATION_COSMOLOGY['OMEGA_M'], load_seed,
    )
    n_part = len(pos_load)
    r_load = _domain_radius(pos_load, geometry)
    m_tot = float(mass.sum())
    log.info('Particle load: %d particles, M_tot=%.6e', n_part, m_tot)

    pair_i, pair_j = _antipodal_pairs(
        pos_load, r_load, r_3d, geometry, boundary_frac,
    )
    log.info('Antipodal boundary pairs: %d', len(pair_i))

    core_mask, core_lo, core_box = _core_selection(
        pos_load, r_load, r_3d, lz, geometry, core_frac,
    )
    core_nvox = _pk_mesh(core_box, pk_nmesh)
    log.info('Core region: %d particles, P(k) mesh %s',
             int(core_mask.sum()), core_nvox.tolist())

    boundary_mask = r_load > (1.0 - boundary_frac) * r_3d
    r_edges = np.linspace(0.0, r_3d, nprof + 1)
    shell_idx = np.clip(
        np.searchsorted(r_edges, r_load, side='right') - 1, 0, nprof - 1,
    )

    # -- cosmology -----------------------------------------------------------
    k_ny = np.pi / delta
    growth = init_cosmology(
        [redshift], float(alphas[-1]) * 2.0 * r_3d, kmax=1.05 * k_ny,
    )[redshift]

    n_a = len(alphas)
    seeds = [seed + i for i in range(nseeds)]

    results: dict[str, np.ndarray] = {
        'meta_geometry': np.array(geometry),
        'meta_r3d': np.float64(r_3d),
        'meta_d4d': np.float64(d_4d),
        'meta_rcrit': np.float64(rcrit),
        'meta_lz': np.float64(lz),
        'meta_ngrid0': np.int64(ngrid0),
        'meta_delta': np.float64(delta),
        'meta_lpt_order': np.int64(lpt_order),
        'meta_redshift': np.float64(redshift),
        'meta_seed': np.int64(seed),
        'meta_nseeds': np.int64(nseeds),
        'meta_load_seed': np.int64(load_seed),
        'meta_n_part': np.int64(n_part),
        'meta_boundary_frac': np.float64(boundary_frac),
        'meta_core_frac': np.float64(core_frac),
        'meta_pk_nmesh': np.int64(pk_nmesh),
        'alphas': alphas_arr,
        'r_edges': r_edges,
        'anti_npairs': np.int64(len(pair_i)),
    }

    prof_rms_dx = np.zeros((nseeds, n_a - 1, nprof))
    prof_p99_dx = np.zeros((nseeds, n_a - 1, nprof))
    prof_rms_dv = np.zeros((nseeds, n_a - 1, nprof))
    prof_p99_dv = np.zeros((nseeds, n_a - 1, nprof))
    ref_rms_psi = np.zeros((nseeds, nprof))
    ref_rms_v = np.zeros((nseeds, nprof))
    anti_c = np.zeros((nseeds, n_a))
    mono_massfrac = np.zeros((nseeds, n_a))
    mono_dr_mean = np.zeros((nseeds, n_a))
    mono_abspsi_r = np.zeros((nseeds, n_a))
    mono_pred_flux = np.zeros((nseeds, n_a))
    pk_all: np.ndarray | None = None
    pk_k: np.ndarray | None = None

    # Surface-to-volume factor for the flux estimate S*<|Psi_r|>/V.
    s_over_v = 3.0 / r_3d if geometry == 'spherical' else 2.0 / r_3d

    rhat = np.zeros_like(pos_load)
    nonzero = r_load > 0
    if geometry == 'spherical':
        rhat[nonzero] = pos_load[nonzero] / r_load[nonzero, None]
    else:
        rhat[nonzero, 0] = pos_load[nonzero, 0] / r_load[nonzero]
        rhat[nonzero, 1] = pos_load[nonzero, 1] / r_load[nonzero]

    t_start = time.time()
    for i_seed, seed_i in enumerate(seeds):
        log.info('%s', '=' * 72)
        log.info('Seed %d [%d/%d]', seed_i, i_seed + 1, nseeds)
        w_k_ref = white_noise(nvox=nvox_ref, seed=seed_i)
        w_real = scipy.fft.irfftn(
            w_k_ref, s=tuple(int(n) for n in nvox_ref),
            axes=(0, 1, 2), workers=-1,
        )
        del w_k_ref

        xpert_ref: ArrayF | None = None
        vpert_ref: ArrayF | None = None
        # Reference (last alpha) first, then the sweep against it.
        for i_a in [n_a - 1] + list(range(n_a - 1)):
            a = float(alphas_arr[i_a])
            nvox = nvox_of(int(n_alpha[i_a]))
            t0 = time.time()
            w_k = _crop_white_noise(w_real, nvox)
            delta_k = generate_delta_k(
                growth.kh_camb, growth.pk_input, nvox, delta, field=w_k,
            )
            del w_k
            # Production path: CIC without MAS compensation (article runs).
            xpert, vpert = run_lpt(
                pos_load, delta_k, nvox, delta, growth,
                lpt_order, 'cic', compensate=False,
            )
            del delta_k

            is_ref = i_a == n_a - 1
            if is_ref:
                xpert_ref, vpert_ref = xpert, vpert
            else:
                dx = np.linalg.norm(xpert - xpert_ref, axis=1)
                dv = np.linalg.norm(vpert - vpert_ref, axis=1)
                for s in range(nprof):
                    in_s = shell_idx == s
                    if not np.any(in_s):
                        continue
                    prof_rms_dx[i_seed, i_a, s] = np.sqrt(np.mean(dx[in_s] ** 2))
                    prof_p99_dx[i_seed, i_a, s] = np.percentile(dx[in_s], 99.0)
                    prof_rms_dv[i_seed, i_a, s] = np.sqrt(np.mean(dv[in_s] ** 2))
                    prof_p99_dv[i_seed, i_a, s] = np.percentile(dv[in_s], 99.0)
                del dx, dv

            disp = xpert - pos_load
            if is_ref:
                psi_mag = np.linalg.norm(disp, axis=1)
                v_mag = np.linalg.norm(vpert, axis=1)
                for s in range(nprof):
                    in_s = shell_idx == s
                    if not np.any(in_s):
                        continue
                    ref_rms_psi[i_seed, s] = np.sqrt(np.mean(psi_mag[in_s] ** 2))
                    ref_rms_v[i_seed, s] = np.sqrt(np.mean(v_mag[in_s] ** 2))
                del psi_mag, v_mag

            # Antipodal boundary-shell displacement correlation.
            di, dj = disp[pair_i], disp[pair_j]
            anti_c[i_seed, i_a] = (
                np.mean(np.sum(di * dj, axis=1))
                / np.sqrt(np.mean(np.sum(di**2, axis=1))
                          * np.mean(np.sum(dj**2, axis=1)))
            )

            # Monopole diagnostics.
            pos_out = xpert.copy()
            if geometry == 'cylindrical':
                # z is genuinely periodic: wrap before the inside test.
                z0 = core_lo[2]
                pos_out[:, 2] = np.mod(pos_out[:, 2] - z0, lz) + z0
            r_after = _domain_radius(pos_out, geometry)
            inside = r_after <= r_3d
            mono_massfrac[i_seed, i_a] = float(mass[inside].sum()) / m_tot
            mono_dr_mean[i_seed, i_a] = float(
                np.mean(r_after[boundary_mask] - r_load[boundary_mask])
            )
            psi_r = np.sum(disp * rhat, axis=1)
            abspsi = float(np.mean(np.abs(psi_r[boundary_mask])))
            mono_abspsi_r[i_seed, i_a] = abspsi
            mono_pred_flux[i_seed, i_a] = s_over_v * abspsi

            # Core-region mass-weighted P(k) (common window across alpha).
            core_pos = wrap(xpert[core_mask] - core_lo, core_box)
            k_c, pk_c, _ = measure_pk(
                core_pos, core_box, nvox=core_nvox, mass=mass[core_mask],
                method='cic', deconvolve=True, interlace=True,
                subtract_shot=False, kmax=0.8 * np.pi / (core_box[0] / core_nvox[0]),
            )
            if pk_all is None:
                pk_k = k_c
                pk_all = np.zeros((nseeds, n_a, len(k_c)))
            pk_all[i_seed, i_a] = pk_c

            log.info(
                '  alpha=%-6g nvox=%s  massfrac=%.6f  anti_c=%+.4f  (%.1f s)',
                a, nvox.tolist(), mono_massfrac[i_seed, i_a],
                anti_c[i_seed, i_a], time.time() - t0,
            )
        del w_real, xpert_ref, vpert_ref

    results.update(
        prof_rms_dx=prof_rms_dx, prof_p99_dx=prof_p99_dx,
        prof_rms_dv=prof_rms_dv, prof_p99_dv=prof_p99_dv,
        ref_rms_psi=ref_rms_psi, ref_rms_v=ref_rms_v,
        anti_c=anti_c,
        mono_massfrac=mono_massfrac, mono_dr_mean=mono_dr_mean,
        mono_abspsi_r=mono_abspsi_r, mono_pred_flux=mono_pred_flux,
        pk_k=pk_k, pk=pk_all,
    )
    atomic_savez(output, **results)
    log.info(
        'Results written to %s (%d arrays, %.1f s total)',
        output, len(results), time.time() - t_start,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Padding-ratio sweep for periodic embedding error.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--geometry', type=str, default='spherical',
                   choices=['spherical', 'cylindrical'])
    p.add_argument('--R3d', type=float, default=500.0, dest='r_3d',
                   help='Domain radius [Mpc/h].')
    p.add_argument('--D4d', type=float, default=75.0, dest='d_4d',
                   help='Stereographic projection diameter [Mpc/h].')
    p.add_argument('--rcrit', type=float, default=100.0,
                   help='Constant-resolution core radius [Mpc/h]; <=0 disables.')
    p.add_argument('--bin-mode', type=str, default='omega',
                   choices=['omega', 'volume'])
    p.add_argument('--nrbins', type=int, default=64,
                   help='Radial shells in the particle load.')
    p.add_argument('--nshell', type=int, default=1024,
                   help='Particles per exterior shell.')
    p.add_argument('--Lz', type=float, default=500.0, dest='lz',
                   help='Cylinder length [Mpc/h] (cylindrical only).')
    p.add_argument('--ngrid0', type=int, default=128,
                   help='Voxels across the domain diameter (sets delta).')
    p.add_argument('--alphas', type=float, nargs='+',
                   default=[1.0, 1.125, 1.25, 1.5, 2.0],
                   help='Padding ratios, ascending; last is the reference.')
    p.add_argument('--lpt', type=int, default=2, choices=[1, 2])
    p.add_argument('--z', type=float, default=31.0, dest='redshift')
    p.add_argument('--seed', type=int, default=137,
                   help='Base white-noise seed.')
    p.add_argument('--nseeds', type=int, default=10,
                   help='Independent noise realizations.')
    p.add_argument('--load-seed', type=int, default=42,
                   help='Seed for the (alpha-independent) particle load.')
    p.add_argument('--boundary-frac', type=float, default=0.05,
                   help='Boundary shell thickness as a fraction of R_3D.')
    p.add_argument('--core-frac', type=float, default=0.5,
                   help='Central-region radius as a fraction of R_3D.')
    p.add_argument('--pk-nmesh', type=int, default=64,
                   help='Mesh cells (shortest axis) for the core P(k).')
    p.add_argument('--nprof', type=int, default=25,
                   help='Radial shells for the error profiles.')
    p.add_argument('-o', '--output', type=str, default='padding_data.npz')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_padding_sweep(
        geometry=args.geometry,
        r_3d=args.r_3d,
        d_4d=args.d_4d,
        rcrit=args.rcrit,
        bin_mode=args.bin_mode,
        nrbins=args.nrbins,
        nshell=args.nshell,
        lz=args.lz,
        ngrid0=args.ngrid0,
        alphas=args.alphas,
        lpt_order=args.lpt,
        redshift=args.redshift,
        seed=args.seed,
        nseeds=args.nseeds,
        load_seed=args.load_seed,
        boundary_frac=args.boundary_frac,
        core_frac=args.core_frac,
        pk_nmesh=args.pk_nmesh,
        nprof=args.nprof,
        output=args.output,
    )
