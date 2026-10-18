#!/usr/bin/env python3
'''Cosmology, growth, and LPT field helpers used by validation scripts.'''

from __future__ import annotations


import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence, TypeAlias, Union

import h5py
import numpy as np
from numpy.typing import NDArray
import scipy.fft
from scipy.interpolate import CubicSpline
import toml

from stepsic.cosmology import (
    CAMBCosmology,
    ColossusCosmology,
    F2_omega,
    F_omega,
    G2_omega,
    hubble_a,
)
from stepsic.field import (
    fourier_kmod,
    fourier_vectors,
    generate_delta_k,
    white_noise,
)
from stepsic.interpolation import KERNELS, GridGeometry, compensation_factors
from stepsic.lpt import lpt1, lpt2
from stepsic.parameters import compute_derived_cosmo
from stepsic.pk import _bin_isotropic_modes, measure_pk_from_delta_k

log = logging.getLogger(__name__)


ArrayF: TypeAlias = NDArray[np.float64]
ArrayI: TypeAlias = NDArray[np.int64]
ArrayC: TypeAlias = Union[NDArray[np.complex128], NDArray[np.complex64]]


VALIDATION_COSMOLOGY_NAME = 'Planck2018EE+BAO'
_VALIDATION_COSMOLOGY_PATH = (
    Path(__file__).parent / 'cosmology'
    / f'{VALIDATION_COSMOLOGY_NAME}.toml'
)
VALIDATION_COSMOLOGY: dict[str, float] = {
    key: float(value)
    for key, value in toml.load(_VALIDATION_COSMOLOGY_PATH).items()
}


@dataclass(frozen=True)
class GrowthData:
    r'''
    Pre-computed growth factors and velocity prefactors for one redshift.

    Attributes
    ----------
    redshift : float
        Target redshift.
    d1 : float
        Linear growth factor :math:`D_1(z)` normalised to :math:`D_1(0) = 1`.
    g1 : float
        First-order Lagrangian growth coefficient (:math:`\equiv 1`).
    g2 : float
        Second-order Lagrangian growth coefficient,
        :math:`-(3/7)\,\Omega_m(a)^{-1/143}`
        [CITE: Bouchet et al. 1995].
    aHf1 : float
        First-order velocity prefactor
        :math:`a\,H(a)\,f_1(a)` [km/s / (Mpc/h)].
    aHf2 : float
        Second-order velocity prefactor
        :math:`a\,H(a)\,f_2(a)` [km/s / (Mpc/h)].
    pk_input : ArrayF
        :math:`P(k, z) = P(k, 0) \cdot D_1(z)^2` evaluated at
        ``kh_camb``.
    kh_camb : ArrayF
        Wavenumber array from CAMB [h/Mpc].
    '''
    redshift: float
    d1: float
    g1: float
    g2: float
    aHf1: float
    aHf2: float
    pk_input: ArrayF
    kh_camb: ArrayF


def init_cosmology(
    redshifts: Sequence[float],
    lbox_max: float,
    *,
    kmin: float | None = None,
    kmax: float | None = None,
    npoints: int = 2048,
    cosmo: dict[str, float] | None = None,
) -> dict[float, GrowthData]:
    '''Set up CAMB + Colossus once and compute growth data per redshift.

    Parameters
    ----------
    redshifts : sequence of float
        Target redshift(s). Duplicates are silently ignored.
    lbox_max : float
        Largest box dimension [Mpc/h], used to set the default
        ``kmin = 1 / lbox_max`` for CAMB.
    kmin : float or None
        Override the minimum wavenumber for the CAMB tabulation.
    kmax : float or None
        Maximum wavenumber for the CAMB tabulation.
    npoints : int
        Number of k-samples in the CAMB output.
    cosmo : dict or None
        Cosmological parameters. Defaults to :data:`VALIDATION_COSMOLOGY`.

    Returns
    -------
    dict mapping float -> GrowthData
        One entry per unique redshift.
    '''
    if cosmo is None:
        cosmo = VALIDATION_COSMOLOGY

    log.info('Initialising Colossus cosmology...')
    cosmo_colossus = ColossusCosmology(
        H0=cosmo['H0'],
        Om0=cosmo['OMEGA_M'],
        Ob0=cosmo['OMEGA_B'],
        Ol0=cosmo['OMEGA_L'],
        sigma8=cosmo['SIGMA8'],
        ns=cosmo['NS'],
        Neff=cosmo['NNU'],
        w0=cosmo['W0'],
        wa=cosmo['WA'],
        Tcmb0=1e-6,
    )

    log.info('Computing CAMB z=0 power spectrum...')
    # stepsic derives the CDM density with the massive-neutrino density removed.
    derived = dict(cosmo)
    compute_derived_cosmo(derived)
    cosmo_camb = CAMBCosmology(
        H0=cosmo['H0'],
        ombh2=derived['OMBH2'],
        omch2=derived['OMCH2'],
        omk=0.0,
        mnu=cosmo['MNU'],
        nnu=cosmo['NNU'],
        YHe=cosmo['YHE'],
        TCMB=cosmo['TCMB'],
        zrei=cosmo['ZREI'],
        w0=cosmo['W0'],
        wa=cosmo['WA'],
        nonlinear=False,
    )
    kh_camb, pk_camb_z0, _ = cosmo_camb.get_spectrum(
        z=0,
        As=cosmo['AS'],
        ns=cosmo['NS'],
        sigma8_init=cosmo['SIGMA8'],
        kmin=kmin if kmin is not None else 1.0 / lbox_max,
        kmax=kmax,
        npoints=npoints,
    )
    pk_z0 = pk_camb_z0[0]

    g1 = 1.0

    growth_table: dict[float, GrowthData] = {}
    for z in sorted(set(redshifts)):
        scale = 1.0 / (z + 1.0)
        g2 = G2_omega(scale, cosmo['OMEGA_M'], cosmo['OMEGA_L'])
        d1 = g1 * cosmo_colossus.Dzplus0(z)
        hz = hubble_a(scale, cosmo['H0'], cosmo['OMEGA_M'], cosmo['OMEGA_L'])
        aHf1 = scale * hz * F_omega(scale, cosmo['OMEGA_M'], cosmo['OMEGA_L'])
        aHf2 = scale * hz * F2_omega(scale, cosmo['OMEGA_M'], cosmo['OMEGA_L'])
        pk_input = pk_z0 * d1**2

        log.info(
            'z = %.1f: D1 = %.6f, g2 = %.6f, aHf1 = %.4f, aHf2 = %.4f',
            z, d1, g2, aHf1, aHf2,
        )
        growth_table[z] = GrowthData(
            redshift=z, d1=d1, g1=g1, g2=g2,
            aHf1=aHf1, aHf2=aHf2,
            pk_input=pk_input, kh_camb=kh_camb,
        )

    return growth_table


def run_lpt(
    pos_grid: ArrayF,
    delta_k: ArrayC,
    nvox: ArrayI,
    dk: float,
    growth: GrowthData,
    lpt_order: int,
    method: str,
    *,
    compensate: bool | None = None,
) -> tuple[ArrayF, ArrayF]:
    r'''Run an LPT displacement and return perturbed positions and velocities.

    Parameters
    ----------
    pos_grid : ndarray, shape (N, 3)
        Lagrangian (unperturbed) particle positions [Mpc/h].
    delta_k : ndarray
        Fourier-space overdensity field.
    nvox : ndarray, shape (3,)
        Number of voxels in each dimension of the grid `(Nx, Ny, Nz)`.
    dk : float
        Physical cell size [Mpc/h].
    growth : GrowthData
        Pre-computed growth factors for the target redshift.
    lpt_order : {1, 2}
        LPT order.
    method : str
        Interpolation method: ``'ngp'``, ``'cic'``, ``'tsc'``.
    compensate : bool or None
        Whether to apply the MAS compensation kernel. If ``None``
        (default), uses the on-grid rule: only TSC genuinely smooths
        on-node particles, so compensate only for TSC.
        [CITE: Sefusatti et al. 2016, Section 3.1]

    Returns
    -------
    xpert : ndarray, shape (N, 3)
        Perturbed (Eulerian) positions [Mpc/h].
    vpert : ndarray, shape (N, 3)
        Peculiar velocities [km/s].
    '''
    if lpt_order not in (1, 2):
        raise ValueError(f'Unsupported LPT order: {lpt_order}.')

    if compensate is None:
        # On a regular SC lattice, CIC interpolation is exact and
        # compensation is unnecessary. TSC genuinely smooths even
        # on-grid particles, so we compensate only for TSC.
        compensate = (method == 'tsc')

    lpt_func = lpt1 if lpt_order == 1 else lpt2
    lpt_kwargs = dict(
        x=pos_grid, delta_k=delta_k, nvox=nvox, dk=dk,
        g1=growth.g1, aHf1=growth.aHf1,
        compensate=compensate, method=method,
    )
    if lpt_order == 2:
        lpt_kwargs.update(g2=growth.g2, aHf2=growth.aHf2)

    return lpt_func(**lpt_kwargs)


def generate_field(
    nvox: ArrayI,
    dk: float,
    growth: GrowthData,
    seed: int,
    *,
    paired: bool = False,
    counter_phase: bool = False,
) -> ArrayC:
    r'''Generate a white-noise realisation and build :math:`\delta_k`.

    The Angulo & Pontzen (2016) paired-fixed scheme has two components:

    * fixed amplitude - each Fourier mode is set to exactly
      :math:`\sqrt{P(k)}` instead of being drawn from a Rayleigh
      distribution.
    * counter-phasing - the second member of the pair is obtained
      by a global sign flip :math:`\delta_k \to -\delta_k`.

    Setting ``paired=True`` activates fixed-amplitude normalisation
    for both members. The ``counter_phase`` flag selects which
    member of the pair is generated: ``False`` for the primary,
    ``True`` for the counter-phased field.

    Parameters
    ----------
    nvox : ndarray, shape (3,)
        Number of voxels in each dimension of the grid `(Nx, Ny, Nz)`.
    dk : float
        Physical cell size [Mpc/h].
    growth : GrowthData
        Pre-computed cosmological quantities (provides ``kh_camb`` and
        ``pk_input``).
    seed : int
        White-noise RNG seed.
    paired : bool
        If ``True``, enable the paired-fixed scheme: amplitudes are
        fixed and the inner loop should call this function once with
        ``counter_phase=False`` and once with ``counter_phase=True``.
    counter_phase : bool
        If ``True``, produce the counter-phased (sign-flipped) member.
        Only meaningful when ``paired=True``; ignored otherwise.

    Returns
    -------
    delta_k : ndarray
        Fourier-space overdensity field.
    '''
    field = white_noise(nvox=nvox, seed=seed)
    return generate_delta_k(
        growth.kh_camb, growth.pk_input, nvox, dk,
        field=field, fixed=paired, paired=counter_phase,
    )


def band_average_reference_pk(
    kh_input: ArrayF,
    pk_input: ArrayF,
    nvox: ArrayI,
    dk: float,
    boxsize: ArrayF,
    *,
    kmin: float | None,
    kmax: float | None,
    dk_bin: float | None,
) -> tuple[ArrayF, ArrayF, ArrayI]:
    '''Band-average the input theory P(k) over the exact discrete Fourier modes of a box.'''
    kmod = fourier_kmod(nvox, dk, hermitian=True)
    pk_spline = CubicSpline(
        np.log(kh_input), np.log(pk_input), extrapolate=False,
    )
    pk_grid = np.zeros_like(kmod)
    mask = kmod > 0.0
    pk_grid[mask] = np.exp(pk_spline(np.log(kmod[mask])))
    return _bin_isotropic_modes(
        pk_grid, kmod, nvox, boxsize, dk,
        kmin=kmin, kmax=kmax, dk_bin=dk_bin,
    )


def load_glass_tile(path: str | Path, tile_side: float) -> ArrayF:
    '''Read a periodic cubic glass and rescale it to a cube of side ``tile_side``.'''
    with h5py.File(path, 'r') as handle:
        positions = handle['PartType1/Coordinates'][:].astype(np.float64)
        glass_side = float(np.atleast_1d(handle['Header'].attrs['BoxSize'])[0])
    return np.mod(positions, glass_side) * (tile_side / glass_side)


def iter_tiled_load(
    tile: ArrayF,
    tile_side: float,
    tiles: int,
    tiles_per_chunk: int,
) -> Iterator[ArrayF]:
    r'''Yield a load of ``tiles``\ :sup:`3` copies of ``tile``, a few copies at a time.

    The copies fill a box of side ``tiles * tile_side``, and the chunks together hold every particle of that load exactly once.
    '''
    index = np.arange(tiles, dtype=np.float64)
    shifts = tile_side * np.stack(
        np.meshgrid(index, index, index, indexing='ij'), axis=-1,
    ).reshape(-1, 3)
    for start in range(0, len(shifts), tiles_per_chunk):
        chunk = shifts[start:start + tiles_per_chunk]
        yield (tile[None, :, :] + chunk[:, None, :]).reshape(-1, 3)


class InterlacedPowerMeter:
    r'''Deposit particles on a mesh in batches and measure their P(k).

    The estimator is the one of :func:`stepsic.pk.measure_pk` with CIC assignment, deconvolution, interlacing and no shot-noise subtraction. Particles are added batch by batch, so the load never has to be in memory at once.
    '''

    _BATCH = 1 << 22

    def __init__(self, nvox: ArrayI, boxsize: ArrayF) -> None:
        self.nvox = np.asarray(nvox, dtype=np.int64)
        self.boxsize = np.asarray(boxsize, dtype=np.float64)
        self.dk = float(self.boxsize[0] / self.nvox[0])
        self._kernel = KERNELS['cic']()
        # The second mesh is shifted by half a cell along every axis.
        self._geometries = [
            GridGeometry(
                nvox=tuple(int(n) for n in self.nvox), boxsize=self.boxsize,
                vox_offset=0.5, origin=origin, periodic=True,
            )
            for origin in (0.0, -0.5 * self.dk)
        ]
        self._grids = [
            np.zeros(tuple(self.nvox), dtype=np.float64) for _ in range(2)
        ]

    def add(self, x: ArrayF) -> None:
        '''Deposit positions ``x`` (any real coordinates; the box is periodic).'''
        nx, ny, nz = (int(n) for n in self.nvox)
        offsets = range(self._kernel.support)
        for start in range(0, len(x), self._BATCH):
            batch = x[start:start + self._BATCH]
            for grid, geometry in zip(self._grids, self._geometries):
                g = geometry.pos_to_grid(batch)
                cell = np.floor(g).astype(np.int64)
                weights = [
                    self._kernel.weights(g[:, axis] - cell[:, axis])
                    for axis in range(3)
                ]
                cell += self._kernel.base_offset
                flat = grid.reshape(-1)
                for ox in offsets:
                    ix = np.mod(cell[:, 0] + ox, nx) * (ny * nz)
                    for oy in offsets:
                        ixy = ix + np.mod(cell[:, 1] + oy, ny) * nz
                        wxy = weights[0][ox] * weights[1][oy]
                        for oz in offsets:
                            np.add.at(
                                flat, ixy + np.mod(cell[:, 2] + oz, nz),
                                wxy * weights[2][oz],
                            )

    def _overdensity_k(self) -> ArrayC:
        grid = self._grids.pop(0)
        grid /= grid.mean()
        grid -= 1.0
        return scipy.fft.rfftn(grid, workers=-1, overwrite_x=True)

    def power(self, *, kmax: float) -> tuple[ArrayF, ArrayF, ArrayI]:
        '''Return ``(k, P, nmodes)`` of everything deposited; the meter is spent afterwards.'''
        delta_k = self._overdensity_k()
        shifted_k = self._overdensity_k()
        # Undo the half-cell shift of the second mesh, then average the two.
        wavevectors = fourier_vectors(self.nvox, self.dk, hermitian=True)
        for axis, k_axis in enumerate(wavevectors):
            shape = [1, 1, 1]
            shape[axis] = -1
            shifted_k *= np.exp(0.5j * self.dk * k_axis).reshape(shape)
        delta_k += shifted_k
        del shifted_k
        delta_k *= 0.5
        factors = compensation_factors(
            self.nvox, self.dk, self.boxsize, method='cic',
        )
        for axis, factor in enumerate(factors):
            shape = [1, 1, 1]
            shape[axis] = -1
            delta_k *= factor.reshape(shape)
        return measure_pk_from_delta_k(
            delta_k, self.boxsize, self.nvox,
            kmin=None, kmax=kmax, dk_bin=None,
        )


def parse_boxsize(lbox_arg: list[float]) -> ArrayF:
    '''Interpret the ``--Lbox`` argument as a (3,) array.

    Accepts one value (cubic box) or exactly three values (anisotropic).

    Parameters
    ----------
    lbox_arg : list of float
        Raw ``--Lbox`` CLI values.

    Returns
    -------
    ndarray of shape (3,)
        Box dimensions [Mpc/h].

    Raises
    ------
    ValueError
        If the number of values is not 1 or 3, or if any are ≤ 0.
    '''
    if len(lbox_arg) == 1:
        boxsize = np.array(lbox_arg * 3, dtype=np.float64)
    elif len(lbox_arg) == 3:
        boxsize = np.array(lbox_arg, dtype=np.float64)
    else:
        raise ValueError(
            f'--Lbox expects 1 or 3 values, got {len(lbox_arg)}.'
        )
    if np.any(boxsize <= 0.0):
        raise ValueError(f'Box dimensions must be positive, got {boxsize}.')
    return boxsize


def z_tag(z: float) -> str:
    '''Format a redshift for use in archive keys (e.g. ``z31``).

    Integers are rendered without a decimal point; others get one
    decimal place.
    '''
    if np.isclose(z, round(z)):
        return f'z{int(round(z))}'
    return f'z{z:.1f}'


def add_cosmology_args(parser: argparse.ArgumentParser) -> None:
    '''Add the shared cosmology and LPT arguments to a parser.'''
    parser.add_argument(
        '--Lbox', type=float, nargs='+', default=[500.0],
        help='Box dimensions [Mpc/h]. One value makes a cube; '
             'three values set Lx, Ly, and Lz.',
    )
    parser.add_argument(
        '--nmesh', type=int, nargs='+', default=[128],
        help='Grid resolution(s): cells along the shortest box dimension.',
    )
    parser.add_argument(
        '--lpt', type=int, nargs='+', default=[2], choices=[1, 2],
        help='LPT order(s).',
    )
    parser.add_argument(
        '--z', type=float, nargs='+', default=[31.0], dest='redshifts',
        help='Target redshift(s).',
    )
    parser.add_argument(
        '--method', type=str, nargs='+', default=['cic'],
        choices=['ngp', 'cic', 'tsc'],
        help='Mass-assignment scheme(s) for LPT interpolation.',
    )
    parser.add_argument(
        '--seed', type=int, default=137,
        help='Base RNG seed.',
    )
    parser.add_argument(
        '--nreal', type=int, default=1,
        help='Number of independent realisations to average. '
             'Seeds: [seed, seed+1, ..., seed+nreal-1].',
    )
    parser.add_argument(
        '--paired', action='store_true',
        help='Enable Angulo & Pontzen (2016) paired-fixed averaging.',
    )
    parser.add_argument(
        '--kmax-frac-ny', type=float, default=0.8,
        dest='kmax_fraction_nyquist',
        help='Maximum measured k as a fraction of the mesh Nyquist frequency.',
    )
    parser.add_argument(
        '-o', '--output', type=str, default='validation_data.npz',
        help='Output .npz file path.',
    )
