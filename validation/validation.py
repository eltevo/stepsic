#!/usr/bin/env python3
'''
Consolidates cosmological parameter definitions, CAMB + Colossus
initialisation, growth-factor computation, and matplotlib styling across
validation runs for the stepsic paper.

Usage
-----
::

    from validation import PLANCK2018, GrowthData, init_cosmology

    growth_table = init_cosmology([31.0, 15.0], lbox_max=500.0)
    gd = growth_table[31.0]
    print(gd.d1, gd.aHf1)
'''

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import logging
from dataclasses import dataclass
from typing import Sequence, TypeAlias, Union

import numpy as np
from numpy.typing import NDArray

from stepsic.cosmology import (
    CAMBCosmology,
    ColossusCosmology,
    F2_omega,
    F_omega,
    hubble_a,
)
from stepsic.field import (
    generate_delta_k,
    white_noise,
)
from stepsic.lpt import lpt1, lpt2

log = logging.getLogger(__name__)


ArrayF: TypeAlias = NDArray[np.float64]
ArrayI: TypeAlias = NDArray[np.int64]
ArrayC: TypeAlias = Union[NDArray[np.complex128], NDArray[np.complex64]]


# Planck 2018 (TT,TE,EE+lowE+lensing, Table 2 of Aghanim et al. 2020)
PLANCK2018: dict[str, float] = {
    'H0':      67.742,
    'OMEGA_M': 0.3099,
    'OMEGA_B': 0.048891,
    'OMEGA_L': 0.6901,
    'NS':      0.96822,
    'AS':      2.1064e-9,
    'SIGMA8':  0.8105,
    'YHE':     0.245421,
    'MNU':     0.06,       # eV, total neutrino mass
    'NNU':     3.046,      # effective number of neutrino species
    'ZREI':    7.89,       # reionization redshift
    'TCMB':    2.7255,     # K, CMB temperature
    'W0':     -1.0,
    'WA':      0.0,
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
        :math:`-(3/7)\,\Omega_m^{-1/143}`
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
        Cosmological parameters. Defaults to :data:`PLANCK2018`.

    Returns
    -------
    dict mapping float → GrowthData
        One entry per unique redshift.
    '''
    if cosmo is None:
        cosmo = PLANCK2018

    h = cosmo['H0'] / 100.0

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
    ombh2 = cosmo['OMEGA_B'] * h**2
    omch2 = (cosmo['OMEGA_M'] - cosmo['OMEGA_B']) * h**2
    cosmo_camb = CAMBCosmology(
        H0=cosmo['H0'],
        ombh2=ombh2,
        omch2=omch2,
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
    g2 = -3.0 / 7.0 * cosmo['OMEGA_M'] ** (-1.0 / 143.0)

    growth_table: dict[float, GrowthData] = {}
    for z in sorted(set(redshifts)):
        scale = 1.0 / (z + 1.0)
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
    r'''Run an LPT displacement and return perturbed positions + velocities.

    Parameters
    ----------
    pos_grid : ndarray, shape (N, 3)
        Lagrangian (unperturbed) particle positions [Mpc/h].
    delta_k : ndarray
        Fourier-space overdensity field.
    nvox : ndarray, shape (3,)
        Number of grid cells per dimension.
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
        Number of grid cells per dimension.
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


def load_archive(path: str) -> dict:
    '''Load an ``.npz`` file and return a plain dict of arrays.'''
    npz = np.load(path, allow_pickle=True)
    return dict(npz)


def setup_matplotlib() -> None:
    '''Configure matplotlib for publication-quality A&A figures.

    Uses the Okabe-Ito colour-blind-safe palette, LaTeX text rendering,
    and inward-facing ticks. Safe to call multiple times.

    .. note::
        Importing matplotlib is deferred to this function so that the
        run scripts can ``import validate_common`` without pulling in
        the entire plotting stack.
    '''
    from cycler import cycler
    import matplotlib.pyplot as plt

    okabe_ito = [
        '#000000', '#E69F00', '#56B4E9', '#009E73',
        '#F0E442', '#0072B2', '#D55E00', '#CC79A7',
    ]

    custom_settings: dict = {
        'axes.prop_cycle': cycler(color=okabe_ito),
        'figure.facecolor': '#ffffff',
        'axes.facecolor': '#ffffff',
        'axes.edgecolor': '0.3',
        'axes.linewidth': 1,
        'axes.grid': False,
        'grid.color': '0.7',
        'grid.linestyle': ':',
        'grid.alpha': 0.6,
        'figure.dpi': 150,
        'savefig.dpi': 300,
    }
    text_settings: dict = {
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'axes.unicode_minus': False,
    }
    for t in ('xtick', 'ytick'):
        custom_settings[f'{t}.direction'] = 'in'
        custom_settings[f'{t}.bottom' if t == 'xtick' else f'{t}.left'] = True
        custom_settings[f'{t}.top' if t == 'xtick' else f'{t}.right'] = True
        custom_settings[f'{t}.color'] = '0.3'
        for m in ('major', 'minor'):
            custom_settings[f'{t}.{m}.width'] = 1
            custom_settings[f'{t}.{m}.size'] = 6 if m == 'major' else 3
    plt.rcParams.update(custom_settings)
    plt.rcParams.update(text_settings)

def histogram(
    values: ArrayF,
    nbins: int,
    *,
    range: tuple[float, float] | None = None,
) -> tuple[ArrayF, ArrayF, ArrayF]:
    '''Compute a histogram and return ``(bin_centres, counts, bin_widths)``.

    Parameters
    ----------
    values : ndarray
        Input data.
    nbins : int
        Number of bins.
    range : tuple of float, optional
        The lower and upper range of the bins. If not provided, the range is
        simply ``(values.min(), values.max())``.

    Returns
    -------
    centres : ndarray
    counts : ndarray (float64)
    widths : ndarray
    '''
    counts, edges = np.histogram(values, bins=nbins, range=range)
    centres = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    return centres, counts.astype(np.float64), widths


def z_tag(z: float) -> str:
    '''Format a redshift for use in archive keys (e.g. ``z31``).

    Integers are rendered without a decimal point; others get one
    decimal place.
    '''
    if np.isclose(z, round(z)):
        return f'z{int(round(z))}'
    return f'z{z:.1f}'


def add_cosmology_args(parser: argparse.ArgumentParser) -> None:
    '''Add the standard cosmology/LPT CLI arguments to parser.

    Covers: ``--Lbox``, ``--nmesh``, ``--lpt``, ``--z``, ``--method``,
    ``--seed``, ``--paired``, ``--kmax-frac-ny``, ``--nreal``, ``-o``.
    Individual scripts can override defaults after calling this.
    '''
    parser.add_argument(
        '--Lbox', type=float, nargs='+', default=[500.0],
        help='Box dimensions [Mpc/h]. One value -> cubic; '
             'three values -> Lx Ly Lz.',
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