#*****************************************************************************#
#  stepsic - An initial condition generator for                               #
#           STEreographically Projected cosmological Simulations              #
#    Copyright (C) 2017-2026 Balazs Pal, Gabor Racz                           #
#                                                                             #
#    This program is free software; you can redistribute it and/or modify     #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation; either version 2 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This program is distributed in the hope that it will be useful,          #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the             #
#    GNU General Public License for more details.                             #
#*****************************************************************************#

from __future__ import annotations

import logging
from typing import Tuple

import scipy
import numpy as np
from numpy.typing import NDArray

from stepsic._typing import ComplexField, FloatVec3, IntVec3, RealField
from stepsic.field import cubic_voxels, fourier_kmod, fourier_vectors
from stepsic.interpolation import compensation_factors, deposit_field

log = logging.getLogger(__name__)


def _cell_size(nvox: NDArray[np.int64], boxsize: NDArray[np.float64]) -> float:
    r'''
    Compute the isotropic cell size for a cubic-voxel mesh.

    Parameters
    ----------
    nvox : ndarray of shape (3,)
        Number of grid cells in each dimension.
    boxsize : ndarray of shape (3,)
        Physical box size [Mpc/h].

    Returns
    -------
    dk : float
        Uniform cell size [Mpc/h], computed as
        ``min(boxsize) / min(nvox)``.

    Raises
    ------
    ValueError
        If the per-axis cell sizes deviate by more than 0.1 % from the
        isotropic value (indicating a non-cubic voxel grid).
    '''
    dcell = boxsize / nvox
    dk = np.min(boxsize) / np.min(nvox)
    if not np.allclose(dcell, dk, rtol=1e-3):
        raise ValueError(
            f'Non-cubic voxels detected: per-axis cell sizes '
            f'{dcell} deviate from isotropic dk={dk:.6f}. '
            f'The periodic P(k) estimator requires cubic voxels. '
            f'Use cubic_voxels() to construct the mesh, or provide '
            f'a commensurate nvox.'
        )
    return float(dk)


def _deposit_overdensity_k(
    x: RealField,
    nvox: IntVec3,
    boxsize: FloatVec3,
    mass: RealField | None,
    method: str,
    origin: FloatVec3,
) -> ComplexField:
    r'''
    Deposit particles and return the Fourier-space overdensity field.

    Parameters
    ----------
    x : ndarray of shape (N, 3)
        Particle positions in physical coordinates [Mpc/h].
    nvox : ndarray of shape (3,)
        Number of grid cells in each dimension.
    boxsize : float or ndarray of shape (3,)
        Physical size of the periodic box [Mpc/h].
    mass : ndarray of shape (N,) or None
        Particle masses (unit weights if ``None``).
    method : str
        Mass-assignment kernel name.
    origin : float or ndarray of shape (3,)
        Physical coordinate of the grid origin corner.

    Returns
    -------
    delta_k : complex ndarray of shape (Nx, Ny, Nz//2+1)
        Fourier-space overdensity from ``rfftn``.
    '''
    N_cells = int(np.prod(nvox))
    rho_grid = deposit_field(
        x, nvox=nvox, boxsize=boxsize, values=mass,
        periodic=True, method=method,
        vox_offset=0.5, origin=origin,
    )
    rho_mean = rho_grid.sum() / N_cells
    if rho_mean <= 0:
        raise ValueError(
            'Mean deposited density is zero or negative. '
            'Check that particle positions are within [0, boxsize).'
        )
    delta_x = rho_grid / rho_mean - 1.0
    log.debug(f'Overdensity: min={delta_x.min():.4f}, max={delta_x.max():.4f}')
    return scipy.fft.rfftn(delta_x, workers=-1)


def _weight_rfft_modes(
    nvox: IntVec3,
    shape: tuple[int, ...],
) -> RealField:
    r'''
    Return conjugate-pair weights for an ``rfftn``-stored Fourier grid.

    Notes
    -----
    For a real-valued field, the full complex Fourier transform obeys
    Hermitian symmetry, so most stored modes in an ``rfftn`` output
    represent both ``+k_z`` and ``-k_z`` and therefore carry weight 2.
    The self-conjugate planes ``k_z = 0`` and, for even ``nvox[2]``, the
    Nyquist plane ``k_z = k_\mathrm{Ny}`` are stored only once and carry
    weight 1.
    '''
    if len(shape) != 3:
        raise ValueError(
            f'Expected a 3D Fourier grid shape, got {shape!r}.'
        )

    nz_stored = nvox[2] // 2 + 1
    if shape[2] != nz_stored:
        raise ValueError(
            'Incompatible rfftn shape for the provided mesh: '
            f'shape[2]={shape[2]}, expected {nz_stored}.'
        )

    weight = 2.0 * np.ones(shape, dtype=np.float64)
    weight[:, :, 0] = 1.0
    if nvox[2] % 2 == 0:
        weight[:, :, nz_stored - 1] = 1.0
    return weight


def _build_k_bin_edges(
    boxsize: FloatVec3,
    dk: float,
    *,
    kmin: float | None = None,
    kmax: float | None = None,
    dk_bin: float | None = None,
) -> RealField:
    r'''
    Construct isotropic shell bin edges for power-spectrum estimation.

    Parameters
    ----------
    boxsize : ndarray of shape (3,)
        Periodic box size [Mpc/h].
    dk : float
        Isotropic cell size [Mpc/h].
    kmin, kmax, dk_bin : float or None
        Requested binning configuration. When omitted, the defaults are
        the fundamental mode, the Nyquist mode, and the fundamental-mode
        spacing, respectively.

    Returns
    -------
    ndarray
        Bin edges in wavenumber [h/Mpc].

    Notes
    -----
    The fundamental mode is defined as ``2*pi / max(boxsize)``, i.e. the
    smallest fundamental frequency across all axes.  This is the most
    conservative default for isotropic binning of an anisotropic box. The
    default Nyquist cutoff is ``pi / dk``.
    '''
    boxsize = np.broadcast_to(boxsize, (3,)).astype(np.float64)

    k_fund = 2.0 * np.pi / np.max(boxsize)
    k_ny = np.pi / dk

    if kmin is None: kmin = k_fund
    if kmax is None: kmax = k_ny
    if dk_bin is None: dk_bin = k_fund

    if kmin <= 0.0:
        raise ValueError(f'kmin must be positive, got {kmin}.')
    if kmax <= kmin:
        raise ValueError(f'kmax must be larger than kmin, got {kmin} >= {kmax}.')
    if dk_bin <= 0.0:
        raise ValueError(f'dk_bin must be positive, got {dk_bin}.')

    bin_edges = np.arange(
        kmin - dk_bin / 2.0, kmax + dk_bin, dk_bin, dtype=np.float64,
    )
    if bin_edges.size < 2:
        raise ValueError('Power-spectrum binning produced fewer than one bin.')
    return bin_edges


def _bin_isotropic_modes(
    values: RealField,
    kmod: RealField,
    nvox: IntVec3,
    boxsize: FloatVec3,
    dk: float,
    *,
    kmin: float | None = None,
    kmax: float | None = None,
    dk_bin: float | None = None,
) -> Tuple[RealField, RealField, NDArray[np.int64]]:
    r'''
    Bin a per-mode scalar field over isotropic Fourier shells.

    Parameters
    ----------
    values : ndarray
        Scalar quantity defined on the stored ``rfftn`` Fourier grid.
    kmod : ndarray
        Magnitude of the Fourier wavevector at each stored grid point
        [h/Mpc]. Must have the same shape as ``values``.
    nvox : ndarray of shape (3,)
        Number of cells in each dimension of the underlying real-space grid.
    boxsize : ndarray of shape (3,)
        Periodic box size [Mpc/h].
    dk : float
        Isotropic cell size [Mpc/h].
    kmin, kmax, dk_bin : float or None
        Shell-binning configuration.

    Returns
    -------
    k_centres : ndarray
        Weighted mean wavenumber in each populated shell [h/Mpc].
    values_binned : ndarray
        Weighted shell average of ``values``.
    nmodes : ndarray
        Number of independent Fourier modes in each populated shell.
    '''
    if values.shape != kmod.shape:
        raise ValueError(
            'values and kmod must have identical shapes, got '
            f'{values.shape!r} and {kmod.shape!r}.'
        )

    bin_edges = _build_k_bin_edges(
        boxsize, dk, kmin=kmin, kmax=kmax, dk_bin=dk_bin,
    )
    n_bins = len(bin_edges) - 1

    weight = _weight_rfft_modes(nvox, kmod.shape)
    k_flat = np.asarray(kmod, dtype=np.float64).ravel()
    values_flat = np.asarray(values, dtype=np.float64).ravel()
    weight_flat = weight.ravel()

    bin_idx = np.digitize(k_flat, bin_edges) - 1
    valid = (
        (bin_idx >= 0)
        & (bin_idx < n_bins)
        & (k_flat > 0.0)
        & np.isfinite(values_flat)
    )

    idx_valid = bin_idx[valid]
    k_valid = k_flat[valid]
    values_valid = values_flat[valid]
    w_valid = weight_flat[valid]

    nmodes = np.bincount(
        idx_valid, weights=w_valid, minlength=n_bins,
    ).astype(np.int64)
    k_sum = np.bincount(
        idx_valid, weights=w_valid * k_valid, minlength=n_bins,
    ).astype(np.float64)
    values_sum = np.bincount(
        idx_valid, weights=w_valid * values_valid, minlength=n_bins,
    ).astype(np.float64)

    populated = nmodes > 0
    k_centres = np.full(n_bins, np.nan, dtype=np.float64)
    values_binned = np.full(n_bins, np.nan, dtype=np.float64)
    k_centres[populated] = k_sum[populated] / nmodes[populated]
    values_binned[populated] = values_sum[populated] / nmodes[populated]
    return k_centres[populated], values_binned[populated], nmodes[populated]


def measure_pk_from_delta_k(
    delta_k: ComplexField,
    boxsize: FloatVec3,
    nvox: IntVec3,
    *,
    subtract_shot: bool = False,
    N_part: int | None = None,
    kmin: float | None = None,
    kmax: float | None = None,
    dk_bin: float | None = None,
) -> Tuple[RealField, RealField, NDArray[np.int64]]:
    r'''
    Measure the isotropic power spectrum directly from a Fourier overdensity field.

    Parameters
    ----------
    delta_k : complex ndarray
        Fourier-space overdensity from ``numpy.fft.rfftn``.
    boxsize : float or ndarray of shape (3,)
        Periodic box size [Mpc/h].
    nvox : ndarray of shape (3,)
        Number of real-space cells underlying ``delta_k``.
    subtract_shot : bool
        Whether to subtract a Poisson shot-noise floor. Disabled by default
        because direct Fourier fields do not in general correspond to a
        particle sampling process.
    N_part : int or None
        Number of particles used to sample the field when shot noise should be
        subtracted. Required when ``subtract_shot=True``.
    kmin, kmax, dk_bin : float or None
        Shell-binning configuration.

    Returns
    -------
    k_centres, pk, nmodes : tuple of ndarray
        Binned isotropic power spectrum and independent mode counts.
    '''
    boxsize = np.broadcast_to(boxsize, (3,)).astype(np.float64)
    nvox = np.asarray(nvox, dtype=np.int64)

    expected_shape = (int(nvox[0]), int(nvox[1]), int(nvox[2] // 2 + 1))
    if delta_k.shape != expected_shape:
        raise ValueError(
            'delta_k has incompatible shape for the provided mesh: '
            f'got {delta_k.shape!r}, expected {expected_shape!r}.'
        )

    if subtract_shot and (N_part is None or N_part <= 0):
        raise ValueError(
            'N_part must be provided and positive when subtract_shot=True.'
        )

    dk = _cell_size(nvox, boxsize)
    kmod = fourier_kmod(nvox, dk, hermitian=True)
    N_cells = int(np.prod(nvox))
    V_box = np.prod(boxsize)

    pk_grid = (V_box / N_cells**2) * np.abs(delta_k)**2

    k_centres, pk_binned, nmodes = _bin_isotropic_modes(
        pk_grid, kmod, nvox, boxsize, dk,
        kmin=kmin, kmax=kmax, dk_bin=dk_bin,
    )

    if subtract_shot:
        P_shot = V_box / N_part
        pk_binned = pk_binned - P_shot
        log.info(f'Subtracted Poisson shot noise: P_shot = {P_shot:.4e} (Mpc/h)^3')

    return k_centres, pk_binned, nmodes


def measure_pk(
    x: NDArray[np.floating],
    boxsize: FloatVec3,
    nmesh: int | None = None,
    *,
    nvox: IntVec3 | None = None,
    mass: RealField | None = None,
    method: str = 'cic',
    deconvolve: bool = True,
    interlace: bool = False,
    subtract_shot: bool = True,
    kmin: float | None = None,
    kmax: float | None = None,
    dk_bin: float | None = None,
) -> Tuple[RealField, RealField, NDArray[np.int64]]:
    r'''
    Measure the isotropic power spectrum from a particle distribution
    in a periodic box.

    Parameters
    ----------
    x : ndarray of shape (N, 3)
        Particle positions in physical coordinates [Mpc/h].
        Must be within ``[0, boxsize)`` (i.e. box-wrapped, not
        centred).  If your particles are centred at the origin
        (``[-L/2, L/2]``), shift them by ``+L/2`` before calling.
    boxsize : float or ndarray of shape (3,)
        Physical size of the periodic box [Mpc/h].  A scalar is
        broadcast to all three dimensions.
    nmesh : int or None
        Number of grid cells per shortest dimension. Used to construct
        ``nvox`` via :func:`~stepsic.field.cubic_voxels` when ``nvox`` is
        not supplied.
    nvox : ndarray of shape (3,), optional
        Explicit mesh shape. When provided, no internal call to
        :func:`~stepsic.field.cubic_voxels` is made. The voxels must be
        cubic (isotropic cell size); use :func:`~stepsic.field.cubic_voxels`
        to guarantee this.
    mass : ndarray of shape (N,) or None
        Particle masses. If ``None``, all particles are assigned unit
        weight (pure number-density field).  For equal-mass particles
        the result is the same either way.
    method : str
        Mass-assignment scheme: ``'ngp'``, ``'cic'``, or ``'tsc'``
        (default ``'cic'``).  Should match the interpolation method
        used during IC generation for a fair comparison.
    deconvolve : bool
        If ``True`` (default), divide out the Fourier-space window of
        the mass-assignment kernel so that the recovered P(k) is not
        suppressed near the Nyquist frequency. When ``interlace=True``,
        deconvolution is always applied (the leading aliasing term that
        normally contaminates naive deconvolution is cancelled by the
        interlacing).
    interlace : bool
        If ``True``, use the interlacing technique of Sefusatti et al.
        (2016) to suppress aliasing. Two density assignments are
        performed - one at the normal grid origin and one shifted by
        half a cell in every direction - then averaged in Fourier space
        with the appropriate phase correction. This doubles the cost
        of the deposit step but allows accurate deconvolution up to
        :math:`\sim 0.9\,k_\mathrm{Ny}`.
    subtract_shot : bool
        If ``True`` (default), subtract the Poisson shot-noise floor
        :math:`P_\mathrm{shot} = V_\mathrm{box} / N_\mathrm{part}`.
        Set to ``False`` to keep the shot-noise contribution, e.g. when
        measuring glass ICs where the intrinsic noise is sub-Poissonian.
    kmin : float or None
        Minimum wavenumber for binning [h/Mpc].  Defaults to the
        fundamental mode :math:`k_f = 2\pi / L_\mathrm{max}`.
    kmax : float or None
        Maximum wavenumber for binning [h/Mpc].  Defaults to the
        Nyquist wavenumber of the shortest axis.
    dk_bin : float or None
        Bin width in wavenumber [h/Mpc].  Defaults to the fundamental
        mode :math:`k_f`.

    Returns
    -------
    k_centres : ndarray of shape (Nbins,)
        Mean wavenumber in each bin [h/Mpc].
    pk : ndarray of shape (Nbins,)
        Estimated power spectrum :math:`\hat{P}(k)` [:math:`(\mathrm{Mpc}/h)^3`].
    nmodes : ndarray of shape (Nbins,)
        Number of independent Fourier modes per bin.

    Notes
    -----
    *   This estimator assumes a periodic box. For spherical or
        cylindrical StePS geometries, use the FKP estimator from StePS
        (``StePS_Pk.py``) which handles non-trivial survey windows.
    *   For variable-mass (multi-resolution) particle loads, ``mass``
        should be provided so that the deposited field is the true
        matter density rather than just the number density.
    *   When ``interlace=False`` and ``deconvolve=True``, the naive
        :math:`\mathrm{sinc}^{-p}` correction amplifies aliased power
        near the Nyquist frequency. Restrict ``kmax`` to
        :math:`\lesssim 2/3\,k_\mathrm{Ny}` or use ``interlace=True``
        for reliable results at higher :math:`k`.

    References
    ----------
    *   Sefusatti et al. 2016, MNRAS 460, 3624; interlacing technique.
    *   Jing 2005, ApJ 620, 559; standard MAS deconvolution and aliasing.
    *   Hockney & Eastwood 1988, §5-3; fundamentals of particle-mesh
        methods and their Fourier-space windows.

    Examples
    --------
    Measure the power spectrum of a periodic grid IC at z = 49:

    >>> import numpy as np
    >>> from stepsic.field import cubic_voxels, white_noise, generate_delta_k
    >>> from stepsic.lpt import lpt1
    >>> from stepsic.analysis import measure_pk
    >>> Lbox = np.array([500.0, 500.0, 500.0])  # Mpc/h
    >>> nmesh = 64
    >>> nvox, dk = cubic_voxels(nmesh, Lbox)
    >>> # ... generate IC with lpt1 ...
    >>> # Shift particles from [-L/2, L/2] to [0, L] if needed:
    >>> pos_shifted = perturbed_pos + Lbox / 2
    >>> k, pk, modes = measure_pk(
    ...     pos_shifted, Lbox, nmesh,
    ...     method='cic', interlace=True,
    ... )
    '''
    boxsize = np.broadcast_to(boxsize, (3,)).astype(np.float64)
    N_part = x.shape[0]

    # --- Mesh setup ---
    if nvox is None:
        if nmesh is None:
            raise ValueError('Either nmesh or nvox must be provided.')
        nvox, dk = cubic_voxels(nmesh, boxsize)
    else:
        nvox = np.asarray(nvox, dtype=np.int64)
        if nvox.shape != (3,):
            raise ValueError(f'nvox must have shape (3,), got {nvox.shape!r}.')
        dk = _cell_size(nvox, boxsize)

    N_cells = int(np.prod(nvox))
    V_box = np.prod(boxsize)

    log.info(
        f'Measure P(k): mesh {nvox[0]}x{nvox[1]}x{nvox[2]}, '
        f'cell size: {dk:.6f} Mpc/h, '
        f'{N_part} particles'
        f'{", interlaced" if interlace else ""}'
    )

    # --- Density deposit & Fourier transform ---
    # Interlacing shifts the mesh by half a cell in each dimension.
    # For cubic voxels the shift is dk/2 along every axis.
    # Reference: Sefusatti et al. 2016, §2.2
    if interlace:
        # Deposit 1: normal grid
        delta_k_A = _deposit_overdensity_k(
            x, nvox, boxsize, mass, method, origin=0,
        )

        # Deposit 2: grid shifted by half a cell in each direction.
        # Shifting the origin by -dk/2 is equivalent to
        # shifting all particles by +dk/2 relative to the grid.
        delta_k_B = _deposit_overdensity_k(
            x, nvox, boxsize, mass, method, origin=-0.5 * dk,
        )

        # Undo the shift in Fourier space. A real-space shift by -dk/2
        # introduces a phase factor exp(-i k*dx). Undo it with
        # exp(+i dk/2 * (kx+ky+kz)).
        # Reference: Sefusatti et al. 2016, eq. 11
        kx, ky, kz = fourier_vectors(nvox, dk, hermitian=True)
        phase_shift = 0.5 * dk * (
            kx[:, None, None] + ky[None, :, None] + kz[None, None, :])
        delta_k_B *= np.exp(1j * phase_shift)

        # Average the two deposits, as this cancels the leading aliasing
        # term of the MAS kernel (the n=+-1 aliases in eq. 18 of Jing 2005).
        delta_k = 0.5 * (delta_k_A + delta_k_B)

    else:
        delta_k = _deposit_overdensity_k(
            x, nvox, boxsize, mass, method, origin=0,
        )

    # --- MAS deconvolution ---
    if deconvolve:
        # Naive sinc deconvolution is always safe with interlacing.
        # Separable kernel applied per axis in place.
        wx, wy, wz = compensation_factors(nvox, dk, boxsize, method=method)
        delta_k *= wx[:, None, None]
        delta_k *= wy[None, :, None]
        delta_k *= wz[None, None, :]

    # |k| magnitude for isotropic shell binning (one transient full array)
    kmod = fourier_kmod(nvox, dk, hermitian=True)

    # --- Power spectrum estimation ---
    # P(k) = V / N_cells^2 * |delta_k|^2
    # The k=0 mode is meaningless here (DC = sum of overdensity ~ 0
    # numerically) but is filtered out during shell binning (k > 0).
    pk_grid = (V_box / N_cells**2) * np.abs(delta_k)**2

    k_centres, pk_binned, nmodes = _bin_isotropic_modes(
        pk_grid, kmod, nvox, boxsize, dk,
        kmin=kmin, kmax=kmax, dk_bin=dk_bin,
    )

    if subtract_shot:
        P_shot = V_box / N_part
        pk_binned = pk_binned - P_shot
        log.info(f'Subtracted Poisson shot noise: P_shot = {P_shot:.4e} (Mpc/h)^3')

    log.info(
        f'Measure P(k): {len(k_centres)} bins, '
        f'k = [{k_centres[0]:.4f}, {k_centres[-1]:.4f}] h/Mpc'
    )
    return k_centres, pk_binned, nmodes