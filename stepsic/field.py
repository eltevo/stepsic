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
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#*****************************************************************************#

from __future__ import annotations
from stepsic._typing import ComplexField, FloatVec3, RealField, Seed

import logging

import scipy
import numpy as np
from scipy.interpolate import CubicSpline
from tabulate import tabulate

from stepsic.rng import RNG

log = logging.getLogger(__name__)


def wrap(x: RealField, boxsize: FloatVec3) -> RealField:
    '''
    Wraps the coordinates in ``x`` to be within the periodic box defined
    by ``boxsize`` to the inverval :math:`(0, L]`

    Parameters
    ----------
    x : ndarray of shape (N, M)
        The coordinates to wrap.
    boxsize : ndarray of shape (M,)
        The size of the periodic box in each dimension.

    Returns
    -------
    wrapped : ndarray
        The wrapped coordinates.
    '''
    return np.mod(x+boxsize/2, boxsize)


def create_grid(nvox: FloatVec3, dk: float) -> tuple[RealField, RealField]:
    '''
    Create a regular grid for the simulation box.

    Parameters
    ----------
    nvox : tuple of int
        The number of voxels in each dimension of the simulation box.
    dk : float
        The uniform step size in each dimension, calculated as the length
        of the shortest dimension divided by the number of voxels in
        that dimension.

    Returns
    -------
    particles : ndarray of shape (N, 3)
        The particle positions in the simulation box, where N is the total
        number of particles (voxels).
    coords : ndarray of shape (3, Nx, Ny, Nz)
        The grid coordinates in each dimension, where Nx, Ny, Nz are the
        number of voxels in each dimension.
    '''
    mesh = tuple(np.arange(-(n-1)*dk/2, n*dk/2, dk) for n in nvox)
    xx, yy, zz = np.meshgrid(*mesh, indexing='ij')
    particles = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    return particles, np.array((xx, yy, zz))


def create_particles(npart: int, boxsize: FloatVec3, seed: Seed = None) -> RealField:
    r'''
    Create a set of particles uniformly distributed in a cubic box.

    Parameters
    ----------
    npart : int
        The number of particles to create.
    boxsize : float or list of float
        Box dimensions in [Mpc]. Can be a scalar for a cubical box or
        an array in the form of `(Lx, Ly, Lz)` for a rectangular cuboid.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    particles : ndarray of shape (npart, 3)
        Particle positions in the simulation box.
    '''
    rng = RNG(seed=seed)
    return rng.uniform(size=(npart, 3), seed=None) * np.array(boxsize)


def cubic_voxels(nmesh: int, boxsize: FloatVec3) -> tuple[FloatVec3, float]:
    '''
    Defines a rectangular cuboid mesh with the specified number of
    voxels in each dimensions, ensuring that the voxels are cubic.
    The function calculates the number of voxels in each dimension
    `(Nx, Ny, Nz)` based on the shortest dimension of the cuboid and
    scales the other dimensions accordingly.

    Parameters
    ----------
    nmesh : int
        Number of voxels in the shortest dimension.
    boxsize : float or tuple of float
        Box dimensions in [Mpc]. Can be a scalar for a cubical box or
        an array in the form of `(Lx, Ly, Lz)` for a rectangular cuboid.

    Returns
    -------
    nvox : ndarray of shape (3,)
        The number of voxels in each dimension of the grid `(Nx, Ny, Nz)`.
    dk : float
        The uniform step size in each dimension, calculated as the length
        of the shortest dimension divided by the number of voxels in
        that dimension.
    '''
    nvox = np.ceil(boxsize / (np.min(boxsize) / nmesh)).astype(int)
    nvox = (nvox + nvox % 2).astype(int)  # Ensure even number of voxels
    dk = np.min(boxsize) / np.min(nvox)
    #log.info('mesh: Nx={}, Ny={}, Nz={}; step size: {}'.format(*nvox, dk))
    return nvox, dk


def anisotropic_voxels(nmesh: int, boxsize: FloatVec3) -> tuple[FloatVec3, float]:
    '''
    Construct a mesh with isotropic physical cell size, anchored to the
    shortest axis (z by convention).

    Parameters
    ----------
    nmesh : int
        Number of voxels in the shortest dimension.
    boxsize : float or tuple of float
        Box dimensions in [Mpc]. Can be a scalar for a cubical box or
        an array in the form of `(Lx, Ly, Lz)` for a rectangular cuboid.

    Returns
    -------
    nvox : ndarray of shape (3,)
        The number of voxels in each dimension of the grid `(Nx, Ny, Nz)`.
    dk : float
        
    '''
    dk = np.min(boxsize) / nmesh
    nvox = np.rint(boxsize / dk).astype(int)
    # enforce exact consistency (avoid drift)
    if not np.allclose(nvox * dk, boxsize, rtol=0, atol=1e-10):
        raise ValueError(
            "Boxsize not compatible with isotropic cell size.\n"
            f"boxsize={boxsize}, dk={dk}, nvox={nvox}"
        )
    return nvox, dk


def fourier_grid(
        nvox: FloatVec3,
        dk: float,
        hermitian: bool = False,
        dtype: np.dtype = np.float64
) -> tuple[RealField, RealField]:
    r'''
    Construct a 3D Fourier space grid.

    This function generates a three-dimensional array of wavevector
    components (``kvec``) and computes the corresponding magnitude
    (``kmod``) for a cubic grid with ``nmesh`` points per side within
    a box of size ``boxsize``.

    The grid is then constructed using the FFT frequencies:
    - For the first two dimensions, the full set of FFT frequencies is
      computed using ``scipy.fft.fftfreq``.
    - For the third dimension, if the input field is real-valued (i.e.
      if Hermitian symmetry is assumed), the reduced set of frequencies
      is computed using ``scipy.fft.rfftfreq``.

    Parameters
    ----------
    nvox : tuple of int
        The number of voxels in each dimension of the grid `(Nx, Ny, Nz)`.
    dk : float
        The uniform step size in each dimension, calculated as the length
        of the shortest dimension divided by the number of voxels in
        that dimension.
    hermitian : bool
        If `True`, assume the field has Hermitian symmetry (i.e. it is
        real-valued) and use the reduced FFT along the last dimension.
    dtype : np.dtype
        The data type of the generated arrays (e.g., `np.float32` or `np.float64`).

    Returns
    -------
    kvec : ndarray
        A three-dimensional array of wavevector components with shape:
          - :math:`(3, {N_x}, {N_y}, {N_z}//2+1)` if ``hermitian`` is `True`.
          - :math:`(3, {N_x}, {N_y}, {N_z})` if ``hermitian`` is `False`.
        Each sub-array corresponds to the ``x``, ``y``, or ``z`` component
        of the wavevector.

    kmod : ndarray
        The magnitude of the wavevector at each grid point, computed as
        :math:`\|\mathbf{k}\| = \sqrt{k_x^2 + k_y^2 + k_z^2}`.
    '''
    kx = scipy.fft.fftfreq(nvox[0]) * 2 * np.pi / dk
    ky = scipy.fft.fftfreq(nvox[1]) * 2 * np.pi / dk
    if hermitian:
        kz = scipy.fft.rfftfreq(nvox[2]) * 2 * np.pi / dk
    else:
        kz = scipy.fft.fftfreq(nvox[2]) * 2 * np.pi / dk
    kvec = np.array(np.meshgrid(kx, ky, kz, indexing='ij'), dtype=dtype)
    kmod = np.linalg.norm(kvec, axis=0).astype(dtype)
    return kvec, kmod


def white_noise(nvox: FloatVec3, seed: Seed = None, dtype: np.dtype = np.float64,
                ref_nvox: int = None) -> RealField:
    r'''
    Return a complex Gaussian array :math:`W(k)` on the ``rfftn()`` grid
    `(Nx, Ny, Nz//2+1)`, obeying Hermitian constraints that guarantee
    :math:`\delta(x)` reconstructed with ``irfftn()`` is real.

    The field is generated in the real space.

    Parameters
    ----------
    nvox : tuple of int
        Number of voxels in each dimension `(Nx, Ny, Nz)`.
    dk : float
        The uniform step size in each dimension, calculated as the length
        of the shortest dimension divided by the number of voxels in
        that dimension.
    seed : int or None, optional
        Random seed for reproducibility. If `None`, uses the default RNG.

    Returns
    -------
    w_k : ndarray
        3D array of white noise values.
    '''
    rng = RNG(seed=seed)
    if ref_nvox:
        # ── Resolution-independent phases ──────────────────────────────────────
        # The default branch below draws in CONFIGURATION space on the nvox^3 grid, so
        # `rng.normal(size=nvox)` fills a differently shaped array when the mesh changes
        # and the SAME seed yields a completely different realization.  Two runs at
        # different NMESH are then incomparable: measured cross-correlation of the z=0
        # density fields fell from ~0.95 (same NMESH) to ~0 (NMESH 256 vs 1024).
        #
        # With ref_nvox set, the field is drawn once on the ref_nvox^3 mesh and its
        # k-modes are cropped to nvox^3, so every mesh <= ref_nvox is a strict subset of
        # one realization -- the standard "fixed phases across resolution" construction.
        # (M/N)^{3/2} restores the per-mode variance of a native M^3 draw, since numpy's
        # unnormalised rfftn gives E|w_k|^2 = N^3.
        N = int(ref_nvox)
        M = int(np.atleast_1d(nvox)[0])
        if M > N:
            raise ValueError(f'PHASE_REF_NMESH ({N}) must be >= NMESH ({M})')
        w_ref = scipy.fft.rfftn(rng.normal(size=(N, N, N), seed=seed).astype(dtype),
                                workers=-1)
        if M < N:
            sel = np.r_[0:M//2, N-M//2:N]
            w_k = w_ref[np.ix_(sel, sel, np.arange(M//2+1))] * (M/N)**1.5
            # cropping breaks Hermitian consistency on the Nyquist planes; restore it
            w_k = scipy.fft.rfftn(scipy.fft.irfftn(w_k, s=(M, M, M), workers=-1),
                                  workers=-1)
        else:
            w_k = w_ref
    else:
        w_k = scipy.fft.rfftn(rng.normal(size=nvox, seed=seed).astype(dtype), workers=-1)
    w_k[0, 0, 0] = 0.0  # set DC=0 (mean density) as we only need fluctuations
    return w_k


def generate_delta_k(
        kh,
        pk,
        nvox: FloatVec3,
        dk: float,
        *,
        field: RealField = None,
        seed: Seed = None,
        fixed: bool = False,
        paired: bool = False,
        complementary: bool = False,
        dtype: np.dtype = np.float64
) -> ComplexField:
    r'''
    Generates the Fourier modes of an arbitrary input field from a
    given power spectrum.

    Parameters
    ----------
    kh : ndarray
        1D array of wavenumbers `k`, in [Mpc].
    pk : ndarray
        1D array of the matter power spectrum `P(k)` at the initial redshift.
    nvox : tuple of int
        The number of voxels in each dimension of the grid `(Nx, Ny, Nz)`.
    dk : float
        The uniform step size in each dimension, calculated as the length
        of the shortest dimension divided by the number of voxels in
        that dimension.
    field : ndarray
        Fourier transform of the real-valued overdensity field
        :math:`\delta(\mathbf{x})` defined on a regular grid, where
        :math:`\mathbf{x}` are the comoving coordinates.
    seed : int
        The seed for the random number generator.
    fixed : bool
        If `True`, replace the random Rayleigh-distributed amplitudes of
        the density modes by the target amplitudes implied by the power
        spectrum, preserving only the phases.
    paired : bool
        If `True`, generates a paired field by applying a global sign
        flip to the Fourier-space density field.
    dtype : np.dtype
        The data type of the generated field (e.g., `np.float32` or `np.float64`).

    Returns
    -------
    delta_k : ndarray
        A 3D complex-valued array of shape `(Nx, Ny, Nz//2+1)` representing
        the Fourier modes of the overdensity field.
    '''
    _, kmod = fourier_grid(nvox, dk, hermitian=True, dtype=dtype)

    # interpolate the power spectrum in log-log space
    spline = CubicSpline(np.log(kh), np.log(pk), extrapolate=True)
    pk_grid = np.zeros_like(kmod, dtype=dtype)
    mask = kmod > 0
    if np.any(mask):
        ktarget_log = np.log(kmod[mask])
        pk_grid[mask] = np.exp(spline(ktarget_log))

    if field is None:
        field = white_noise(nvox=nvox, seed=seed, dtype=dtype)

    # Sirko 2005; Bagla & Padmanabhan 1997; Klypin & Holtzman 1997
    target_A = np.sqrt(pk_grid / dk**3, dtype=dtype)
    if fixed:
        # Flips phase and sets amplitude to 1 for every mode.
        # Angulo & Pontzen 2016
        amp = np.abs(field)
        phase = np.zeros_like(field)
        phase[amp > 0.0] = field[amp > 0.0] / amp[amp > 0.0]
        delta_k = phase * (np.sqrt(np.prod(nvox)) * target_A)
    else:
        delta_k = field * target_A

    if complementary:
        # ── Complementary initial conditions ───────────────────────────────────
        # Racz, Kiessling, Csabai & Szapudi (2022), arXiv:2210.15077.
        # Rescale each |k| shell so that the PAIR (this run + the original) averages
        # to the target spectrum, their eq. (9):
        #       P_C(k) = 2 P_target(k) - P_IC(k).
        # With r = P_IC/P_target this is a real per-shell factor on the original field:
        #       compensated   (r < 2):  delta_C = delta * sqrt(2/r - 1)   -> P_C = 2P_t - P_IC
        #       uncompensated (r >= 2): delta_C = delta / r               -> P_C = P_t^2/P_IC
        # The second branch is the fallback where exact compensation would demand a
        # negative power; it still pulls the shell towards the target without the
        # beat-coupling of an over-corrected mode.  Phases are preserved here and then
        # flipped below (the sign flip leaves P(k) untouched).
        kf_shell = np.min(2*np.pi/(np.asarray(nvox)*dk))
        ish = np.rint(kmod/kf_shell).astype(np.int64)
        pw = np.abs(delta_k)**2
        tg = target_A**2
        nb = int(ish.max()) + 1
        # delta_k = field * target_A with E|field|^2 = prod(nvox) (unnormalised rfftn),
        # so the target shell power carries that same factor -- cf. the `fixed` branch
        # above, which scales by sqrt(prod(nvox)).
        num = np.bincount(ish.ravel(), weights=pw.ravel(), minlength=nb)
        den = np.bincount(ish.ravel(), weights=tg.ravel(), minlength=nb) * np.prod(nvox)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(den > 0, num/den, 1.0)
        r = np.where(np.isfinite(r) & (r > 0), r, 1.0)
        fac = np.where(r < 2.0, np.sqrt(np.maximum(2.0/r - 1.0, 0.0)), 1.0/r)
        delta_k = delta_k * fac[ish].astype(delta_k.dtype)

    delta_k[0, 0, 0] = 0.0  # set DC=0 (mean density) as we only need fluctuations

    if paired or complementary:
        delta_k = -delta_k
    return delta_k


def create_nres_mass_map(
        n_grid_samples: int,
        mass_list: RealField,
        M_box: float,
        boxsize: FloatVec3
) -> tuple[RealField, RealField]:
    '''
    Creates a lookup table for the number of voxels per mass bin
    for a variable resolution grid in a regular StePS simulation.

    Parameters
    ----------
    n_grid_samples : int
        Number of grids with different resolutions.
    mass_list : ndarray
        Array containing the sorted unique particle masses.
    M_box : float
        Total mass in the simulation box (in 1e11 Msol).
    boxsize : ndarray
        Box dimensions in [Mpc]. Can be a scalar for a cubical box or
        an array in the form of `(Lx, Ly, Lz)` for a rectangular cuboid.

    Returns
    -------
    nres_tab : ndarray of shape (n_grid_samples,)
        Array containing the number of resolution elements for each grid.
    mass_tab : ndarray of shape (n_grid_samples,)
        Array containing the mass values corresponding to each grid.
    '''
    nres_list = np.min(boxsize) // np.cbrt(np.prod(boxsize) * mass_list / M_box)
    idx = np.linspace(
        0, mass_list.size - 1, n_grid_samples, endpoint=True, dtype=int)
    
    # Populate lookup table by starting with the outermost mass bin
    nres_tab = nres_list[idx[::-1]]
    mass_tab = mass_list[idx[::-1]]

    log.info('The generated resolution-mass map:') #The full nresx x nresy x nresz grid resolution is printed here
    nvox = np.zeros((n_grid_samples, 3), dtype=int)
    for i in range(n_grid_samples):
        nvox[i], dk = cubic_voxels(nres_tab[i], boxsize)
    print(tabulate([*zip(nvox[:,0], nvox[:,1], nvox[:,2], mass_tab)],
                   headers=['Resolution_x', 'Resolution_y', 'Resolution_z', 'Mass [1e11 Msol/h]'],
                   floatfmt=('.0f', '.0f', '.0f', '.6f')))
    return nres_tab, mass_tab