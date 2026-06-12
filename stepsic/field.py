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


def fourier_vectors(
        nvox: FloatVec3,
        dk: float,
        hermitian: bool = False,
        dtype: np.dtype = np.float64
) -> tuple[RealField, RealField, RealField]:
    r'''
    Construct the 1D per-axis wavevectors of a Fourier-space grid.

    Parameters
    ----------
    nvox : tuple of int
        The number of voxels in each dimension of the grid `(Nx, Ny, Nz)`.
    dk : float
        The uniform step size in each dimension, calculated as the length
        of the shortest dimension divided by the number of voxels in
        that dimension.
    hermitian : bool
        If `True`, use the reduced ``rfftfreq`` frequencies along the last
        axis (length `Nz//2+1`); otherwise the full ``fftfreq`` set.
    dtype : np.dtype
        The data type of the returned vectors (e.g. `np.float32` or
        `np.float64`).

    Returns
    -------
    kx, ky, kz : ndarray
        1D wavevector components of length `Nx`, `Ny`, and
        `Nz//2+1` (if ``hermitian``) or `Nz`, respectively.
    '''
    kx = (scipy.fft.fftfreq(nvox[0]) * 2 * np.pi / dk).astype(dtype)
    ky = (scipy.fft.fftfreq(nvox[1]) * 2 * np.pi / dk).astype(dtype)
    if hermitian:
        kz = (scipy.fft.rfftfreq(nvox[2]) * 2 * np.pi / dk).astype(dtype)
    else:
        kz = (scipy.fft.fftfreq(nvox[2]) * 2 * np.pi / dk).astype(dtype)
    return kx, ky, kz


def fourier_kmod(
        nvox: FloatVec3,
        dk: float,
        hermitian: bool = False,
        dtype: np.dtype = np.float64
) -> RealField:
    r'''
    Compute only the wavevector magnitude :math:`|\mathbf{k}|`.

    Parameters
    ----------
    nvox : tuple of int
        The number of voxels in each dimension of the grid `(Nx, Ny, Nz)`.
    dk : float
        The uniform step size in each dimension.
    hermitian : bool
        If `True`, assume Hermitian symmetry (reduced last axis).
    dtype : np.dtype
        The data type of the returned magnitude array.

    Returns
    -------
    kmod : ndarray
        The wavevector magnitude at each grid point,
        :math:`\sqrt{k_x^2 + k_y^2 + k_z^2}`.
    '''
    kx, ky, kz = fourier_vectors(nvox, dk, hermitian=hermitian, dtype=dtype)
    kmod = np.sqrt(
        kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2)
    return kmod.astype(dtype, copy=False)


def white_noise(nvox: FloatVec3, seed: Seed = None, dtype: np.dtype = np.float64) -> RealField:
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
    kmod = fourier_kmod(nvox, dk, hermitian=True, dtype=dtype)

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

    delta_k[0, 0, 0] = 0.0  # set DC=0 (mean density) as we only need fluctuations

    if paired:
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