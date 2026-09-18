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
from stepsic._typing import ComplexField, FloatVec3, IntVec3, RealField, Seed

import logging

import scipy
import numpy as np
from scipy.interpolate import CubicSpline

from stepsic.rng import philox_complex_gaussian

log = logging.getLogger(__name__)

# Target byte size for real-valued x-slabs used in spectral calculations.
_SPECTRAL_SLAB_BUDGET_BYTES = 64_000_000


def _x_slabs(shape, dtype):
    """Yield x-axis slices within the spectral scratch-memory budget."""
    plane_values = int(np.prod(shape[1:], dtype=np.int64))
    plane_bytes = max(1, plane_values * np.dtype(dtype).itemsize)
    depth = max(1, _SPECTRAL_SLAB_BUDGET_BYTES // plane_bytes)
    for start in range(0, int(shape[0]), depth):
        yield slice(start, min(start + depth, int(shape[0])))


def wrap(x: RealField, boxsize: FloatVec3) -> RealField:
    """
    Shift centered coordinates by half a box and wrap to ``[0, L)``.

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
    """
    return np.mod(x+boxsize/2, boxsize)


def cubic_voxels(nmesh: int, boxsize: FloatVec3) -> tuple[IntVec3, float]:
    """
    Choose even mesh dimensions with cubic cells.

    The shortest box dimension sets the cell spacing; other dimensions
    are rounded up to an even number of cells.

    Parameters
    ----------
    nmesh : int
        Number of voxels in the shortest dimension.
    boxsize : float or tuple of float
        Box dimensions as an array ``(Lx, Ly, Lz)`` in a common length unit.

    Returns
    -------
    nvox : ndarray of shape (3,)
        The number of voxels in each dimension of the grid `(Nx, Ny, Nz)`.
    dk : float
        Physical cell spacing in the same length unit as the box.
    """
    nmesh = nmesh + nmesh % 2
    nvox = np.ceil(boxsize / (np.min(boxsize) / nmesh)).astype(int)
    nvox = (nvox + nvox % 2).astype(int)  # Ensure even number of voxels
    dk = np.min(boxsize) / np.min(nvox)
    return nvox, dk


def fourier_vectors(
        nvox: IntVec3,
        dk: float,
        hermitian: bool = False,
        dtype: np.dtype = np.float64
) -> tuple[RealField, RealField, RealField]:
    r"""
    Construct the 1D per-axis wavevectors of a Fourier-space grid.

    Parameters
    ----------
    nvox : tuple of int
        The number of voxels in each dimension of the grid `(Nx, Ny, Nz)`.
    dk : float
        Physical cell spacing in the same length unit as the box.
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
    """
    kx = (scipy.fft.fftfreq(nvox[0]) * 2 * np.pi / dk).astype(dtype)
    ky = (scipy.fft.fftfreq(nvox[1]) * 2 * np.pi / dk).astype(dtype)
    if hermitian:
        kz = (scipy.fft.rfftfreq(nvox[2]) * 2 * np.pi / dk).astype(dtype)
    else:
        kz = (scipy.fft.fftfreq(nvox[2]) * 2 * np.pi / dk).astype(dtype)
    return kx, ky, kz


def fourier_kmod(
        nvox: IntVec3,
        dk: float,
        hermitian: bool = False,
        dtype: np.dtype = np.float64
) -> RealField:
    r"""
    Compute only the wavevector magnitude :math:`|\mathbf{k}|`.

    Parameters
    ----------
    nvox : tuple of int
        The number of voxels in each dimension of the grid `(Nx, Ny, Nz)`.
    dk : float
        Physical cell spacing in the same length unit as the box.
    hermitian : bool
        If `True`, assume Hermitian symmetry (reduced last axis).
    dtype : np.dtype
        The data type of the returned magnitude array.

    Returns
    -------
    kmod : ndarray
        The wavevector magnitude at each grid point,
        :math:`\sqrt{k_x^2 + k_y^2 + k_z^2}`.
    """
    kx, ky, kz = fourier_vectors(nvox, dk, hermitian=hermitian, dtype=dtype)
    kmod = np.sqrt(
        kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2)
    return kmod.astype(dtype, copy=False)


def white_noise(nvox: IntVec3, seed: Seed = None, dtype: np.dtype = np.float64) -> ComplexField:
    r"""
    Return a complex Gaussian array W(k) on the rfftn grid
    (Nx, Ny, Nz//2+1), obeying Hermitian constraints that guarantee
    irfftn(W) is real.

    Modes are drawn directly in Fourier space with Philox4x64-10. The
    counter is the grid-independent signed integer mode triple
    (ix, iy, iz), stored as uint64 two's-complement words, with counter
    word 3 reserved at zero. The Philox key is (seed, 0). If seed is
    None, a random uint64 seed is drawn and logged.

    The returned field is W(k) = sqrt(prod(nvox)) * a(k), where a(k) is
    a unit complex Gaussian. Thus W(k) / sqrt(prod(nvox)) is identical
    for every shared integer mode across resolutions with the same seed.
    Each grid's own Nyquist planes and DC mode are set exactly to zero.

    Parameters
    ----------
    nvox : tuple of int
        Number of voxels in each dimension (Nx, Ny, Nz).
    seed : int or None, optional
        Random seed for reproducibility. If None, a random uint64 seed is
        logged and used for this field.
    dtype : dtype
        Real precision controlling the returned complex dtype. float32
        returns complex64; float64 returns complex128.

    Returns
    -------
    w_k : ndarray
        Complex white-noise values on the rfftn grid.
    """
    nvox = np.asarray(nvox, dtype=np.int64)
    if nvox.shape != (3,):
        raise ValueError('nvox must contain exactly three axes')
    assert np.all(nvox % 2 == 0), 'white_noise requires even nvox'
    nx, ny, nz = map(int, nvox)

    if seed is None:
        seed = np.random.default_rng().integers(
            0, np.iinfo(np.uint64).max, dtype=np.uint64
        )
        log.info('white_noise seed: %d', int(seed))
    key = np.array((np.uint64(seed), np.uint64(0)), dtype=np.uint64)

    ix = np.rint(scipy.fft.fftfreq(nx) * nx).astype(np.int64)
    iy = np.rint(scipy.fft.fftfreq(ny) * ny).astype(np.int64)
    iz = np.rint(scipy.fft.rfftfreq(nz) * nz).astype(np.int64)
    nz_half = nz // 2 + 1

    w_k = np.empty((nx, ny, nz_half), dtype=np.result_type(dtype, np.complex64))
    norm = float(np.sqrt(np.prod(nvox, dtype=np.float64)))

    iy_grid, iz_grid = np.meshgrid(iy, iz, indexing='ij')
    ncounter = ny * nz_half
    counters = np.empty((ncounter, 4), dtype=np.uint64)
    counters[:, 1] = iy_grid.ravel().astype(np.uint64)
    counters[:, 2] = iz_grid.ravel().astype(np.uint64)
    counters[:, 3] = 0

    for i, ix_mode in enumerate(ix):
        counters[:, 0] = np.array(ix_mode, dtype=np.int64).astype(np.uint64)
        w_k[i] = (philox_complex_gaussian(counters, key) * norm).reshape(
            ny, nz_half
        )

    w_k[nx // 2, :, :] = 0.0
    w_k[:, ny // 2, :] = 0.0
    w_k[:, :, nz // 2] = 0.0

    ix_signed = ix[:, None]
    iy_signed = iy[None, :]
    canonical = (ix_signed > 0) | ((ix_signed == 0) & (iy_signed > 0))
    flip_i = (-np.arange(nx)) % nx
    flip_j = (-np.arange(ny)) % ny
    plane = w_k[:, :, 0]
    mirrored = np.conj(plane[flip_i[:, None], flip_j[None, :]])
    w_k[:, :, 0] = np.where(canonical, plane, mirrored)
    w_k[0, 0, 0] = 0.0
    return w_k


def generate_delta_k(
        kh,
        pk,
        nvox: IntVec3,
        dk: float,
        *,
        field: ComplexField | None = None,
        seed: Seed = None,
        fixed: bool = False,
        paired: bool = False,
        dtype: np.dtype = np.float64
) -> ComplexField:
    r"""
    Generate Fourier-space overdensity modes from a power spectrum.

    Parameters
    ----------
    kh : ndarray
        Positive wavenumbers in h/Mpc, in strictly increasing order.
    pk : ndarray
        Positive power spectrum values in (Mpc/h)^3 at the target redshift.
    nvox : tuple of int
        The number of voxels in each dimension of the grid `(Nx, Ny, Nz)`.
    dk : float
        Physical cell spacing in Mpc/h.
    field : ndarray
        Complex white-noise modes with the normalization returned by
        :func:`white_noise`, of shape ``(Nx, Ny, Nz//2+1)``. If omitted,
        generate them with ``seed``.
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
        Real precision used for the spectrum grid and generated white noise.

    Returns
    -------
    delta_k : ndarray
        A 3D complex-valued array of shape `(Nx, Ny, Nz//2+1)` representing
        the Fourier modes of the overdensity field.
    """
    dtype = np.dtype(dtype)
    nvox = np.asarray(nvox, dtype=np.int64)
    shape = (int(nvox[0]), int(nvox[1]), int(nvox[2] // 2 + 1))
    kx, ky, kz = fourier_vectors(nvox, dk, hermitian=True, dtype=dtype)
    spline = CubicSpline(np.log(kh), np.log(pk), extrapolate=True)

    if field is None:
        field = white_noise(nvox=nvox, seed=seed, dtype=dtype)
    field = np.asarray(field)
    if field.shape != shape:
        raise ValueError(f'field must have shape {shape}')

    complex_dtype = np.result_type(field.dtype, dtype)
    delta_k = np.empty(shape, dtype=complex_dtype)
    norm = np.sqrt(np.prod(nvox))

    for sl in _x_slabs(shape, dtype):
        amplitude = np.sqrt(
            kx[sl, None, None]**2
            + ky[None, :, None]**2
            + kz[None, None, :]**2
        )
        nonzero_k = amplitude > 0.0
        np.log(amplitude, out=amplitude, where=nonzero_k)
        spectrum = np.exp(spline(amplitude))
        amplitude[...] = spectrum
        amplitude[~nonzero_k] = 0.0
        np.divide(amplitude, dk**3, out=amplitude)
        np.sqrt(amplitude, out=amplitude)

        if fixed:
            field_amplitude = np.abs(field[sl])
            nonzero_field = field_amplitude > 0.0
            delta_k[sl].fill(0.0)
            np.divide(
                field[sl], field_amplitude, out=delta_k[sl],
                where=nonzero_field,
            )
            np.multiply(delta_k[sl], norm * amplitude, out=delta_k[sl])
            del field_amplitude, nonzero_field
        else:
            np.multiply(field[sl], amplitude, out=delta_k[sl])
        del amplitude, nonzero_k, spectrum

    delta_k[0, 0, 0] = 0.0  # set DC=0 (mean density) as we only need fluctuations

    if paired:
        np.negative(delta_k, out=delta_k)
    return delta_k
