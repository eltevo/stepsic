#*******************************************************************************#
#  stepsic - An initial condition generator for                                 #
#           STEreographically Projected cosmological Simulations                #
#    Copyright (C) 2017-2026 Balazs Pal, Gabor Racz                             #
#                                                                               #
#    This program is free software; you can redistribute it and/or modify       #
#    it under the terms of the GNU General Public License as published by       #
#    the Free Software Foundation; either version 2 of the License, or          #
#    (at your option) any later version.                                        #
#                                                                               #
#    This program is distributed in the hope that it will be useful,            #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              #
#    GNU General Public License for more details.                               #
#*******************************************************************************#

from __future__ import annotations

import logging

import scipy
import numpy as np

from stepsic._typing import ComplexField, IntVec3, RealField
from stepsic.field import _x_slabs, fourier_vectors
from stepsic.interpolation import _interpolate_field_chunks, compensation_factors

log = logging.getLogger(__name__)

# Number of particle rows used per displacement and velocity statistics batch.
_STATISTICS_PARTICLES_PER_CHUNK = 1_000_000


def _poisson_solve(
    source_k: ComplexField,
    wavevectors: tuple[RealField, RealField, RealField],
    dtype: np.dtype,
    *,
    out: ComplexField | None = None,
) -> ComplexField:
    """Return ``-source_k / |k|^2`` using bounded real scratch slabs."""
    if out is None:
        out = np.empty(source_k.shape, dtype=np.result_type(source_k.dtype, dtype))
    kx, ky, kz = wavevectors
    for sl in _x_slabs(source_k.shape, dtype):
        k2 = (
            kx[sl, None, None]**2
            + ky[None, :, None]**2
            + kz[None, None, :]**2
        )
        if sl.start == 0:
            k2[0, 0, 0] = np.inf
        np.divide(source_k[sl], k2, out=out[sl])
        np.negative(out[sl], out=out[sl])
        del k2
    return out


def _apply_compensation(
    field_k: ComplexField,
    factors: tuple[RealField, RealField, RealField] | None,
) -> None:
    if factors is None:
        return
    wx, wy, wz = factors
    field_k *= wx[:, None, None]
    field_k *= wy[None, :, None]
    field_k *= wz[None, None, :]


def _real_gradient(
    potential_k: ComplexField,
    wavevector: RealField,
    multiplier: complex,
    scratch_k: ComplexField,
    nvox: tuple[int, int, int],
    factors: tuple[RealField, RealField, RealField] | None,
) -> RealField:
    np.multiply(potential_k, multiplier * wavevector, out=scratch_k)
    _apply_compensation(scratch_k, factors)
    return scipy.fft.irfftn(
        scratch_k, s=nvox, axes=(0, 1, 2), workers=-1,
    )


def log_lpt(
    x: RealField,
    xpert: RealField,
    vpert: RealField,
    *,
    title: str | None = None
) -> None:
    """Log per-axis displacement and velocity statistics for an LPT step.

    Parameters
    ----------
    x : ndarray of shape (N, 3)
        Unperturbed particle positions [Mpc/h].
    xpert : ndarray of shape (N, 3)
        Perturbed particle positions [Mpc/h].
    vpert : ndarray of shape (N, 3)
        Velocity values as supplied; raw LPT values precede the driver
        conversion to km/s and the snapshot convention.
    title : str, optional
        Label prefix for log messages (e.g. ``'1LPT'``).
    """
    if x.shape[0] == 0:
        raise ValueError('cannot log LPT statistics for an empty particle set')
    xmax = np.zeros(3, dtype=np.float64)
    xsum = np.zeros(3, dtype=np.float64)
    vmax = np.zeros(3, dtype=np.float64)
    vsum = np.zeros(3, dtype=np.float64)
    for start in range(0, x.shape[0], _STATISTICS_PARTICLES_PER_CHUNK):
        stop = min(start + _STATISTICS_PARTICLES_PER_CHUNK, x.shape[0])
        for axis in range(3):
            values = np.subtract(xpert[start:stop, axis], x[start:stop, axis])
            np.abs(values, out=values)
            xmax[axis] = np.maximum(xmax[axis], np.max(values))
            xsum[axis] += np.sum(values, dtype=np.float64)
            values = np.abs(vpert[start:stop, axis])
            vmax[axis] = np.maximum(vmax[axis], np.max(values))
            vsum[axis] += np.sum(values, dtype=np.float64)
    xavg = xsum / x.shape[0]
    vavg = vsum / x.shape[0]
    for i, xi in enumerate(('x', 'y', 'z')):
        log.info(
            f"{title} '{xi}' displacements: "
            f"d_max = {xmax[i]*1e3:.3f} kpc/h; d_avg = {xavg[i]*1e3:.3f} kpc/h ")
        log.info(
            f"{title} 'v{xi}' velocities:   "
            f"v_max = {vmax[i]:.3f} km/s; v_avg = {vavg[i]:.3f} km/s")
    return


def lpt1(
    x: RealField,
    delta_k: ComplexField,
    nvox: IntVec3,
    dk: float,
    g1: float,
    aHf1: float,
    compensate: bool = False,
    method: str = 'cic',
    dtype: np.dtype = np.float64,
) -> tuple[RealField, RealField]:
    r"""
    Apply the Zel'dovich approximation (1LPT) to displace particles.

    Computes first-order displacements and velocities from the Fourier-space
    overdensity field via the gravitational potential, as given by
    Bernardeau et al. (2002):

    .. math::
        \mathbf{\Psi}^{(1)}(\mathbf{k}) =
            i \frac{\mathbf{k}}{|\mathbf{k}|^2} \delta(\mathbf{k}),
        \qquad
        \mathbf{x} = \mathbf{q} + D_1 \mathbf{\Psi}^{(1)},
        \qquad
        \mathbf{v} = D_1 a H f \, \mathbf{\Psi}^{(1)}.

    Parameters
    ----------
    x : ndarray of shape (N, 3)
        Unperturbed (Lagrangian) particle positions [Mpc/h].
    delta_k : ndarray of shape (Nx, Ny, Nz//2+1)
        Fourier-space overdensity field (half-complex).
    nvox : (int, int, int)
        Grid dimensions ``(Nx, Ny, Nz)``.
    dk : float
        Physical cell spacing [Mpc/h].
    g1 : float
        First-order growth coefficient (usually 1).
    aHf1 : float
        Velocity prefactor :math:`a H(a) f(a)` [km/s/Mpc].
    compensate : bool
        Apply MAS deconvolution before interpolation. Required for
        off-grid particle loads (e.g. glass); must be False for
        regular lattices where particles sit on grid nodes.
    method : {'ngp', 'cic', 'tsc'}
        Interpolation kernel (default: ``'cic'``).
    dtype : np.dtype
        Numerical precision for Fourier-space calculations (default: ``np.float64``).

    Returns
    -------
    xpert : ndarray of shape (N, 3)
        Perturbed positions [Mpc/h].
    vpert : ndarray of shape (N, 3)
        Raw velocities: displacement [Mpc/h] times the velocity prefactor.
        The driver divides by ``h`` for km/s and by ``sqrt(a)`` for
        the Gadget snapshot convention.
    """
    dtype = np.dtype(dtype)
    nvox = tuple(map(int, nvox))
    wavevectors = fourier_vectors(nvox, dk, hermitian=True, dtype=dtype)
    K = (
        wavevectors[0][:, None, None],
        wavevectors[1][None, :, None],
        wavevectors[2][None, None, :],
    )
    boxsize = np.asarray(nvox, dtype=dtype) * dk
    factors = (
        compensation_factors(nvox, dk, boxsize, method=method, dtype=dtype)
        if compensate else None
    )

    phi_k = _poisson_solve(delta_k, wavevectors, dtype)
    scratch_k = np.empty_like(phi_k)
    xpert = np.array(x, dtype=np.result_type(x.dtype, dtype), copy=True)
    vpert = np.zeros(x.shape, dtype=dtype)
    for axis in range(3):
        displacement = _real_gradient(
            phi_k, K[axis], -1j, scratch_k, nvox, factors,
        )
        for sl, values in _interpolate_field_chunks(
            x, (displacement,), boxsize, origin=-boxsize / 2,
            method=method, vox_offset=0.5, periodic=True, dtype=dtype,
        ):
            component = values[:, 0]
            xpert[sl, axis] += g1 * component
            vpert[sl, axis] = g1 * aHf1 * component
        del displacement
    return xpert, vpert


def lpt2(
    x: RealField,
    delta_k: ComplexField,
    nvox: IntVec3,
    dk: float,
    g1: float,
    g2: float,
    aHf1: float,
    aHf2: float,
    compensate: bool = False,
    method: str = 'cic',
    dtype: np.dtype = np.float64
) -> tuple[RealField, RealField]:
    r"""
    Apply second-order LPT (2LPT) to displace particles.

    Extends the Zel'dovich approximation with the second-order correction
    as given by Bernardeau et al. (2002):

    .. math::
        \mathbf{x} = \mathbf{q}
            + D_1 \mathbf{\Psi}^{(1)}
            + D_2 \mathbf{\Psi}^{(2)},

    where :math:`\mathbf{\Psi}^{(2)} = \nabla\phi^{(2)}` is sourced by
    the quadratic invariant of the 1LPT deformation tensor:

    .. math::
        S(\mathbf{x}) = \sum_{i<j} \left(
            \Psi^{(1)}_{i,i}\,\Psi^{(1)}_{j,j}
            - \bigl(\Psi^{(1)}_{i,j}\bigr)^2
        \right),
        \qquad
        \phi^{(2)}(\mathbf{k}) = -\frac{S(\mathbf{k})}{|\mathbf{k}|^2}.

    Parameters
    ----------
    x : ndarray of shape (N, 3)
        Unperturbed (Lagrangian) particle positions [Mpc/h].
    delta_k : ndarray of shape (Nx, Ny, Nz//2+1)
        Fourier-space overdensity field (half-complex).
    nvox : (int, int, int)
        Grid dimensions ``(Nx, Ny, Nz)``.
    dk : float
        Physical cell spacing [Mpc/h].
    g1 : float
        First-order growth coefficient (usually 1).
    g2 : float
        Second-order growth coefficient
        (typically :math:`-\tfrac{3}{7}\,\Omega_m^{-1/143}`).
    aHf1 : float
        First-order velocity prefactor :math:`a H f_1` [km/s/Mpc].
    aHf2 : float
        Second-order velocity prefactor :math:`a H f_2` [km/s/Mpc].
    compensate : bool
        Apply MAS deconvolution before interpolation (see `lpt1`).
    method : {'ngp', 'cic', 'tsc'}
        Interpolation kernel (default: ``'cic'``).
    dtype : np.dtype
        Numerical precision for Fourier-space calculations (default: ``np.float64``).

    Returns
    -------
    xpert : ndarray of shape (N, 3)
        Perturbed positions [Mpc/h].
    vpert : ndarray of shape (N, 3)
        Raw velocities: displacement [Mpc/h] times the velocity prefactor.
        The driver divides by ``h`` for km/s and by ``sqrt(a)`` for
        the Gadget snapshot convention.

    See Also
    --------
    lpt1 : First-order (Zel'dovich) displacement only.
    """
    dtype = np.dtype(dtype)
    nvox = tuple(map(int, nvox))
    wavevectors = fourier_vectors(nvox, dk, hermitian=True, dtype=dtype)
    K = (
        wavevectors[0][:, None, None],
        wavevectors[1][None, :, None],
        wavevectors[2][None, None, :],
    )
    boxsize = np.asarray(nvox, dtype=dtype) * dk
    factors = (
        compensation_factors(nvox, dk, boxsize, method=method, dtype=dtype)
        if compensate else None
    )

    phi1_k = _poisson_solve(delta_k, wavevectors, dtype)
    scratch_k = np.empty_like(phi1_k)

    def deformation(i, j):
        np.multiply(phi1_k, K[i], out=scratch_k)
        np.multiply(scratch_k, K[j], out=scratch_k)
        return scipy.fft.irfftn(
            scratch_k, s=nvox, axes=(0, 1, 2), workers=-1,
        )

    dPxx = deformation(0, 0)
    dPyy = deformation(1, 1)
    # The diagonal contribution is dPxx*dPyy + dPzz*(dPxx + dPyy).
    source = np.multiply(dPxx, dPyy)
    np.add(dPxx, dPyy, out=dPxx)
    del dPyy
    dPzz = deformation(2, 2)
    np.multiply(dPxx, dPzz, out=dPxx)
    np.add(source, dPxx, out=source)
    del dPxx, dPzz

    for i, j in ((0, 1), (0, 2), (1, 2)):
        dPij = deformation(i, j)
        np.square(dPij, out=dPij)
        np.subtract(source, dPij, out=source)
        del dPij

    phi2_k = scipy.fft.rfftn(source, workers=-1, overwrite_x=True)
    del source
    _poisson_solve(phi2_k, wavevectors, dtype, out=phi2_k)

    xpert = np.array(x, dtype=np.result_type(x.dtype, dtype), copy=True)
    vpert = np.zeros(x.shape, dtype=dtype)
    for axis in range(3):
        displacement1 = _real_gradient(
            phi1_k, K[axis], -1j, scratch_k, nvox, factors,
        )
        displacement2 = _real_gradient(
            phi2_k, K[axis], 1j, scratch_k, nvox, factors,
        )
        for sl, values in _interpolate_field_chunks(
            x, (displacement1, displacement2), boxsize,
            origin=-boxsize / 2, method=method, vox_offset=0.5,
            periodic=True, dtype=dtype,
        ):
            first = values[:, 0]
            second = values[:, 1]
            xpert[sl, axis] += g1 * first
            xpert[sl, axis] += g2 * second
            vpert[sl, axis] = g1 * aHf1 * first
            vpert[sl, axis] += g2 * aHf2 * second
        del displacement1, displacement2
    return xpert, vpert
