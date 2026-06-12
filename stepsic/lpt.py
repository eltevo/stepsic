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
from typing import Tuple

import scipy
import numpy as np

from stepsic._typing import ComplexField, IntVec3, RealField
from stepsic.field import fourier_vectors
from stepsic.interpolation import compensation_factors, interpolate_field

log = logging.getLogger(__name__)


def log_lpt(
    x: RealField,
    xpert: RealField,
    vpert: RealField,
    *,
    title: str | None = None
) -> None:
    '''Log per-axis displacement and velocity statistics for an LPT step.

    Parameters
    ----------
    x : ndarray of shape (N, 3)
        Unperturbed particle positions [Mpc/h].
    xpert : ndarray of shape (N, 3)
        Perturbed particle positions [Mpc/h].
    vpert : ndarray of shape (N, 3)
        Particle velocities [km/s].
    title : str, optional
        Label prefix for log messages (e.g. ``'1LPT'``).
    '''
    xabs, vabs = np.abs(xpert - x), np.abs(vpert)
    xmax, xavg = np.max(xabs, axis=0), np.mean(xabs, axis=0)
    vmax, vavg = np.max(vabs, axis=0), np.mean(vabs, axis=0)
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
) -> Tuple[RealField, RealField]:
    r'''
    Apply the Zel'dovich approximation (1LPT) to displace particles.

    Computes first-order displacements and velocities from the Fourier-space
    overdensity field via the gravitational potential, as given by
    Bernardeau et al. (2002):

    .. math::
        \mathbf{\Psi}^{(1)}(\mathbf{k}) =
            -i \frac{\mathbf{k}}{|\mathbf{k}|^2} \delta(\mathbf{k}),
        \qquad
        \mathbf{x} = \mathbf{q} + D_1 \mathbf{\Psi}^{(1)},
        \qquad
        \mathbf{v} = a H f \, \mathbf{\Psi}^{(1)}.

    Parameters
    ----------
    x : ndarray of shape (N, 3)
        Unperturbed (Lagrangian) particle positions [Mpc/h].
    delta_k : ndarray of shape (Nx, Ny, Nz//2+1)
        Fourier-space overdensity field (half-complex).
    nvox : (int, int, int)
        Grid dimensions ``(Nx, Ny, Nz)``.
    dk : float
        Fourier-space spacing [h/Mpc].
    g1 : float
        First-order growth coefficient (usually 1).
    aHf1 : float
        Velocity prefactor :math:`a H(a) f(a)` [km/s / (Mpc/h)].
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
        Peculiar velocities [km/s].
    '''
    kx, ky, kz = fourier_vectors(nvox, dk, hermitian=True, dtype=dtype)
    K = (kx[:, None, None], ky[None, :, None], kz[None, None, :])
    boxsize = np.asarray(nvox, dtype=dtype) * dk

    # Gravitational potential in Fourier space: phi(k) = -delta(k) / |k|^2.
    # |k|^2 is built on the fly from the 1D vectors (one transient array).
    # Setting the DC mode to inf forces phi(0)=0 without a boolean mask;
    # delta_k[0, 0, 0] is already 0, so this only avoids the 0/0 there.
    k2 = K[0]**2 + K[1]**2 + K[2]**2
    k2[0, 0, 0] = np.inf
    phi_k = -delta_k / k2  # inherits delta_k's complex dtype (complex64/128)

    # Separable MAS deconvolution to pre-sharpen the field before interpolation
    if compensate:
        wx, wy, wz = compensation_factors(
            nvox, dk, boxsize, method=method, dtype=dtype)

    # Build the displacement field one axis at a time using the identity
    #       psi_i(k) = -i k_i phi(k).
    disp_field = np.empty((3, *nvox), dtype=dtype)
    for i in range(3):
        psi_i = (-1j * K[i]) * phi_k
        if compensate:
            psi_i *= wx[:, None, None]
            psi_i *= wy[None, :, None]
            psi_i *= wz[None, None, :]
        disp_field[i] = scipy.fft.irfftn(
            psi_i, s=nvox, axes=(0, 1, 2), workers=-1)
    disp_field_interp = interpolate_field(
        x=x, field=disp_field, boxsize=boxsize,
        origin=-boxsize / 2, method=method, vox_offset=0.5, periodic=True, dtype=dtype)
    xpert = x + g1 * disp_field_interp  # Bernardeau et al. 2002, eq. 98
    vpert = g1 * aHf1 * disp_field_interp  # Bernardeau et al. 2002, eq. 99
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
    r'''
    Apply second-order LPT (2LPT) to displace particles.

    Extends the Zel'dovich approximation with the second-order correction
    as given by Bernardeau et al. (2002):

    .. math::
        \mathbf{x} = \mathbf{q}
            + D_1 \mathbf{\Psi}^{(1)}
            + D_2 \mathbf{\Psi}^{(2)},

    where :math:`\mathbf{\Psi}^{(2)} = -\nabla\phi^{(2)}` is sourced by
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
        Fourier-space spacing [h/Mpc].
    g1 : float
        First-order growth coefficient (usually 1).
    g2 : float
        Second-order growth coefficient
        (typically :math:`-\tfrac{3}{7}\,\Omega_m^{-1/143}`).
    aHf1 : float
        First-order velocity prefactor :math:`a H f_1` [km/s / (Mpc/h)].
    aHf2 : float
        Second-order velocity prefactor [km/s / (Mpc/h)].
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
        Peculiar velocities [km/s].

    See Also
    --------
    lpt1 : First-order (Zel'dovich) displacement only.
    '''
    kx, ky, kz = fourier_vectors(nvox, dk, hermitian=True, dtype=dtype)
    K = (kx[:, None, None], ky[None, :, None], kz[None, None, :])
    boxsize = np.asarray(nvox, dtype=dtype) * dk

    # Separable MAS deconvolution factors (applied per axis, in place below)
    if compensate:
        wx, wy, wz = compensation_factors(
            nvox, dk, boxsize, method=method, dtype=dtype)

    # |k|^2 on the fly; DC -> inf forces phi(0)=0 without a boolean mask
    # (delta_k[0, 0, 0] is already 0). Reused for both Poisson solves below.
    k2 = K[0]**2 + K[1]**2 + K[2]**2
    k2[0, 0, 0] = np.inf

    # -- 1. First-order displacement (Psi^(1)) --------------------------------
    phi1_k = -delta_k / k2  # gravitational potential; inherits delta_k complex dtype
    disp_field1 = np.empty((3, *nvox), dtype=dtype)
    for i in range(3):
        psi_i = (-1j * K[i]) * phi1_k  # -i k_i phi(k)
        if compensate:  # pre-sharpen before interpolation
            psi_i *= wx[:, None, None]
            psi_i *= wy[None, :, None]
            psi_i *= wz[None, None, :]
        disp_field1[i] = scipy.fft.irfftn(
            psi_i, s=nvox, axes=(0, 1, 2), workers=-1)

    # -- 2. Second-order source S(x) from the 1LPT deformation tensor ---------
    # The derivatives d(Psi_i)/dx_j use the UNCOMPENSATED field. The
    # source S(x) lives on the original grid, so it must use the
    # physically correct displacement. Using the identity
    #       1j * psi1_k[i] * k_j == phi1_k * k_i * k_j ,
    # each derivative is built directly from phi1_k and the 1D vectors.
    def _dP(i, j):
        return scipy.fft.irfftn(
            phi1_k * (K[i] * K[j]), s=nvox, axes=(0, 1, 2), workers=-1)

    # Accumulate S incrementally so at most three deformation arrays are alive.
    dPxx, dPyy, dPzz = _dP(0, 0), _dP(1, 1), _dP(2, 2)
    S = dPxx * dPyy + dPxx * dPzz + dPyy * dPzz
    del dPxx, dPyy, dPzz
    S -= _dP(0, 1)**2  # (d Psi_x/dy)^2
    S -= _dP(0, 2)**2  # (d Psi_x/dz)^2
    S -= _dP(1, 2)**2  # (d Psi_y/dz)^2
    S_k = scipy.fft.rfftn(S, workers=-1)
    del S

    # -- 3. Second-order displacement (Psi^(2)) -------------------------------
    # Poisson equation for the source: phi2(k) = -S(k) / |k|^2.
    phi2_k = -S_k / k2  # inherits S_k complex dtype
    disp_field2 = np.empty((3, *nvox), dtype=dtype)
    for i in range(3):
        psi_i = (-1j * K[i]) * phi2_k
        if compensate:
            psi_i *= wx[:, None, None]
            psi_i *= wy[None, :, None]
            psi_i *= wz[None, None, :]
        disp_field2[i] = scipy.fft.irfftn(
            psi_i, s=nvox, axes=(0, 1, 2), workers=-1)

    # -- 4. Interpolate and update particle positions and velocities ----------
    # For each spatial axis, interpolate the displacement fields (both
    # first- and second-order) from the grid to the particle positions.
    disp_field1_interp = interpolate_field(
        x=x, field=disp_field1, boxsize=boxsize,
        origin=-boxsize / 2, method=method, vox_offset=0.5, periodic=True, dtype=dtype)
    disp_field2_interp = interpolate_field(
        x=x, field=disp_field2, boxsize=boxsize,
        origin=-boxsize / 2, method=method, vox_offset=0.5, periodic=True, dtype=dtype)

    xpert = x + g1 * disp_field1_interp + g2 * disp_field2_interp
    vpert = g1 * aHf1 * disp_field1_interp + g2 * aHf2 * disp_field2_interp
    return xpert, vpert