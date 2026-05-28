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
from stepsic.field import fourier_grid
from stepsic.interpolation import compensation_kernel, interpolate_field

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
    kvec, kmod = fourier_grid(nvox, dk, hermitian=True, dtype=dtype)
    mask = kmod > 0.0  # Avoid division by zero at k = 0
    if dtype == np.float32:
        phi_k = np.zeros_like(kmod, dtype=np.complex64)
    else:
        phi_k = np.zeros_like(kmod, dtype=np.complex128)
    phi_k[mask] = -delta_k[mask] / kmod[mask]**2  # Gravitational potential in Fourier space
    psi1_k = -1j * phi_k[np.newaxis, ...] * kvec  # Displacement field in Fourier space
    boxsize = np.asarray(nvox, dtype=dtype) * dk
    # Apply deconvolution to pre-sharpen the field before interpolation
    if compensate:
        W_inv = compensation_kernel(kvec, nvox, boxsize, method=method, dtype=dtype)
        psi1_k *= W_inv[np.newaxis, ...]
    disp_field = scipy.fft.irfftn(psi1_k, s=nvox, axes=(-3, -2, -1), workers=-1)
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
    kvec, kmod = fourier_grid(nvox, dk, hermitian=True, dtype=dtype)
    mask = kmod > 0.0  # Avoid division by zero at k = 0
    boxsize = np.asarray(nvox, dtype=dtype) * dk

    # Precompute deconvolution kernel if needed
    if compensate:
        W_inv = compensation_kernel(kvec, nvox, boxsize, method=method, dtype=dtype)

    # -- 1. First-order displacement (Psi^(1)) --------------------------------
    if dtype == np.float32:
        phi1_k = np.zeros_like(kmod, dtype=np.complex64)
    else:
        phi1_k = np.zeros_like(kmod, dtype=np.complex128)
    phi1_k[mask] = -delta_k[mask] / kmod[mask]**2  # Gravitational potential in Fourier space
    psi1_k = -1j * phi1_k[np.newaxis, ...] * kvec  # Displacement field in Fourier space
    # Apply deconvolution to pre-sharpen before interpolation
    if compensate:
        psi1_k_comp = psi1_k * W_inv[np.newaxis, ...]
    else:
        psi1_k_comp = psi1_k
    disp_field1 = scipy.fft.irfftn(psi1_k_comp, s=nvox, axes=(-3, -2, -1), workers=-1)

    # -- 2. Compute derivatives of Psi^(1) for the second-order source --------
    # NOTE: The derivatives for the 2LPT source term use the uncompensated
    # `psi1_k`. The compensation corrects for interpolation artifacts,
    # but the source term S(x) is computed on the grid (no interpolation
    # involved), so it must use the physically correct (uncompensated)
    # displacement field.
    axes = (0, 1, 2)
    dPxx = scipy.fft.irfftn(1j * psi1_k[0] * kvec[0], s=nvox, axes=axes, workers=-1)  # d(Psi_x)/dx
    dPxy = scipy.fft.irfftn(1j * psi1_k[0] * kvec[1], s=nvox, axes=axes, workers=-1)  # d(Psi_x)/dy
    dPxz = scipy.fft.irfftn(1j * psi1_k[0] * kvec[2], s=nvox, axes=axes, workers=-1)  # d(Psi_x)/dz
    # --
    dPyy = scipy.fft.irfftn(1j * psi1_k[1] * kvec[1], s=nvox, axes=axes, workers=-1)  # d(Psi_y)/dy
    dPyz = scipy.fft.irfftn(1j * psi1_k[1] * kvec[2], s=nvox, axes=axes, workers=-1)  # d(Psi_y)/dz
    # --
    dPzz = scipy.fft.irfftn(1j * psi1_k[2] * kvec[2], s=nvox, axes=axes, workers=-1)  # d(Psi_z)/dz

    # Compute the quadratic source S(x) and its Fourier transform S(k)
    S = dPxx * dPyy + dPxx * dPzz + dPyy * dPzz - (dPxy**2 + dPxz**2 + dPyz**2)
    S_k = scipy.fft.rfftn(S, workers=-1)

    # -- 3. Second-order displacement (Psi^(2)) -------------------------------
    # Solve the Poisson equation in Fourier space, now for the source term S(k)
    #
    #     phi2(k) = -S(k) / |k|^2
    #
    phi2_k = np.zeros_like(S_k, dtype=complex)
    phi2_k[mask] = -S_k[mask] / kmod[mask]**2
    psi2_k = -1j * phi2_k[np.newaxis, ...] * kvec  # Displacement field in Fourier space
    # Compensate the second-order field as well
    # We can simply overwrite psi2_k here, because it will not be reused
    if compensate:
        psi2_k *= W_inv[np.newaxis, ...]
    disp_field2 = scipy.fft.irfftn(psi2_k, s=nvox, axes=(-3, -2, -1), workers=-1)

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