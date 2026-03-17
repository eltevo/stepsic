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
from typing import Tuple
from stepsic._typing import RealField, ComplexField, IntVec3

import numpy as np

from stepsic.field import fourier_grid
from stepsic.interpolation import interpolate_field, compensation_kernel

import logging
log = logging.getLogger(__name__)


def log_lpt(
    x: RealField, xpert: RealField, vpert: RealField, *, title: str = None
) -> None:
    '''TODO'''
    xabs, vabs = np.abs(xpert - x), np.abs(vpert)
    xmax, xavg = np.max(xabs, axis=0), np.mean(xabs, axis=0)
    vmax, vavg = np.max(vabs, axis=0), np.mean(vabs, axis=0)
    for i, xi in enumerate(('x', 'y', 'z')):
        log.info(f"{title} '{xi}' displacements: "
                 f"d_max = {xmax[i]*1e3:.3f} kpc/h; d_avg = {xavg[i]*1e3:.3f} kpc/h ")
        log.info(f"{title} 'v{xi}' velocities:   "
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
    method: str = 'cic'
) -> Tuple[RealField, RealField]:
    r'''
    Apply first-order Lagrangian Perturbation Theory (LPT), i.e., the
    Zel'dovich approximation, to generate perturbed particle positions
    and velocities.

    In this approximation, particles are displaced from their initial
    (Lagrangian) positions :math:`\mathbf{q}` to their final (Eulerian)
    positions :math:`\mathbf{x}` using a displacement field
    :math:`\mathbf{\Psi}`:

    .. math::
        \mathbf{x}(\mathbf{q}, t) = \mathbf{q} + \mathbf{\Psi}(\mathbf{q}),

    where :math:`D_1(t)` is the linear growing model and
    :math:`\mathbf{\Psi}(\mathbf{q})` is the displacement field computed
    from the initial density perturbations. The displacement field is
    related to the gravitational potential, and in Fourier space, it is
    calculated from the overdensity field :math:`\delta(\mathbf{k})`:

    .. math::
        \mathbf{\Psi}(\mathbf{k}) =
            -i \, \frac{\mathbf{k}}{|\mathbf{k}|^2} \, \delta(\mathbf{k})
        \quad \text{for } |\mathbf{k}| > 0\,,

    where :math:`\mathbf{k}` is the wavevector.
    
    Parameters
    ----------
    x : ndarray of shape (N, 3)
        Initial unperturbed particle positions in physical units.
    delta_k : ndarray
        A 3D complex-valued array of shape ``(Nx, Ny, Nz//2+1)`` representing
        the Fourier modes of the overdensity field.
    nvox : tuple of int
        The number of voxels in each dimension of the grid ``(Nx, Ny, Nz)``.
    dk : float
        The uniform step size in each dimension, calculated as the length
        of the shortest dimension divided by the number of voxels in
        that dimension.
    g1 : float
        The first-order Lagrangian growth coefficient, typically
        :math:`g_1 = 1`.
    aHf1 : float
        A prefactor for the velocity calculation, typically related to the
        time derivative of the growth factor (e.g. :math:`\dot{D}_1` or
        :math:`H(a)f(a)` where :math:`f` is the growth rate).
    compensate : bool
        If True, applies a deconvolution kernel in Fourier space before
        the inverse FFT to compensate for the smoothing introduced by
        the interpolation scheme. Required when particles do not sit on
        grid nodes (e.g. glass initial conditions). For regular lattices
        (SC, BCC, FCC) where particles coincide with grid points, this
        should be False.
    method : str
        Interpolation method: 'ngp', 'cic', or 'tsc' (default: 'cic').
    
    Returns
    -------
    xpert : ndarray of shape (N, 3)
        Perturbed particle positions in physical [Mpc/h].
    vpert : ndarray of shape (N, 3)
        Particle peculiar velocities in [km/s]. The velocity is computed
        according to the specific formula implemented in this function:

        .. math::
            \mathbf{v} = \frac{\dot{D}(a)}{D(1)} \cdot \mathbf{\Psi}\,,
        
        where :math:`\dot{D}(a)` is approximated by the velocity prefactor

        .. math::
            v_{\mathrm{fac}} = a H(a) f(a)\,.

    Notes
    -----
    **Overview of the Implementation:**

    1.  **Fourier Grid Setup:** A 3D Fourier grid (``kvec``) is constructed
        for a mesh of size ``(Nx, Ny, Nz)``. For a real-valued field,
        according to the Hermitian constraints, a reduced FFT is used,
        so the k-space arrays have a shape of ``(Nx, Ny, Nz//2 + 1)``.
        Regardless of the mesh size, we always cut along the z-axis in
        this implementation.

    2.  **Displacement Field Calculation:**
        - The gravitational potential :math:`\phi(\mathbf{k})` is computed
          in Fourier space from the overdensity field :math:`\delta(\mathbf{k})`
          using the relation:
          .. math::
              \phi(\mathbf{k}) = -\frac{\delta(\mathbf{k})}{|\mathbf{k}|^2}\,,
          where :math:`|\mathbf{k}|` is the magnitude of the wavevector.
        - The Fourier-space displacement field :math:`\mathbf{\Psi}(\mathbf{k})`
          is computed for each spatial component. Division by zero at the
          DC mode (:math:`|\mathbf{k}| = 0`) is avoided.
        - If ``compensate=True``, a deconvolution kernel is applied to
          pre-sharpen the Fourier modes before the inverse FFT.
        - An inverse FFT converts :math:`\mathbf{\Psi}(\mathbf{k})` back
          to a real-space grid.

    3.  **Interpolation and Particle Update:**
        - The gridded displacement field :math:`\mathbf{\Psi}` is
          interpolated to the Lagrangian particle positions :math:`\mathbf{q}`.
        - Particle positions are updated to their Eulerian coordinates:

          .. math::
              \mathbf{x}_{\text{pert}} = \mathbf{q} + \mathbf{\Psi}(\mathbf{q})

        - Particle velocities are computed using the interpolated
          displacement field :math:`\mathbf{\Psi}`, the growing mode
          ``D1``, and the velocity prefactor ``aHf1``.
    '''
    kvec, kmod = fourier_grid(nvox, dk, hermitian=True)
    mask = kmod > 0.0  # Avoid division by zero at k = 0
    phi_k = np.zeros_like(kmod, dtype=complex)
    phi_k[mask] = -delta_k[mask] / kmod[mask]**2  # Gravitational potential in Fourier space
    psi1_k = -1j * phi_k[np.newaxis, ...] * kvec  # Displacement field in Fourier space
    boxsize = np.asarray(nvox) * dk
    # Apply deconvolution to pre-sharpen the field before interpolation
    if compensate:
        W_inv = compensation_kernel(kvec, nvox, boxsize, method=method)
        psi1_k *= W_inv[np.newaxis, ...]
    disp_field = np.fft.irfftn(psi1_k, s=nvox, axes=(-3, -2, -1))
    disp_field_interp = interpolate_field(
        x=x, field=disp_field, boxsize=boxsize,
        origin=-boxsize / 2, method=method, vox_offset=0.5,
        periodic=True).astype(np.float32)
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
) -> tuple[RealField, RealField]:
    r'''
    Apply second-order Lagrangian Perturbation Theory (2LPT) to generate
    perturbed particle positions and velocities.

    This method provides a more accurate description of particle
    trajectories than the Zel'dovich approximation (1LPT) by including
    the second-order term in the displacement. The final (Eulerian)
    position :math:`\mathbf{x}` is computed from the initial (Lagrangian)
    position :math:`\mathbf{q}` as:

    .. math::
        \mathbf{x}(\mathbf{q}, t) = \mathbf{q} - D_1(t)\mathbf{\Psi}^{(1)}(\mathbf{q}) + D_2(t)\mathbf{\Psi}^{(2)}(\mathbf{q})

    where :math:`\mathbf{\Psi}^{(1)}` and :math:`\mathbf{\Psi}^{(2)}`
    are the first- and second-order displacement fields, and :math:`D_1`
    and :math:`D_2` are the corresponding linear and second-order growing
    modes.

    The **first-order field** is calculated from the overdensity :math:`\delta`
    as in 1LPT:
    
    .. math::
        \mathbf{\Psi}^{(1)}(\mathbf{k}) = -i \frac{\mathbf{k}}{|\mathbf{k}|^2} \delta(\mathbf{k}).

    The **second-order field** is derived from a scalar potential :math:`\phi^{(2)}`,
    where :math:`\mathbf{\Psi}^{(2)} = -\nabla\phi^{(2)}`. The potential
    itself is sourced by a quadratic source term :math:`S(\mathbf{x})`
    from the spatial derivatives of the first-order displacement.
    Following standard 2LPT theory, one may compute

    .. math::
        S(\mathbf{x}) =
            \frac{\partial \Psi^{(1)}_x}{\partial x}\,\frac{\partial \Psi^{(1)}_y}{\partial y}
            + \frac{\partial \Psi^{(1)}_x}{\partial x}\,\frac{\partial \Psi^{(1)}_z}{\partial z}
            + \frac{\partial \Psi^{(1)}_y}{\partial y}\,\frac{\partial \Psi^{(1)}_z}{\partial z}
            - \left[
                \left(\frac{\partial \Psi^{(1)}_x}{\partial y}\right)^2
                + \left(\frac{\partial \Psi^{(1)}_x}{\partial z}\right)^2
                + \left(\frac{\partial \Psi^{(1)}_y}{\partial z}\right)^2
            \right].
         
    In the code we denote:
        - $dPxx = \frac{\partial \Psi^{(1)}_x}{\partial x}$,
        - $dPxy = \frac{\partial \Psi^{(1)}_x}{\partial y}$,
        - $dPxz = \frac{\partial \Psi^{(1)}_x}{\partial z}$,
        - $dPyy = \frac{\partial \Psi^{(1)}_y}{\partial y}$,
        - $dPyz = \frac{\partial \Psi^{(1)}_y}{\partial z}$,
        - $dPzz = \frac{\partial \Psi^{(1)}_z}{\partial z}$,
         
    and then set

    .. math::
        S(\mathbf{x}) =
            dPxx\,dPyy + dPxx\,dPzz + dPyy\,dPzz - (dPxy^2 + dPxz^2 + dPyz^2).
        
    The Poisson equation in Fourier space is solved for the second-order
    potential:

    .. math::
        \phi^{(2)}(\mathbf{k}) = -\frac{S(\mathbf{k})}{|\mathbf{k}|^2},
         
    with the $k=0$ mode appropriately masked. The second-order displacement
    in Fourier space is then given by

    .. math::
        \Psi^{(2)}_i(\mathbf{k}) = i\,k_i\,\phi^{(2)}(\mathbf{k}),
         
    and an inverse FFT yields the real-space second-order displacement
    field.
    

    Parameters
    ----------
    x : ndarray of shape (N, 3)
        Initial unperturbed particle positions in physical [Mpc/h].
    delta_k : ndarray
        A 3D complex-valued array of shape (Nx, Ny, Nz//2+1) representing
        the Fourier modes of the overdensity field.
    nvox : tuple of int
        The number of voxels in each dimension of the grid (Nx, Ny, Nz).
    dk : float
        The uniform step size in each dimension, calculated as the length
        of the shortest dimension divided by the number of voxels in
        that dimension.
    g1 : float
        The first-order Lagrangian growth coefficient, typically
        :math:`g_1 = 1`.
    g2 : float
        The second-order Lagrangian growth coefficient, typically
        :math:`g_2 = -\frac{3}{7} \Omega_M^{-1/143}`.
    aHf1 : float
        A prefactor for the velocity calculation, typically related to the
        time derivative of the growth factor (e.g. :math:`\dot{D}_1` or
        :math:`H(a)f(a)` where :math:`f` is the growth rate).
    aHf2 : float
        A prefactor for the second-order velocity term.

    Returns
    -------
    xpert : ndarray of shape (N, 3)
        Perturbed particle positions in physical [Mpc/h].
    vpert : ndarray of shape (N, 3)
        Particle peculiar velocities in [km/s], computed as:

        .. math::
            \mathbf{v} = -D_1 \cdot dD_1 \cdot \mathbf{\Psi}^{(1)} + D_2 \cdot dD_2 \cdot \mathbf{\Psi}^{(2)}.

    Notes
    -----
    **Overview of the Implementation:**

    1.  **First-Order Displacement:** The first-order displacement field,
        :math:`\mathbf{\Psi}^{(1)}`, is computed from the Fourier-space
        overdensity field, identical to the Zel'dovich approximation.

    2.  **Second-Order Source:** The spatial derivatives of the first-order
        field (i.e., the deformation tensor :math:`\partial \Psi_i^{(1)} / \partial q_j`)
        are calculated using FFTs. These are then combined to form the
        source term for the second-order potential.

    3.  **Second-Order Displacement:** The Poisson equation for the
        second-order potential :math:`\phi^{(2)}` is solved in Fourier
        space. The second-order displacement field, :math:`\mathbf{\Psi}^{(2)}`,
        is then found by taking the gradient of this potential, again in
        Fourier space.

    4.  **Interpolation and Particle Update:**
        - Both the first- and second-order displacement fields are
          interpolated from the grid to the Lagrangian particle positions.
        - Particle positions and velocities are updated by combining the
          interpolated first- and second-order contributions.
    '''
    kvec, kmod = fourier_grid(nvox, dk, hermitian=True)
    mask = kmod > 0.0  # Avoid division by zero at k = 0
    boxsize = np.asarray(nvox) * dk

    # Precompute deconvolution kernel if needed
    if compensate:
        W_inv = compensation_kernel(kvec, nvox, boxsize, method=method)

    # -- 1. First-order displacement (Psi^(1)) --------------------------------
    phi1_k = np.zeros_like(kmod, dtype=complex)
    phi1_k[mask] = -delta_k[mask] / kmod[mask]**2  # Gravitational potential in Fourier space
    psi1_k = -1j * phi1_k[np.newaxis, ...] * kvec  # Displacement field in Fourier space
    # Apply deconvolution to pre-sharpen before interpolation
    if compensate:
        psi1_k_comp = psi1_k * W_inv[np.newaxis, ...]
    else:
        psi1_k_comp = psi1_k
    disp_field1 = np.fft.irfftn(psi1_k_comp, s=nvox, axes=(-3, -2, -1))

    # -- 2. Compute derivatives of Psi^(1) for the second-order source --------
    # NOTE: The derivatives for the 2LPT source term use the
    # uncompensated psi1_k. The compensation corrects for
    # interpolation artifacts, but the source term S(x) is computed
    # on the grid (no interpolation involved), so it must use the
    # physically correct (uncompensated) displacement field.
    axes = (0, 1, 2)
    dPxx = np.fft.irfftn(1j * psi1_k[0] * kvec[0], s=nvox, axes=axes)  # d(Psi_x)/dx
    dPxy = np.fft.irfftn(1j * psi1_k[0] * kvec[1], s=nvox, axes=axes)  # d(Psi_x)/dy
    dPxz = np.fft.irfftn(1j * psi1_k[0] * kvec[2], s=nvox, axes=axes)  # d(Psi_x)/dz
    # --
    dPyy = np.fft.irfftn(1j * psi1_k[1] * kvec[1], s=nvox, axes=axes)  # d(Psi_y)/dy
    dPyz = np.fft.irfftn(1j * psi1_k[1] * kvec[2], s=nvox, axes=axes)  # d(Psi_y)/dz
    # --
    dPzz = np.fft.irfftn(1j * psi1_k[2] * kvec[2], s=nvox, axes=axes)  # d(Psi_z)/dz

    # Compute the quadratic source S(x) and its Fourier transform S(k)
    S = dPxx * dPyy + dPxx * dPzz + dPyy * dPzz - (dPxy**2 + dPxz**2 + dPyz**2)
    S_k = np.fft.rfftn(S)

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
    disp_field2 = np.fft.irfftn(psi2_k, s=nvox, axes=(-3, -2, -1))

    # -- 4. Interpolate and update particle positions and velocities ----------
    # For each spatial axis, interpolate the displacement fields (both
    # first- and second-order) from the grid to the particle positions.
    disp_field1_interp = interpolate_field(
        x=x, field=disp_field1, boxsize=boxsize,
        origin=-boxsize / 2, method=method, vox_offset=0.5,
        periodic=True).astype(np.float32)
    disp_field2_interp = interpolate_field(
        x=x, field=disp_field2, boxsize=boxsize,
        origin=-boxsize / 2, method=method, vox_offset=0.5,
        periodic=True).astype(np.float32)

    xpert = x + g1 * disp_field1_interp + g2 * disp_field2_interp
    vpert = g1 * aHf1 * disp_field1_interp + g2 * aHf2 * disp_field2_interp
    return xpert, vpert