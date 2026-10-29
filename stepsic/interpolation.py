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

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from stepsic._typing import BoolVec3, FloatVec3, IntVec3, RealField
from stepsic.field import fourier_vectors


@dataclass(frozen=True)
class GridGeometry:
    """
    Defines the mapping between physical coordinates and grid indices.

    Parameters
    ----------
    nvox : array of shape (3,)
        Number of voxels in each dimension.
    boxsize : tuple of float or array of shape (3,)
        Physical size of the box.
    vox_offset : float
        Grid offset: 0.5 for voxel-centered, 0.0 for node-centered.
        Convention: grid point i is located at physical position
        origin + (i + vox_offset) * cell_size.
        - vox_offset = 0.0: node-centered (grid points at cell corners)
        - vox_offset = 0.5: voxel-centered (grid points at voxel centers)
    origin : float, array of shape (3,) or None
        Physical coordinate of the box corner, before ``vox_offset`` is
        applied. ``None`` selects zero. For a centered grid, use -boxsize/2.
    periodic : array of shape (3,) of bool
        Periodicity in each dimension.
    """
    nvox: IntVec3
    boxsize: FloatVec3
    vox_offset: float
    origin: FloatVec3 | None
    periodic: BoolVec3

    def __post_init__(self) -> None:
        """Normalize inputs to 1D arrays of length 3."""
        nvox = np.broadcast_to(self.nvox, (3,)).astype(np.int32)
        object.__setattr__(self, 'nvox', nvox)
        boxsize = np.broadcast_to(self.boxsize, (3,)).astype(np.float64)
        object.__setattr__(self, 'boxsize', boxsize)
        periodic = np.broadcast_to(self.periodic, (3,)).astype(np.bool_)
        object.__setattr__(self, 'periodic', periodic)
        if self.origin is None:
            object.__setattr__(self, 'origin', np.zeros(3, dtype=np.float64))
        else:
            origin = np.broadcast_to(self.origin, (3,)).astype(np.float64)
            object.__setattr__(self, 'origin', origin)

    @property
    def cell_size(self) -> RealField:
        return self.boxsize / self.nvox

    def pos_to_grid(self, x: RealField) -> RealField:
        """Convert physical positions to fractional grid coordinates."""
        x = x - self.origin  # shift to box-local coordinates [0, boxsize)
        box = self.boxsize
        per = self.periodic
        bshape = (1,) * (x.ndim - 1) + (3,)
        x = np.where(per.reshape(bshape), np.mod(x, box.reshape(bshape)), x)
        return x / self.cell_size - self.vox_offset


# Interpolation kernels
# 
# Define separable interpolation kernels with properties:
# - support: number of grid points spanned by the kernel
# - floor_shift: shift added to the grid coordinate g before its floor is
#   taken; 0.5 for odd support, so the kernel centres on the nearest node
# - base_offset: offset from floor(g + floor_shift) to the first grid point involved
# - weights(dx): function returning interpolation weights for the fractional
#   position dx = g + floor_shift - floor(g + floor_shift) in [0, 1)
#
# You can define new kernels by subclassing `InterpolationKernel` and adding them
# to the `KERNELS` registry.

class InterpolationKernel(ABC):
    """Base class for separable interpolation kernels."""

    @property
    @abstractmethod
    def support(self) -> int:
        """Number of grid points the kernel spans."""
        pass

    @property
    @abstractmethod
    def base_offset(self) -> int:
        """Offset from floor(g) to the first grid point involved."""
        pass

    @abstractmethod
    def weights(self, dx: RealField) -> tuple[RealField, ...]:
        """Compute interpolation weights for fractional position dx in [0, 1)."""
        pass

    @property
    def floor_shift(self) -> float:
        """Shift added to the grid coordinate before its floor is taken.

        A kernel of odd support is centred on the nearest grid point, one of
        even support on the cell that contains the position.
        """
        return 0.5 * (self.support % 2)

    @property
    def order(self) -> int:
        """Interpolation order."""
        return self.support - 1


class NGPKernel(InterpolationKernel):
    """Implements Nearest Grid Point (NGP) interpolation."""
    @property
    def support(self) -> int:
        return 1

    @property
    def base_offset(self) -> int:
        return 0

    def weights(self, dx: RealField) -> tuple[RealField, ...]:
        return (np.ones_like(dx),)


class CICKernel(InterpolationKernel):
    """Implements Cloud-in-Cell (CIC) interpolation."""

    @property
    def support(self) -> int:
        return 2

    @property
    def base_offset(self) -> int:
        return 0

    def weights(self, dx: RealField) -> tuple[RealField, ...]:
        return (1.0 - dx, dx)


class TSCKernel(InterpolationKernel):
    """Implements Triangular Shaped Cloud (TSC) interpolation."""

    @property
    def support(self) -> int:
        return 3

    @property
    def base_offset(self) -> int:
        return -1

    def weights(self, dx: RealField) -> tuple[RealField, ...]:
        w0 = 0.5 * (1.0 - dx)**2
        w1 = 0.75 - (dx - 0.5)**2
        w2 = 0.5 * dx**2
        return (w0, w1, w2)

# Kernel registry
KERNELS: dict[str, type[InterpolationKernel]] = {
    'ngp': NGPKernel,
    'cic': CICKernel,
    'tsc': TSCKernel,
}


def compensation_factors(
    nvox: IntVec3,
    dk: float,
    boxsize: FloatVec3,
    hermitian: bool = True,
    method: str | InterpolationKernel = 'cic',
    dtype: np.dtype = np.float64,
) -> tuple[RealField, RealField, RealField]:
    r"""
    Compute per-axis Fourier factors for mass-assignment deconvolution.

    The MAS deconvolution kernel factorises over the axes,

    .. math::
        W^{-1}(\mathbf{k}) = \prod_{i=x,y,z}
            \mathrm{sinc}\!\left(\frac{k_i}{2 k_{\mathrm{Ny},i}}\right)^{-(p+1)},

    so it can be applied to a Fourier-space field one axis at a time without
    ever building the full 3D kernel::

        wx, wy, wz = compensation_factors(nvox, dk, boxsize, method=method)
        arr *= wx[:, None, None]
        arr *= wy[None, :, None]
        arr *= wz[None, None, :]

    Each returned factor is already raised to the ``-(p+1)`` power. Because
    ``sinc`` is bounded in :math:`[2/\pi, 1]` over the half-open FFT frequency
    range :math:`[-1/2, 1/2)`, it never vanishes and no division guard is
    needed when ``boxsize / nvox`` equals ``dk``.

    Parameters
    ----------
    nvox : tuple of int or array of shape (3,)
        Number of grid cells in each dimension.
    dk : float
        Physical cell spacing in the same length unit as ``boxsize``.
    boxsize : tuple of float or array of shape (3,)
        Physical size of the box.
    hermitian : bool
        If `True`, use the reduced last axis (length `Nz//2+1`).
    method : str or InterpolationKernel
        Interpolation method: 'ngp', 'cic', 'tsc', or a kernel instance.
    dtype : np.dtype
        Numerical precision for the returned factors.

    Returns
    -------
    wx, wy, wz : ndarray
        1D deconvolution factors of length `Nx`, `Ny`, and `Nz//2+1`
        (if ``hermitian``) or `Nz`. Their outer product is the full
        three-dimensional deconvolution factor.
    """
    if isinstance(method, str):
        kernel = KERNELS[method.lower()]()
    else:
        kernel = method
    power = kernel.order + 1  # = kernel.support

    nvox_f = np.asarray(nvox, dtype=dtype)
    boxsize = np.asarray(boxsize, dtype=dtype)
    k_ny = np.pi * nvox_f / boxsize  # Nyquist wavenumber per axis, shape (3,)

    kx, ky, kz = fourier_vectors(nvox, dk, hermitian=hermitian, dtype=dtype)
    wx = (1.0 / np.sinc(kx / (2.0 * k_ny[0]))**power).astype(dtype, copy=False)
    wy = (1.0 / np.sinc(ky / (2.0 * k_ny[1]))**power).astype(dtype, copy=False)
    wz = (1.0 / np.sinc(kz / (2.0 * k_ny[2]))**power).astype(dtype, copy=False)
    return wx, wy, wz


class FieldInterpolator:
    """
    Interpolates a 3D field to particle positions using separable kernels.

    This is the adjoint of :class:`FieldDepositor`.

    Parameters
    ----------
    kernel : str or InterpolationKernel
        Interpolation kernel: 'ngp', 'cic', 'tsc', or a custom kernel instance.
    geometry : GridGeometry
        Grid geometry specification.
    dtype : np.dtype
        Floating-point precision used for interpolation computations and output.

    Examples
    --------
    >>> geom = GridGeometry(
    ...     boxsize=np.array([100.0, 100.0, 100.0]),
    ...     nvox=(64, 64, 64),
    ...     vox_offset=0.0,  # node-centered (FFT output)
    ...     origin=np.array([-50.0, -50.0, -50.0]),  # centered grid
    ...     periodic=True,
    ... )
    >>> interp = FieldInterpolator('cic', geom)
    >>> values = interp(field, particle_positions)
    """

    # Number of particle coordinates and interpolation weights processed together.
    _PARTICLES_PER_CHUNK = 1_000_000

    def __init__(
        self, 
        kernel: str | InterpolationKernel,
        geometry: GridGeometry,
        dtype: np.dtype = np.float64,
    ):
        if isinstance(kernel, str):
            kernel = KERNELS[kernel.lower()]()
        self.kernel = kernel
        self.geometry = geometry
        self.dtype = np.dtype(dtype)
        if not np.issubdtype(self.dtype, np.floating):
            raise TypeError('dtype must be a floating-point type')

    def __call__(
            self,
            field: RealField,
            x: RealField
        ) -> RealField:
        """
        Evaluate the gridded vector ``field`` at arbitrary positions.

        Parameters
        ----------
        field : ndarray
            Grid values, shape ``(ncomp, Nx, Ny, Nz)``. Scalar fields
            require an explicit leading axis of length one.
        x : ndarray of shape (N, 3)
            Particle positions in physical coordinates.

        Returns
        -------
        ndarray
            Values of shape ``(N, ncomp)`` with all singleton axes removed.
            One particle gives ``(ncomp,)``; one component gives ``(N,)``;
            one particle and one component give a zero-dimensional array.
        """
        field = np.asarray(field, dtype=self.dtype)
        result = np.zeros((x.shape[0], field.shape[0]), dtype=self.dtype)
        for sl, values in self._iter_component_chunks(tuple(field), x):
            result[sl] = values

        return result.squeeze()

    def _iter_component_chunks(
            self,
            fields: tuple[RealField, ...],
            x: RealField,
        ):
        """Yield two-dimensional interpolation results for bounded chunks."""
        if not fields:
            raise ValueError('at least one field component is required')
        fields = tuple(np.asarray(field, dtype=self.dtype) for field in fields)
        grid_shape = tuple(self.geometry.nvox)
        if any(field.shape != grid_shape for field in fields):
            raise ValueError(f'field components must have shape {grid_shape}')

        for start in range(0, x.shape[0], self._PARTICLES_PER_CHUNK):
            stop = min(start + self._PARTICLES_PER_CHUNK, x.shape[0])
            sl = slice(start, stop)
            result = np.zeros((stop - start, len(fields)), dtype=self.dtype)
            self._eval_component_chunk(fields, x[sl], result)
            yield sl, result

    def _eval_component_chunk(
            self,
            fields: tuple[RealField, ...],
            x: RealField,
            result: RealField
        ) -> None:
        """Accumulate interpolated values for one particle chunk into ``result``."""
        # Convert positions to fractional grid coordinates
        g = self.geometry.pos_to_grid(x) + self.kernel.floor_shift  # (N, 3)
        g_int = np.floor(g).astype(np.int32, copy=False)
        g_frc = (g - g_int).astype(self.dtype, copy=False)

        # Compute 1-D weights along each axis; shape (support, N)
        wx, wy, wz = (
            np.vstack(self.kernel.weights(g_frc[:, i])) for i in range(3)
        )

        # Loop over all combinations of kernel offsets
        nvox = self.geometry.nvox
        periodic = self.geometry.periodic

        for dx in range(self.kernel.support):
            ix = g_int[:, 0] + self.kernel.base_offset + dx
            ix = ix % nvox[0] if periodic[0] else np.clip(ix, 0, nvox[0]-1)

            for dy in range(self.kernel.support):
                iy = g_int[:, 1] + self.kernel.base_offset + dy
                iy = iy % nvox[1] if periodic[1] else np.clip(iy, 0, nvox[1]-1)

                for dz in range(self.kernel.support):
                    iz = g_int[:, 2] + self.kernel.base_offset + dz
                    iz = iz % nvox[2] if periodic[2] else np.clip(iz, 0, nvox[2]-1)

                    w = wx[dx] * wy[dy] * wz[dz]    # shape (N,)
                    for component, field in enumerate(fields):
                        fvals = field[ix, iy, iz]
                        np.multiply(fvals, w, out=fvals)
                        result[:, component] += fvals
    

class FieldDepositor:
    r"""
    Deposit particle-carried quantities onto a regular 3D grid using
    separable mass-assignment kernels.
 
    This is the adjoint of :class:`FieldInterpolator`.
 
    Parameters
    ----------
    kernel : str or InterpolationKernel
        Mass-assignment kernel: ``'ngp'``, ``'cic'``, ``'tsc'``, or
        a custom kernel instance.
    geometry : GridGeometry
        Grid geometry specification.
    dtype : np.dtype
        Floating-point precision used for deposition computations and output.
 
    Examples
    --------
    Deposit unit-weight particles onto a 64³ periodic grid:
 
    >>> geom = GridGeometry(
    ...     boxsize=np.array([100.0, 100.0, 100.0]),
    ...     nvox=(64, 64, 64),
    ...     vox_offset=0.5,
    ...     origin=np.zeros(3),
    ...     periodic=True,
    ... )
    >>> dep = FieldDepositor('cic', geom)
    >>> density = dep(x, values=None)  # number-count field
    """
 
    def __init__(
        self,
        kernel: str | InterpolationKernel,
        geometry: GridGeometry,
        dtype: np.dtype = np.float64,
    ):
        if isinstance(kernel, str):
            kernel = KERNELS[kernel.lower()]()
        self.kernel = kernel
        self.geometry = geometry
        self.dtype = np.dtype(dtype)
        if not np.issubdtype(self.dtype, np.floating):
            raise TypeError('dtype must be a floating-point type')
 
    def __call__(
        self,
        x: RealField,
        values: RealField | None = None,
    ) -> RealField:
        r"""
        Scatter particle values onto the grid.
 
        Parameters
        ----------
        x : ndarray of shape (N, 3)
            Particle positions in physical coordinates.
        values : ndarray of shape (N,) or (N, ncomp), or None
            Quantity to deposit at each particle position.  If ``None``,
            unit weights are used.
 
        Returns
        -------
        grid : ndarray of shape (nx, ny, nz) or (ncomp, nx, ny, nz)
            The deposited field.  If ``values`` is 1-D or ``None``, the
            output is a scalar grid ``(nx, ny, nz)``.  If ``values`` has
            ``ncomp > 1`` columns, the output has a leading component
            axis matching :class:`FieldInterpolator`'s field layout.
        """
        N = x.shape[0]
        nvox = self.geometry.nvox
 
        # Handle scalar vs. vector deposit
        if values is None:
            values = np.ones((N, 1), dtype=self.dtype)
        else:
            values = np.asarray(values, dtype=self.dtype)
            if values.ndim == 1:
                values = values[:, np.newaxis]  # (N,) -> (N, 1)
        ncomp = values.shape[1]
 
        grid = np.zeros((ncomp, *nvox), dtype=self.dtype)
 
        # Convert positions to fractional grid coordinates
        g = self.geometry.pos_to_grid(x) + self.kernel.floor_shift  # (N, 3)
        g_int = np.floor(g).astype(np.int32, copy=False)
        g_frc = (g - g_int).astype(self.dtype, copy=False)

        # Compute 1-D weights along each axis; shape (support, N)
        wx, wy, wz = (
            np.vstack(self.kernel.weights(g_frc[:, i])) for i in range(3)
        )
 
        # Scatter contributions onto the grid
        periodic = self.geometry.periodic
        for dx in range(self.kernel.support):
            ix = g_int[:, 0] + self.kernel.base_offset + dx
            ix = ix % nvox[0] if periodic[0] else np.clip(ix, 0, nvox[0] - 1)
 
            for dy in range(self.kernel.support):
                iy = g_int[:, 1] + self.kernel.base_offset + dy
                iy = iy % nvox[1] if periodic[1] else np.clip(iy, 0, nvox[1] - 1)
 
                for dz in range(self.kernel.support):
                    iz = g_int[:, 2] + self.kernel.base_offset + dz
                    iz = iz % nvox[2] if periodic[2] else np.clip(iz, 0, nvox[2] - 1)
 
                    w = wx[dx] * wy[dy] * wz[dz]  # (N,)
                    for c in range(ncomp):
                        np.add.at(grid[c], (ix, iy, iz), w * values[:, c])
 
        return grid.squeeze()


def _interpolate_field_chunks(
    x: RealField,
    fields: tuple[RealField, ...],
    boxsize: FloatVec3,
    periodic: BoolVec3 = True,
    method: str = 'cic',
    vox_offset: float = 0.0,
    origin: FloatVec3 | None = None,
    dtype: np.dtype = np.float64,
):
    """Yield bounded interpolation results without joining component grids."""
    if not fields:
        raise ValueError('at least one field component is required')
    geometry = GridGeometry(
        boxsize=boxsize,
        nvox=fields[0].shape,
        vox_offset=vox_offset,
        origin=origin,
        periodic=periodic,
    )
    interpolator = FieldInterpolator(method, geometry, dtype=dtype)
    yield from interpolator._iter_component_chunks(fields, x)


def interpolate_field(
    x: RealField,
    field: RealField,
    boxsize: FloatVec3,
    periodic: BoolVec3 = True,
    method: str = 'cic',
    vox_offset: float = 0.0,
    origin: FloatVec3 | None = None,
    dtype: np.dtype = np.float64,
) -> RealField:
    """
    Interpolate a 3D field at particle positions.

    Parameters
    ----------
    x : ndarray of shape (N, 3)
        Particle positions in physical coordinates.
    field : ndarray of shape (ncomp, nx, ny, nz)
        Values of a vector field on the grid. Use ``ncomp=1`` for scalar fields.
    boxsize : ndarray of shape (3,)
        Physical size of the periodic box.
    periodic : bool or ndarray of shape (3,) of bool
        Periodicity in each dimension (default: all True).
    method : str
        Interpolation method: 'ngp', 'cic', or 'tsc' (default: 'cic').
    vox_offset : float
        Grid offset: 0.5 for voxel-centered, 0.0 for node-centered (default: 0.0).
    origin : ndarray of shape (3,) or None
        Physical coordinate of the box corner. Default is (0, 0, 0).
        For a centered grid, use -boxsize/2.

    Returns
    -------
    values : ndarray
        Interpolated values of shape ``(N, ncomp)`` with singleton axes
        removed, including a zero-dimensional array when both are one.
    """
    geometry = GridGeometry(
        boxsize=boxsize,
        nvox=field.shape[1:],
        vox_offset=vox_offset,
        origin=origin,
        periodic=periodic,
    )
    interpolator = FieldInterpolator(method, geometry, dtype=dtype)
    return interpolator(field, x)


def deposit_field(
    x: RealField,
    nvox: IntVec3,
    boxsize: FloatVec3,
    values: RealField | None = None,
    periodic: BoolVec3 = True,
    method: str = 'cic',
    vox_offset: float = 0.5,
    origin: FloatVec3 | None = None,
) -> RealField:
    r"""
    Deposit particle quantities onto a regular 3D grid.
 
    Parameters
    ----------
    x : ndarray of shape (N, 3)
        Particle positions in physical coordinates.
    nvox : tuple of int or ndarray of shape (3,)
        Number of grid cells ``(Nx, Ny, Nz)``.
    boxsize : ndarray of shape (3,)
        Physical size of the periodic box.
    values : ndarray of shape (N,) or (N, ncomp), or None
        Quantity to deposit.  If ``None``, unit weights are used
        (number-count field).
    periodic : bool or ndarray of shape (3,) of bool
        Periodicity in each dimension (default: all True).
    method : str
        Mass-assignment kernel: ``'ngp'``, ``'cic'``, or ``'tsc'``
        (default ``'cic'``).
    vox_offset : float
        Grid offset: 0.5 for voxel-centred, 0.0 for node-centred
        (default 0.5).
    origin : float, ndarray of shape (3,) or None
        Physical coordinate of the box corner.  Default is ``None``.
 
    Returns
    -------
    grid : ndarray of shape (nx, ny, nz) or (ncomp, nx, ny, nz)
        The deposited field.
    """
    geometry = GridGeometry(
        boxsize=boxsize,
        nvox=nvox,
        vox_offset=vox_offset,
        origin=origin,
        periodic=periodic,
    )
    depositor = FieldDepositor(method, geometry)
    return depositor(x, values)
