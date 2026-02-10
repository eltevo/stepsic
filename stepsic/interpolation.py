#*******************************************************************************#
#  stepsic - An initial condition generator for                                 #
#            STEreographically Projected cosmological Simulations               #
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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple, Type, Union, Optional

import numpy as np
from numpy.typing import NDArray

from scipy.interpolate import RegularGridInterpolator


@dataclass(frozen=True)
class GridGeometry:
    '''
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
    origin : array of shape (3,) or None
        Physical coordinate of the box corner (where index 0 maps to).
        Default is (0, 0, 0). For a centered grid, use -boxsize/2.
    periodic : array of shape (3,) of bool
        Periodicity in each dimension.
    '''
    nvox: Union[Tuple[int, int, int], NDArray[np.integer]]
    boxsize: Union[Tuple[float, float, float], NDArray[np.floating]]
    vox_offset: float
    origin: Optional[Union[Tuple[float, float, float], NDArray[np.floating]]] = None
    periodic: Union[bool, Tuple[float, float, float], NDArray[np.bool_]] = False

    def __post_init__(self) -> None:
        '''Normalize inputs to 1D arrays of length 3.'''
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
    def cell_size(self) -> NDArray[np.floating]:
        return self.boxsize / self.nvox

    def pos_to_grid(self, pos: NDArray[np.floating]) -> NDArray[np.floating]:
        '''Convert physical positions to fractional grid coordinates.'''
        x = pos - self.origin  # shift to box-local coordinates [0, boxsize)
        box = self.boxsize
        per = self.periodic
        bshape = (1,) * (x.ndim - 1) + (3,)
        x = np.where(per.reshape(bshape), np.mod(x, box.reshape(bshape)), x)
        return x / self.cell_size - self.vox_offset

    def wrap_indices(self, idx: NDArray[np.integer]) -> NDArray[np.integer]:
        '''Apply periodic or clamped wrapping to grid indices.'''
        out = idx.copy()

        per = self.periodic
        out[..., per] = out[..., per] % self.nvox[per]
        out[..., not per] = np.clip(out[..., not per], 0, self.nvox[not per] - 1)

        return out


# Interpolation kernels
# 
# Define separable interpolation kernels with properties:
# - support: number of grid points spanned by the kernel
# - base_offset: offset from floor(grid) to the first grid point involved
# - weights(dx): function returning interpolation weights for fractional position dx in [0, 1)
#
# You can define new kernels by subclassing `InterpolationKernel` and adding them
# to the `KERNELS` registry.

class InterpolationKernel(ABC):
    '''Base class for separable interpolation kernels.'''

    @property
    @abstractmethod
    def support(self) -> int:
        '''Number of grid points the kernel spans.'''
        pass

    @property
    @abstractmethod
    def base_offset(self) -> int:
        '''Offset from floor(g) to the first grid point involved.'''
        pass

    @abstractmethod
    def weights(self, dx: NDArray[np.floating]) -> Tuple[NDArray[np.floating], ...]:
        '''Compute interpolation weights for fractional position dx in [0, 1).'''
        pass

    @property
    def order(self) -> int:
        '''Interpolation order.'''
        return self.support - 1


class NGPKernel(InterpolationKernel):
    '''
    Implements Nearest Grid Point (NGP) interpolation in a hacky way.
    Defines a 2-point kernel with a step weight function to select the
    nearest grid point based on the fractional position.
    '''

    @property
    def support(self) -> int:
        return 2

    @property
    def base_offset(self) -> int:
        return 0

    def weights(self, dx: NDArray[np.floating]) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        w1 = (dx >= 0.5).astype(dx.dtype)
        w0 = 1.0 - w1
        return (w0, w1)


class CICKernel(InterpolationKernel):
    '''Implements Cloud-in-Cell (CIC) interpolation.'''

    @property
    def support(self) -> int:
        return 2

    @property
    def base_offset(self) -> int:
        return 0

    def weights(self, dx: NDArray[np.floating]) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        return (1.0 - dx, dx)


class TSCKernel(InterpolationKernel):
    '''Implements Triangular Shaped Cloud (TSC) interpolation.'''

    @property
    def support(self) -> int:
        return 3

    @property
    def base_offset(self) -> int:
        return -1

    def weights(self, dx: NDArray[np.floating]) -> Tuple[NDArray[np.floating], ...]:
        w0 = 0.5 * (1.0 - dx)**2
        w1 = 0.75 - (dx - 0.5)**2
        w2 = 0.5 * dx**2
        return (w0, w1, w2)

# Kernel registry
KERNELS: Dict[str, Type[InterpolationKernel]] = {
    'ngp': NGPKernel,
    'cic': CICKernel,
    'tsc': TSCKernel,
}


def compensation_kernel(
    kvec: NDArray[np.floating],
    nvox: Union[Tuple[int, int, int], NDArray[np.integer]],
    boxsize: Union[Tuple[float, float, float], NDArray[np.floating]],
    method: Union[str, InterpolationKernel] = 'cic',
) -> NDArray[np.floating]:
    r'''
    Compute the deconvolution kernel that compensates for the smoothing
    introduced by a particle-mesh interpolation scheme.

    An interpolation kernel of order :math:`p` (NGP: 0, CIC: 1, TSC: 2)
    acts as a low-pass filter in Fourier space with transfer function
    :math:`\tilde{W}(\mathbf{k}) = \prod_i \mathrm{sinc}^{p+1}(\pi k_i / 2 k_{\mathrm{Ny},i})`.
    To recover unsmoothed field values after interpolation, the Fourier
    modes must be multiplied by :math:`\tilde{W}^{-1}` **before** the
    inverse FFT.

    Parameters
    ----------
    kvec : ndarray of shape (3, nx, ny, nz_half)
        Wavevector components from ``fourier_grid`` (Hermitian layout).
    nvox : tuple of int or array of shape (3,)
        Number of grid cells in each dimension.
    boxsize : tuple of float or array of shape (3,)
        Physical size of the box.
    method : str or InterpolationKernel
        Interpolation method: 'ngp', 'cic', 'tsc', or a kernel instance.

    Returns
    -------
    ndarray, real, same shape as ``kvec[0]``
        Multiplicative correction factor :math:`W^{-1}(\mathbf{k})`.

    Notes
    -----
    The correction is:

    .. math::
        W^{-1}(\mathbf{k}) = \left[
            \prod_{i=x,y,z} \mathrm{sinc}\!\left(
                \frac{\pi\, k_i}{2\, k_{\mathrm{Ny},i}}
            \right)
        \right]^{-(p+1)}

    where :math:`k_{\mathrm{Ny},i} = \pi N_i / L_i` and :math:`p` is
    the interpolation order (``kernel.support - 1``).

    This should be applied to Fourier-space fields **before** ``irfftn``
    when interpolating onto particles that do not coincide with grid
    nodes (e.g., glass initial conditions or arbitrary distributions).
    For particles on a regular lattice matching the grid, the correction
    is unnecessary.

    References
    ----------
    Hockney & Eastwood (1988), §5-3; Jing (2005), eq. 20;
    monofonIC implementation in ``grid_interpolate.hh``.
    '''
    if isinstance(method, str):
        kernel = KERNELS[method.lower()]()
    else:
        kernel = method

    nvox = np.asarray(nvox, dtype=np.float64)
    boxsize = np.asarray(boxsize, dtype=np.float64)

    # Nyquist wavenumber per axis: k_Ny = pi * N / L
    k_ny = np.pi * nvox / boxsize  # shape (3,)

    # sinc(pi * k_i / (2 * k_Ny_i)) = sinc(k_i * L_i / (2 * N_i))
    # Using np.sinc which computes sin(pi*x)/(pi*x), so we need the
    # argument WITHOUT the pi factor:
    #   sinc_arg_i = k_i / (2 * k_Ny_i) = k_i * L_i / (2 * pi * N_i)
    #
    # But we want sin(pi * k_i / (2*k_Ny)) / (pi * k_i / (2*k_Ny)),
    # which is np.sinc(k_i / (2*k_Ny)):
    power = kernel.order + 1  # = kernel.support

    W = np.ones_like(kvec[0])
    for i in range(3):
        arg = kvec[i] / (2.0 * k_ny[i])  # dimensionless, in [0, 0.5]
        W *= np.sinc(arg) ** power  # np.sinc(x) = sin(pi*x)/(pi*x)

    # Return the inverse: the deconvolution factor
    # Guard against division by zero (should not happen for k < k_Ny,
    # but protect anyway)
    return np.where(np.abs(W) > 1e-15, 1.0 / W, 1.0)


class FieldInterpolator:
    '''
    Interpolates a 3D field to particle positions using separable kernels.

    Parameters
    ----------
    kernel : str or InterpolationKernel
        Interpolation kernel: 'ngp', 'cic', 'tsc', or a custom kernel instance.
    geometry : GridGeometry
        Grid geometry specification.

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
    '''

    def __init__(
        self, 
        kernel: Union[str, InterpolationKernel],
        geometry: GridGeometry,
    ):
        if isinstance(kernel, str):
            kernel = KERNELS[kernel.lower()]()
        self.kernel = kernel
        self.geometry = geometry

    def __call__(
            self,
            field: NDArray[np.floating],
            pos: NDArray[np.floating]
        ) -> NDArray[np.floating]:
        '''
        Evaluate the gridded vector ``field`` at arbitrary positions.

        Parameters
        ----------
        field : ndarray
            Shape of (ncomp, nx, ny, nz) for vector.
        pos : ndarray of shape (N, 3)
            Physical positions.

        Returns
        -------
        ndarray
            Interpolated field values at the input positions, shape (N, ncomp).
        '''
        # Convert positions to fractional grid coordinates
        grid = self.geometry.pos_to_grid(pos)
        grid_int = np.floor(grid).astype(np.int32, copy=False)
        grid_frc = grid - grid_int

        # Compute weights along each axis; shape (support, N)
        wx, wy, wz = (
            np.vstack(self.kernel.weights(grid_frc[:, i])) for i in range(3)
        )

        # Promote result type to float to avoid integer truncation; shape (N, ncomp)
        res_dtype = np.result_type(field.dtype, np.float64)
        result = np.zeros((pos.shape[0], field.shape[0]), dtype=res_dtype)

        # Loop over all combinations of kernel offsets
        nvox = self.geometry.nvox
        periodic = self.geometry.periodic

        for dx in range(self.kernel.support):
            ix = grid_int[:, 0] + self.kernel.base_offset + dx
            ix = ix % nvox[0] if periodic[0] else np.clip(ix, 0, nvox[0]-1)

            for dy in range(self.kernel.support):
                iy = grid_int[:, 1] + self.kernel.base_offset + dy
                iy = iy % nvox[1] if periodic[1] else np.clip(iy, 0, nvox[1]-1)

                for dz in range(self.kernel.support):
                    iz = grid_int[:, 2] + self.kernel.base_offset + dz
                    iz = iz % nvox[2] if periodic[2] else np.clip(iz, 0, nvox[2]-1)

                    w = wx[dx] * wy[dy] * wz[dz]    # shape (N,)
                    fvals = field[:, ix, iy, iz].T  # shape (N, ncomp)
                    result += w[:, np.newaxis] * fvals

        return result.squeeze()


def interpolate_field(
    pos: NDArray[np.floating],
    field: NDArray[np.floating],
    boxsize: Union[Tuple[float, float, float], NDArray[np.floating]],
    periodic: Union[bool, Tuple[float, float, float], NDArray[np.bool_]] = True,
    method: str = 'cic',
    vox_offset: float = 0.0,
    origin: Optional[Union[Tuple[float, float, float], NDArray[np.floating]]] = None,
) -> NDArray[np.floating]:
    '''
    Interpolate a 3D field at particle positions.

    Drop-in replacement for existing code. For repeated interpolations
    on the same grid, prefer creating a FieldInterpolator directly.

    Parameters
    ----------
    pos : array of shape (N, 3)
        Particle positions in physical coordinates.
    field : array of shape (ncomp, nx, ny, nz)
        Values of a vector field on the grid. Use ``ncomp=1`` for scalar fields.
    boxsize : array of shape (3,)
        Physical size of the periodic box.
    periodic : bool or array of shape (3,) of bool
        Periodicity in each dimension (default: all True).
    method : str
        Interpolation method: 'ngp', 'cic', or 'tsc' (default: 'cic').
    vox_offset : float
        Grid offset: 0.5 for voxel-centered, 0.0 for node-centered (default: 0.0).
    origin : array of shape (3,) or None
        Physical coordinate of the box corner. Default is (0, 0, 0).
        For a centered grid, use -boxsize/2.

    Returns
    -------
    values : array of shape (N, ncomp)
        Interpolated field values.
    '''
    geometry = GridGeometry(
        boxsize=boxsize,
        nvox=field.shape[1:],
        vox_offset=vox_offset,
        origin=origin,
        periodic=periodic,
    )
    interp = FieldInterpolator(method, geometry)
    return interp(field, pos)


def interpolate_field_deprecated(x, field, dk, method='linear'):
    r'''
    Interpolate a grid-based field onto particle positions using periodic
    boundaries.

    Parameters
    ----------
    x : ndarray of shape (N, 3)
        Particle positions in physical [Mpc].
    field : ndarray
        The grid-based field (e.g. a displacement field) defined on a
        regular grid.
    dk : float
        The uniform step size in each dimension, calculated as the length
        of the shortest dimension divided by the number of voxels in
        that dimension.
    method : str
        The interpolation method to use. This can be 'linear', 'nearest',
        or 'cubic'. The default is 'linear'.

    Returns
    -------
    interp_values : ndarray of shape (N,)
        Field values interpolated at the particle positions.
    '''
    nvox = field.shape
    mesh = tuple(np.arange(-(n-1)*dk/2, n*dk/2, dk) for n in nvox)
    interpolator = RegularGridInterpolator(
        points=mesh,
        values=field,
        method=method,
        bounds_error=False,
        fill_value=None  # Extrapolate using periodic wrapping if needed
    )
    return interpolator(x)