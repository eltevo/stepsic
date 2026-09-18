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

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root


class SphericalBinner(ABC):
    r"""
    Base class for radial binning strategies in the non-compact
    (:math:`\mathbb{R}^3`) space of a spherical StePS simulation.

    Subclasses implement :meth:`r_limit` to define the radial bin edges.
    """

    @abstractmethod
    def r_limit(self, i: int) -> float:
        """Return the outer edge of the i-th radial bin."""
        ...


    def shell_volume(self, i: int) -> float:
        r"""Volume of the i-th spherical shell.

        .. math::
            V_i = \frac{4\pi}{3}\left(r_{i+1}^3 - r_i^3\right)
        """
        r0, r1 = self.r_limit(i), self.r_limit(i + 1)
        return 4.0 / 3.0 * np.pi * (r1**3 - r0**3)


    @staticmethod
    def invert_x_minus_sin_x(
        y: float | NDArray[np.floating],
        *,
        method: str = 'hybr',
        tol: float = 1e-06,
    ) -> NDArray[np.floating]:
        r"""
        Solve :math:`x - \sin(x) = y` for :math:`x \in [0, 2\pi]`.

        The function :math:`f(x) = x - \sin(x)` is monotonically
        increasing on :math:`[0, 2\pi]` with :math:`f(0) = 0` and
        :math:`f(2\pi) = 2\pi`.  This solver handles the full range
        needed by constant-volume binning, where the doubled-angle form
        :math:`2\omega - \sin(2\omega)` can reach :math:`2\pi` when the
        simulation radius is large relative to :math:`R_{4D}`.

        Uses ``scipy.optimize.root`` with an initial guess of
        :math:`x_0 = y + \sin(y)` (first-order Kepler-equation trick).

        Parameters
        ----------
        y : float or ndarray
            Target value(s); must satisfy :math:`0 \le y \le 2\pi`.
        method : str, optional
            Root-finding algorithm (default ``'hybr'``).
        tol : float, optional
            Absolute tolerance for the solver.

        Returns
        -------
        x : ndarray
            Solution(s) in :math:`[0, 2\pi]`.
        """
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        if not np.all((y >= 0.0) & (y <= 2 * np.pi + 1e-10)):
            raise ValueError(f'y must be in [0, 2π], got y={y}')

        # Kepler-equation starter in the interior. Near either endpoint the
        # derivative vanishes, so use x - sin(x) ~ x^3/6 (and its reflection)
        # to keep the solver away from the flat endpoint.
        x0 = y + np.sin(y)
        near_lower = y < 0.1
        near_upper = y > 2 * np.pi - 0.1
        x0 = np.where(near_lower, np.cbrt(6.0 * y), x0)
        x0 = np.where(
            near_upper,
            2 * np.pi - np.cbrt(6.0 * (2 * np.pi - y)),
            x0,
        )
        x0 = np.clip(x0, 0.0, 2 * np.pi)

        sol = root(lambda x: x - np.sin(x) - y,
                   x0=x0, method=method, tol=tol)
        if not sol.success:
            raise RuntimeError(f'Root finding failed: {sol.message}')
        return sol.x


class SphericalLinear(SphericalBinner):
    r"""
    Equal angular-step binning in the non-compact :math:`\mathbb{R}^3`
    space.

    The angular step is :math:`\Delta\alpha`, and the radial limit
    of the *i*-th bin is:

    .. math::
        r_i = R_{4D} \, \tan(i \, \Delta\alpha)

    where :math:`R_{4D} = D_{4D}/2`. The angular coordinate here is
    defined by :math:`\alpha = \arctan(r/R_{4D})`.

    Parameters
    ----------
    r_4d : float
        Radius of the compactification sphere (:math:`D_{4D}/2`).
    n_bins : int
        Number of radial bins.
    last_cell_size : float
        Fractional size of the outermost cell relative to the
        uniform angular step.
    """

    def __init__(self, r_4d: float, n_bins: int, last_cell_size: float):
        self.r_4d = r_4d
        self.n_bins = n_bins
        self.last_cell_size = last_cell_size
        self.d_omega = np.pi / (2 * (self.n_bins + self.last_cell_size))

    def r_limit(self, i: int) -> float:
        omega = i * self.d_omega
        return self.r_4d * np.tan(omega)


class SphericalConstantVolume(SphericalBinner):
    r"""
    Constant-volume binning on the compact 3-sphere.

    Each bin encloses the same volume on :math:`S^3`, which translates
    to unequal radial steps in the non-compact :math:`\mathbb{R}^3`
    space.  The bin boundaries are found by inverting the relation
    :math:`x - \sin(x) = y`.

    The stereographic projection is:

    .. math::
        r = D_s \, \tan\!\left(\frac{\omega}{2}\right), \qquad
        \omega = 2 \arctan\!\left(\frac{r}{D_s}\right),

    where :math:`D_s = 2\,R_{4D}` is the diameter of the compactification
    sphere and :math:`\omega \in [0, \pi)` is the hyperspherical arc
    angle measured from the tangent point.

    Parameters
    ----------
    r_4d : float
        Radius of the compactification sphere (:math:`R_{4D} = D_{4D}/2`).
    n_bins : int
        Number of radial bins.
    r_3d : float
        Maximum simulation radius in the non-compact space.
    """

    def __init__(self, r_4d: float, n_bins: int, r_3d: float):
        self.r_4d = r_4d
        self.n_bins = n_bins
        self.omega_max = 2 * np.arctan(r_3d / (2 * r_4d))
        self.unit_bin = (
            2 * self.omega_max - np.sin(2 * self.omega_max)
        ) / n_bins

    def r_limit(self, i: int) -> float:
        # unit_bin is in terms of 2ω − sin(2ω), so after inverting
        # f(x) = x − sin(x) we obtain 2ω; divide by 2 to get ω.
        result = self.invert_x_minus_sin_x(i * self.unit_bin)
        omega = float(np.squeeze(result)) / 2
        return 2.0 * self.r_4d * np.tan(omega / 2)


class CylindricalBinner(ABC):
    r"""
    Base class for radial binning strategies in the non-compact
    :math:`\mathbb{R}^2` plane of a cylindrical
    (:math:`S^1 \times \mathbb{R}^2`) StePS simulation.
    """

    @abstractmethod
    def r_limit(self, i: int) -> float:
        """Return the outer edge of the *i*-th radial bin."""
        ...


    def shell_volume(self, i: int, Lz: float) -> float:
        r"""Volume of the i-th cylindrical annulus.

        .. math::
            V_i = \pi\,(r_{i+1}^2 - r_i^2) \, L_z

        Parameters
        ----------
        Lz : float
            Height of the cylinder (periodic direction).
        """
        r0, r1 = self.r_limit(i), self.r_limit(i + 1)
        return np.pi * (r1**2 - r0**2) * Lz


class CylindricalLinear(CylindricalBinner):
    r"""
    Equal angular-step binning for a cylindrical simulation.

    Uses the same internal half-angle parametrization as
    :class:`SphericalLinear`, applied to the 2D non-compact plane.
    The radial limit of the *i*-th bin is:

    .. math::
        r_i = R_{4D} \, \tan(i \, \Delta\alpha)

    where :math:`\alpha = \arctan(r/R_{4D})`
    (see :class:`SphericalLinear` for details).

    Parameters
    ----------
    r_4d : float
        Radius of the compactification sphere (:math:`D_{4D}/2`).
    n_bins : int
        Number of radial bins.
    last_cell_size : float
        Fractional size of the outermost cell relative to the
        uniform angular step.
    """

    def __init__(self, r_4d: float, n_bins: int, last_cell_size: float):
        self.r_4d = r_4d
        self.n_bins = n_bins
        self.last_cell_size = last_cell_size
        self.d_omega = np.pi / (2 * (self.n_bins + self.last_cell_size))

    def r_limit(self, i: int) -> float:
        omega = i * self.d_omega
        return self.r_4d * np.tan(omega)


class CylindricalConstantVolume(CylindricalBinner):
    r"""
    Constant-volume binning for a cylindrical simulation.

    Each annulus encloses the same area on the compact 2-sphere
    :math:`S^2`.  The cap area between angles :math:`0` and
    :math:`\omega` is :math:`2\pi(1 - \cos\omega)`, so equal-area bins
    satisfy :math:`1 - \cos\omega_i = i \cdot \Delta`, with closed-form

    .. math::
        \tan^2(\omega/2) = \frac{1 - \cos\omega}{1 + \cos\omega}
                         = \frac{u}{2 - u}, \qquad u = 1 - \cos\omega.

    The stereographic projection follows Racz (2018), Eq. (4-5):

    .. math::
        r = D_s \, \tan\!\left(\frac{\omega}{2}\right), \qquad
        \omega = 2 \arctan\!\left(\frac{r}{D_s}\right),

    where :math:`D_s = 2\,R_{4D}`.

    Parameters
    ----------
    r_4d : float
        Radius of the compactification sphere (:math:`D_{4D}/2`).
    n_bins : int
        Number of radial bins.
    r_3d : float
        Maximum simulation radius in the non-compact space.
    """

    def __init__(self, r_4d: float, n_bins: int, r_3d: float):
        self.r_4d = r_4d
        self.n_bins = n_bins
        self.omega_max = 2 * np.arctan(r_3d / (2 * r_4d))
        self.unit_bin = (1 - np.cos(self.omega_max)) / n_bins

    def r_limit(self, i: int) -> float:
        u = i * self.unit_bin
        return 2.0 * self.r_4d * np.sqrt(u / (2 - u))


def create_binner(params: dict) -> SphericalBinner | CylindricalBinner:
    r"""
    Create the appropriate radial binner from a parameter dictionary.

    Parameters
    ----------
    params : dict
        Must contain at minimum ``'GEOMETRY'``, ``'BIN_MODE'``,
        ``'D_4D'``, ``'NRBINS'``, and ``'R_3D'``.

    Returns
    -------
    SphericalBinner or CylindricalBinner
        The constructed binner instance.
    """
    geometry = params['GEOMETRY']
    bin_mode = params['BIN_MODE']
    r_4d = params['D_4D'] / 2.0
    n_bins = params['NRBINS']
    r_3d = params['R_3D']

    if geometry == 'spherical':
        if bin_mode == 'omega':
            # Linear binner uses internal half-angle alpha = arctan(r/r_4d),
            # NOT the paper omega = 2*arctan(r/D_s). The half-angle at the
            # simulation edge is alpha_max = arctan(r_3d / r_4d), and d_alpha
            # is chosen so that (n_bins + last_cell_size) * d_alpha = pi/2.
            last_cell_size = (
                n_bins * np.pi / (2 * np.arctan(r_3d / r_4d)) - n_bins
            )
            return SphericalLinear(r_4d, n_bins, last_cell_size)
        elif bin_mode == 'volume':
            return SphericalConstantVolume(r_4d, n_bins, r_3d)
        else:
            raise ValueError(f"Unknown BIN_MODE '{bin_mode}' for spherical geometry.")

    elif geometry == 'cylindrical':
        if bin_mode == 'omega':
            # Same half-angle convention as spherical linear; see above.
            last_cell_size = (
                n_bins * np.pi / (2 * np.arctan(r_3d / r_4d)) - n_bins
            )
            return CylindricalLinear(r_4d, n_bins, last_cell_size)
        elif bin_mode == 'volume':
            return CylindricalConstantVolume(r_4d, n_bins, r_3d)
        else:
            raise ValueError(f"Unknown BIN_MODE '{bin_mode}' for cylindrical geometry.")

    elif geometry == 'cubical':
        raise ValueError(
            "Cubical geometry does not use radial binning. "
            "Use TYPE='grid' or TYPE='random' instead of TYPE='shell'."
        )
    else:
        raise ValueError(f"Unknown GEOMETRY '{geometry}'.")
