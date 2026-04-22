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

import functools
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

from stepsic.rng import RNG

log = logging.getLogger(__name__)


class SphericalBinner(ABC):
    r'''
    Base class for radial binning strategies in the non-compact
    (:math:`\mathbb{R}^3`) space of a spherical StePS simulation.

    Subclasses must implement :meth:`r_limit` and :meth:`r_centroid`
    to define the bin edges and mass-weighted centroids respectively.
    '''

    @abstractmethod
    def r_limit(self, i: int) -> float:
        '''Return the outer edge of the i-th radial bin.'''
        ...

    @abstractmethod
    def r_centroid(self, i: int) -> float:
        '''Return the mass-weighted centroid of the i-th bin.'''
        ...

    def shell_volume(self, i: int) -> float:
        r'''Volume of the i-th spherical shell.

        .. math::
            V_i = \frac{4\pi}{3}\left(r_{i+1}^3 - r_i^3\right)
        '''
        r0, r1 = self.r_limit(i), self.r_limit(i + 1)
        return 4.0 / 3.0 * np.pi * (r1**3 - r0**3)

    @staticmethod
    def centroid(r0: float, r1: float) -> float:
        r'''
        Mass-weighted centroid of a spherical shell between :math:`r_0`
        and :math:`r_1`.

        For a uniform-density shell, the mass-weighted radial centroid
        is:

        .. math::
            \bar{r} = \frac{\int_{r_0}^{r_1} r \cdot r^2 \, dr}
                           {\int_{r_0}^{r_1} r^2 \, dr}
                    = \frac{3}{4} \;
                      \frac{r_1^4 - r_0^4}{r_1^3 - r_0^3}

        which can be rewritten as the form used in the code:

        .. math::
            \bar{r} = \frac{r_1 - r_0}{4} \;
                      \frac{r_0^2 + 2\,r_0\,r_1 + 3\,r_1^2}
                           {r_0^2 + r_0\,r_1 + r_1^2}
                      + r_0

        Parameters
        ----------
        r0 : float
            Inner edge of the shell.
        r1 : float
            Outer edge of the shell.

        Returns
        -------
        float
            The mass-weighted radial centroid.
        '''
        # Rewritten from 3/4 * (r1^4 - r0^4) / (r1^3 - r0^3) for
        # numerical stability when r0 ~ r1.
        return (
            0.25 * (r1 - r0)
            * (r0*r0 + 2*r0*r1 + 3*r1*r1)
            / (r0*r0 + r0*r1 + r1*r1)
            + r0
        )

    @staticmethod
    def invert_x_minus_sin_x(
        y: Union[float, NDArray[np.floating]],
        *,
        method: str = 'hybr',
        tol: float = 1e-06,
    ) -> NDArray[np.floating]:
        r'''
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

        Raises
        ------
        ValueError
            If any element of ``y`` is outside :math:`[0, 2\pi]`.
        RuntimeError
            If the root finder fails to converge.
        '''
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        if not np.all((y >= 0.0) & (y <= 2 * np.pi + 1e-10)):
            raise ValueError(f'y must be in [0, 2π], got y={y}')

        # Better initial guess via Kepler equation starter
        x0 = y + np.sin(y)
        x0 = np.clip(x0, 0.0, 2 * np.pi)

        sol = root(lambda x: x - np.sin(x) - y,
                   x0=x0, method=method, tol=tol)
        if not sol.success:
            raise RuntimeError(f'Root finding failed: {sol.message}')
        return sol.x


class SphericalLinear(SphericalBinner):
    r'''
    Equal angular-step binning in the non-compact :math:`\mathbb{R}^3`
    space.

    Uses an internal half-angle parametrization
    :math:`\alpha \equiv \omega/2`, where :math:`\omega` is the
    hyperspherical arc angle from Racz (2018). The angular step is
    :math:`\Delta\alpha`, and the radial limit of the `i`-th bin is:

    .. math::
        r_i = R_{4D} \, \tan(i \, \Delta\alpha)

    where :math:`R_{4D} = D_{4D}/2`.  This is equivalent to the paper
    formula :math:`r = 2 R_{4D} \tan(\omega/2)` with
    :math:`\alpha = \omega/2`.

    Parameters
    ----------
    r_4d : float
        Radius of the compactification sphere (:math:`D_{4D}/2`).
    n_bins : int
        Number of radial bins.
    last_cell_size : float
        Fractional size of the outermost cell relative to the
        uniform angular step.
    '''

    def __init__(self, r_4d: float, n_bins: int, last_cell_size: float):
        self.r_4d = r_4d
        self.n_bins = n_bins
        self.last_cell_size = last_cell_size
        self.d_omega = np.pi / (2 * (self.n_bins + self.last_cell_size))

    def r_limit(self, i: int) -> float:
        omega = i * self.d_omega
        return self.r_4d * np.tan(omega)

    def r_centroid(self, i: int) -> float:
        return self.centroid(self.r_limit(i), self.r_limit(i + 1))


class SphericalConstantVolume(SphericalBinner):
    r'''
    Constant-volume binning on the compact 3-sphere.

    Each bin encloses the same volume on :math:`S^3`, which translates
    to unequal radial steps in the non-compact :math:`\mathbb{R}^3`
    space.  The bin boundaries are found by inverting the relation
    :math:`x - \sin(x) = y`.

    The stereographic projection follows:

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
    '''

    def __init__(self, r_4d: float, n_bins: int, r_3d: float):
        self.r_4d = r_4d
        self.n_bins = n_bins
        self.omega_max = 2 * np.arctan(r_3d / (2 * r_4d))
        self.unit_bin = (
            2 * self.omega_max - np.sin(2 * self.omega_max)
        ) / n_bins

    def r_limit(self, i: int) -> float:
        # unit_bin is in terms of 2*omega − sin(2*omega), so after
        # inverting f(x) = x − sin(x) we obtain 2*omega.
        result = self.invert_x_minus_sin_x(i * self.unit_bin)
        omega = float(np.squeeze(result)) / 2
        return 2.0 * self.r_4d * np.tan(omega / 2)

    def r_centroid(self, i: int) -> float:
        return self.centroid(self.r_limit(i), self.r_limit(i + 1))


class CylindricalBinner(ABC):
    r'''
    Base class for radial binning strategies in the non-compact
    :math:`\mathbb{R}^2` plane of a cylindrical
    (:math:`S^1 \times \mathbb{R}^2`) StePS simulation.
    '''

    @abstractmethod
    def r_limit(self, i: int) -> float:
        '''Return the outer edge of the `i`-th radial bin.'''
        ...

    @abstractmethod
    def r_centroid(self, i: int) -> float:
        '''Return the mass-weighted centroid of the `i`-th bin.'''
        ...

    def shell_volume(self, i: int, Lz: float) -> float:
        r'''Volume of the i-th cylindrical annulus.

        .. math::
            V_i = \pi\,(r_{i+1}^2 - r_i^2) \, L_z

        Parameters
        ----------
        Lz : float
            Height of the cylinder (periodic direction).
        '''
        r0, r1 = self.r_limit(i), self.r_limit(i + 1)
        return np.pi * (r1**2 - r0**2) * Lz

    @staticmethod
    def centroid(r0: float, r1: float) -> float:
        r'''
        Mass-weighted centroid of a cylindrical annulus between
        :math:`r_0` and :math:`r_1` (assuming uniform height).

        .. math::
            \bar{r} = \frac{\int_{r_0}^{r_1} r \cdot r \, dr}
                           {\int_{r_0}^{r_1} r \, dr}
                    = \frac{2}{3} \;
                      \frac{r_1^3 - r_0^3}{r_1^2 - r_0^2}

        Parameters
        ----------
        r0 : float
            Inner edge of the annulus.
        r1 : float
            Outer edge of the annulus.

        Returns
        -------
        float
            The mass-weighted radial centroid.
        '''
        return 2.0 / 3.0 * (r1**3 - r0**3) / (r1**2 - r0**2)


class CylindricalLinear(CylindricalBinner):
    r'''
    Equal angular-step binning for a cylindrical simulation.

    Uses the same internal half-angle parametrization as
    :class:`SphericalLinear`, applied to the 2D non-compact plane.
    The radial limit of the `i`-th bin is:

    .. math::
        r_i = R_{4D} \, \tan(i \, \Delta\alpha)

    where :math:`\alpha = \omega/2` is the half-angle
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
    '''

    def __init__(self, r_4d: float, n_bins: int, last_cell_size: float):
        self.r_4d = r_4d
        self.n_bins = n_bins
        self.last_cell_size = last_cell_size
        self.d_omega = np.pi / (2 * (self.n_bins + self.last_cell_size))

    def r_limit(self, i: int) -> float:
        omega = i * self.d_omega
        return self.r_4d * np.tan(omega)

    def r_centroid(self, i: int) -> float:
        return self.centroid(self.r_limit(i), self.r_limit(i + 1))


class CylindricalConstantVolume(CylindricalBinner):
    r'''
    Constant-volume binning for a cylindrical simulation.

    Each annulus encloses the same area on the compact 2-sphere,
    following the same :math:`x - \sin(x)` inversion as the spherical
    case but applied to 2D stereographic angles.

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
    '''

    def __init__(self, r_4d: float, n_bins: int, r_3d: float):
        self.r_4d = r_4d
        self.n_bins = n_bins
        self.omega_max = 2 * np.arctan(r_3d / (2 * r_4d))
        self.unit_bin = (
            2 * self.omega_max - np.sin(2 * self.omega_max)
        ) / n_bins

    def r_limit(self, i: int) -> float:
        result = SphericalBinner.invert_x_minus_sin_x(i * self.unit_bin)
        omega = float(np.squeeze(result)) / 2
        return 2.0 * self.r_4d * np.tan(omega / 2)

    def r_centroid(self, i: int) -> float:
        return self.centroid(self.r_limit(i), self.r_limit(i + 1))


def create_binner(params: dict) -> SphericalBinner | CylindricalBinner:
    r'''
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

    Raises
    ------
    ValueError
        On invalid geometry or binning mode combinations.
    '''
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
    else:  # parameters.py already checks this, but you can never be too careful
        raise ValueError(f"Unknown GEOMETRY '{geometry}'.")
    
def _random_unit_vectors_sphere(n: int, rng: RNG, seed: int | None = None) -> NDArray:
    r'''
    Generate ``n`` uniformly distributed unit vectors on the 2-sphere.

    Parameters
    ----------
    n : int
        Number of unit vectors to generate.
    rng : RNG
        Random number generator instance.
    seed : int or None
        Optional seed override.

    Returns
    -------
    ndarray of shape (n, 3)
        Unit vectors uniformly distributed on :math:`S^2`.
    '''
    vectors = np.empty((n, 3), dtype=np.float64)
    filled = 0
    while filled < n:
        # Generate in batches for efficiency
        batch_size = min(2 * (n - filled), 100_000)
        candidates = rng.uniform(size=(batch_size, 3), seed=seed) - 0.5
        radii = np.linalg.norm(candidates, axis=1)
        valid = radii <= 0.5
        valid_vecs = candidates[valid] / radii[valid, np.newaxis]
        take = min(valid_vecs.shape[0], n - filled)
        vectors[filled : filled + take] = valid_vecs[:take]
        filled += take
    return vectors


def _random_unit_vectors_cylinder(
    n: int,
    Lz: float,
    rng: RNG,
    seed: int | None = None,
) -> NDArray:
    r'''
    Legacy function.

    Generate ``n`` points uniformly distributed on the surface of a
    unit-radius cylinder of height ``Lz``, for use as cylindrical shell
    particles.

    The (x, y) components are unit vectors on the circle (shell surface),
    and the z component is uniform in :math:`[0, L_z]`.

    Parameters
    ----------
    n : int
        Number of points to generate.
    Lz : float
        Height of the cylinder (periodic z-direction).
    rng : RNG
        Random number generator instance.
    seed : int or None
        Optional seed override.

    Returns
    -------
    ndarray of shape (n, 3)
        Points with unit-radius (x, y) and z in [0, Lz].

    Notes
    -----
    Legacy function to match the original cylindrical shell particle
    generation in StePS_IC. Unused.
    '''
    vectors = np.empty((n, 3), dtype=np.float64)
    filled = 0
    while filled < n:
        batch_size = min(2 * (n - filled), 100_000)
        xy = rng.uniform(size=(batch_size, 2), seed=seed) - 0.5
        radii = np.linalg.norm(xy, axis=1)
        valid = radii <= 0.5
        unit_xy = xy[valid] / radii[valid, np.newaxis]
        take = min(unit_xy.shape[0], n - filled)
        vectors[filled : filled + take, :2] = unit_xy[:take]
        vectors[filled : filled + take, 2] = (
            rng.uniform(size=(take,), seed=seed) * Lz
        )
        filled += take
    return vectors


def shell_masses(
    binner: SphericalBinner | CylindricalBinner,
    n_bins: int,
    n_per_shell: int,
    rho_mean: float,
    Lz: float | None = None,
) -> NDArray[np.float64]:
    r'''
    Compute the particle mass in each radial bin.

    .. math::
        m_i = \frac{\rho_\mathrm{mean} \, V_i}{N_\mathrm{shell}}

    Parameters
    ----------
    binner : SphericalBinner or CylindricalBinner
        The radial binning strategy.
    n_bins : int
        Number of radial bins.
    n_per_shell : int
        Number of particles in each shell (constant across bins).
    rho_mean : float
        Mean matter density in internal units.
    Lz : float or None
        Height of the cylinder; required for cylindrical binners.

    Returns
    -------
    masses : ndarray of shape (n_bins,)
        Particle mass for each bin in internal mass units.
    '''
    masses = np.empty(n_bins, dtype=np.float64)
    for i in range(n_bins):
        if isinstance(binner, SphericalBinner):
            V = binner.shell_volume(i)
        elif isinstance(binner, CylindricalBinner):
            if Lz is None:
                raise ValueError(
                    "Lz (cylinder height) is required for cylindrical binners."
                )
            V = binner.shell_volume(i, Lz)
        else:
            raise TypeError(f"Unknown binner type: {type(binner)}")
        masses[i] = rho_mean * V / n_per_shell
    return masses


def _bin_index_for_radius(
    binner: SphericalBinner | CylindricalBinner,
    r: float,
) -> int:
    r'''
    Find the bin index containing radius ``r``.

    Parameters
    ----------
    binner : SphericalBinner or CylindricalBinner
        The radial binning strategy.
    r : float
        Target radius in the non-compact space.

    Returns
    -------
    int
        Zero-based bin index.
    '''
    i = 0
    while binner.r_limit(i + 1) <= r:
        i += 1
    return i


@dataclass(frozen=True, slots=True)
class _RcritZones:
    r'''
    Pre-computed two-zone layout for a constant-resolution inner volume.

    Parameters
    ----------
    i_crit : int
        First bin index that lies outside the constant-resolution
        region.  Shells ``[0, i_crit)`` are replaced by the uniform
        interior pool.
    n_inside : int
        Number of particles in the constant-resolution interior.
    mass_inside : float
        Per-particle mass inside the constant-resolution region.
    n_outside : int
        Number of particles in the multiresolution exterior shells.
    masses_outside : ndarray of shape (n_bins - i_crit,)
        Per-particle mass for each exterior shell.
    '''
    i_crit: int
    n_inside: int
    mass_inside: float
    n_outside: int
    masses_outside: NDArray[np.float64]


def _compute_rcrit_zones(
    binner: SphericalBinner | CylindricalBinner,
    n_bins: int,
    n_per_shell: int,
    rho_mean: float,
    r_crit: float,
    Lz: float | None = None,
) -> _RcritZones:
    r'''
    Plan the two-zone particle layout for a constant-resolution interior.

    Given a critical radius, this function determines how many particles
    go inside and outside, and what their masses are.

    Parameters
    ----------
    binner : SphericalBinner or CylindricalBinner
        The radial binning strategy.
    n_bins : int
        Total number of radial bins.
    n_per_shell : int
        Number of particles per exterior shell.
    rho_mean : float
        Mean matter density in internal units.
    r_crit : float
        Critical radius for the constant-resolution zone.
    Lz : float or None
        Height of the cylinder; required for cylindrical binners.

    Returns
    -------
    _RcritZones
        The pre-computed zone layout.

    Raises
    ------
    ValueError
        If ``r_crit`` falls outside the binning range.
    '''
    i_crit = _bin_index_for_radius(binner, r_crit)

    if i_crit >= n_bins:
        raise ValueError(
            f"RCRIT={r_crit} exceeds the simulation volume "
            f"(last bin edge: {binner.r_limit(n_bins):.4f}). "
            f"Reduce RCRIT or increase R_3D."
        )

    # Reference mass: what a particle in shell i_crit would weigh
    if isinstance(binner, SphericalBinner):
        V_ref = binner.shell_volume(i_crit)
    else:
        V_ref = binner.shell_volume(i_crit, Lz)
    mass_ref = rho_mean * V_ref / n_per_shell

    # Interior volume bounded by the outer edge of the last interior bin
    r_boundary = binner.r_limit(i_crit)
    if isinstance(binner, SphericalBinner):
        V_inside = 4.0 / 3.0 * np.pi * r_boundary**3
    else:
        V_inside = np.pi * r_boundary**2 * Lz

    # Number of interior particles at the reference mass
    n_inside = int(V_inside * rho_mean / mass_ref)
    if n_inside < 1:
        raise ValueError(
            f"RCRIT={r_crit} is too small: the interior volume "
            f"would contain fewer than 1 particle."
        )

    # Re-derive exact mass for self-consistency (integer rounding shifts it)
    mass_inside = V_inside * rho_mean / n_inside

    # Exterior masses: standard multiresolution shells from i_crit onwards
    n_ext = n_bins - i_crit
    masses_outside = shell_masses(
        binner, n_bins, n_per_shell, rho_mean, Lz=Lz,
    )[i_crit:]

    n_outside = n_ext * n_per_shell

    log.info(
        f'RCRIT zone plan: i_crit={i_crit}, '
        f'r_boundary={r_boundary:.4f} Mpc/h, '
        f'N_inside={n_inside}, N_outside={n_outside}, '
        f'N_total={n_inside + n_outside}'
    )
    log.info(
        f'Interior mass resolution: {mass_inside*1e11:.6e} Msol'
    )

    return _RcritZones(
        i_crit=i_crit,
        n_inside=n_inside,
        mass_inside=mass_inside,
        n_outside=n_outside,
        masses_outside=masses_outside,
    )


def _fill_spherical_shell(
    r0: float,
    r1: float,
    n: int,
    rng: RNG,
) -> NDArray:
    r'''
    Place *n* particles uniformly in a spherical shell :math:`[r_0, r_1]`.

    Uses cube-root sampling in :math:`r^3` to ensure uniform volume
    density, with directions drawn from :meth:`_random_unit_vectors_sphere`.

    .. math::
        r = \left(u \cdot (r_1^3 - r_0^3) + r_0^3\right)^{1/3}

    Parameters
    ----------
    r0 : float
        Inner shell edge.
    r1 : float
        Outer shell edge.
    n : int
        Number of particles.
    rng : RNG
        Random number generator instance.

    Returns
    -------
    ndarray of shape (n, 3)
        Cartesian particle positions.
    '''
    unit_vecs = _random_unit_vectors_sphere(n, rng)
    u = rng.uniform(size=(n,))
    radii = np.cbrt(u * (r1**3 - r0**3) + r0**3)
    return radii[:, np.newaxis] * unit_vecs


def _fill_cylindrical_annulus(
    r0: float,
    r1: float,
    n: int,
    rng: RNG,
    *,
    Lz: float,
) -> NDArray:
    r'''
    Place *n* particles uniformly in a cylindrical annulus
    :math:`[r_0, r_1] \times [0, L_z]`.

    Uses square-root sampling in :math:`r^2` to ensure uniform area
    density in the (x, y) plane, with z drawn uniformly in
    :math:`[0, L_z]`.

    .. math::
        r = \sqrt{u \cdot (r_1^2 - r_0^2) + r_0^2}

    Parameters
    ----------
    r0 : float
        Inner annulus edge.
    r1 : float
        Outer annulus edge.
    n : int
        Number of particles.
    rng : RNG
        Random number generator instance.
    Lz : float
        Height of the cylinder (periodic z-direction).

    Returns
    -------
    ndarray of shape (n, 3)
        Cartesian particle positions.
    '''
    pos = np.empty((n, 3), dtype=np.float64)
    theta = rng.uniform(size=(n,)) * 2 * np.pi
    u = rng.uniform(size=(n,))
    radii = np.sqrt(u * (r1**2 - r0**2) + r0**2)
    pos[:, 0] = radii * np.cos(theta)
    pos[:, 1] = radii * np.sin(theta)
    pos[:, 2] = rng.uniform(size=(n,)) * Lz
    return pos


def _create_shells(
    binner: SphericalBinner | CylindricalBinner,
    n_bins: int,
    n_per_shell: int,
    rho_mean: float,
    fill_fn: Callable[[float, float, int, RNG], NDArray],
    *,
    seed: int | None = None,
    r_crit: float | None = None,
    Lz: float | None = None,
    shell_label: str = 'shell',
) -> tuple[NDArray, NDArray]:
    r'''
    Unified particle generation for concentric-shell geometries.

    Parameters
    ----------
    binner : SphericalBinner or CylindricalBinner
        The radial binning strategy defining shell/annulus edges.
    n_bins : int
        Number of radial bins.
    n_per_shell : int
        Number of particles per shell (exterior shells when ``r_crit``
        is set, all shells otherwise).
    rho_mean : float
        Mean matter density in internal units.
    fill_fn : callable
        ``(r0, r1, n, rng) -> ndarray(n, 3)`` — places *n* particles
        uniformly in the volume between radii *r0* and *r1*.
    seed : int or None
        Random seed for reproducibility.
    r_crit : float or None
        If set, defines the radius of the constant-resolution inner
        volume.
    Lz : float or None
        Height of the cylinder; required for cylindrical binners
        (threaded through to :func:`_compute_rcrit_zones` and
        :func:`shell_masses`).
    shell_label : str
        Word to use in log messages (``'shell'`` or ``'annulus'``).

    Returns
    -------
    pos : ndarray of shape (N_total, 3)
        Particle positions in Cartesian coordinates.
    mass : ndarray of shape (N_total,)
        Particle masses in internal mass units.
    '''
    rng = RNG(seed=seed)

    if r_crit is not None:
        zones = _compute_rcrit_zones(
            binner, n_bins, n_per_shell, rho_mean, r_crit, Lz=Lz,
        )
        N_total = zones.n_inside + zones.n_outside
        pos = np.empty((N_total, 3), dtype=np.float64)
        mass = np.empty(N_total, dtype=np.float64)

        log.info(
            f'Generating {N_total} particles: {zones.n_inside} inside '
            f'RCRIT + {zones.n_outside} in {n_bins - zones.i_crit} '
            f'exterior {shell_label}s ({n_per_shell} per {shell_label})...'
        )

        # Interior: uniform fill inside r_boundary (shell with r0=0)
        r_boundary = binner.r_limit(zones.i_crit)
        pos[:zones.n_inside] = fill_fn(0.0, r_boundary, zones.n_inside, rng)
        mass[:zones.n_inside] = zones.mass_inside

        # Exterior: standard multiresolution shells
        offset = zones.n_inside
        for j_ext, j_bin in enumerate(range(zones.i_crit, n_bins)):
            r0 = binner.r_limit(j_bin)
            r1 = binner.r_limit(j_bin + 1)
            start = offset + j_ext * n_per_shell
            end = start + n_per_shell

            pos[start:end] = fill_fn(r0, r1, n_per_shell, rng)
            mass[start:end] = zones.masses_outside[j_ext]

    else:
        N_total = n_bins * n_per_shell
        pos = np.empty((N_total, 3), dtype=np.float64)
        mass = np.empty(N_total, dtype=np.float64)

        masses_per_bin = shell_masses(
            binner, n_bins, n_per_shell, rho_mean, Lz=Lz,
        )
        log.info(
            f'Generating {N_total} particles in {n_bins} {shell_label}s '
            f'({n_per_shell} per {shell_label})...'
        )
        log.info(
            f'Mass range: [{masses_per_bin[0]*1e11:.6e}, '
            f'{masses_per_bin[-1]*1e11:.6e}] Msol'
        )

        for j in range(n_bins):
            r0 = binner.r_limit(j)
            r1 = binner.r_limit(j + 1)
            start = j * n_per_shell
            end = start + n_per_shell

            pos[start:end] = fill_fn(r0, r1, n_per_shell, rng)
            mass[start:end] = masses_per_bin[j]

    log.info(f'Total mass: {np.sum(mass)*1e11:.6e} Msol')
    return pos, mass


def create_shell_particles(params: dict) -> tuple[NDArray, NDArray]:
    r'''
    High-level factory for shell-based particle generation.

    Constructs the appropriate binner from the parameter dictionary
    and generates particles matching the StePS stereographic geometry.

    When ``RCRIT`` is present and ``BIN_MODE='omega'``, the inner volume
    is filled with constant-mass particles.

    This function should be called when ``TYPE = 'shell'`` is selected
    in the configuration.

    Parameters
    ----------
    params : dict
        Full parameter dictionary.  Must contain:
        ``'GEOMETRY'``, ``'BIN_MODE'``, ``'D_4D'``, ``'NRBINS'``,
        ``'R_3D'``, ``'NSHELL'``, ``'SEED'``, ``'H0'``, ``'RHO_MEAN'``,
        ``'OMEGA_M'``, ``'H'``, and ``'LBOX'`` (for cylindrical).
        Optional: ``'RCRIT'`` (only for ``BIN_MODE='omega'``).

    Returns
    -------
    pos : ndarray of shape (N, 3)
        Particle positions.
    mass : ndarray of shape (N,)
        Particle masses in internal mass units.

    Raises
    ------
    ValueError
        If ``GEOMETRY`` is ``'cubical'``.
    '''
    geometry = params['GEOMETRY']
    if geometry == 'cubical':
        raise ValueError(
            "TYPE='shell' is not valid for cubical geometry. "
            "Use TYPE='grid' or TYPE='random' instead."
        )

    binner = create_binner(params)

    rho_mean = params['RHO_MEAN']
    n_bins = params['NRBINS']
    n_per_shell = params['NSHELL']
    seed = params.get('SEED', None)
    r_crit = params.get('RCRIT', None)

    Lz = None
    if geometry == 'spherical':
        fill_fn = _fill_spherical_shell
        shell_label = 'spherical shell'
    elif geometry == 'cylindrical':
        Lz = np.min(params['LBOX'])
        fill_fn = functools.partial(
            _fill_cylindrical_annulus, Lz=Lz,
        )
        shell_label = 'cylindrical annulus'
    else:
        raise ValueError(f"Unknown GEOMETRY '{geometry}'.")

    return _create_shells(
        binner, n_bins, n_per_shell, rho_mean, fill_fn,
        seed=seed, r_crit=r_crit, Lz=Lz, shell_label=shell_label,
    )