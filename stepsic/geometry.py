#*****************************************************************************#
#  stepsic - An initial condition generator for                               #
#           STEreographically Projected cosmological Simulations              #
#    Copyright (C) 2017-2026 Gabor Racz, Balazs Pal                           #
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
from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

from stepsic.rng import RNG

import logging
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
        simulation radius is large relative to :math:`D_S`.

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

    The stereographic angle :math:`\omega` is divided into equal steps
    of size :math:`\Delta\omega`, and the radial limit of the *i*-th
    bin is:

    .. math::
        r_i = D_S \, \tan(i \, \Delta\omega)

    where :math:`D_S = D_{4D}/2` is the radius of the compactification
    sphere.

    Parameters
    ----------
    d_s : float
        Radius of the compactification sphere (:math:`D_{4D}/2`).
    n_bins : int
        Number of radial bins.
    last_cell_size : float
        Fractional size of the outermost cell relative to the
        uniform angular step.
    '''

    def __init__(self, d_s: float, n_bins: int, last_cell_size: float):
        self.d_s = d_s
        self.n_bins = n_bins
        self.last_cell_size = last_cell_size
        self.d_omega = np.pi / (2 * (self.n_bins + self.last_cell_size))

    def r_limit(self, i: int) -> float:
        omega = i * self.d_omega
        return self.d_s * np.tan(omega)

    def r_centroid(self, i: int) -> float:
        return self.centroid(self.r_limit(i), self.r_limit(i + 1))


class SphericalConstantVolume(SphericalBinner):
    r'''
    Constant-volume binning on the compact 3-sphere.

    Each bin encloses the same volume on :math:`S^3`, which translates
    to unequal radial steps in the non-compact :math:`\mathbb{R}^3`
    space.  The bin boundaries are found by inverting the relation
    :math:`x - \sin(x) = y`.

    Parameters
    ----------
    d_s : float
        Radius of the compactification sphere (:math:`D_{4D}/2`).
    n_bins : int
        Number of radial bins.
    R_sim : float
        Maximum simulation radius in the non-compact space.
    '''

    def __init__(self, d_s: float, n_bins: int, R_sim: float):
        self.d_s = d_s
        self.n_bins = n_bins
        self.omega_max = 2 * np.arctan(R_sim / d_s)
        self.unit_bin = (
            2 * self.omega_max - np.sin(2 * self.omega_max)
        ) / n_bins

    def r_limit(self, i: int) -> float:
        # unit_bin is in terms of 2ω − sin(2ω), so after inverting
        # f(x) = x − sin(x) we obtain 2ω; divide by 2 to get ω.
        result = self.invert_x_minus_sin_x(i * self.unit_bin)
        omega = float(np.squeeze(result)) / 2
        return self.d_s * np.tan(omega)

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
        '''Return the outer edge of the *i*-th radial bin.'''
        ...

    @abstractmethod
    def r_centroid(self, i: int) -> float:
        '''Return the mass-weighted centroid of the *i*-th bin.'''
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

    Uses the same stereographic :math:`\omega` parametrization as
    :class:`SphericalLinear`, applied to the 2D non-compact plane.
    The radial limit of the *i*-th bin is:

    .. math::
        r_i = D_S \, \tan(i \, \Delta\omega)

    Notes
    -----
    This matches the legacy ``Calculate_rlimits_i`` function used for
    both spherical and cylindrical geometries in StePS_IC.

    Parameters
    ----------
    d_s : float
        Radius of the compactification sphere (:math:`D_{4D}/2`).
    n_bins : int
        Number of radial bins.
    last_cell_size : float
        Fractional size of the outermost cell relative to the
        uniform angular step.
    '''

    def __init__(self, d_s: float, n_bins: int, last_cell_size: float):
        self.d_s = d_s
        self.n_bins = n_bins
        self.last_cell_size = last_cell_size
        self.d_omega = np.pi / (2 * (self.n_bins + self.last_cell_size))

    def r_limit(self, i: int) -> float:
        omega = i * self.d_omega
        return self.d_s * np.tan(omega)

    def r_centroid(self, i: int) -> float:
        return self.centroid(self.r_limit(i), self.r_limit(i + 1))


class CylindricalConstantVolume(CylindricalBinner):
    r'''
    Constant-volume binning for a cylindrical simulation.

    Each annulus encloses the same area on the compact 2-sphere,
    following the same :math:`x - \sin(x)` inversion as the spherical
    case but applied to 2D stereographic angles.

    Notes
    -----
    This matches the legacy ``Calculate_rlimits_i_2D_cvol`` function
    in StePS_IC.

    Parameters
    ----------
    d_s : float
        Radius of the compactification sphere (:math:`D_{4D}/2`).
    n_bins : int
        Number of radial bins.
    R_sim : float
        Maximum simulation radius in the non-compact space.
    '''

    def __init__(self, d_s: float, n_bins: int, R_sim: float):
        self.d_s = d_s
        self.n_bins = n_bins
        self.omega_max = 2 * np.arctan(R_sim / d_s)
        self.unit_bin = (
            2 * self.omega_max - np.sin(2 * self.omega_max)
        ) / n_bins

    def r_limit(self, i: int) -> float:
        result = SphericalBinner.invert_x_minus_sin_x(i * self.unit_bin)
        omega = float(np.squeeze(result)) / 2
        return self.d_s * np.tan(omega)

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
    d_s = params['D_4D'] / 2.0
    n_bins = params['NRBINS']
    R_sim = params['R_3D']

    if geometry == 'spherical':
        if bin_mode == 'omega':
            last_cell_size = (
                n_bins * np.pi / (2 * np.arctan(R_sim / d_s)) - n_bins
            )
            return SphericalLinear(d_s, n_bins, last_cell_size)
        elif bin_mode == 'volume':
            return SphericalConstantVolume(d_s, n_bins, R_sim)
        else:
            raise ValueError(f"Unknown BIN_MODE '{bin_mode}' for spherical geometry.")

    elif geometry == 'cylindrical':
        if bin_mode == 'omega':
            last_cell_size = (
                n_bins * np.pi / (2 * np.arctan(R_sim / d_s)) - n_bins
            )
            return CylindricalLinear(d_s, n_bins, last_cell_size)
        elif bin_mode == 'volume':
            return CylindricalConstantVolume(d_s, n_bins, R_sim)
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

    Notes
    -----
    Uses rejection sampling within the unit cube, matching the legacy
    ``Generate_random_shell`` implementation in StePS_IC.

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


def create_spherical_shells(
    binner: SphericalBinner,
    n_bins: int,
    n_per_shell: int,
    rho_mean: float,
    seed: int | None = None,
) -> tuple[NDArray, NDArray]:
    r'''
    Generate particles in concentric spherical shells for a spherical
    (:math:`\mathbb{R}^3`) StePS simulation.

    Each shell contains ``n_per_shell`` particles placed at random
    angular positions on the sphere, with radial positions drawn
    uniformly within the shell volume (i.e., uniform in :math:`r^3`
    between shell edges).

    Parameters
    ----------
    binner : SphericalBinner
        The radial binning strategy defining shell edges.
    n_bins : int
        Number of radial bins.
    n_per_shell : int
        Number of particles per shell.
    rho_mean : float
        Mean matter density in internal units.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    pos : ndarray of shape (N_total, 3)
        Particle positions in Cartesian coordinates [Mpc].
    mass : ndarray of shape (N_total,)
        Particle masses in internal mass units.

    Notes
    -----
    The radial placement within each shell uses the cube-root sampling

    .. math::
        r = \left(u \cdot (r_1^3 - r_0^3) + r_0^3\right)^{1/3}

    where :math:`u \sim \mathrm{Uniform}(0, 1)`, to ensure uniform
    volume density.
    
    Notes
    -----
    This matches the legacy implementation in StePS_IC.
    '''
    rng = RNG(seed=seed)
    N_total = n_bins * n_per_shell
    pos = np.empty((N_total, 3), dtype=np.float64)
    mass = np.empty(N_total, dtype=np.float64)

    masses_per_bin = shell_masses(binner, n_bins, n_per_shell, rho_mean)
    log.info(
        f'Generating {N_total} particles in {n_bins} spherical shells '
        f'({n_per_shell} per shell)...'
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

        # Random unit direction vectors on S^2
        unit_vecs = _random_unit_vectors_sphere(n_per_shell, rng)

        # Radial positions: uniform in volume; sample r^3 uniformly
        u = rng.uniform(size=(n_per_shell,))
        radii = np.cbrt(u * (r1**3 - r0**3) + r0**3)

        pos[start:end] = radii[:, np.newaxis] * unit_vecs
        mass[start:end] = masses_per_bin[j]

    log.info(f'Total mass: {np.sum(mass)*1e11:.6e} Msol')
    return pos, mass


def create_cylindrical_shells(
    binner: CylindricalBinner,
    n_bins: int,
    n_per_shell: int,
    Lz: float,
    rho_mean: float,
    seed: int | None = None,
) -> tuple[NDArray, NDArray]:
    r'''
    Generate particles in concentric cylindrical annuli for a
    cylindrical (:math:`S^1 \times \mathbb{R}^2`) StePS simulation.

    Each annulus contains ``n_per_shell`` particles placed at random
    angular positions on the circle and uniform z-positions within
    :math:`[0, L_z]`, with radial positions drawn uniformly within the
    annulus area (i.e., uniform in :math:`r^2` between annulus edges).

    Parameters
    ----------
    binner : CylindricalBinner
        The radial binning strategy defining annulus edges.
    n_bins : int
        Number of radial bins.
    n_per_shell : int
        Number of particles per annulus.
    Lz : float
        Height of the cylinder (periodic z-direction) [Mpc].
    rho_mean : float
        Mean matter density in internal units.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    pos : ndarray of shape (N_total, 3)
        Particle positions in Cartesian coordinates [Mpc].
        The z-coordinate is in :math:`[0, L_z]`.
    mass : ndarray of shape (N_total,)
        Particle masses in internal mass units.

    Notes
    -----
    The radial placement within each annulus uses

    .. math::
        r = \sqrt{u \cdot (r_1^2 - r_0^2) + r_0^2}

    where :math:`u \sim \mathrm{Uniform}(0, 1)`, to ensure uniform
    area density in the (x, y) plane.

    '''
    rng = RNG(seed=seed)
    N_total = n_bins * n_per_shell
    pos = np.empty((N_total, 3), dtype=np.float64)
    mass = np.empty(N_total, dtype=np.float64)

    masses_per_bin = shell_masses(
        binner, n_bins, n_per_shell, rho_mean, Lz=Lz
    )
    log.info(
        f'Generating {N_total} particles in {n_bins} cylindrical annuli '
        f'({n_per_shell} per annulus, Lz={Lz:.2f} Mpc)...'
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

        # Random angular positions on the circle
        theta = rng.uniform(size=(n_per_shell,)) * 2 * np.pi

        # Radial positions: uniform in area; sample r^2 uniformly
        u = rng.uniform(size=(n_per_shell,))
        radii = np.sqrt(u * (r1**2 - r0**2) + r0**2)

        pos[start:end, 0] = radii * np.cos(theta)
        pos[start:end, 1] = radii * np.sin(theta)
        pos[start:end, 2] = rng.uniform(size=(n_per_shell,)) * Lz

        mass[start:end] = masses_per_bin[j]

    log.info(f'Total mass: {np.sum(mass)*1e11:.6e} Msol')
    return pos, mass


def create_shell_particles(params: dict) -> tuple[NDArray, NDArray]:
    r'''
    High-level factory for shell-based particle generation.

    Constructs the appropriate binner from the parameter dictionary
    and generates particles matching the StePS stereographic geometry.

    This function should be called when ``TYPE = 'shell'`` is selected
    in the configuration.

    Parameters
    ----------
    params : dict
        Full parameter dictionary.  Must contain:
        ``'GEOMETRY'``, ``'BIN_MODE'``, ``'D_4D'``, ``'NRBINS'``,
        ``'R_3D'``, ``'NSHELL'``, ``'SEED'``, ``'H0'``, ``'RHO_MEAN'``,
        ``'OMEGA_M'``, ``'H'``, and ``'LBOX'`` (for cylindrical).

    Returns
    -------
    pos : ndarray of shape (N, 3)
        Particle positions [Mpc].
    mass : ndarray of shape (N,)
        Particle masses in internal mass units.

    Raises
    ------
    ValueError
        If ``GEOMETRY`` is ``'cubical'`` (use grid/random instead).
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

    if geometry == 'spherical':
        pos, mass = create_spherical_shells(
            binner, n_bins, n_per_shell, rho_mean, seed=seed,
        )
    elif geometry == 'cylindrical':
        Lz = np.min(params['LBOX'])
        pos, mass = create_cylindrical_shells(
            binner, n_bins, n_per_shell, Lz, rho_mean, seed=seed,
        )
    else:
        raise ValueError(f"Unknown GEOMETRY '{geometry}'.")

    return pos, mass