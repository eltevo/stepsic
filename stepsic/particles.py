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

"""Construct regular, random, spherical, and cylindrical particle loads."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stepsic._typing import FloatVec3, IntVec3, RealField, Seed
from stepsic.geometry import CylindricalBinner, SphericalBinner, create_binner
from stepsic.rng import RNG

log = logging.getLogger(__name__)


def create_grid(
        nvox: IntVec3,
        dk: float,
        dtype: np.dtype = np.float64,
        return_coords: bool = True
) -> tuple[RealField, RealField] | RealField:
    """
    Create a regular grid for the simulation box.

    Parameters
    ----------
    nvox : tuple of int
        The number of voxels in each dimension of the simulation box.
    dk : float
        Physical cell spacing in the same length unit as the box.
    dtype : np.dtype
        The data type of the returned particle positions. The grid axes
        are always computed in float64 and cast on assignment, so
        ``dtype=np.float32`` gives values identical to casting the
        float64 result.
    return_coords : bool
        If `True` (default), also build and return the full 3D coordinate
        array. Pass `False` to skip that allocation when only the
        particle positions are needed.

    Returns
    -------
    particles : ndarray of shape (N, 3)
        The particle positions in the simulation box, where N is the total
        number of particles (voxels).
    coords : ndarray of shape (3, Nx, Ny, Nz)
        The grid coordinates in each dimension, where Nx, Ny, Nz are the
        number of voxels in each dimension. Only returned when
        ``return_coords`` is `True`.
    """
    mesh = tuple(np.arange(-(n-1)*dk/2, n*dk/2, dk) for n in nvox)
    shape = tuple(m.size for m in mesh)
    bcast = (
        mesh[0][:, None, None],
        mesh[1][None, :, None],
        mesh[2][None, None, :],
    )
    grid = np.empty((*shape, 3), dtype=dtype)
    for i in range(3):
        grid[..., i] = bcast[i]
    particles = grid.reshape(-1, 3)
    if not return_coords:
        return particles
    coords = np.empty((3, *shape), dtype=np.float64)
    for i in range(3):
        coords[i] = bcast[i]
    return particles, coords


def create_particles(npart: int, boxsize: FloatVec3, seed: Seed = None) -> RealField:
    r"""
    Create particles uniformly distributed in ``[0, boxsize)``.

    Parameters
    ----------
    npart : int
        The number of particles to create.
    boxsize : float or list of float
        Box dimensions in [Mpc]. Can be a scalar for a cubical box or
        an array in the form of `(Lx, Ly, Lz)` for a rectangular cuboid.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    particles : ndarray of shape (npart, 3)
        Particle positions in the simulation box.
    """
    rng = RNG(seed=seed)
    return rng.uniform(size=(npart, 3), seed=None) * np.array(boxsize)


def _random_unit_vectors_sphere(n: int, rng: RNG, seed: int | None = None) -> NDArray:
    r"""
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
    """
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


def shell_masses(
    binner: SphericalBinner | CylindricalBinner,
    n_bins: int,
    n_per_shell: int,
    rho_mean: float,
    Lz: float | None = None,
) -> NDArray[np.float64]:
    r"""
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
    """
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
    r"""
    Find the bin index containing radius ``r``.

    Performs a linear scan from bin 0, returning the first *i* such
    that ``binner.r_limit(i+1) > r``.  This is equivalent to
    ``floor(arctan(r/r_4d) / d_omega)`` for linear binners but works
    for any binner type.

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
    """
    i = 0
    while binner.r_limit(i + 1) <= r:
        i += 1
    return i


@dataclass(frozen=True, slots=True)
class _RcritZones:
    r"""
    Pre-computed two-zone layout for a constant-resolution inner volume.

    Parameters
    ----------
    i_crit : int
        First bin index that lies *outside* the constant-resolution
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
    """
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
    r"""
    Compute the two-zone particle layout for a constant-resolution interior.

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
    """
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
        f'RCRIT zone layout: i_crit={i_crit}, '
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


def create_spherical_shells(
    binner: SphericalBinner,
    n_bins: int,
    n_per_shell: int,
    rho_mean: float,
    seed: int | None = None,
    r_crit: float | None = None,
) -> tuple[NDArray, NDArray]:
    r"""
    Generate particles in concentric spherical shells for a spherical
    (:math:`\mathbb{R}^3`) StePS simulation.

    Each shell contains ``n_per_shell`` particles placed at random
    angular positions on the sphere, with radial positions drawn
    uniformly within the shell volume (i.e., uniform in :math:`r^3`
    between shell edges).

    When ``r_crit`` is given, the volume inside the critical radius is
    filled with a single pool of equal-mass particles (constant
    resolution), and only the exterior shells follow the standard
    multiresolution scheme.

    Parameters
    ----------
    binner : SphericalBinner
        The radial binning strategy defining shell edges.
    n_bins : int
        Number of radial bins.
    n_per_shell : int
        Number of particles per shell (used for exterior shells when
        ``r_crit`` is set, or for all shells otherwise).
    rho_mean : float
        Mean matter density in internal units.
    seed : int or None
        Random seed for reproducibility.
    r_crit : float or None
        If set, defines the radius of the constant-resolution inner
        volume.  Only valid for ``BIN_MODE='omega'``.

    Returns
    -------
    pos : ndarray of shape (N_total, 3)
        Particle positions in Cartesian coordinates [Mpc/h].
    mass : ndarray of shape (N_total,)
        Particle masses in internal mass units.
    """
    rng = RNG(seed=seed)

    if r_crit is not None:
        zones = _compute_rcrit_zones(
            binner, n_bins, n_per_shell, rho_mean, r_crit,
        )
        N_total = zones.n_inside + zones.n_outside
        pos = np.empty((N_total, 3), dtype=np.float64)
        mass = np.empty(N_total, dtype=np.float64)

        log.info(
            f'Generating {N_total} particles: {zones.n_inside} inside '
            f'RCRIT + {zones.n_outside} in {n_bins - zones.i_crit} '
            f'exterior shells ({n_per_shell} per shell)...'
        )

        # --- Interior: uniform random fill inside r_boundary ---
        r_boundary = binner.r_limit(zones.i_crit)
        unit_vecs = _random_unit_vectors_sphere(zones.n_inside, rng)
        u = rng.uniform(size=(zones.n_inside,))
        radii = np.cbrt(u * r_boundary**3)
        pos[:zones.n_inside] = radii[:, np.newaxis] * unit_vecs
        mass[:zones.n_inside] = zones.mass_inside

        # --- Exterior: standard multiresolution shells ---
        offset = zones.n_inside
        for j_ext, j_bin in enumerate(range(zones.i_crit, n_bins)):
            r0 = binner.r_limit(j_bin)
            r1 = binner.r_limit(j_bin + 1)
            start = offset + j_ext * n_per_shell
            end = start + n_per_shell

            unit_vecs = _random_unit_vectors_sphere(n_per_shell, rng)
            u = rng.uniform(size=(n_per_shell,))
            radii = np.cbrt(u * (r1**3 - r0**3) + r0**3)

            pos[start:end] = radii[:, np.newaxis] * unit_vecs
            mass[start:end] = zones.masses_outside[j_ext]

    else:
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

            unit_vecs = _random_unit_vectors_sphere(n_per_shell, rng)
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
    r_crit: float | None = None,
) -> tuple[NDArray, NDArray]:
    r"""
    Generate particles in concentric cylindrical annuli for a
    cylindrical (:math:`S^1 \times \mathbb{R}^2`) StePS simulation.

    Each annulus contains ``n_per_shell`` particles placed at random
    angular positions on the circle and uniform z-positions within
    :math:`[0, L_z]`, with radial positions drawn uniformly within the
    annulus area (i.e., uniform in :math:`r^2` between annulus edges).

    When ``r_crit`` is given, the volume inside the critical radius is
    filled with a single pool of equal-mass particles (constant
    resolution), and only the exterior annuli follow the standard
    multiresolution scheme.

    Parameters
    ----------
    binner : CylindricalBinner
        The radial binning strategy defining annulus edges.
    n_bins : int
        Number of radial bins.
    n_per_shell : int
        Number of particles per annulus (used for exterior annuli when
        ``r_crit`` is set, or for all annuli otherwise).
    Lz : float
        Height of the cylinder (periodic z-direction) [Mpc/h].
    rho_mean : float
        Mean matter density in internal units.
    seed : int or None
        Random seed for reproducibility.
    r_crit : float or None
        If set, defines the radius of the constant-resolution inner
        volume. Only valid for ``BIN_MODE='omega'``.

    Returns
    -------
    pos : ndarray of shape (N_total, 3)
        Particle positions in Cartesian coordinates [Mpc/h].
        The z-coordinate is in :math:`[0, L_z]`.
    mass : ndarray of shape (N_total,)
        Particle masses in internal mass units.
    """
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
            f'exterior annuli ({n_per_shell} per annulus, Lz={Lz:.2f})...'
        )

        # --- Interior: uniform random fill inside r_boundary cylinder ---
        r_boundary = binner.r_limit(zones.i_crit)
        n_in = zones.n_inside

        theta = rng.uniform(size=(n_in,)) * 2 * np.pi
        u = rng.uniform(size=(n_in,))
        radii = np.sqrt(u * r_boundary**2)

        pos[:n_in, 0] = radii * np.cos(theta)
        pos[:n_in, 1] = radii * np.sin(theta)
        pos[:n_in, 2] = rng.uniform(size=(n_in,)) * Lz
        mass[:n_in] = zones.mass_inside

        # --- Exterior: standard multiresolution annuli ---
        offset = n_in
        for j_ext, j_bin in enumerate(range(zones.i_crit, n_bins)):
            r0 = binner.r_limit(j_bin)
            r1 = binner.r_limit(j_bin + 1)
            start = offset + j_ext * n_per_shell
            end = start + n_per_shell

            theta = rng.uniform(size=(n_per_shell,)) * 2 * np.pi
            u = rng.uniform(size=(n_per_shell,))
            radii = np.sqrt(u * (r1**2 - r0**2) + r0**2)

            pos[start:end, 0] = radii * np.cos(theta)
            pos[start:end, 1] = radii * np.sin(theta)
            pos[start:end, 2] = rng.uniform(size=(n_per_shell,)) * Lz

            mass[start:end] = zones.masses_outside[j_ext]

    else:
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

            theta = rng.uniform(size=(n_per_shell,)) * 2 * np.pi
            u = rng.uniform(size=(n_per_shell,))
            radii = np.sqrt(u * (r1**2 - r0**2) + r0**2)

            pos[start:end, 0] = radii * np.cos(theta)
            pos[start:end, 1] = radii * np.sin(theta)
            pos[start:end, 2] = rng.uniform(size=(n_per_shell,)) * Lz

            mass[start:end] = masses_per_bin[j]

    log.info(f'Total mass: {np.sum(mass)*1e11:.6e} Msol')
    return pos, mass


def create_shell_particles(params: dict) -> tuple[NDArray, NDArray]:
    r"""
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
        ``'R_3D'``, ``'NSHELL'``, ``'RHO_MEAN'``, and ``'LBOX'``
        (for cylindrical). Optional: ``'SEED'`` and ``'RCRIT'``
        (only for ``BIN_MODE='omega'``).

    Returns
    -------
    pos : ndarray of shape (N, 3)
        Particle positions [Mpc/h].
    mass : ndarray of shape (N,)
        Particle masses in internal mass units.
    """
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

    if geometry == 'spherical':
        pos, mass = create_spherical_shells(
            binner, n_bins, n_per_shell, rho_mean,
            seed=seed, r_crit=r_crit,
        )
    elif geometry == 'cylindrical':
        Lz = np.min(params['LBOX'])
        pos, mass = create_cylindrical_shells(
            binner, n_bins, n_per_shell, Lz, rho_mean,
            seed=seed, r_crit=r_crit,
        )
    else:
        raise ValueError(f"Unknown GEOMETRY '{geometry}'.")

    return pos, mass
