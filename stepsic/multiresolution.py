"""Production LPT fields accumulated at fixed particle masses."""

from __future__ import annotations

import logging

import numpy as np
from tabulate import tabulate

from stepsic._typing import FloatVec3, RealField

from stepsic.field import cubic_voxels, generate_delta_k, white_noise
from stepsic.lpt import lpt1, lpt2, log_lpt


log = logging.getLogger(__name__)

# Number of selected particle rows added to the final arrays per batch.
_ACCUMULATION_PARTICLES_PER_CHUNK = 1_000_000


def _mass_interp_weights(
        mass: RealField,
        mass_tab: RealField
) -> tuple[np.ndarray, np.ndarray, RealField]:
    """
    Bracketing sample indices and weights for linear interpolation in mass.

    Parameters
    ----------
    mass : ndarray of shape (N,)
        Particle masses.
    mass_tab : ndarray of shape (S,)
        Sample masses ordered from the largest to the smallest, as
        produced by :func:`create_nres_mass_map`.

    Returns
    -------
    j_lo, j_hi : ndarray of shape (N,) of int
        Indices into ``mass_tab`` of the two samples bracketing each
        particle's mass.
    w_hi : ndarray of shape (N,) of float
        Weight of the ``j_hi`` sample, such that
        ``(1 - w_hi) * samples[j_lo] + w_hi * samples[j_hi]`` reproduces
        ``np.interp`` on the ascending mass axis (linear inside the
        table, constant extrapolation outside).
    """
    mass = np.asarray(mass, dtype=np.float64)
    mass_tab = np.asarray(mass_tab, dtype=np.float64)
    nsamp = mass_tab.size
    if nsamp == 1:
        zeros = np.zeros(mass.shape, dtype=np.intp)
        return zeros, zeros, np.zeros(mass.shape)

    asc = mass_tab[::-1]  # ascending masses
    hi = np.clip(np.searchsorted(asc, mass, side='right'), 1, nsamp - 1)
    lo = hi - 1
    span = asc[hi] - asc[lo]
    # A zero span only occurs for duplicated table masses;
    # -> weight 0 then selects the lower duplicate.
    w_hi = np.where(span > 0, (mass - asc[lo]) / np.where(span > 0, span, 1.0), 0.0)
    w_hi = np.clip(w_hi, 0.0, 1.0)  # constant extrapolation outside the table
    return nsamp - 1 - lo, nsamp - 1 - hi, w_hi


def _common_cell_count(boxsize: FloatVec3, limit: float) -> int:
    """
    Smallest even cell count of the shortest side that fits every side.

    With this many cubic cells along the shortest side, every side of the
    box holds an even whole number of cells, and so does every mesh whose
    count is a multiple of it.

    Parameters
    ----------
    boxsize : ndarray
        Box lengths ``(Lx, Ly, Lz)``.
    limit : float
        Largest count to accept.

    Returns
    -------
    int
        The cell count.

    Raises
    ------
    ValueError
        If no count up to ``limit`` fits every side.
    """
    ratio = np.asarray(boxsize, dtype=np.float64) / np.min(boxsize)
    for count in range(2, int(limit) + 1, 2):
        cells = ratio * count
        whole = np.rint(cells)
        if np.all(np.abs(cells - whole) <= 1e-9 * cells) and np.all(whole % 2 == 0):
            return count
    raise ValueError(
        f'The box {np.asarray(boxsize).tolist()} has no mesh of cubic cells that spans it '
        f'with at most {int(limit)} cells along its shortest side, the count of the '
        f'coarsest level. Choose sides with a simple ratio, e.g. a cylinder diameter '
        f'that is a multiple of half the period.')


def create_nres_mass_map(
        n_grid_samples: int,
        mass_list: RealField,
        M_box: float,
        boxsize: FloatVec3,
        silent=False
) -> tuple[RealField, RealField]:
    """
    Map ascending particle masses to a coarse-to-fine mesh hierarchy.

    Parameters
    ----------
    n_grid_samples : int
        Number of grids with different resolutions.
    mass_list : ndarray
        Nonempty array of unique particle masses in ascending order.
    M_box : float
        Total mass in the simulation box, in the same units as ``mass_list``.
    boxsize : ndarray
        Box lengths ``(Lx, Ly, Lz)`` in Mpc/h.

    Returns
    -------
    nres_tab : ndarray of shape (n_grid_samples,)
        Shortest-axis cell counts, stored as floating-point values. Each
        request is rounded up to a multiple of the count returned by
        :func:`_common_cell_count`, so that every level spans the same box.
        Pass each to :func:`~stepsic.field.cubic_voxels` to obtain three
        even mesh dimensions and the physical cell spacing.
    mass_tab : ndarray of shape (n_grid_samples,)
        Corresponding sample masses in descending order. The coarsest
        mesh represents the largest mass.

    Raises
    ------
    ValueError
        If the box sides admit no common count up to that of the coarsest
        level.
    """
    nres_list = np.min(boxsize) // np.cbrt(np.prod(boxsize) * mass_list / M_box)
    idx = np.linspace(
        0, mass_list.size - 1, n_grid_samples, endpoint=True, dtype=int)

    # Populate lookup table by starting with the outermost mass bin
    nres_tab = nres_list[idx[::-1]]
    mass_tab = mass_list[idx[::-1]]

    # White-noise modes are shared between levels by integer index, which is
    # the same physical wave on every level only if every mesh spans the box.
    coarsest = nres_tab.min() + nres_tab.min() % 2
    multiple = _common_cell_count(boxsize, coarsest)
    nres_tab = multiple * np.ceil(nres_tab / multiple)

    log.info('The generated resolution-mass map:')
    nvox = np.zeros((n_grid_samples, 3), dtype=int)
    for i in range(n_grid_samples):
        nvox[i], dk = cubic_voxels(nres_tab[i], boxsize)
    if not silent:
        print(tabulate([*zip(nvox[:,0], nvox[:,1], nvox[:,2], mass_tab)],
                   headers=['Resolution_x', 'Resolution_y', 'Resolution_z', 'Mass [1e11 Msol/h]'],
                   floatfmt=('.0f', '.0f', '.0f', '.6f')))
    return nres_tab, mass_tab


def density_samples(k_h_mpc, power_mpc_h3, hierarchy, params):
    """Yield Fourier-space density fields in hierarchy order.

    Parameters
    ----------
    k_h_mpc : ndarray
        Positive input wavenumbers in h/Mpc, in strictly increasing order.
    power_mpc_h3 : ndarray
        Positive power spectrum values in (Mpc/h)^3 at the target redshift.
    hierarchy : sequence of tuple
        Each entry contains three integer mesh dimensions and a physical
        cell spacing in Mpc/h, ordered from coarse to fine.
    params : dict
        ``SEED`` and ``DTYPE`` control white noise; ``PAIRED`` and ``FIXED``
        control density generation. Each mesh uses the same seed.

    Yields
    ------
    ndarray of complex
        Density modes of shape ``(Nx, Ny, Nz//2+1)`` for the current mesh.
        The generator releases its field reference before advancing.
    """
    for nvox, cell_mpc_h in hierarchy:
        field = white_noise(nvox=nvox, seed=params['SEED'], dtype=params['DTYPE'])
        delta = generate_delta_k(
            k_h_mpc, power_mpc_h3, nvox, cell_mpc_h, field=field,
            paired=params['PAIRED'], fixed=params['FIXED'],
            dtype=params['DTYPE'])
        del field
        yield delta
        del delta


def multiresolution_lpt(
    positions_mpc_h, masses, mass_levels, hierarchy, density_fields, *,
    order, g1, g2, aHf1, aHf2, method, compensate,
):
    """Interpolate LPT displacements and velocities between mass levels.

    Parameters
    ----------
    positions_mpc_h : ndarray of shape (N, 3)
        Unperturbed particle positions in Mpc/h.
    masses : ndarray of shape (N,)
        Particle masses, in the same units as ``mass_levels``.
    mass_levels : ndarray
        Sample masses in descending order, one per hierarchy entry.
    hierarchy : sequence of tuple
        Nonempty coarse-to-fine sequence of (three mesh dimensions,
        physical cell spacing in Mpc/h). Sampling aligns FFT index zero
        with the finest mesh's voxel center.
    density_fields : iterable of ndarray
        Fourier-space density fields in hierarchy order, each of shape
        ``(Nx, Ny, Nz//2+1)``. Consumed one at a time.
    order : {1, 2}
        LPT order.
    g1, g2 : float
        First- and second-order displacement growth coefficients.
    aHf1, aHf2 : float
        Velocity prefactors in km/s/Mpc for each order.
    method : {'ngp', 'cic', 'tsc'}
        Particle interpolation method.
    compensate : bool
        Apply mass-assignment deconvolution before interpolation.

    Returns
    -------
    displacement : ndarray of shape (N, 3)
        Unwrapped displacements in Mpc/h, linearly interpolated in mass.
        Masses outside the table use the nearest endpoint.
    velocity : ndarray of shape (N, 3)
        Raw LPT velocities. The driver divides by ``h`` for km/s and by
        ``sqrt(a)`` for the Gadget snapshot convention.
    """
    if order not in (1, 2) or len(hierarchy) != len(mass_levels) or not hierarchy:
        raise ValueError('LPT order and mass hierarchy must be valid')
    if not np.all(np.isfinite(positions_mpc_h)) or not np.all(np.isfinite(masses)):
        raise ValueError('particle positions and masses must be finite')
    j_lo, j_hi, w_hi = _mass_interp_weights(masses, mass_levels)
    finest_cell = min(cell for _, cell in hierarchy)
    displacement = np.zeros_like(positions_mpc_h)
    velocity = np.zeros_like(positions_mpc_h)
    fields = iter(density_fields)
    for si, (nvox, cell) in enumerate(hierarchy):
        delta_k = next(fields)
        candidate = (j_lo == si) | (j_hi == si)
        indices = np.flatnonzero(candidate)
        coeff = (
            (j_lo[indices] == si) * (1.0 - w_hi[indices])
            + (j_hi[indices] == si) * w_hi[indices]
        )
        nonzero = coeff > 0.0
        indices = indices[nonzero]
        coeff = coeff[nonzero]
        if indices.size == 0:
            del delta_k
            continue

        sample_x = positions_mpc_h[indices].copy()
        sample_x += (cell - finest_cell) / 2
        kwargs = dict(delta_k=delta_k, nvox=nvox, dk=cell, g1=g1,
                      aHf1=aHf1, method=method, compensate=compensate,
                      dtype=positions_mpc_h.dtype)
        if order == 2:
            kwargs.update(g2=g2, aHf2=aHf2)
        xpert, vpert = (lpt1 if order == 1 else lpt2)(sample_x, **kwargs)
        log_lpt(sample_x, xpert, vpert, title=f'{order}LPT sample {si + 1}')
        np.subtract(xpert, sample_x, out=xpert)
        np.multiply(xpert, coeff[:, None], out=xpert)
        np.multiply(vpert, coeff[:, None], out=vpert)
        for start in range(0, indices.size, _ACCUMULATION_PARTICLES_PER_CHUNK):
            stop = min(start + _ACCUMULATION_PARTICLES_PER_CHUNK, indices.size)
            selected = indices[start:stop]
            displacement[selected] += xpert[start:stop]
            velocity[selected] += vpert[start:stop]
        del delta_k, kwargs, xpert, vpert, sample_x
    return displacement, velocity
