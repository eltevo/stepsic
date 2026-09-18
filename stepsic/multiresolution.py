"""Production LPT fields accumulated at fixed particle masses."""

from __future__ import annotations

import logging

import numpy as np
from tabulate import tabulate

from stepsic._typing import FloatVec3, RealField

from stepsic.field import cubic_voxels, generate_delta_k, white_noise
from stepsic.lpt import lpt1, lpt2, log_lpt


log = logging.getLogger(__name__)


def _mass_interp_weights(
        mass: RealField,
        mass_tab: RealField
) -> tuple[np.ndarray, np.ndarray, RealField]:
    '''
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
    '''
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


def create_nres_mass_map(
        n_grid_samples: int,
        mass_list: RealField,
        M_box: float,
        boxsize: FloatVec3
) -> tuple[RealField, RealField]:
    '''
    Creates a lookup table for the number of voxels per mass bin
    for a variable resolution grid in a regular StePS simulation.

    Parameters
    ----------
    n_grid_samples : int
        Number of grids with different resolutions.
    mass_list : ndarray
        Array containing the sorted unique particle masses.
    M_box : float
        Total mass in the simulation box (in 1e11 Msol).
    boxsize : ndarray
        Box dimensions in [Mpc]. Can be a scalar for a cubical box or
        an array in the form of `(Lx, Ly, Lz)` for a rectangular cuboid.

    Returns
    -------
    nres_tab : ndarray of shape (n_grid_samples,)
        Array containing the number of resolution elements for each grid.
    mass_tab : ndarray of shape (n_grid_samples,)
        Array containing the mass values corresponding to each grid.
    '''
    nres_list = np.min(boxsize) // np.cbrt(np.prod(boxsize) * mass_list / M_box)
    idx = np.linspace(
        0, mass_list.size - 1, n_grid_samples, endpoint=True, dtype=int)

    # Populate lookup table by starting with the outermost mass bin
    nres_tab = nres_list[idx[::-1]]
    mass_tab = mass_list[idx[::-1]]

    log.info('The generated resolution-mass map:') #The full nresx x nresy x nresz grid resolution is printed here
    nvox = np.zeros((n_grid_samples, 3), dtype=int)
    for i in range(n_grid_samples):
        nvox[i], dk = cubic_voxels(nres_tab[i], boxsize)
    print(tabulate([*zip(nvox[:,0], nvox[:,1], nvox[:,2], mass_tab)],
                   headers=['Resolution_x', 'Resolution_y', 'Resolution_z', 'Mass [1e11 Msol/h]'],
                   floatfmt=('.0f', '.0f', '.0f', '.6f')))
    return nres_tab, mass_tab


def density_samples(k_h_mpc, power_mpc_h3, hierarchy, params):
    for nvox, cell_mpc_h in hierarchy:
        field = white_noise(nvox=nvox, seed=params['SEED'], dtype=params['DTYPE'])
        delta = generate_delta_k(
            k_h_mpc, power_mpc_h3, nvox, cell_mpc_h, field=field,
            paired=params['PAIRED'], fixed=params['FIXED'])
        del field
        yield delta
        del delta


def multiresolution_lpt(
    positions_mpc_h, masses, mass_levels, hierarchy, density_fields, *,
    order, g1, g2, aHf1, aHf2, method, compensate,
):
    """Return unwrapped displacements [Mpc/h] and raw LPT velocities.

    Each hierarchy entry is (three mesh dimensions, physical cell size in
    Mpc/h). FFT index zero is registered to the finest mesh's voxel centre.
    Raw velocities receive the Hubble and snapshot scaling in the driver.
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
        sample_x = positions_mpc_h + (cell - finest_cell) / 2
        delta_k = next(fields)
        kwargs = dict(delta_k=delta_k, nvox=nvox, dk=cell, g1=g1,
                      aHf1=aHf1, method=method, compensate=compensate)
        if order == 2:
            kwargs.update(g2=g2, aHf2=aHf2)
        xpert, vpert = (lpt1 if order == 1 else lpt2)(sample_x, **kwargs)
        log_lpt(sample_x, xpert, vpert, title=f'{order}LPT sample {si + 1}')
        coeff = (np.where(j_lo == si, 1.0 - w_hi, 0.0)
                 + np.where(j_hi == si, w_hi, 0.0))
        sel = coeff != 0.0
        displacement[sel] += coeff[sel, None] * (xpert[sel] - sample_x[sel])
        velocity[sel] += coeff[sel, None] * vpert[sel]
        del delta_k, kwargs, xpert, vpert
    return displacement, velocity
