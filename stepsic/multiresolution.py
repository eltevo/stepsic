"""Production LPT fields accumulated at fixed particle masses."""

import numpy as np

from stepsic.field import _mass_interp_weights, generate_delta_k, white_noise
from stepsic.lpt import lpt1, lpt2, log_lpt


def _density_samples(k_h_mpc, power_mpc_h3, hierarchy, params):
    for nvox, cell_mpc_h in hierarchy:
        field = white_noise(nvox=nvox, seed=params['SEED'], dtype=params['DTYPE'])
        delta = generate_delta_k(
            k_h_mpc, power_mpc_h3, nvox, cell_mpc_h, field=field,
            paired=params['PAIRED'], fixed=params['FIXED'])
        del field
        yield delta
        del delta


def _multiresolution_lpt(
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
