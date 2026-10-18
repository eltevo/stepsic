"""Read StePS_Pk ASCII output."""

from __future__ import annotations

import numpy as np


def load_pk_measured(path: str) -> tuple[np.ndarray, np.ndarray]:
    '''Load the k, P(k) columns from a StePS_Pk ASCII output file.

    Parameters
    ----------
    path : str
        Path to the whitespace-delimited ASCII file with columns
        k [1/Mpc] and P(k) [Mpc^3].

    Returns
    -------
    k : ndarray
        Wavenumber array (positive, finite entries only).
    pk : ndarray
        Power spectrum values (positive, finite entries only).
    '''
    data = np.atleast_2d(np.loadtxt(path, comments="#"))
    k = data[:, 0]
    pk = data[:, 1]
    mask = np.isfinite(k) & np.isfinite(pk) & (k > 0) & (pk > 0)
    return k[mask], pk[mask]
