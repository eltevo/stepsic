"""Seed-level uncertainty and numerical bounds for relative field errors."""

from __future__ import annotations

import numpy as np
from scipy.stats import t


def simultaneous_band(relative_errors, numerical_error, *, confidence=0.95):
    """Bonferroni Student intervals for independent seed means.

    Columns are the complete, predeclared comparison family. Paired members
    must be averaged before calling. Coverage assumes normally distributed
    seed means; for nonlinear/windowed statistics it is asymptotic. The
    non-negative numerical bound widens both ends of each interval.
    """
    samples = np.asarray(relative_errors, dtype=np.float64)
    error = np.asarray(numerical_error, dtype=np.float64)
    if samples.ndim != 2 or min(samples.shape) < 1:
        raise ValueError('relative errors must be a non-empty seed by bin array')
    if not np.all(np.isfinite(samples)):
        raise ValueError('relative errors must be finite')
    if error.shape != samples.shape[1:] or np.any(error < 0) or not np.all(np.isfinite(error)):
        raise ValueError('numerical error must be a finite non-negative bound per bin')
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError('confidence must lie between zero and one')
    mean = samples.mean(axis=0)
    if len(samples) < 2:
        return mean, None, None
    quantile = t.ppf(1 - (1-confidence)/(2*samples.shape[1]), len(samples)-1)
    width = quantile * samples.std(axis=0, ddof=1) / np.sqrt(len(samples)) + error
    return mean, mean - width, mean + width


def supported_intervals(coordinates, accepted):
    """Return all contiguous supported intervals on an ordered measured axis."""
    coordinate = np.asarray(coordinates, dtype=np.float64)
    mask = np.asarray(accepted)
    if coordinate.ndim != 1 or len(coordinate) == 0 or mask.shape != coordinate.shape:
        raise ValueError('coordinates and acceptance must be non-empty one-dimensional peers')
    if mask.dtype != np.bool_ or not np.all(np.isfinite(coordinate)) or np.any(np.diff(coordinate) <= 0):
        raise ValueError('coordinates must increase and acceptance must be boolean')
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return np.column_stack((coordinate[np.flatnonzero(edges == 1)],
                            coordinate[np.flatnonzero(edges == -1)-1]))
