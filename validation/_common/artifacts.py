"""Helpers for numerical results stored in NumPy archives."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]


def load_archive(path: str) -> dict[str, np.ndarray]:
    """Load a typed NumPy archive without permitting pickle payloads."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def histogram(
    values: ArrayF,
    nbins: int,
    *,
    range: tuple[float, float] | None = None,
) -> tuple[ArrayF, ArrayF, ArrayF]:
    """Return histogram bin centres, float64 counts, and bin widths."""
    if nbins <= 0:
        raise ValueError("nbins must be positive")
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("histogram values must be non-empty and finite")
    counts, edges = np.histogram(values, bins=nbins, range=range)
    return 0.5 * (edges[:-1] + edges[1:]), counts.astype(np.float64), np.diff(edges)
