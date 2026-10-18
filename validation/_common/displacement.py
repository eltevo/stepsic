"""Mass-weighted pair measurements in undisplaced particle catalogues."""

import numpy as np
from scipy.stats.sampling import DiscreteAliasUrn


def pair_separations(positions_mpc_h, pairs, periodic_length_mpc_h=None):
    q = np.asarray(positions_mpc_h, dtype=np.float64)
    pairs = np.asarray(pairs)
    if q.ndim != 2 or q.shape[1] != 3 or not np.all(np.isfinite(q)):
        raise ValueError('positions must be finite three-vectors')
    if (pairs.ndim != 2 or pairs.shape[1] != 2
            or not np.issubdtype(pairs.dtype, np.integer)
            or np.any(pairs < 0) or np.any(pairs >= len(q))):
        raise ValueError('pairs must contain valid particle indices')
    delta = q[pairs[:, 0]] - q[pairs[:, 1]]
    if periodic_length_mpc_h is not None:
        length = periodic_length_mpc_h
        if not np.isfinite(length) or length <= 0:
            raise ValueError('periodic length must be finite and positive')
        delta[:, 2] -= length * np.rint(delta[:, 2] / length)
    return np.linalg.norm(delta, axis=1)


def sample_pairs(positions_mpc_h, masses, edges_mpc_h, count, *,
                 max_separation_mpc_h, seed, periodic_length_mpc_h=None):
    """Condition independent mass-weighted endpoints on each separation bin.

    Rejection preserves the product of endpoint masses within each bin.
    The geometric upper bound marks unsupported bins; a sampling shortfall
    elsewhere raises rather than being reported as an empty bin.
    """
    q = np.asarray(positions_mpc_h, dtype=np.float64)
    mass = np.asarray(masses, dtype=np.float64)
    edges = np.asarray(edges_mpc_h, dtype=np.float64)
    pair_separations(q, np.empty((0, 2), dtype=int), periodic_length_mpc_h)
    if (len(q) < 2 or mass.shape != (len(q),) or not np.all(np.isfinite(mass))
            or np.any(mass <= 0) or edges.ndim != 1 or len(edges) < 2
            or not np.all(np.isfinite(edges)) or edges[0] <= 0
            or np.any(np.diff(edges) <= 0) or not isinstance(count, int) or count < 1
            or not np.isfinite(max_separation_mpc_h) or max_separation_mpc_h <= 0):
        raise ValueError('invalid masses, separation bins, pair count or extent')
    supported = edges[:-1] < max_separation_mpc_h
    pairs = np.full((len(edges)-1, count, 2), -1, dtype=np.int64)
    counts = np.zeros(len(edges)-1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    distribution = DiscreteAliasUrn(mass / np.sum(mass, dtype=np.float64), random_state=rng)
    for _ in range(100000):
        if np.all(counts[supported] == count):
            return pairs, counts
        candidates = distribution.rvs(size=(262144, 2))
        separation = pair_separations(q, candidates, periodic_length_mpc_h)
        bins = np.searchsorted(edges, separation, side='right') - 1
        for b in np.flatnonzero(supported & (counts < count)):
            selected = candidates[bins == b][:count-counts[b]]
            pairs[b, counts[b]:counts[b]+len(selected)] = selected
            counts[b] += len(selected)
    raise RuntimeError(f'pair sampling did not fill supported bins: {counts.tolist()}')


def structure_components(displacement_mpc_h, pairs, counts):
    """Mean squared Cartesian relative displacements [(Mpc/h)^2]."""
    psi = np.asarray(displacement_mpc_h, dtype=np.float64)
    if psi.ndim != 2 or psi.shape[1] != 3 or not np.all(np.isfinite(psi)):
        raise ValueError('displacements must be finite three-vectors')
    pairs, counts = np.asarray(pairs), np.asarray(counts)
    if (pairs.ndim != 3 or pairs.shape[2] != 2 or counts.shape != pairs.shape[:1]
            or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 0) or np.any(counts > pairs.shape[1])):
        raise ValueError('invalid pair bins or counts')
    result = np.full((len(counts), 3), np.nan)
    for b, n in enumerate(counts):
        selected = pairs[b, :n]
        if not len(selected):
            continue
        pair_separations(psi, selected)
        delta = psi[selected[:, 0]] - psi[selected[:, 1]]
        result[b] = np.mean(delta * delta, axis=0, dtype=np.float64)
    return result


def rms_changes(mean_s):
    """Adjacent box changes in percent; box size is the first axis."""
    values = np.asarray(mean_s, dtype=np.float64)
    if values.ndim < 1 or values.shape[0] != 3 or np.any(np.isinf(values)):
        raise ValueError('three box sizes with finite supported variances are required')
    missing = np.isnan(values)
    if not np.all(missing == missing[:1]) or not np.any(~missing):
        raise ValueError('missing bins must agree across box sizes')
    if np.any(values[~missing] <= 0):
        raise ValueError('supported variances must be positive')
    return 100 * (np.sqrt(values[1:] / values[:-1]) - 1)


def simultaneous_intervals(seed_s, *, nbootstrap, seed, confidence):
    """Joint seed bootstrap, max-standardized deviations over displayed bins.

    Input axes are independent seed, box, then all plotted geometries/regions/
    separations. Sign partners must already have been averaged in S.
    """
    values = np.asarray(seed_s, dtype=np.float64)
    if values.ndim < 2 or values.shape[0] < 2 or nbootstrap < 2:
        raise ValueError('at least two seeds and bootstrap resamples are required')
    missing = np.isnan(values)
    if not np.all(missing == missing[:1]):
        raise ValueError('missing bins must agree across seeds')
    for value in values:
        rms_changes(value)
    estimate = rms_changes(np.mean(values, axis=0))
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(values), size=(nbootstrap, len(values)))
    boot = np.array([rms_changes(np.mean(values[index], axis=0)) for index in draws])
    valid = np.isfinite(estimate)
    spread = np.std(boot[:, valid], axis=0, ddof=1)
    active = spread > 1e-12
    maxima = np.zeros(nbootstrap)
    if np.any(active):
        maxima = np.max(abs(boot[:, valid][:, active] - estimate[valid][active])
                        / spread[active], axis=1)
    critical = np.quantile(maxima, confidence)
    halfwidth = np.full_like(estimate, np.nan)
    halfwidth[valid] = critical * spread
    return estimate, estimate-halfwidth, estimate+halfwidth
