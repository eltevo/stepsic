r"""
Eigenmodes of the Laplacian on the Poincaré Dodecahedral Space S^3/I* (Phase 7B).

The cosmological signature of PDS lives in the discrete, group-restricted eigenmode
spectrum of S^3/I*, not in a flat-space P(k).  The Laplacian eigenvalues on S^3 (unit
radius) are -n(n+2), n=0,1,2,...; the degree-n eigenspace is the SO(4) representation
V_n (x) V_n of dimension (n+1)^2.  On the quotient S^3/I* (binary icosahedral group I*
acting by left multiplication) only the I*-invariant modes survive.

The number of I*-invariant vectors in the SU(2) spin-n/2 irrep V_n is the character/Molien
sum

    dim(V_n^{I*}) = (1/120) sum_{g in I*} chi_n(g),
    chi_n(g) = sin((n+1) psi_g) / sin(psi_g),   cos(psi_g) = (scalar part of g),

which is 0 for n=2,4,6,8,10 and first becomes 1 at **n=12** (then 20,24,30,32,36,...) —
the famous suppression of the largest fluctuation modes.

An explicit I*-invariant degree-n harmonic is obtained by projecting any degree-n
hyperspherical harmonic onto the invariant subspace,

    (P f)(q) = (1/120) sum_{g in I*} f(g . q),

which is exact and evaluable at arbitrary unit quaternions q.  A Gaussian random field on
the invariant subspace with power P(k_n) (k_n = sqrt(n(n+2))/R_curv) is then

    delta(q) = sum_n sqrt(P(k_n)) * P[ random degree-n harmonic ](q).

This module provides the multiplicities, the harmonics, the projection, and the GRF
synthesis.  Validated: ``invariant_multiplicity`` and the projected-norm test both place
the first non-trivial mode at n=12 (see ``tests``/the module self-test).
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import warnings

from stepsic import pds


# Y_lm(theta, phi), portable across the SciPy 1.15 API rename.  SciPy 1.15 added
# sph_harm_y(l, m, theta, phi) and deprecated sph_harm(m, l, phi, theta); the latter
# was removed in SciPy 1.17.  The two are numerically identical -- only the name and
# the argument order/meaning changed.  Resolved once, at import time.
try:                                    # SciPy >= 1.15
    from scipy.special import sph_harm_y as _sph_harm_y
except ImportError:                     # SciPy < 1.15
    from scipy.special import sph_harm as _sph_harm_legacy

    def _sph_harm_y(l, m, theta, phi):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')     # scipy sph_harm deprecation
            return _sph_harm_legacy(m, l, phi, theta)


__all__ = [
    'invariant_multiplicity', 'invariant_n_list', 'wavenumber',
    'quat_to_angles', 'hyperspherical_harmonic',
    'project_invariant', 'random_invariant_field', 'synthesize_grf',
]


# --- spectrum / multiplicities ------------------------------------------------------
def invariant_multiplicity(n: int) -> int:
    """dim(V_n^{I*}) — number of I*-invariant vectors in the spin-n/2 SU(2) irrep.

    Character sum over the 120 elements of I*.  Returns 0 for n=2..10, 1 at n=12,20,24,
    30,32,36,... (the Poincaré dodecahedral spectrum).
    """
    g = pds.istar()
    q0 = np.clip(g[:, 0], -1.0, 1.0)
    psi = np.arccos(q0)
    s = np.sin(psi)
    chi = np.empty(len(psi))
    near = np.abs(s) < 1e-9                          # g = +-1 (psi = 0 or pi)
    chi[~near] = np.sin((n + 1) * psi[~near]) / s[~near]
    chi[near] = (n + 1) * np.where(q0[near] > 0, 1.0, (-1.0) ** n)
    return int(round(chi.sum() / len(g)))


def invariant_n_list(n_max: int) -> list[int]:
    """The degrees n in [0, n_max] that carry I*-invariant modes (n=0,12,20,24,...)."""
    return [n for n in range(0, n_max + 1) if invariant_multiplicity(n) > 0]


def wavenumber(n: int, R_curv: float) -> float:
    """Comoving wavenumber of the degree-n S^3 eigenmode: k_n = sqrt(n(n+2)) / R_curv."""
    return np.sqrt(n * (n + 2.0)) / R_curv


# --- hyperspherical harmonics -------------------------------------------------------
def quat_to_angles(q: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """unit quaternion(s) (...,4) -> (chi, theta, phi), with q = (cos chi, sin chi nhat)."""
    q = np.asarray(q, dtype=np.float64)
    chi = np.arccos(np.clip(q[..., 0], -1.0, 1.0))
    s = np.sin(chi); s = np.where(s < 1e-12, 1e-12, s)
    theta = np.arccos(np.clip(q[..., 3] / s, -1.0, 1.0))
    phi = np.arctan2(q[..., 2], q[..., 1])
    return chi, theta, phi


def hyperspherical_harmonic(n: int, l: int, m: int, q: NDArray) -> NDArray:
    """Degree-n hyperspherical harmonic Q_{nlm} on S^3 (l=0..n, m=-l..l).

    Q_{nlm} = sin^l(chi) C_{n-l}^{(l+1)}(cos chi) Y_{lm}(theta, phi)  (radial norm omitted;
    irrelevant for the GRF construction, which only needs a spanning set per degree n).
    Eigenvalue of the S^3 Laplacian: -n(n+2).
    """
    from scipy.special import eval_gegenbauer
    chi, theta, phi = quat_to_angles(q)
    radial = np.sin(chi) ** l * eval_gegenbauer(n - l, l + 1, np.cos(chi))
    Y = _sph_harm_y(l, m, theta, phi)
    return radial * Y


# --- I*-invariant projection & GRF --------------------------------------------------
def project_invariant(field_fn, q: NDArray, chunk: int = 200000) -> NDArray:
    """Project a function onto the I*-invariant subspace: (1/120) sum_g field_fn(g.q).

    ``field_fn`` maps quaternions (M,4) -> values (M,).  Evaluated over the 120 I* images
    of each q in memory-bounded chunks.  Returns the invariant projection at q.
    """
    g = pds.istar()                                  # (120,4)
    q = np.asarray(q, dtype=np.float64)
    out = np.zeros(len(q), dtype=complex)
    for s in range(0, len(q), chunk):
        qc = q[s:s + chunk]
        imgs = pds.quat_mult(g, qc[:, None, :]).reshape(-1, 4)   # (n*120, 4)
        out[s:s + chunk] = field_fn(imgs).reshape(len(qc), 120).mean(1)
    return out


def random_invariant_field(n: int, rng: np.random.Generator, q: NDArray) -> NDArray:
    """A single random I*-invariant degree-n eigenfunction sampled at q.

    Draws iid complex Gaussian coefficients for the (n+1)^2 degree-n harmonics and projects
    onto the invariant subspace.  Real part is a real invariant eigenfunction.
    """
    coeffs = {(l, m): rng.normal() + 1j * rng.normal()
              for l in range(n + 1) for m in range(-l, l + 1)}

    def fn(qq):
        out = np.zeros(len(qq), dtype=complex)
        for (l, m), c in coeffs.items():
            out += c * hyperspherical_harmonic(n, l, m, qq)
        return out

    return project_invariant(fn, q)


def synthesize_grf(q: NDArray, pk_func, R_curv: float, n_max: int,
                   rng: np.random.Generator) -> NDArray:
    """Discrete-spectrum Gaussian random overdensity field on S^3/I* at points q.

        delta(q) = sum_{n: invariant} sqrt(P(k_n)) * Re P[random degree-n harmonic](q)

    Parameters
    ----------
    q : (M,4) unit quaternions (particle positions on S^3).
    pk_func : callable k[1/L] -> P(k); the cosmological power spectrum.
    R_curv : curvature radius (same length units as 1/k).
    n_max : highest degree to include (n=12 is the first non-trivial mode).
    rng : numpy Generator.

    Returns the real overdensity field (M,).  (Normalisation is set by pk_func up to the
    per-mode count; intended for relative/spectral studies and as the LPT source.)
    """
    delta = np.zeros(len(q))
    for n in invariant_n_list(n_max):
        if n == 0:
            continue                                  # the constant mode carries no fluctuation
        amp = np.sqrt(max(pk_func(wavenumber(n, R_curv)), 0.0))
        delta += amp * random_invariant_field(n, rng, q).real
    return delta


if __name__ == '__main__':
    # self-test: the first non-trivial invariant mode is at n=12
    print("n : dim(V_n^I*)")
    for n in range(0, 37, 2):
        print(f"  {n:2d} : {invariant_multiplicity(n)}")
    print("invariant degrees up to 40:", invariant_n_list(40))
