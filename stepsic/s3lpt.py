r"""
Zel'dovich (1LPT) displacement field from the discrete S^3/I* eigenmode spectrum
(Phase 7B), for the large-scale / topology-carrying part of a Poincaré-dodecahedral IC.

The overdensity GRF on the I*-invariant modes is (see :mod:`stepsic.s3harmonics`)

    delta(q) = sum_{n in invariant} A_n * Re psi_n(q),   A_n = sqrt(P(k_n)),

with psi_n a unit-variance random I*-invariant degree-n harmonic and
k_n = sqrt(n(n+2))/R_curv.  The Zel'dovich displacement is Psi = -grad phi with
nabla^2 phi = delta; this is exact mode-by-mode because nabla^2 psi_n = -k_n^2 psi_n, so

    phi(q)  = sum_n  (-A_n / k_n^2) Re psi_n(q),
    Psi(q)  = -grad phi(q)  (a tangent vector field on S^3, perp to q),

and the LPT consistency div Psi = -delta holds by construction.  The gradient is taken
with a central finite difference in an orthonormal tangent frame at each q.

Because these modes are smooth and large-scale (low n), the field is evaluated on a
**coarse set of points** and interpolated to the particle load; the small-scale power is
supplied by the existing flat-space FFT LPT (the hybrid, assembled by the IC pipeline).
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from stepsic import s3harmonics as s3
from stepsic import pds

__all__ = ['tangent_basis', 'make_invariant_potential', 'displacement_field']


def tangent_basis(q: NDArray) -> NDArray:
    """Orthonormal tangent frame at unit quaternion(s) q: (...,3,4), each row perp to q."""
    q = np.asarray(q, dtype=np.float64)
    M = q.shape[:-1]

    # Project the 4 Euclidean axes into the tangent space at q:  cand_k = e_k - q_k q,
    # which has norm sqrt(1 - q_k^2).  Discard the axis most nearly parallel to q
    # (argmax|q_k|): since sum_k q_k^2 = 1, every *retained* candidate then has
    # q_k^2 <= 1/2, i.e. norm >= 1/sqrt(2).  The frame is therefore uniformly
    # well-conditioned and has no degenerate case.
    #
    # Taking the first three axes unconditionally instead is wrong: at a coordinate-axis
    # quaternion (e.g. q = (1,0,0,0), the stereographic origin -- a particle at the box
    # centre) the parallel candidate vanishes, and near one it is a tiny difference of
    # large numbers.  Either way a gradient direction is silently lost from the frame,
    # and Psi = sum_a (grad_a f) e_a then misses that direction entirely.
    cand = np.broadcast_to(np.eye(4), M + (4, 4)).copy()
    cand = cand - (cand @ q[..., None])[..., 0][..., None] * q[..., None, :]

    drop = np.argmax(np.abs(q), axis=-1)                      # (M,)
    order = np.argsort(np.arange(4) == drop[..., None], axis=-1, kind='stable')
    cand = np.take_along_axis(cand, order[..., :3, None], axis=-2)

    # Modified Gram-Schmidt with one re-orthogonalization pass; each pass also
    # re-projects out q so the rows stay perpendicular to it to machine precision.
    e = np.zeros(M + (3, 4))
    for a in range(3):
        v = cand[..., a, :]
        for _ in range(2):
            v = v - np.sum(v * q, axis=-1, keepdims=True) * q
            for j in range(a):
                v = v - np.sum(v * e[..., j, :], axis=-1, keepdims=True) * e[..., j, :]
        e[..., a, :] = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return e


def _raw_invariant_field_fn(n: int, rng: np.random.Generator):
    """Closure evaluating one random (unnormalised) I*-invariant degree-n field f_n(q),
    f_n = P[ sum_lm c_lm Q_nlm ],  with the random coefficients fixed once."""
    cf = {(l, m): rng.normal() + 1j * rng.normal()
          for l in range(n + 1) for m in range(-l, l + 1)}

    def gfield(qq):
        v = np.zeros(qq.shape[:-1], dtype=complex)
        for (l, m), c in cf.items():
            v += c * s3.hyperspherical_harmonic(n, l, m, qq)
        return v

    return lambda q: s3.project_invariant(gfield, np.asarray(q).reshape(-1, 4)).real.reshape(np.asarray(q).shape[:-1])


def displacement_field(q: NDArray, pk_func, R_curv: float, n_split: int,
                       rng: np.random.Generator, eps: float = 1e-4,
                       in_domain_mask: NDArray | None = None):
    r"""Cosmologically-normalised Zel'dovich displacement on the discrete S^3/I* modes.

    For each invariant degree n (0 < n <= n_split) a random invariant field f_n is built
    and normalised so the degree-n overdensity has the right variance,

        delta_n = f_n * sqrt( P(k_n) * m_n / (V_domain * var_domain(f_n)) ),
        m_n = (n+1) * dim(V_n^{I*}),   V_domain = pi^2 R^3 / 60,

    matching the continuous-limit power P(k_n) per mode.  The Zel'dovich displacement is
    the **physical** gradient of phi (nabla^2 phi = delta, phi_n = -delta_n/k_n^2):

        Psi = -grad_phys phi = (1/R) sum_n (norm_n / k_n^2) grad_unit f_n,

    a 4D tangent field (perp to q), in length units (so Psi is a comoving displacement to
    be scaled by the LPT growth, already folded into P(k) via D1^2 back-scaling).

    Returns (Psi (M,4), info).  ``in_domain_mask`` (bool, M) restricts the variance
    estimate to in-domain points; if None all q are used.
    """
    q = np.asarray(q, dtype=np.float64)
    degrees = [n for n in s3.invariant_n_list(n_split) if n > 0]
    V_domain = np.pi ** 2 * R_curv ** 3 / 60.0
    E = tangent_basis(q)                              # (M,3,4)
    mask = np.ones(len(q), bool) if in_domain_mask is None else in_domain_mask
    Psi = np.zeros_like(q)
    used = {}
    for n in degrees:
        kn = s3.wavenumber(n, R_curv)
        m_n = (n + 1) * s3.invariant_multiplicity(n)
        f_fn = _raw_invariant_field_fn(n, rng)
        f0 = f_fn(q)
        var = np.var(f0[mask])
        if var < 1e-30:
            continue
        norm_n = np.sqrt(max(pk_func(kn), 0.0) * m_n / (V_domain * var))
        weight = norm_n / (kn * kn) / R_curv          # (norm_n/k_n^2) * (1/R) physical grad
        for a in range(3):
            ea = E[:, a, :]
            qp = q + eps * ea; qp /= np.linalg.norm(qp, axis=-1, keepdims=True)
            qm = q - eps * ea; qm /= np.linalg.norm(qm, axis=-1, keepdims=True)
            grad_a = (f_fn(qp) - f_fn(qm)) / (2 * eps)  # unit-S^3 directional derivative
            Psi += weight * grad_a[:, None] * ea        # Psi = +(1/R)(norm/k^2) grad f
        used[n] = dict(k_n=kn, m_n=m_n, norm=norm_n)
    info = {'degrees': degrees, 'used': used, 'V_domain': V_domain,
            'k_split': s3.wavenumber(n_split, R_curv)}
    return Psi, info


def pds_discrete_displacement(pos_stereo: NDArray, pk_func, R_curv: float, n_split: int,
                              rng: np.random.Generator, ngrid_coarse: int = 10,
                              half: float | None = None):
    """Discrete-spectrum Zel'dovich displacement at PDS particle positions (stereographic).

    The smooth low-n field is evaluated on a coarse regular stereographic grid and
    interpolated to the particles; the result is the stereographic-coordinate displacement
    Delta x (N,3) to add to the flat-LPT (high-passed) displacement.

    pos_stereo : (N,3) particle stereographic coordinates.
    Returns (dx_stereo (N,3), info).
    """
    from scipy.interpolate import RegularGridInterpolator
    pos_stereo = np.asarray(pos_stereo, dtype=np.float64)
    if half is None:
        half = 1.05 * np.abs(pos_stereo).max()
    ax = np.linspace(-half, half, ngrid_coarse)
    GX, GY, GZ = np.meshgrid(ax, ax, ax, indexing='ij')
    gstereo = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], 1)
    gq = pds.inverse_stereo(gstereo, R_curv)
    mask = pds.in_domain(gq)
    Psi, info = displacement_field(gq, pk_func, R_curv, n_split, rng, in_domain_mask=mask)
    # interpolate the 4D-tangent displacement to particle positions
    interps = [RegularGridInterpolator((ax, ax, ax), Psi[:, k].reshape((ngrid_coarse,) * 3),
                                       bounds_error=False, fill_value=0.0) for k in range(4)]
    Psi_p = np.stack([ip(pos_stereo) for ip in interps], 1)            # (N,4)
    # apply along S^3: q_new = normalize(q + Psi_phys/R); stereo displacement
    pq = pds.inverse_stereo(pos_stereo, R_curv)
    qn = pq + Psi_p / R_curv
    qn /= np.linalg.norm(qn, axis=1, keepdims=True)
    dx = pds.stereo_project(qn, R_curv) - pos_stereo
    return dx, info


def highpass_delta_k(delta_k: NDArray, nvox, dk: float, k_split: float,
                     width_dex: float = 0.15) -> NDArray:
    """Smoothly remove modes below k_split from an rfftn-layout overdensity field delta_k,
    so the flat-LPT path supplies only the small scales of the hybrid IC (and the flat
    power at PDS-forbidden large scales is removed, to be replaced by the discrete modes).

    W(k) = 0.5 (1 + tanh( ln(k/k_split) / width_dex )); delta_k *= W.  Uses stepsic's own
    ``fourier_grid`` so the k-grid exactly matches what ``lpt1`` assumes.
    """
    from stepsic.field import fourier_grid
    _, kmod = fourier_grid(np.asarray(nvox), dk, hermitian=True)
    with np.errstate(divide='ignore'):
        W = 0.5 * (1.0 + np.tanh(np.log(np.maximum(kmod, 1e-30) / k_split) / width_dex))
    return delta_k * W.astype(delta_k.real.dtype)


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 4)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    Psi, info = displacement_field(v, lambda k: k ** -2.0, 3100.0, n_split=24, rng=rng)
    perp = np.abs(np.einsum('ij,ij->i', Psi, v)).max() / np.sqrt((Psi ** 2).sum(1)).mean()
    print('degrees used:', info['degrees'])
    print(f'Psi rel. tangency = {perp:.2e}  (should be ~0)')
