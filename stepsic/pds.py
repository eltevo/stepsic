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

r'''
Mathematical primitives for Poincaré Dodecahedral Space (PDS).

PDS = :math:`S^3 / I^*` where :math:`I^*` is the binary icosahedral group
(120 unit quaternions).  This module is a NumPy port of the StePS C++ header
``StePS/src/pds_group.h`` and mirrors its conventions exactly:

- Positions are unit quaternions ``q = (q0, q1, q2, q3)`` on the unit
  :math:`S^3`; the fundamental domain is the Voronoi cell of the north pole
  ``e0 = (1, 0, 0, 0)`` under the I* action.
- Geodesic distance: :math:`\chi(p, q) = \arccos(p \cdot q) \in [0, \pi]`;
  physical separation is :math:`r = R_{\rm curv}\,\chi`.
- Stereographic projection (StePS convention):
  :math:`x_i = R\,q_i / (1 + q_0)`, inverse
  :math:`q = ((R^2 - r^2),\, 2Rx,\, 2Ry,\, 2Rz) / (R^2 + r^2)`.
- Force kernels are per unit source mass with G = 1 and no :math:`4\pi`
  factor, identical to ``pds_green()`` in the C++ code.

Most functions accept arrays of quaternions with shape ``(..., 4)`` and
broadcast.
'''

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

#: Golden ratio
TAU = (1.0 + np.sqrt(5.0)) / 2.0

#: Order of the binary icosahedral group I*
N_ISTAR = 120

#: Componentwise equality tolerance for unit quaternions (matches PDS_QUAT_EPS)
QUAT_EPS = 1e-9

#: North pole of S^3 (centre of the fundamental domain)
E0 = np.array([1.0, 0.0, 0.0, 0.0])

# Generators of I* satisfying s^3 = t^5 = (st)^2 = -1, plus their inverses
# (same as PDS_GEN in pds_group.h)
_GENERATORS = np.array([
    [0.5, 0.5, 0.5, 0.5],                    # s
    [0.5, -0.5, -0.5, -0.5],                 # s^-1
    [TAU / 2.0, 1.0 / (2.0 * TAU), 0.5, 0.0],   # t
    [TAU / 2.0, -1.0 / (2.0 * TAU), -0.5, 0.0],  # t^-1
])

_ISTAR_CACHE: NDArray | None = None


def quat_mult(p: NDArray, q: NDArray) -> NDArray:
    '''Hamilton product p ⊗ q for arrays of shape (..., 4) (broadcasts).'''
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    r = np.empty(np.broadcast_shapes(p.shape, q.shape))
    r[..., 0] = p[..., 0]*q[..., 0] - p[..., 1]*q[..., 1] - p[..., 2]*q[..., 2] - p[..., 3]*q[..., 3]
    r[..., 1] = p[..., 0]*q[..., 1] + p[..., 1]*q[..., 0] + p[..., 2]*q[..., 3] - p[..., 3]*q[..., 2]
    r[..., 2] = p[..., 0]*q[..., 2] - p[..., 1]*q[..., 3] + p[..., 2]*q[..., 0] + p[..., 3]*q[..., 1]
    r[..., 3] = p[..., 0]*q[..., 3] + p[..., 1]*q[..., 2] - p[..., 2]*q[..., 1] + p[..., 3]*q[..., 0]
    return r


def quat_conj(q: NDArray) -> NDArray:
    '''Quaternion conjugate (= inverse for unit quaternions).'''
    q = np.asarray(q, dtype=np.float64)
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def normalize(q: NDArray) -> NDArray:
    '''Return q scaled to unit norm along the last axis.'''
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def istar() -> NDArray:
    '''
    The 120 unit quaternions of the binary icosahedral group I*,
    generated once by BFS closure from the two generators (cached).

    Returns an array of shape (120, 4); element 0 is the identity.
    '''
    global _ISTAR_CACHE
    if _ISTAR_CACHE is not None:
        return _ISTAR_CACHE

    elements = [np.array([1.0, 0.0, 0.0, 0.0])]
    queue = [np.array([1.0, 0.0, 0.0, 0.0])]
    while queue:
        cur = queue.pop(0)
        for gen in _GENERATORS:
            new = normalize(quat_mult(cur, gen))
            if not any(np.all(np.abs(new - e) < QUAT_EPS) for e in elements):
                elements.append(new)
                queue.append(new)
    if len(elements) != N_ISTAR:
        raise RuntimeError(
            f'I* BFS generation produced {len(elements)} elements, expected {N_ISTAR}'
        )
    _ISTAR_CACHE = np.array(elements)
    _ISTAR_CACHE.setflags(write=False)
    return _ISTAR_CACHE


def chi(p: NDArray, q: NDArray) -> NDArray:
    '''Geodesic distance χ(p, q) = arccos(p·q) ∈ [0, π] on the unit S³.'''
    d = np.sum(np.asarray(p, dtype=np.float64) * np.asarray(q, dtype=np.float64), axis=-1)
    return np.arccos(np.clip(d, -1.0, 1.0))


def green_bare(chi_val: NDArray, R_curv: float) -> NDArray:
    '''
    Uncompensated S³ force kernel 1/(R² sin²χ) per unit source mass
    (Gauss's law on S³ for an isolated point mass; G = 1, no 4π factor).

    Returns 0 at χ = 0 (no self-force) and at χ ≥ π (antipode).
    Identical to ``pds_green()`` in the C++ code.
    '''
    chi_val = np.asarray(chi_val, dtype=np.float64)
    s = np.sin(chi_val)
    with np.errstate(divide='ignore', invalid='ignore'):
        f = 1.0 / (R_curv * R_curv * s * s)
    return np.where((chi_val < 1e-12) | (chi_val > np.pi - 1e-12), 0.0, f)


def green_compensated(chi_val: NDArray, R_curv: float) -> NDArray:
    r'''
    Background-compensated S³ force kernel per unit source mass:

    .. math::
        F(\chi) = \frac{1 - V(\chi)/V_{S^3}}{R^2 \sin^2\chi},
        \qquad \frac{V(\chi)}{V_{S^3}} = \frac{2\chi - \sin 2\chi}{2\pi}

    This is the force sourced by a point mass plus a uniform negative
    background of equal total mass (the homogeneous mean density is already
    accounted for by the Friedmann expansion in comoving simulations).
    Finite at the antipode (→ 0) and → 1/r² as χ → 0.
    '''
    chi_val = np.asarray(chi_val, dtype=np.float64)
    s = np.sin(chi_val)
    frac = (2.0 * chi_val - np.sin(2.0 * chi_val)) / (2.0 * np.pi)
    with np.errstate(divide='ignore', invalid='ignore'):
        f = (1.0 - frac) / (R_curv * R_curv * s * s)
    # chi -> pi limit: (1 - frac) ~ (4/3)(pi-chi)^3/(2 pi), sin^2 ~ (pi-chi)^2
    # so f -> (2/(3 pi)) (pi - chi) / R^2 -> 0; the cutoff below is safe.
    return np.where((chi_val < 1e-12) | (chi_val > np.pi - 1e-12), 0.0, f)


def in_domain(q: NDArray, eps: float = QUAT_EPS) -> NDArray:
    '''
    Test whether unit quaternion(s) q lie in the fundamental domain
    (Voronoi cell of e0): q[..., 0] ≥ q·g for all g ∈ I*.

    Accepts shape (..., 4); returns boolean array of shape (...).
    '''
    q = np.asarray(q, dtype=np.float64)
    dots = np.tensordot(q, istar(), axes=([-1], [1]))  # (..., 120)
    return np.all(dots <= q[..., 0][..., None] + eps, axis=-1)


def wrap(q: NDArray) -> NDArray:
    '''
    Map unit quaternion(s) q to the canonical representative in the
    fundamental domain: find g* = argmax_g (q·g), return conj(g*) ⊗ q.

    Accepts shape (..., 4); returns the same shape.
    '''
    q = np.asarray(q, dtype=np.float64)
    group = istar()
    dots = np.tensordot(q, group, axes=([-1], [1]))   # (..., 120)
    best = np.argmax(dots, axis=-1)                   # (...)
    gconj = quat_conj(group[best])                    # (..., 4)
    return normalize(quat_mult(gconj, q))


def apply_group(g_idx: int, q: NDArray) -> NDArray:
    '''The g-th I* image of position q: q_img = I*[g_idx] ⊗ q.'''
    return quat_mult(istar()[g_idx], q)


def images(q: NDArray) -> NDArray:
    '''All 120 I* images of position(s) q: shape (..., 120, 4).'''
    q = np.asarray(q, dtype=np.float64)
    return quat_mult(istar(), q[..., None, :])


def force_direction(p: NDArray, q: NDArray) -> NDArray:
    '''
    Unit tangent vector at p pointing toward q along the geodesic:
    t = (q − (p·q)p) / |q − (p·q)p|.  Zero vector if p = ±q.
    Broadcasts over (..., 4).
    '''
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    d = np.sum(p * q, axis=-1, keepdims=True)
    t = q - d * p
    len2 = np.sum(t * t, axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_unit = t / np.sqrt(len2)
    return np.where(len2 < 1e-24, 0.0, t_unit)


def exact_force(p: NDArray, q_src: NDArray, R_curv: float,
                kernel=green_compensated) -> NDArray:
    '''
    Exact gravitational acceleration (per unit source mass) at position p
    from ALL 120 I* images of a source at q_src, as a 4D tangent vector at p.

    The 3D stereographic force used by StePS is the (1, 2, 3) components.
    p: (..., 4); q_src: (4,) or broadcastable; returns (..., 4).
    '''
    p = np.asarray(p, dtype=np.float64)
    imgs = images(q_src)                                   # (..., 120, 4)
    chis = chi(p[..., None, :], imgs)                      # (..., 120)
    mags = kernel(chis, R_curv)                            # (..., 120)
    dirs = force_direction(p[..., None, :], imgs)          # (..., 120, 4)
    return np.sum(mags[..., None] * dirs, axis=-2)


# --- Stereographic projection (StePS convention) ----------------------------

def stereo_project(q: NDArray, R_curv: float) -> NDArray:
    '''Forward stereographic projection x_i = R q_i / (1 + q0); (..., 4) → (..., 3).'''
    q = np.asarray(q, dtype=np.float64)
    return R_curv * q[..., 1:] / (1.0 + q[..., 0])[..., None]


def inverse_stereo(x: NDArray, R_curv: float) -> NDArray:
    '''Inverse stereographic projection q = ((R²−r²), 2Rx) / (R²+r²); (..., 3) → (..., 4).'''
    x = np.asarray(x, dtype=np.float64)
    r2 = np.sum(x * x, axis=-1)
    denom = R_curv * R_curv + r2
    q = np.empty(x.shape[:-1] + (4,))
    q[..., 0] = (R_curv * R_curv - r2) / denom
    q[..., 1:] = 2.0 * R_curv * x / denom[..., None]
    return q


def conformal_factor(x: NDArray, R_curv: float) -> NDArray:
    r'''
    Conformal factor Ω(r) = 2R² / (R² + r²) of the stereographic projection:
    a Euclidean length element dx at radius r corresponds to the physical
    length Ω(r)·dx on S³, and a Euclidean volume element dx³ to Ω³·dx³.
    (∫ Ω³ d³x over all of R³ = 2π²R³ = Vol(S³) exactly.)
    '''
    x = np.asarray(x, dtype=np.float64)
    r2 = np.sum(x * x, axis=-1)
    return 2.0 * R_curv * R_curv / (R_curv * R_curv + r2)


def stereo_vel_transform(q_in: NDArray, q_out: NDArray,
                         x_in: NDArray, x_out: NDArray,
                         v: NDArray, R_curv: float) -> NDArray:
    '''
    Transform the stereographic velocity v when wrap() maps q_in → q_out
    (exact Jacobian of x_out = f(ḡ · f⁻¹(x_in)), ḡ = q_out ⊗ conj(q_in)).
    Identity when ḡ = 1.  Vectorized over (..., 3)/(..., 4); returns v_out.

    Port of ``pds_stereo_vel_transform()`` in pds_group.h.
    '''
    q_in, q_out = np.asarray(q_in, dtype=np.float64), np.asarray(q_out, dtype=np.float64)
    x_in, x_out = np.asarray(x_in, dtype=np.float64), np.asarray(x_out, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    # Step 1 - lift v to the S^3 tangent at q_in (inverse-stereo Jacobian)
    r2 = np.sum(x_in * x_in, axis=-1)
    D = R_curv * R_curv + r2
    D2 = D * D
    xv = np.sum(x_in * v, axis=-1)
    u = np.empty(np.broadcast_shapes(x_in.shape[:-1], v.shape[:-1]) + (4,))
    u[..., 0] = -4.0 * R_curv * R_curv * xv / D2
    u[..., 1:] = (2.0 * R_curv * (D[..., None] * v - 2.0 * x_in * xv[..., None])
                  / D2[..., None])

    # Step 2 - transport by the wrapping isometry g_bar = q_out * conj(q_in)
    gbar = quat_mult(q_out, quat_conj(q_in))
    w = quat_mult(gbar, u)

    # Step 3 - project back to R^3 (forward-stereo Jacobian at q_out)
    s = 1.0 / (1.0 + q_out[..., 0])
    return s[..., None] * (R_curv * w[..., 1:] - x_out * w[..., 0][..., None])
