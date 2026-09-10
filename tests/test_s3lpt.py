'''
Unit tests for stepsic.s3lpt — the S³ tangent frame used to build LPT displacements
on the Poincaré Dodecahedral Space.

Run with:  python -m pytest tests/test_s3lpt.py -v
       or: python tests/test_s3lpt.py
'''
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stepsic.s3lpt import tangent_basis  # noqa: E402


def _assert_valid_frame(e, q, ctx):
    '''e must be 3 orthonormal 4-vectors, each perpendicular to q.'''
    assert np.linalg.matrix_rank(e, tol=1e-8) == 3, f'{ctx}: frame is rank-deficient'
    assert np.abs(e @ q).max() < 1e-10, f'{ctx}: basis not perpendicular to q'
    assert np.abs(e @ e.T - np.eye(3)).max() < 1e-10, f'{ctx}: basis not orthonormal'


def test_tangent_basis_at_coordinate_axes():
    '''Regression: at a coordinate-axis quaternion one Gram-Schmidt candidate projects
    to exactly zero.  It must not consume an output row -- otherwise the frame comes
    back rank-2 with a missing physical gradient direction.

    q = (1,0,0,0) is the stereographic origin, i.e. a particle at the box centre, so
    this is reachable by an ordinary grid load rather than a measure-zero curiosity.'''
    for i in range(4):
        for sign in (+1.0, -1.0):
            q = np.zeros(4)
            q[i] = sign
            _assert_valid_frame(tangent_basis(q), q, f'axis {sign:+.0f}*e{i}')


def test_tangent_basis_near_degenerate():
    '''Frames stay well-conditioned as a candidate approaches degeneracy.

    Naive Gram-Schmidt over the first three axes is not merely wrong *at* the axes:
    just off them the parallel candidate is a tiny difference of large numbers, so
    the frame loses orthogonality by catastrophic cancellation.  Sweep the offset
    direction as well as its size -- an axis-aligned probe alone misses this.'''
    rng = np.random.default_rng(0)
    worst_perp = worst_gram = 0.0
    for _ in range(2000):
        i = rng.integers(4)
        eps = 10.0 ** rng.uniform(-14, -1)
        d = rng.normal(size=4)
        d[i] = 0.0
        q = np.zeros(4)
        q[i] = 1.0
        q = q + eps * d / np.linalg.norm(d)
        q /= np.linalg.norm(q)
        e = tangent_basis(q)
        worst_perp = max(worst_perp, np.abs(e @ q).max())
        worst_gram = max(worst_gram, np.abs(e @ e.T - np.eye(3)).max())
    assert worst_perp < 1e-12, f'near-axis perpendicularity degraded to {worst_perp:.2e}'
    assert worst_gram < 1e-12, f'near-axis orthonormality degraded to {worst_gram:.2e}'


def test_tangent_basis_resolves_tangent_projector():
    '''sum_a e_a e_a^T must equal the tangent projector I - q q^T.

    This is the property displacement_field() actually relies on: it accumulates
    Psi = sum_a (grad_a f) e_a, so a frame that fails here drops a physical gradient
    direction from the Zel'dovich displacement.  It is also what makes the result
    independent of *which* orthonormal frame is chosen.'''
    rng = np.random.default_rng(2)
    q = rng.normal(size=(5000, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q = np.vstack([q, np.eye(4), -np.eye(4)])

    e = tangent_basis(q)
    proj = np.einsum('nij,nik->njk', e, e)
    tangent = np.eye(4) - np.einsum('ni,nj->nij', q, q)
    assert np.abs(proj - tangent).max() < 1e-12


def test_tangent_basis_random_and_batched():
    '''Vectorized path returns a valid frame at every point, including the axes.'''
    rng = np.random.default_rng(0)
    q = rng.normal(size=(4000, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q = np.vstack([q, np.eye(4), -np.eye(4)])

    e = tangent_basis(q)
    assert e.shape == q.shape[:-1] + (3, 4)
    assert np.linalg.matrix_rank(e, tol=1e-8).min() == 3
    assert np.abs(np.einsum('nij,nj->ni', e, q)).max() < 1e-10
    assert np.abs(np.einsum('nij,nkj->nik', e, e) - np.eye(3)).max() < 1e-10


def test_tangent_basis_spans_tangent_space():
    '''Any tangent vector at q is reconstructed exactly from its frame components --
    this is what fails when the frame silently loses a direction.'''
    rng = np.random.default_rng(1)
    for q in (np.array([1.0, 0, 0, 0]), np.array([0, 1.0, 0, 0]),
              rng.normal(size=4) / np.linalg.norm(rng.normal(size=4))):
        q = q / np.linalg.norm(q)
        e = tangent_basis(q)
        v = rng.normal(size=4)
        v = v - (v @ q) * q                      # project into the tangent space
        assert np.abs(e.T @ (e @ v) - v).max() < 1e-10


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'{name}: PASS')
