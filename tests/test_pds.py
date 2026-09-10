'''
Unit tests for stepsic.pds — NumPy port of the StePS PDS primitives
(binary icosahedral group I*, fundamental domain, S³ force kernels,
stereographic projection).

Run with:  python -m pytest tests/test_pds.py -v   (if pytest is installed)
       or: python tests/test_pds.py                (no pytest required)
'''

import os
import sys

import numpy as np

# stepsic is not pip-installed; make the repo root importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stepsic import pds  # noqa: E402


def test_group_order_and_unit_norm():
    g = pds.istar()
    assert g.shape == (120, 4)
    norms = np.linalg.norm(g, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-12)


def test_group_contains_plus_minus_identity():
    g = pds.istar()
    e = np.array([1.0, 0.0, 0.0, 0.0])
    assert any(np.allclose(row, e, atol=1e-9) for row in g)
    assert any(np.allclose(row, -e, atol=1e-9) for row in g)


def test_group_closure_and_inverses():
    g = pds.istar()
    # Closure: g_i * g_j must be in the group for a sample of pairs
    rng = np.random.default_rng(42)
    for _ in range(200):
        i, j = rng.integers(0, 120, size=2)
        prod = pds.quat_mult(g[i], g[j])
        dists = np.max(np.abs(g - prod), axis=1)
        assert dists.min() < 1e-9
    # Inverses: conj(g_i) must be in the group for every element
    for row in g:
        inv = pds.quat_conj(row)
        dists = np.max(np.abs(g - inv), axis=1)
        assert dists.min() < 1e-9


def test_nearest_nontrivial_element_at_36_degrees():
    # The nearest I* image of the domain centre is at chi = 36 degrees
    g = pds.istar()
    chis = np.degrees(pds.chi(pds.E0, g))
    nonzero = np.sort(chis[chis > 1e-6])
    assert abs(nonzero[0] - 36.0) < 1e-9
    # and there are exactly 12 of them (the face-pairing isometries)
    assert np.sum(np.abs(chis - 36.0) < 1e-6) == 12


def test_in_domain_across_face_boundary():
    # Walk from the centre toward a face midpoint: inside until chi_in = 18 deg
    g = pds.istar()
    chis = pds.chi(pds.E0, g)
    nearest = g[np.argsort(chis)[1]]  # a 36-degree neighbour
    direction = pds.normalize(nearest[1:])  # spatial direction to the face
    for chi_deg, expected in [(5.0, True), (17.5, True), (18.5, False), (30.0, False)]:
        c = np.radians(chi_deg)
        q = np.concatenate([[np.cos(c)], np.sin(c) * direction])
        assert bool(pds.in_domain(q)) == expected, f'chi = {chi_deg} deg'


def test_wrap_idempotent_and_lands_in_domain():
    rng = np.random.default_rng(1)
    q = pds.normalize(rng.normal(size=(500, 4)))
    w = pds.wrap(q)
    assert np.all(pds.in_domain(w, eps=1e-9))
    w2 = pds.wrap(w)
    assert np.allclose(w, w2, atol=1e-12)
    # wrapping must preserve the geodesic structure: q and w differ by an isometry
    assert np.allclose(np.linalg.norm(w, axis=1), 1.0, atol=1e-12)


def test_wrap_in_domain_is_identity():
    # A point already inside the domain must be returned unchanged
    c = np.radians(10.0)
    q = np.array([np.cos(c), np.sin(c), 0.0, 0.0])
    assert np.allclose(pds.wrap(q), q, atol=1e-12)


def test_stereo_round_trip():
    R = 3100.0
    rng = np.random.default_rng(2)
    x = rng.uniform(-500, 500, size=(100, 3))
    q = pds.inverse_stereo(x, R)
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-12)
    x_back = pds.stereo_project(q, R)
    assert np.allclose(x, x_back, atol=1e-9)


def test_stereo_face_midpoint_radius():
    # chi = 18 deg (face midpoint) corresponds to stereo radius R*tan(9 deg)
    R = 3100.0
    c = np.radians(18.0)
    q = np.array([np.cos(c), np.sin(c), 0.0, 0.0])
    x = pds.stereo_project(q, R)
    assert abs(np.linalg.norm(x) - R * np.tan(np.radians(9.0))) < 1e-9


def test_conformal_factor_volume_integral():
    # The shell integral of Omega^3 over r in [0, inf) gives Vol(S^3) = 2 pi^2 R^3
    R = 3100.0
    r = np.linspace(0, 400 * R, 4_000_001)
    omega = pds.conformal_factor(np.stack([r, np.zeros_like(r), np.zeros_like(r)], axis=-1), R)
    # np.trapz was renamed np.trapezoid in NumPy 2.0 and removed in NumPy 2.4
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    integral = trapezoid(omega**3 * 4 * np.pi * r**2, r)
    assert abs(integral / (2 * np.pi**2 * R**3) - 1) < 1e-3


def test_green_kernels_flat_space_limit():
    R = 3100.0
    c = 1e-3  # r = 3.1 Mpc
    r = R * c
    for kernel in (pds.green_bare, pds.green_compensated):
        f = kernel(c, R)
        assert abs(f / (1.0 / r**2) - 1.0) < 1e-3, kernel.__name__


def test_green_compensated_antipode_and_positivity():
    R = 3100.0
    chis = np.linspace(0.01, np.pi - 0.01, 1000)
    f = pds.green_compensated(chis, R)
    assert np.all(f > 0)
    # smooth approach to zero at the antipode:
    # f(pi - eps) ~ (2/(3 pi)) eps / R^2, so f(pi-0.01)/f(pi/2) ~ 4.2e-3
    assert f[-1] < 1e-2 * f[len(f) // 2]
    assert pds.green_compensated(np.pi, R) == 0.0
    # bare kernel instead diverges toward the antipode
    assert pds.green_bare(chis[-1], R) > 1e3 * pds.green_compensated(chis[-1], R)


def test_force_direction_tangency():
    rng = np.random.default_rng(3)
    p = pds.normalize(rng.normal(size=(50, 4)))
    q = pds.normalize(rng.normal(size=(50, 4)))
    t = pds.force_direction(p, q)
    # tangent at p: orthogonal to p, unit norm
    assert np.allclose(np.sum(t * p, axis=1), 0.0, atol=1e-12)
    assert np.allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-12)
    # degenerate case p = q gives the zero vector
    assert np.allclose(pds.force_direction(p[0], p[0]), 0.0)


def test_exact_force_cancels_at_domain_centre():
    # All 119 self-images of a particle at e0 cancel by I* symmetry
    R = 3100.0
    f = pds.exact_force(pds.E0, pds.E0, R, kernel=pds.green_compensated)
    assert np.max(np.abs(f)) < 1e-12 / R**2 * 1e6  # tiny compared to typical scales


def test_exact_force_newtonian_limit():
    # Two nearby particles: total force dominated by the nearest image, ~ 1/r^2
    R = 3100.0
    c = 5.0 / R  # 5 Mpc separation
    p = np.array([np.cos(c), np.sin(c), 0.0, 0.0])
    f = pds.exact_force(p, pds.E0, R, kernel=pds.green_compensated)
    f_newton = 1.0 / (R * c) ** 2
    # force points back toward e0 (negative q1-tangent direction at p)
    assert abs(np.linalg.norm(f) / f_newton - 1.0) < 1e-4
    t_toward_e0 = pds.force_direction(p, pds.E0)
    assert np.dot(f / np.linalg.norm(f), t_toward_e0) > 0.999999


def test_vel_transform_identity_for_trivial_element():
    R = 3100.0
    x = np.array([100.0, -200.0, 50.0])
    q = pds.inverse_stereo(x, R)
    v = np.array([123.0, -45.0, 6.0])
    v_out = pds.stereo_vel_transform(q, q, x, x, v, R)
    assert np.allclose(v_out, v, atol=1e-9)


def test_vel_transform_preserves_physical_speed():
    # The wrapping isometry preserves the S^3 metric: |Omega(x) v| is invariant
    R = 3100.0
    rng = np.random.default_rng(4)
    for _ in range(20):
        x_in = rng.uniform(550, 800, size=3)  # outside the domain for sure
        q_in = pds.inverse_stereo(x_in, R)
        q_out = pds.wrap(q_in)
        if np.allclose(q_in, q_out, atol=1e-12):
            continue
        x_out = pds.stereo_project(q_out, R)
        v_in = rng.normal(size=3) * 100.0
        v_out = pds.stereo_vel_transform(q_in, q_out, x_in, x_out, v_in, R)
        speed_in = pds.conformal_factor(x_in, R) * np.linalg.norm(v_in)
        speed_out = pds.conformal_factor(x_out, R) * np.linalg.norm(v_out)
        assert abs(speed_out / speed_in - 1.0) < 1e-9


if __name__ == '__main__':
    import sys
    import traceback
    try:
        import pytest
        sys.exit(pytest.main([__file__, '-v']))
    except ImportError:
        failed = 0
        tests = [f for name, f in sorted(globals().items()) if name.startswith('test_')]
        for f in tests:
            try:
                f()
                print(f'PASS  {f.__name__}')
            except Exception:
                failed += 1
                print(f'FAIL  {f.__name__}')
                traceback.print_exc()
        print(f'\n{len(tests) - failed}/{len(tests)} tests passed')
        sys.exit(1 if failed else 0)
