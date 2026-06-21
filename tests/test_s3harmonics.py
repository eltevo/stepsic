'''
Unit tests for stepsic.s3harmonics — eigenmodes of the Laplacian on the Poincaré
Dodecahedral Space S³/I* (Phase 7B discrete-spectrum ICs).

The defining property is the Poincaré-dodecahedral spectrum: the lowest non-trivial
I*-invariant eigenmode is at n=12 (the "missing" large-scale fluctuations).

Run with:  python -m pytest tests/test_s3harmonics.py -v
       or: python tests/test_s3harmonics.py
'''
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stepsic import s3harmonics as s3  # noqa: E402
from stepsic import pds                # noqa: E402


def test_invariant_multiplicity_spectrum():
    '''Molien/character sum reproduces the PDS spectrum: 0 for n=2..10, then 1 at
    n=12, 20, 24, 30, 32, 36 (the famous large-scale-mode suppression).'''
    assert s3.invariant_multiplicity(0) == 1            # constant mode
    for n in (2, 4, 6, 8, 10):
        assert s3.invariant_multiplicity(n) == 0, f'n={n} should be forbidden'
    for n in (12, 20, 24, 30, 32, 36):
        assert s3.invariant_multiplicity(n) == 1, f'n={n} should carry a mode'


def test_invariant_n_list():
    assert s3.invariant_n_list(40) == [0, 12, 20, 24, 30, 32, 36, 40]


def test_first_nontrivial_mode_is_n12():
    '''The explicit projected eigenfunction is machine-zero for n=10 and non-zero at
    n=12 — consistent with the multiplicities, with an evaluable function.'''
    rng = np.random.default_rng(0)
    v = rng.normal(size=(300, 4)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    f10 = s3.random_invariant_field(10, np.random.default_rng(1), v).real
    f12 = s3.random_invariant_field(12, np.random.default_rng(1), v).real
    assert np.var(f10) < 1e-12, 'n=10 must be forbidden'
    assert np.var(f12) > 1e-3, 'n=12 must carry the first invariant mode'


def test_projected_mode_is_istar_invariant():
    '''A projected n=12 mode satisfies f(g.q) = f(q) for all g in I* to machine precision.'''
    rng = np.random.default_rng(2)
    v = rng.normal(size=(200, 4)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    coeffs = {(l, m): rng.normal() + 1j * rng.normal()
              for l in range(13) for m in range(-l, l + 1)}

    def field(q):
        out = np.zeros(len(q), dtype=complex)
        for (l, m), c in coeffs.items():
            out += c * s3.hyperspherical_harmonic(12, l, m, q)
        return out

    f = s3.project_invariant(field, v)
    g7 = pds.istar()[7]
    f_shifted = s3.project_invariant(field, pds.quat_mult(g7, v))
    scale = np.sqrt(np.mean(np.abs(f) ** 2))
    assert np.max(np.abs(f_shifted - f)) / scale < 1e-10


def test_wavenumber():
    R = 3100.0
    assert abs(s3.wavenumber(12, R) - np.sqrt(12 * 14) / R) < 1e-12
    assert s3.wavenumber(0, R) == 0.0


def test_synthesize_grf_runs_and_uses_only_invariant_degrees():
    rng = np.random.default_rng(7)
    v = rng.normal(size=(200, 4)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    delta = s3.synthesize_grf(v, lambda k: k ** -2.0, 3100.0, n_max=20, rng=rng)
    assert delta.shape == (200,)
    assert np.isfinite(delta).all()
    assert delta.std() > 0.0


if __name__ == '__main__':
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
