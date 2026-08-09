from __future__ import annotations

import numpy as np
import pytest

from stepsic.rng import RNG, philox4x64_10, philox_complex_gaussian


UINT64_MASK = (1 << 64) - 1
UINT256_MASK = (1 << 256) - 1


def _counter_minus_one(counter: tuple[int, int, int, int]) -> tuple[int, ...]:
    value = sum(int(word) << (64 * i) for i, word in enumerate(counter))
    value = (value - 1) & UINT256_MASK
    return tuple((value >> (64 * i)) & UINT64_MASK for i in range(4))


def _numpy_philox_block(
    counter: tuple[int, int, int, int],
    key: tuple[int, int],
) -> np.ndarray:
    bitgen = np.random.Philox(
        key=np.asarray(key, dtype=np.uint64),
        counter=np.asarray(_counter_minus_one(counter), dtype=np.uint64),
    )
    return bitgen.random_raw(4)


@pytest.mark.parametrize(
    ("counter", "key"),
    [
        ((1, 0, 0, 0), (1, 0)),
        ((0, 0, 0, 0), (123, 456)),
        ((2**64 - 1, 0, 7, 0), (2**64 - 1, 2**63)),
        ((0, 0, 0, 2**64 - 1), (0, 2**64 - 1)),
    ],
)
def test_philox4x64_10__matches_numpy_fixed_and_boundary_vectors(
    counter, key
) -> None:
    """NumPy's independently implemented Philox bit-generator is the oracle."""
    actual = philox4x64_10(np.asarray(counter, dtype=np.uint64), key)
    np.testing.assert_array_equal(actual, _numpy_philox_block(counter, key))


def test_philox4x64_10__matches_numpy_vectorized_counters() -> None:
    """every vectorized block is compared exactly with NumPy Philox."""
    oracle_rng = np.random.default_rng(20260717)
    counters = oracle_rng.integers(
        0, np.iinfo(np.uint64).max, size=(128, 4), dtype=np.uint64
    )
    key = (0x0123456789ABCDEF, 0xFEDCBA9876543210)
    expected = np.stack(
        [_numpy_philox_block(tuple(map(int, counter)), key) for counter in counters]
    )
    np.testing.assert_array_equal(philox4x64_10(counters, key), expected)


def test_philox_complex_gaussian__is_deterministic_finite_and_vectorized() -> None:
    """identical counter/key inputs must give identical finite outputs."""
    counters = np.arange(32, dtype=np.uint64).reshape(8, 4)
    key = np.array([42, 0], dtype=np.uint64)
    first = philox_complex_gaussian(counters, key)
    second = philox_complex_gaussian(counters, key)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (8,)
    assert np.all(np.isfinite(first.real))
    assert np.all(np.isfinite(first.imag))


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("uniform", {"size": 16}),
        ("normal", {"mean": -2.0, "std": 3.0, "size": 16}),
        ("integers", {"low": -4, "high": 9, "size": 16}),
    ],
)
def test_rng__same_seed_produces_same_stream(method, kwargs) -> None:
    """seeded stream identity is an exact determinism invariant."""
    first = getattr(RNG(112358), method)(**kwargs)
    second = getattr(RNG(112358), method)(**kwargs)
    np.testing.assert_array_equal(first, second)
