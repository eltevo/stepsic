from __future__ import annotations

import numpy as np
import pytest

from stepsic.field import generate_delta_k, white_noise
from stepsic.rng import philox4x64_10


def test_white_noise__shared_modes_match_across_resolutions() -> None:
    """Every mesh of a multiresolution IC draws the same realisation.

    Modes are keyed by their signed integer index: each mode of a mesh is a
    distinct draw, and W(k)/sqrt(N) is identical on the modes that two meshes
    share; otherwise the levels mixed in mass are different universes. 24 and
    48 cells are sizes where fftfreq(n) * n is not integer-exact. The kz = 0
    plane must be the conjugate mirror of itself so that irfftn keeps all of
    its power, and the Nyquist planes and DC mode must vanish on every mesh.
    """
    seed = 2**63 + 5
    coarse = white_noise((24, 24, 24), seed=seed)
    fine = white_noise((48, 48, 48), seed=seed)

    shared = np.arange(-11, 12)
    ix, iy, iz = np.meshgrid(shared, shared, np.arange(12), indexing="ij")
    np.testing.assert_allclose(
        coarse[ix % 24, iy % 24, iz] / np.sqrt(24**3),
        fine[ix % 48, iy % 48, iz] / np.sqrt(48**3),
        # The same unit Gaussian scaled by sqrt(N) and back: float64 roundoff.
        rtol=1e-15,
        atol=0.0,
    )

    for w, n in ((coarse, 24), (fine, 48)):
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        np.testing.assert_array_equal(w[:, :, 0], np.conj(w[-i % n, -j % n, 0]))
        assert not np.any(w[n // 2]) and not np.any(w[:, n // 2]) and not np.any(w[:, :, n // 2])
        assert w[0, 0, 0] == 0.0
        drawn = w[:, :, 1 : n // 2][w[:, :, 1 : n // 2] != 0]
        assert np.unique(drawn).size == drawn.size

    single = white_noise((24, 24, 24), seed=seed, dtype=np.float32)
    np.testing.assert_array_equal(single, coarse.astype(np.complex64))


def _numpy_philox_block(counter, key) -> np.ndarray:
    """NumPy's Philox4x64-10 block for `counter` (NumPy pre-increments)."""
    value = sum(int(word) << (64 * i) for i, word in enumerate(counter)) - 1
    value &= (1 << 256) - 1
    words = [(value >> (64 * i)) & ((1 << 64) - 1) for i in range(4)]
    bitgen = np.random.Philox(
        key=np.asarray(key, dtype=np.uint64), counter=np.asarray(words, dtype=np.uint64)
    )
    return bitgen.random_raw(4)


def test_philox4x64_10__matches_numpy_philox() -> None:
    """The white-noise bit source equals NumPy's independent Philox4x64-10."""
    max64 = 2**64 - 1
    cases = [
        ((1, 0, 0, 0), (1, 0)),
        ((0, 0, 0, 0), (123, 456)),
        ((max64, 0, 7, 0), (max64, 2**63)),
        ((0, 0, 0, max64), (0, max64)),
    ]
    for counter, key in cases:
        np.testing.assert_array_equal(
            philox4x64_10(np.asarray(counter, dtype=np.uint64), key),
            _numpy_philox_block(counter, key),
        )
    counters = np.random.default_rng(1).integers(0, max64, size=(64, 4), dtype=np.uint64)
    key = (0x0123456789ABCDEF, 0xFEDCBA9876543210)
    np.testing.assert_array_equal(
        philox4x64_10(counters, key),
        np.stack([_numpy_philox_block(tuple(map(int, c)), key) for c in counters]),
    )


def test_generate_delta_k__mode_power_follows_power_law_spectrum(tiny_chunks) -> None:
    """The density modes carry the requested P(k).

    With the discrete convention P(k) = V |delta_k|^2 / N^2 and V = N dx^3,
    the mode powers |delta_k|^2 dx^3 / (N P(|k|)) are independent unit
    exponentials, so their mean is 1 with standard error 1/sqrt(M). A power
    law (rather than a constant P) also exposes a wrongly scaled |k|.
    """
    n, dx = 32, 0.5
    kh = np.geomspace(1e-2, 1e2, 64)
    delta = generate_delta_k(kh, 3.0 * kh**-2, (n, n, n), dx, seed=42)

    k1 = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=dx)
    kmod = np.sqrt(k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + kz[None, None, :] ** 2)
    # Independent modes: drop the self-mirrored kz = 0 plane and Nyquist planes.
    keep = np.ones(delta.shape, dtype=bool)
    keep[n // 2] = keep[:, n // 2] = False
    keep[:, :, 0] = keep[:, :, n // 2] = False
    ratio = np.abs(delta[keep]) ** 2 * dx**3 / (n**3 * 3.0 * kmod[keep] ** -2)
    assert abs(ratio.mean() - 1.0) <= 5.0 / np.sqrt(ratio.size)
