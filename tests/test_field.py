from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.fft

from conftest import SEEDS
from stepsic.field import (
    _mass_interp_weights,
    anisotropic_voxels,
    create_grid,
    create_nres_mass_map,
    cubic_voxels,
    fourier_kmod,
    fourier_vectors,
    generate_delta_k,
    white_noise,
    wrap,
)


GOLDEN = Path(__file__).parent / "goldens" / "white_noise_portability.npz"


def test_wrap__range_and_periodic_idempotence_at_boundaries() -> None:
    """modulo projection stays in-box and is invariant under box shifts."""
    length = 8.0
    eps = np.finfo(np.float64).eps * length
    x = np.array([-eps, 0.0, length, 2.0 * length, -length]).reshape(-1, 1)
    wrapped = wrap(x, np.array([length]))
    assert np.all(wrapped >= 0.0)
    assert np.all(wrapped < length)
    np.testing.assert_allclose(
        wrapped,
        wrap(x + 3.0 * length, np.array([length])),
        # Adding 3L to -eps loses one float64 ulp before the modulo operation.
        rtol=0.0,
        atol=eps,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_create_grid__has_analytic_spacing_order_and_singleton(dtype) -> None:
    """expected coordinates are the hand-enumerated Cartesian product."""
    expected = np.array(
        [
            [-0.25, 0.0, -0.25],
            [-0.25, 0.0, 0.25],
            [0.25, 0.0, -0.25],
            [0.25, 0.0, 0.25],
        ],
        dtype=dtype,
    )
    particles, coords = create_grid((2, 1, 2), 0.5, dtype=dtype)
    np.testing.assert_array_equal(particles, expected)
    np.testing.assert_array_equal(
        coords, np.moveaxis(expected.reshape(2, 1, 2, 3), -1, 0)
    )
    singleton, singleton_coords = create_grid((1, 1, 1), 0.5, dtype=dtype)
    np.testing.assert_array_equal(singleton, np.zeros((1, 3), dtype=dtype))
    np.testing.assert_array_equal(singleton_coords, np.zeros((3, 1, 1, 1)))


@pytest.mark.parametrize("voxelizer", [cubic_voxels, anisotropic_voxels])
def test_voxelizers__cell_volumes_partition_box(voxelizer) -> None:
    """N_cells times the analytic cubic cell volume equals box volume."""
    boxsize = np.array([4.0, 6.0, 8.0])
    nvox, cell_size = voxelizer(4, boxsize)
    np.testing.assert_allclose(
        np.prod(nvox) * cell_size**3,
        np.prod(boxsize),
        # Products are ordered differently, so allow float64 roundoff only.
        rtol=1e-15,
        atol=0.0,
    )


@pytest.mark.parametrize("hermitian", [False, True])
def test_fourier_helpers__match_numpy_frequency_and_norm_oracles(hermitian) -> None:
    """np.fft.fftfreq/rfftfreq and np.linalg.norm are external oracles."""
    nvox = np.array([4, 6, 8])
    cell_size = 0.5
    actual = fourier_vectors(nvox, cell_size, hermitian=hermitian)
    expected = [
        np.fft.fftfreq(nvox[0], d=cell_size) * 2.0 * np.pi,
        np.fft.fftfreq(nvox[1], d=cell_size) * 2.0 * np.pi,
        (np.fft.rfftfreq if hermitian else np.fft.fftfreq)(
            nvox[2], d=cell_size
        )
        * 2.0
        * np.pi,
    ]
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)
    mesh = np.stack(np.meshgrid(*expected, indexing="ij"))
    np.testing.assert_array_equal(
        fourier_kmod(nvox, cell_size, hermitian=hermitian),
        np.linalg.norm(mesh, axis=0),
    )


def test_white_noise__has_hermitian_dc_and_nyquist_structure(wn32) -> None:
    """rFFT Hermitian symmetry plus exact DC/Nyquist nulls are invariants."""
    n = 32
    np.testing.assert_array_equal(wn32[n // 2], 0.0)
    np.testing.assert_array_equal(wn32[:, n // 2], 0.0)
    np.testing.assert_array_equal(wn32[:, :, n // 2], 0.0)
    assert wn32[0, 0, 0] == 0.0
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    np.testing.assert_array_equal(
        wn32[:, :, 0], np.conj(wn32[(-i) % n, (-j) % n, 0])
    )
    assert np.isrealobj(scipy.fft.irfftn(wn32, s=(n, n, n)))


def test_white_noise__shared_modes_are_resolution_portable(wn16, wn32) -> None:
    """normalized shared integer modes are resolution-independent."""
    modes = [
        (ix, iy, iz)
        for ix in range(-7, 8)
        for iy in range(-7, 8)
        for iz in range(8)
    ]
    low = np.array([wn16[ix % 16, iy % 16, iz] for ix, iy, iz in modes]) / np.sqrt(
        16**3
    )
    high = np.array([wn32[ix % 32, iy % 32, iz] for ix, iy, iz in modes]) / np.sqrt(
        32**3
    )
    np.testing.assert_allclose(
        low,
        high,
        # Multiplication then division by non-power-of-two sqrt(32**3) rounds.
        rtol=5e-16,
        atol=5e-16,
    )


def test_white_noise__seed_and_dtype_are_deterministic() -> None:
    """seed identity and float32 casting are exact determinism promises."""
    first = white_noise((16, 16, 16), seed=42)
    second = white_noise((16, 16, 16), seed=42)
    different = white_noise((16, 16, 16), seed=43)
    single = white_noise((16, 16, 16), seed=42, dtype=np.float32)
    np.testing.assert_array_equal(first, second)
    assert np.any(first != different)
    assert single.dtype == np.complex64
    np.testing.assert_array_equal(single, first.astype(np.complex64))


@pytest.mark.parametrize("seed", SEEDS)
def test_white_noise__statistics_follow_derived_five_sigma_bounds(seed) -> None:
    """Gaussian sample mean/std/variance use N-derived five-sigma bounds."""
    n = 32
    n_cells = n**3
    field = white_noise((n, n, n), seed=seed)
    real = scipy.fft.irfftn(field, s=(n, n, n))
    expected_std = np.sqrt(1.0 - 3.0 / n)
    # For N Gaussian cells, SE(sample std) ~= sigma/sqrt(2N); use 5 sigma.
    std_bound = 5.0 * expected_std / np.sqrt(2.0 * n_cells)
    assert abs(float(real.std()) - expected_std) <= std_bound
    # Exact zero DC makes the mean zero; this bound is FFT summation roundoff.
    assert abs(float(real.mean())) <= 5.0 * np.finfo(np.float64).eps

    keep = np.arange(n) != n // 2
    normalized = field[:, :, 1 : n // 2] / np.sqrt(n_cells)
    normalized = normalized[keep][:, keep]
    n_complex = normalized.size
    # Re/Im are N(0, 1/2); SE(sample variance)=0.5*sqrt(2/(N-1)).
    variance_bound = 5.0 * 0.5 * np.sqrt(2.0 / (n_complex - 1))
    assert abs(float(normalized.real.var()) - 0.5) <= variance_bound
    assert abs(float(normalized.imag.var()) - 0.5) <= variance_bound


def test_white_noise__matches_portability_golden_with_provenance() -> None:
    """bit portability is pinned by the checked-in regeneration artifact."""
    with np.load(GOLDEN, allow_pickle=False) as golden:
        required = {
            "w",
            "git_commit",
            "numpy_version",
            "scipy_version",
            "generated_date",
            "generator",
            "seed",
            "nvox",
        }
        assert required <= set(golden.files)
        assert str(golden["git_commit"])
        assert str(golden["numpy_version"])
        assert str(golden["scipy_version"])
        assert str(golden["generated_date"])
        assert str(golden["generator"]) == "tests/regen/regen_white_noise_portability.py"
        assert int(golden["seed"]) == 42
        np.testing.assert_array_equal(golden["nvox"], (16, 16, 16))
        np.testing.assert_array_equal(
            white_noise((16, 16, 16), seed=42), golden["w"]
        )


def test_generate_delta_k__quadruple_power_doubles_amplitude() -> None:
    """the analytic amplitude-scaling identity delta[4P] = 2 delta[P]."""
    kh = np.geomspace(1e-3, 1e2, 64)
    pk = 3.0 * kh**-1.25
    nvox = np.array([16, 16, 16])
    field = white_noise(nvox, seed=42)
    baseline = generate_delta_k(kh, pk, nvox, 0.5, field=field)
    scaled = generate_delta_k(kh, 4.0 * pk, nvox, 0.5, field=field)
    np.testing.assert_allclose(
        scaled,
        2.0 * baseline,
        # Log-spline exponentiation introduces only float64 roundoff.
        rtol=5e-15,
        atol=0.0,
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_generate_delta_k__mode_variance_matches_input_power(seed) -> None:
    """independent complex-mode powers are exponential with mean P."""
    n = 32
    nvox = np.array([n, n, n])
    cell_size = 0.5
    power = 7.0
    kh = np.geomspace(1e-3, 1e3, 32)
    delta = generate_delta_k(
        kh, np.full_like(kh, power), nvox, cell_size, seed=seed
    )
    normalized = (
        np.abs(delta[:, :, 1 : n // 2]) ** 2
        * cell_size**3
        / (n**3 * power)
    )
    keep = np.arange(n) != n // 2
    samples = normalized[keep][:, keep].ravel()
    # Unit exponential samples have SE(mean)=1/sqrt(N); use 5 sigma.
    assert abs(float(samples.mean()) - 1.0) <= 5.0 / np.sqrt(samples.size)


@pytest.mark.parametrize(
    "mass",
    [
        np.array([], dtype=np.float64),
        np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0]),
    ],
)
def test_mass_interp_weights__match_numpy_interp_boundaries(mass) -> None:
    """np.interp supplies the independent interior/extrapolation oracle."""
    mass_tab = np.array([8.0, 4.0, 2.0, 1.0])
    j_lo, j_hi, w_hi = _mass_interp_weights(mass, mass_tab)
    actual = (1.0 - w_hi) * mass_tab[j_lo] + w_hi * mass_tab[j_hi]
    expected = np.interp(mass, mass_tab[::-1], mass_tab[::-1])
    np.testing.assert_array_equal(actual, expected)


def test_mass_interp_weights__singleton_selects_only_sample() -> None:
    """a singleton lookup table has the sole exact partition of unity."""
    mass = np.array([-1.0, 4.0, 10.0])
    j_lo, j_hi, w_hi = _mass_interp_weights(mass, np.array([4.0]))
    np.testing.assert_array_equal(j_lo, 0)
    np.testing.assert_array_equal(j_hi, 0)
    np.testing.assert_array_equal(w_hi, 0.0)


def test_create_nres_mass_map__conserves_analytic_box_mass(capsys) -> None:
    """cell mass M/N^3 times N^3 equals the hand-chosen box mass."""
    box_mass = 4096.0
    mass_list = np.array([1.0, 8.0, 64.0])
    nres, mass = create_nres_mass_map(
        3, mass_list, box_mass, np.array([32.0, 32.0, 32.0])
    )
    capsys.readouterr()
    np.testing.assert_array_equal(nres, (4.0, 8.0, 16.0))
    np.testing.assert_array_equal(mass * nres**3, box_mass)
