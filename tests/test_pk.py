from __future__ import annotations

import numpy as np
import pytest

from conftest import SEEDS
from stepsic.field import generate_delta_k
from stepsic.pk import (
    _bin_isotropic_modes,
    _build_k_bin_edges,
    _weight_rfft_modes,
    measure_pk,
    measure_pk_from_delta_k,
)


@pytest.mark.parametrize("nvox", [np.array([8, 8, 8]), np.array([7, 9, 11])])
def test_weight_rfft_modes__sums_to_real_grid_cell_count(nvox) -> None:
    """T1: rFFT pair weights exactly reconstruct the full Fourier grid."""
    shape = (int(nvox[0]), int(nvox[1]), int(nvox[2] // 2 + 1))
    assert _weight_rfft_modes(nvox, shape).sum() == np.prod(nvox)


def test_build_k_bin_edges__matches_hand_arithmetic() -> None:
    """T2: half-bin edges are hand-derived from kmin=1, kmax=3, dk=1/2."""
    actual = _build_k_bin_edges(
        np.array([10.0, 10.0, 10.0]),
        1.0,
        kmin=1.0,
        kmax=3.0,
        dk_bin=0.5,
    )
    np.testing.assert_array_equal(actual, np.arange(0.75, 3.5, 0.5))


def test_bin_isotropic_modes__matches_full_grid_numpy_histogram() -> None:
    """T3: np.histogram on the unreduced full FFT grid is the oracle."""
    n = 8
    nvox = np.array([n, n, n])
    boxsize = np.array([8.0, 8.0, 8.0])
    cell_size = 1.0
    kfund = 2.0 * np.pi / boxsize[0]
    axes = [np.fft.fftfreq(n, d=cell_size) * 2.0 * np.pi for _ in range(3)]
    kfull = np.sqrt(
        axes[0][:, None, None] ** 2
        + axes[1][None, :, None] ** 2
        + axes[2][None, None, :] ** 2
    )
    values_full = 1.0 + kfull**2
    khalf = kfull[:, :, : n // 2 + 1]
    values_half = values_full[:, :, : n // 2 + 1]
    actual_k, actual_values, actual_counts = _bin_isotropic_modes(
        values_half,
        khalf,
        nvox,
        boxsize,
        cell_size,
        kmin=kfund,
        kmax=3.0 * kfund,
        dk_bin=kfund,
    )
    edges = (np.arange(4) + 0.5) * kfund
    mask = kfull > 0.0
    counts, _ = np.histogram(kfull[mask], bins=edges)
    ksum, _ = np.histogram(kfull[mask], bins=edges, weights=kfull[mask])
    value_sum, _ = np.histogram(
        kfull[mask], bins=edges, weights=values_full[mask]
    )
    populated = counts > 0
    expected_k = ksum[populated] / counts[populated]
    expected_values = value_sum[populated] / counts[populated]
    np.testing.assert_array_equal(actual_counts, counts[populated])
    # Sequential sums of N positive float64 terms have a worst-case N*eps bound.
    k_bound = counts[populated] * np.finfo(np.float64).eps * expected_k
    value_bound = counts[populated] * np.finfo(np.float64).eps * expected_values
    assert np.all(np.abs(actual_k - expected_k) <= k_bound)
    assert np.all(np.abs(actual_values - expected_values) <= value_bound)


def test_measure_pk_from_delta_k__single_mode_shell_has_analytic_normalization() -> None:
    """T2: Parseval normalization V|delta_k|^2/N_cells^2 is hand-derived."""
    n = 8
    nvox = np.array([n, n, n])
    boxsize = np.array([8.0, 8.0, 8.0])
    amplitude = 11.0
    delta = np.zeros((n, n, n // 2 + 1), dtype=np.complex128)
    # Populate all six fundamental directions so the isotropic shell is uniform.
    delta[1, 0, 0] = amplitude
    delta[-1, 0, 0] = amplitude
    delta[0, 1, 0] = amplitude
    delta[0, -1, 0] = amplitude
    delta[0, 0, 1] = amplitude
    kfund = 2.0 * np.pi / boxsize[0]
    k, power, nmodes = measure_pk_from_delta_k(
        delta,
        boxsize,
        nvox,
        kmin=kfund,
        kmax=1.1 * kfund,
        dk_bin=0.4 * kfund,
    )
    expected_power = np.prod(boxsize) * amplitude**2 / np.prod(nvox) ** 2
    np.testing.assert_array_equal(k, (kfund,))
    np.testing.assert_array_equal(power, (expected_power,))
    np.testing.assert_array_equal(nmodes, (6,))


@pytest.mark.parametrize("seed", SEEDS)
def test_generate_measure_pk__recovers_constant_power_at_five_sigma(seed) -> None:
    """T4: P(k)=7 k^0 shell means have Gaussian 5sqrt(2/Nmodes) errors."""
    n = 32
    nvox = np.array([n, n, n])
    boxsize = np.array([32.0, 32.0, 32.0])
    target_power = 7.0
    kh = np.geomspace(1e-3, 1e3, 64)
    delta = generate_delta_k(
        kh, np.full_like(kh, target_power), nvox, 1.0, seed=seed
    )
    k, measured, nmodes = measure_pk_from_delta_k(delta, boxsize, nvox)
    selected = (nmodes >= 40) & (k < 0.6 * np.pi)
    relative_error = np.abs(measured[selected] / target_power - 1.0)
    # A Gaussian-field shell estimator has relative sigma=sqrt(2/N_modes).
    five_sigma = 5.0 * np.sqrt(2.0 / nmodes[selected])
    assert np.all(relative_error <= five_sigma), (
        seed,
        k[selected],
        relative_error,
        five_sigma,
    )


@pytest.mark.slow
def test_measure_pk__interlacing_reduces_known_high_frequency_alias() -> None:
    """T1: Sefusatti et al. (2016) interlacing suppresses the folded mode."""
    coarse = 16
    fine = 64
    boxsize = np.ones(3)
    xaxis = (np.arange(fine) + 0.5) / fine
    yzaxis = (np.arange(coarse) + 0.5) / coarse
    x, y, z = np.meshgrid(xaxis, yzaxis, yzaxis, indexing="ij")
    positions = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    high_mode = 10
    masses = 1.0 + 0.8 * np.cos(2.0 * np.pi * high_mode * positions[:, 0])
    alias_mode = coarse - high_mode
    k_alias = 2.0 * np.pi * alias_mode
    common = dict(
        x=positions,
        mass=masses,
        boxsize=boxsize,
        nvox=np.array([coarse, coarse, coarse]),
        method="cic",
        deconvolve=True,
        subtract_shot=False,
        kmin=k_alias,
        kmax=k_alias + 0.1 * 2.0 * np.pi,
        dk_bin=0.2 * 2.0 * np.pi,
    )
    _, plain, _ = measure_pk(interlace=False, **common)
    _, interlaced, _ = measure_pk(interlace=True, **common)
    assert interlaced[0] < plain[0]
