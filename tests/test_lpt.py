from __future__ import annotations

import numpy as np
import pytest
import scipy.fft

from stepsic.field import create_grid
from stepsic.lpt import lpt1, lpt2


def _plane_density(
    positions: np.ndarray,
    boxsize: float,
    modes: tuple[int, int, int],
    amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    wavevector = 2.0 * np.pi * np.asarray(modes) / boxsize
    # Continuity gives delta=-div(Psi), so this sign makes Psi=+A k/k^2 sin(k.q).
    density = -amplitude * np.cos(positions @ wavevector)
    return density, wavevector


@pytest.mark.parametrize("modes", [(1, 0, 0), (1, 1, 1)])
def test_lpt1__matches_single_plane_wave_solution(modes) -> None:
    """Bernardeau et al. (2002), eq. 94, supplies the plane-wave oracle."""
    nvox = np.array([16, 16, 16])
    cell_size = 0.5
    boxsize = nvox[0] * cell_size
    amplitude = 0.2
    growth = 0.7
    velocity_factor = 3.0
    positions, _ = create_grid(nvox, cell_size)
    density, wavevector = _plane_density(positions, boxsize, modes, amplitude)
    delta_k = scipy.fft.rfftn(density.reshape(tuple(nvox)))
    xpert, vpert = lpt1(
        positions,
        delta_k,
        nvox,
        cell_size,
        growth,
        velocity_factor,
        method="ngp",
    )
    phase = positions @ wavevector
    displacement = (
        amplitude
        * np.sin(phase)[:, None]
        * wavevector[None, :]
        / np.dot(wavevector, wavevector)
    )
    np.testing.assert_allclose(
        xpert,
        positions + growth * displacement,
        # FFT roundoff for a single exactly represented Fourier mode.
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        vpert,
        growth * velocity_factor * displacement,
        # FFT roundoff for a single exactly represented Fourier mode.
        rtol=2e-13,
        atol=2e-13,
    )


def test_lpt1__matches_direct_dft_reference_on_small_grid() -> None:
    """an explicit O(N^2) full-complex DFT is the independent oracle."""
    n = 8
    nvox = np.array([n, n, n])
    cell_size = 0.75
    positions, _ = create_grid(nvox, cell_size)
    rng = np.random.default_rng(42)
    density = rng.normal(size=(n, n, n))
    density -= density.mean()
    delta_full = np.fft.fftn(density)
    # LPT inputs produced by white_noise have exact-null Nyquist planes.
    delta_full[n // 2, :, :] = 0.0
    delta_full[:, n // 2, :] = 0.0
    delta_full[:, :, n // 2] = 0.0
    density = np.fft.ifftn(delta_full).real
    delta_rfft = scipy.fft.rfftn(density)
    delta_full = np.fft.fftn(density)
    axes = [np.fft.fftfreq(n, d=cell_size) * 2.0 * np.pi for _ in range(3)]
    kx, ky, kz = np.meshgrid(*axes, indexing="ij")
    k2 = kx**2 + ky**2 + kz**2
    nonzero = k2 > 0.0
    reference = np.zeros_like(positions)
    for particle, position in enumerate(positions):
        # FFT array index zero is the first voxel centre, not physical x=0.
        local_position = position - positions[0]
        phase = np.exp(
            1j
            * (
                kx * local_position[0]
                + ky * local_position[1]
                + kz * local_position[2]
            )
        )
        for axis, component in enumerate((kx, ky, kz)):
            # Bernardeau et al. eq. 94 evaluated by direct summation, not FFT.
            coefficient = np.zeros_like(delta_full)
            coefficient[nonzero] = (
                1j * component[nonzero] * delta_full[nonzero] / k2[nonzero]
            )
            reference[particle, axis] = np.real(np.sum(coefficient * phase)) / n**3
    xpert, vpert = lpt1(
        positions, delta_rfft, nvox, cell_size, 1.0, 1.0, method="ngp"
    )
    np.testing.assert_allclose(
        xpert - positions,
        reference,
        # Direct DFT and FFT sum 512 complex terms in different orders.
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        vpert,
        reference,
        # Direct DFT and FFT sum 512 complex terms in different orders.
        rtol=1e-13,
        atol=1e-13,
    )


def test_lpt1__shared_physical_mode_matches_across_resolutions() -> None:
    """the same represented physical mode gives equal LPT on shared sites."""
    low_nvox = np.array([16, 16, 16])
    high_nvox = np.array([32, 32, 32])
    cell_size = 1.0
    low_positions, _ = create_grid(low_nvox, cell_size)
    high_positions, _ = create_grid(high_nvox, cell_size)
    wavelength = 8.0
    wave = 2.0 * np.pi / wavelength
    low_density = -0.2 * np.cos(wave * low_positions[:, 0])
    high_density = -0.2 * np.cos(wave * high_positions[:, 0])
    low_delta = scipy.fft.rfftn(low_density.reshape(tuple(low_nvox)))
    high_delta = scipy.fft.rfftn(high_density.reshape(tuple(high_nvox)))
    low_x, low_v = lpt1(
        low_positions, low_delta, low_nvox, cell_size, 1.0, 2.0, method="ngp"
    )
    high_x, high_v = lpt1(
        low_positions, high_delta, high_nvox, cell_size, 1.0, 2.0, method="ngp"
    )
    np.testing.assert_allclose(
        low_x,
        high_x,
        # Independent FFT sizes differ only through float64 summation roundoff.
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        low_v,
        high_v,
        # Independent FFT sizes differ only through float64 summation roundoff.
        rtol=1e-13,
        atol=1e-13,
    )


def test_lpt2__single_mode_second_order_source_is_null() -> None:
    """the 2LPT quadratic invariant vanishes for one plane wave."""
    nvox = np.array([16, 16, 16])
    cell_size = 0.5
    positions, _ = create_grid(nvox, cell_size)
    density, _ = _plane_density(positions, 8.0, (1, 1, 0), 0.2)
    delta_k = scipy.fft.rfftn(density.reshape(tuple(nvox)))
    first_x, first_v = lpt1(
        positions, delta_k, nvox, cell_size, 0.8, 2.0, method="ngp"
    )
    second_x, second_v = lpt2(
        positions,
        delta_k,
        nvox,
        cell_size,
        0.8,
        -3.0 / 7.0,
        2.0,
        1.0,
        method="ngp",
    )
    np.testing.assert_allclose(
        second_x,
        first_x,
        # The analytic source is zero; residuals are chained FFT roundoff.
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        second_v,
        first_v,
        # The analytic source is zero; residuals are chained FFT roundoff.
        rtol=1e-13,
        atol=1e-13,
    )


def test_lpt2__matches_two_orthogonal_mode_solution() -> None:
    """standard 2LPT quadratic source for orthogonal plane waves.

    The independent oracle uses laplacian(phi2)=S and Psi2=+grad(phi2).
    """
    nvox = np.array([16, 16, 16])
    cell_size = 0.5
    boxsize = 8.0
    positions, _ = create_grid(nvox, cell_size)
    amplitude_x, amplitude_y = 0.2, 0.3
    wave = 2.0 * np.pi / boxsize
    density = -amplitude_x * np.cos(wave * positions[:, 0])
    density -= amplitude_y * np.cos(wave * positions[:, 1])
    delta_k = scipy.fft.rfftn(density.reshape(tuple(nvox)))
    g1, g2 = 0.8, -3.0 / 7.0
    aHf1, aHf2 = 2.0, 1.5
    xpert, vpert = lpt2(
        positions,
        delta_k,
        nvox,
        cell_size,
        g1,
        g2,
        aHf1,
        aHf2,
        method="ngp",
    )
    psi1 = np.zeros_like(positions)
    psi1[:, 0] = amplitude_x * np.sin(wave * positions[:, 0]) / wave
    psi1[:, 1] = amplitude_y * np.sin(wave * positions[:, 1]) / wave
    # S=A_x A_y cos(kx)cos(ky); solving laplacian(phi2)=S at
    # |k|^2=2k^2 and taking Psi2=+grad(phi2) gives the positive prefactor below.
    psi2 = np.zeros_like(positions)
    prefactor = amplitude_x * amplitude_y / (2.0 * wave)
    psi2[:, 0] = (
        prefactor
        * np.sin(wave * positions[:, 0])
        * np.cos(wave * positions[:, 1])
    )
    psi2[:, 1] = (
        prefactor
        * np.cos(wave * positions[:, 0])
        * np.sin(wave * positions[:, 1])
    )
    np.testing.assert_allclose(
        xpert,
        positions + g1 * psi1 + g2 * psi2,
        # A few exactly represented modes leave only chained FFT roundoff.
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        vpert,
        g1 * aHf1 * psi1 + g2 * aHf2 * psi2,
        # A few exactly represented modes leave only chained FFT roundoff.
        rtol=2e-13,
        atol=2e-13,
    )


@pytest.mark.slow
@pytest.mark.parametrize("compensate", [False, True])
def test_lpt1__cic_plane_wave_converges_at_second_order(compensate) -> None:
    """linear CIC has O(h^2) error for a smooth plane wave."""
    positions = np.random.default_rng(42).uniform(-0.5, 0.5, size=(128, 3))
    wave = 2.0 * np.pi
    expected = np.zeros_like(positions)
    expected[:, 0] = 0.2 * np.sin(wave * positions[:, 0]) / wave
    errors = []
    for n in (16, 32, 64):
        nvox = np.array([n, n, n])
        cell_size = 1.0 / n
        grid, _ = create_grid(nvox, cell_size)
        density = -0.2 * np.cos(wave * grid[:, 0])
        delta_k = scipy.fft.rfftn(density.reshape(tuple(nvox)))
        xpert, _ = lpt1(
            positions,
            delta_k,
            nvox,
            cell_size,
            1.0,
            1.0,
            compensate=compensate,
            method="cic",
        )
        errors.append(np.sqrt(np.mean(((xpert - positions) - expected) ** 2)))
    ratios = np.asarray(errors[:-1]) / np.asarray(errors[1:])
    # Second order predicts 4x refinement ratios; 3 allows pre-asymptotic terms.
    assert np.all(ratios >= 3.0), (errors, ratios)
