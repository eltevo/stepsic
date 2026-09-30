from __future__ import annotations

import numpy as np
import pytest
import scipy.fft

from stepsic.field import generate_delta_k
from stepsic.lpt import lpt1, lpt2
from stepsic.particles import create_grid


N = 16
BOX = 16.0
CELL = BOX / N
K_FUND = 2.0 * np.pi / BOX


def _lattice() -> np.ndarray:
    return create_grid((N, N, N), CELL, return_coords=False)


def _density_mode(positions: np.ndarray, modes, weights=None) -> np.ndarray:
    """Fourier modes (1/N) sum_j w_j exp(-i k.x_j) of a particle set or sampled field.

    With unit weights this is the Eulerian density contrast carried by
    equal-mass particles at every k != 0 of a regular lattice.
    """
    k = K_FUND * np.asarray(modes, dtype=np.float64)
    if weights is None:
        weights = np.ones(len(positions))
    return np.concatenate([
        weights @ np.exp(-1j * (positions @ kb.T)) / len(positions)
        for kb in np.array_split(k, max(1, len(k) // 128))
    ])


def test_lpt1__particle_density_reproduces_input_modes(tiny_chunks) -> None:
    """The displaced lattice carries the requested linear density field.

    Continuity gives delta = -div(Psi) at first order, so the Eulerian
    density of the Zel'dovich particles must equal the input delta mode
    by mode, up to terms of relative size delta_rms ~ 1e-4. Sign, gradient
    axis, Poisson normalisation, or registration of the particle lattice
    with the FFT mesh all break this identity at O(1).
    """
    kh = np.geomspace(1e-3, 1e2, 64)
    delta_k = generate_delta_k(kh, 1e-8 / kh, (N, N, N), CELL, seed=42)
    delta = scipy.fft.irfftn(delta_k, s=(N, N, N)).ravel()
    q = _lattice()
    x, _ = lpt1(q, delta_k, (N, N, N), CELL, 1.0, 0.0, method="cic")

    # Every non-DC mode of the half space below the Nyquist index.
    grid = np.arange(-(N // 2) + 1, N // 2)
    modes = np.array(
        [(i, j, l) for i in grid for j in grid for l in range(N // 2) if l > 0 or (i, j) > (0, 0)]
    )
    wanted = _density_mode(q, modes, weights=delta)
    got = _density_mode(x, modes)
    scale = np.sqrt(np.mean(np.abs(wanted) ** 2))
    # Second-order terms are ~delta_rms (1e-4) of the first-order mode.
    assert np.max(np.abs(got - wanted)) <= 1e-3 * scale


def test_lpt2__second_order_density_matches_perturbation_theory_kernel(tiny_chunks) -> None:
    """2LPT reproduces the second-order Eulerian density kernel F2.

    For two orthogonal waves of equal |k|, standard perturbation theory
    gives F2(k1, k2) = 5/7 (Bernardeau et al. 2002, eq. 45 at mu = 0),
    while Zel'dovich alone gives 1/2 and a reversed second-order term
    gives 2/7. With D2 = -(3/7) D1^2 (Einstein-de Sitter) the (k1 + k2)
    mode of the particle density is 2 F2 delta(k1) delta(k2), independent
    of the sign conventions used inside lpt2.
    """
    eps = 1e-2
    q = _lattice()
    delta = eps * (np.cos(K_FUND * q[:, 0]) + np.cos(K_FUND * q[:, 1]))
    delta_k = scipy.fft.rfftn(delta.reshape(N, N, N))
    x, _ = lpt2(q, delta_k, (N, N, N), CELL, 1.0, -3.0 / 7.0, 0.0, 0.0, method="cic")

    d1x, d1y = _density_mode(q, [(1, 0, 0), (0, 1, 0)], weights=delta)
    d2 = _density_mode(x, [(1, 1, 0)])[0]
    F2 = d2 / (2.0 * d1x * d1y)
    # Fourth-order corrections are eps^2 = 1e-4 of the second-order mode.
    np.testing.assert_allclose(F2, 5.0 / 7.0, rtol=1e-3, atol=0.0)


@pytest.mark.parametrize("compensate", [False, True])
@pytest.mark.parametrize(("method", "order"), [("ngp", 1), ("cic", 2), ("tsc", 3)])
def test_lpt1__off_grid_particles_receive_their_own_displacement(
    method, order, compensate
) -> None:
    """Particles between mesh nodes get the displacement of their own position.

    The mesh value i describes the field at the voxel centre, where the
    lattice particles sit. Interpolating a plane wave to arbitrary positions
    must therefore reproduce Psi(x) without a phase shift, attenuated only by
    the kernel window sinc(k dx / 2)^p (Hockney & Eastwood 1988, sec. 5-3),
    and with unit amplitude once `compensate` divides that window out. A
    kernel centred off the node hands every particle the displacement of a
    point displaced by a fraction of this mesh's cell; meshes of different
    cell size then disagree, which breaks multiresolution registration.
    """
    amplitude = 0.1
    q = _lattice()
    delta = -amplitude * np.cos(K_FUND * q[:, 0])
    delta_k = scipy.fft.rfftn(delta.reshape(N, N, N))
    samples = 256 * N
    x = np.zeros((samples, 3))
    x[:, 0] = -BOX / 2 + (np.arange(samples) + 0.5) * BOX / samples
    xpert, _ = lpt1(
        x, delta_k, (N, N, N), CELL, 1.0, 0.0, method=method, compensate=compensate
    )
    psi = xpert[:, 0] - x[:, 0]

    # Analytic solution Psi_x = (A / k) sin(k x); project the measured field on it.
    in_phase = 2.0 * np.mean(psi * np.sin(K_FUND * x[:, 0])) * K_FUND / amplitude
    quadrature = 2.0 * np.mean(psi * np.cos(K_FUND * x[:, 0])) * K_FUND / amplitude
    shift_in_cells = np.arctan2(-quadrature, in_phase) / K_FUND / CELL
    transfer = np.hypot(in_phase, quadrature)
    window = 1.0 if compensate else np.sinc(K_FUND * CELL / (2.0 * np.pi)) ** order
    # 256 samples per cell integrate the piecewise-polynomial field to ~1e-5.
    assert abs(shift_in_cells) <= 1e-3
    np.testing.assert_allclose(transfer, window, rtol=1e-4, atol=0.0)
