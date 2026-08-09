from __future__ import annotations

import numpy as np
import pytest

from conftest import SEEDS
from stepsic.interpolation import (
    CICKernel,
    FieldDepositor,
    FieldInterpolator,
    KERNELS,
    NGPKernel,
    TSCKernel,
    compensation_kernel,
)


def _triple_loop_interpolate(field, x, geometry, method):
    """Structurally independent particle/stencil/component reference loop."""
    kernel = KERNELS[method]()
    expected = np.zeros((len(x), field.shape[0]))
    grid_positions = geometry.pos_to_grid(x)
    for particle, grid_position in enumerate(grid_positions):
        lower = np.floor(grid_position).astype(int)
        fractions = grid_position - lower
        weights = [
            np.array([w[0] for w in kernel.weights(np.array([fraction]))])
            for fraction in fractions
        ]
        for ox in range(kernel.support):
            for oy in range(kernel.support):
                for oz in range(kernel.support):
                    index = lower + kernel.base_offset + (ox, oy, oz)
                    for axis in range(3):
                        if geometry.periodic[axis]:
                            index[axis] %= geometry.nvox[axis]
                        else:
                            index[axis] = np.clip(
                                index[axis], 0, geometry.nvox[axis] - 1
                            )
                    weight = weights[0][ox] * weights[1][oy] * weights[2][oz]
                    for component in range(field.shape[0]):
                        expected[particle, component] += (
                            weight * field[(component, *index)]
                        )
    return expected


@pytest.mark.parametrize("method", ["ngp", "cic", "tsc"])
@pytest.mark.parametrize("seed", SEEDS)
def test_kernel_weights__partition_unity(method, seed) -> None:
    """every one-dimensional mass-assignment stencil partitions unity."""
    dx = np.random.default_rng(seed).uniform(size=64)
    total = np.sum(KERNELS[method]().weights(dx), axis=0)
    np.testing.assert_allclose(
        total,
        1.0,
        # Adding up to three float64 weights incurs ordinary summation roundoff.
        rtol=1e-15,
        atol=1e-15,
    )


@pytest.mark.parametrize(
    ("kernel", "expected"),
    [
        (NGPKernel(), (1.0,)),
        (CICKernel(), (0.75, 0.25)),
        (TSCKernel(), (0.28125, 0.6875, 0.03125)),
    ],
)
def test_kernel_weights__match_hockney_eastwood_spot_values(
    kernel, expected
) -> None:
    """Hockney & Eastwood (1988), section 5-3, at dx=1/4."""
    actual = np.array(kernel.weights(np.array([0.25]))).ravel()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("kernel", "offsets", "centroid_shift"),
    [
        (CICKernel(), np.array([0.0, 1.0]), 0.0),
        (TSCKernel(), np.array([-1.0, 0.0, 1.0]), -0.5),
    ],
)
def test_kernel_weights__satisfy_first_moment(
    kernel, offsets, centroid_shift
) -> None:
    """Hockney-Eastwood CIC/TSC stencils reproduce their linear centroid."""
    dx = np.linspace(0.0, 1.0, 17, endpoint=False)
    weights = np.array(kernel.weights(dx))
    actual = offsets @ weights
    np.testing.assert_allclose(
        actual,
        dx + centroid_shift,
        # Weighted sums use up to three float64 products.
        rtol=1e-15,
        atol=1e-15,
    )


@pytest.mark.parametrize("method", ["ngp", "cic", "tsc"])
def test_field_interpolator__matches_independent_triple_loop(
    method, grid_geom
) -> None:
    """an in-test scalar triple loop is the independent implementation."""
    geometry = grid_geom()
    rng = np.random.default_rng(42)
    field = rng.normal(size=(2, 8, 8, 8))
    positions = rng.uniform(0.0, 8.0, size=(20, 3))
    actual = FieldInterpolator(method, geometry)(field, positions)
    expected = _triple_loop_interpolate(field, positions, geometry, method)
    np.testing.assert_allclose(
        actual,
        expected,
        # Vectorized and scalar accumulation orders differ across <=27 terms.
        rtol=1e-14,
        atol=1e-14,
    )


@pytest.mark.parametrize("method", ["ngp", "cic", "tsc"])
def test_field_depositor__conserves_particle_mass(method, grid_geom) -> None:
    """partition of unity makes total deposited mass an invariant."""
    geometry = grid_geom()
    rng = np.random.default_rng(42)
    positions = rng.uniform(0.0, 8.0, size=(100, 3))
    masses = rng.uniform(0.1, 2.0, size=100)
    deposited = FieldDepositor(method, geometry)(positions, masses)
    np.testing.assert_allclose(
        deposited.sum(),
        masses.sum(),
        # Scatter accumulation reorders 100 float64 additions.
        rtol=1e-14,
        atol=1e-14,
    )


def test_field_depositor__ngp_matches_numpy_histogramdd(grid_geom) -> None:
    """np.histogramdd is the independent NGP binning oracle."""
    geometry = grid_geom(vox_offset=0.0)
    rng = np.random.default_rng(112358)
    positions = rng.uniform(0.0, 8.0, size=(200, 3))
    masses = rng.uniform(0.1, 2.0, size=200)
    actual = FieldDepositor("ngp", geometry)(positions, masses)
    expected, _ = np.histogramdd(
        positions, bins=(8, 8, 8), range=((0, 8), (0, 8), (0, 8)), weights=masses
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("method", ["ngp", "cic", "tsc"])
def test_interpolate_deposit__are_adjoint(method, grid_geom) -> None:
    """<interpolate(f),m> = <f,deposit(m)> is the defining adjoint identity."""
    geometry = grid_geom()
    rng = np.random.default_rng(0)
    field = rng.normal(size=(2, 8, 8, 8))
    positions = rng.uniform(0.0, 8.0, size=(40, 3))
    values = rng.normal(size=(40, 2))
    interpolated = FieldInterpolator(method, geometry)(field, positions)
    deposited = FieldDepositor(method, geometry)(positions, values)
    np.testing.assert_allclose(
        np.vdot(interpolated, values),
        np.vdot(field, deposited),
        # The two dot products accumulate identical terms in different orders.
        rtol=1e-13,
        atol=1e-13,
    )


def test_field_depositor__periodic_stencil_wraps_into_cell_zero(grid_geom) -> None:
    """a CIC stencil crossing the upper periodic face wraps to index zero."""
    geometry = grid_geom(nvox=(4, 4, 4), boxsize=(4.0, 4.0, 4.0))
    eps = np.finfo(np.float64).eps
    grid = FieldDepositor("cic", geometry)(
        np.array([[4.0 - eps, 0.5, 0.5]]), np.array([1.0])
    )
    assert grid[0, 0, 0] > 0.0
    np.testing.assert_array_equal(grid.sum(), 1.0)


@pytest.mark.parametrize("method", ["ngp", "cic", "tsc"])
def test_compensation_kernel__matches_hockney_eastwood_sinc(method) -> None:
    """Hockney & Eastwood (1988) section 5-3 supplies the sinc oracle."""
    nvox = np.array([4, 6, 8])
    boxsize = np.array([2.0, 3.0, 4.0])
    axes = [
        np.fft.fftfreq(nvox[0], d=0.5) * 2.0 * np.pi,
        np.fft.fftfreq(nvox[1], d=0.5) * 2.0 * np.pi,
        np.fft.rfftfreq(nvox[2], d=0.5) * 2.0 * np.pi,
    ]
    kvec = np.stack(np.broadcast_arrays(
        axes[0][:, None, None], axes[1][None, :, None], axes[2][None, None, :]
    ))
    power = KERNELS[method]().support
    expected = np.ones(kvec.shape[1:])
    for axis in range(3):
        k_nyquist = np.pi * nvox[axis] / boxsize[axis]
        expected /= np.sinc(kvec[axis] / (2.0 * k_nyquist)) ** power
    actual = compensation_kernel(kvec, nvox, boxsize, method=method)
    assert actual[0, 0, 0] == 1.0
    np.testing.assert_allclose(
        actual,
        expected,
        # Both evaluations use float64 sinc products in different loop layouts.
        rtol=1e-15,
        atol=0.0,
    )


def test_grid_geometry_pos_to_grid__maps_boundary_coordinates(grid_geom) -> None:
    """the documented origin/cell-size/offset coordinate definition."""
    geometry = grid_geom(
        nvox=(4, 4, 4),
        boxsize=(8.0, 8.0, 8.0),
        origin=(-4.0, -4.0, -4.0),
        vox_offset=0.5,
    )
    positions = np.array(
        [
            [-4.0, -4.0, -4.0],
            [0.0, 0.0, 0.0],
            [4.0, 4.0, 4.0],
            [12.0, 12.0, 12.0],
        ]
    )
    expected = np.array(
        [
            [-0.5, -0.5, -0.5],
            [1.5, 1.5, 1.5],
            [-0.5, -0.5, -0.5],
            [-0.5, -0.5, -0.5],
        ]
    )
    np.testing.assert_array_equal(geometry.pos_to_grid(positions), expected)
