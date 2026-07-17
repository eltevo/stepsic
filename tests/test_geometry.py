from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import chi2

from conftest import SEEDS
from stepsic.geometry import (
    CylindricalBinner,
    CylindricalConstantVolume,
    CylindricalLinear,
    SphericalBinner,
    SphericalConstantVolume,
    SphericalLinear,
    _bin_index_for_radius,
    _random_unit_vectors_sphere,
    create_binner,
    create_cylindrical_shells,
    create_spherical_shells,
    shell_masses,
)
from stepsic.rng import RNG


@pytest.mark.parametrize(
    "y",
    [
        0.0,
        1e-6,
        0.1,
        1.0,
        np.pi,
        2.0 * np.pi - 1.0,
        2.0 * np.pi - 1e-6,
        2.0 * np.pi,
    ],
)
def test_invert_x_minus_sin_x__round_trips_and_matches_brentq(y) -> None:
    """T1/T3: algebraic round trip plus scipy.brentq's bracketed root oracle."""
    actual = float(SphericalBinner.invert_x_minus_sin_x(y, tol=1e-10)[0])
    expected = brentq(
        lambda x: x - np.sin(x) - y,
        0.0,
        2.0 * np.pi,
        xtol=1e-14,
    )
    np.testing.assert_allclose(
        actual,
        expected,
        # hybr stops on residual while brentq stops on bracket width.
        rtol=0.0,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        actual - np.sin(actual),
        y,
        # Near x=0 the cubic inverse is ill-conditioned; 2e-8 follows root tol.
        rtol=0.0,
        atol=2e-8,
    )


@pytest.mark.parametrize("y", [-1.0, 2.0 * np.pi + 1.0])
def test_invert_x_minus_sin_x__rejects_out_of_range_boundaries(y) -> None:
    """T1: the documented closed domain rejects limit-minus/plus-one inputs."""
    with pytest.raises(ValueError, match=r"must be in \[0, 2π\]"):
        SphericalBinner.invert_x_minus_sin_x(y)


def test_shell_volume__matches_analytic_sphere_and_cylinder() -> None:
    """T2: Euclidean shell volumes are 4pi/3(r1^3-r0^3) and pi(r1^2-r0^2)L."""
    spherical = SphericalLinear(10.0, 8, 2.0)
    cylindrical = CylindricalLinear(10.0, 8, 2.0)
    index = 3
    sr0, sr1 = spherical.r_limit(index), spherical.r_limit(index + 1)
    cr0, cr1 = cylindrical.r_limit(index), cylindrical.r_limit(index + 1)
    expected_sphere = 4.0 * np.pi * (sr1**3 - sr0**3) / 3.0
    expected_cylinder = np.pi * (cr1**2 - cr0**2) * 7.0
    np.testing.assert_allclose(
        spherical.shell_volume(index),
        expected_sphere,
        # Multiplication/division ordering differs by at most float64 roundoff.
        rtol=3e-16,
        atol=0.0,
    )
    np.testing.assert_allclose(
        cylindrical.shell_volume(index, 7.0),
        expected_cylinder,
        # Multiplication/division ordering differs by at most float64 roundoff.
        rtol=3e-16,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("binner_type", "weight"),
    [
        (SphericalBinner, lambda r: r**2),
        (CylindricalBinner, lambda r: r),
    ],
)
def test_centroid__matches_scipy_quadrature(binner_type, weight) -> None:
    """T3: scipy.integrate.quad independently computes the radial first moment."""
    r0, r1 = 1.25, 3.75
    numerator = quad(lambda r: r * weight(r), r0, r1, epsabs=1e-13)[0]
    denominator = quad(weight, r0, r1, epsabs=1e-13)[0]
    expected = numerator / denominator
    np.testing.assert_allclose(
        binner_type.centroid(r0, r1),
        expected,
        # QUADPACK and the closed form are both float64 evaluations.
        rtol=2e-15,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "binner",
    [
        SphericalLinear(10.0, 8, 2.0),
        SphericalConstantVolume(10.0, 8, 30.0),
        CylindricalLinear(10.0, 8, 2.0),
        CylindricalConstantVolume(10.0, 8, 30.0),
    ],
)
def test_binner_families__are_monotone_with_centroids_inside_bins(binner) -> None:
    """T1: ordered edges and an in-bin mass centroid are structural invariants."""
    edges = np.array([binner.r_limit(i) for i in range(9)])
    assert np.all(np.diff(edges) > 0.0)
    centroids = np.array([binner.r_centroid(i) for i in range(8)])
    assert np.all(centroids > edges[:-1])
    assert np.all(centroids < edges[1:])


@pytest.mark.parametrize(
    ("binner", "compact_measure"),
    [
        (
            SphericalConstantVolume(10.0, 8, 30.0),
            lambda r: (
                lambda omega: 2.0 * omega - np.sin(2.0 * omega)
            )(2.0 * np.arctan(r / 20.0)),
        ),
        (
            CylindricalConstantVolume(10.0, 8, 30.0),
            lambda r: 1.0 - np.cos(2.0 * np.arctan(r / 20.0)),
        ),
    ],
)
def test_constant_volume_binners__have_equal_compact_volume(
    binner, compact_measure
) -> None:
    """T2: Racz (2018), eqs. 4-5, give equal compact-volume increments."""
    measures = np.array([compact_measure(binner.r_limit(i)) for i in range(9)])
    increments = np.diff(measures)
    np.testing.assert_allclose(
        increments,
        increments[0],
        # Root inversion and trigonometric round trips accumulate float64 error.
        rtol=2e-8,
        atol=2e-10,
    )


@pytest.mark.parametrize(
    ("params", "expected_type"),
    [
        ({"GEOMETRY": "spherical", "BIN_MODE": "omega"}, SphericalLinear),
        ({"GEOMETRY": "spherical", "BIN_MODE": "volume"}, SphericalConstantVolume),
        ({"GEOMETRY": "cylindrical", "BIN_MODE": "omega"}, CylindricalLinear),
        (
            {"GEOMETRY": "cylindrical", "BIN_MODE": "volume"},
            CylindricalConstantVolume,
        ),
    ],
)
def test_create_binner__dispatches_all_four_families(
    params, expected_type, params_factory
) -> None:
    """T1: geometry/bin-mode pairs map to the specified concrete family."""
    assert isinstance(create_binner(params_factory(**params)), expected_type)


@pytest.mark.parametrize(
    ("binner", "length"),
    [
        (SphericalLinear(10.0, 8, 2.0), None),
        (CylindricalLinear(10.0, 8, 2.0), 7.0),
    ],
)
def test_shell_masses__conserve_density_times_total_volume(binner, length) -> None:
    """T1: particle masses sum exactly to rho times the represented volume."""
    rho_mean = 2.5
    n_per_shell = 32
    masses = shell_masses(binner, 8, n_per_shell, rho_mean, Lz=length)
    if isinstance(binner, SphericalBinner):
        volume = 4.0 * np.pi * binner.r_limit(8) ** 3 / 3.0
    else:
        volume = np.pi * binner.r_limit(8) ** 2 * length
    np.testing.assert_allclose(
        masses.sum() * n_per_shell,
        rho_mean * volume,
        # Summing eight shell differences incurs float64 cancellation.
        rtol=1e-14,
        atol=0.0,
    )


def test_bin_index_for_radius__assigns_edge_to_outer_bin() -> None:
    """T1: bins are left-closed/right-open at zero, edge-eps, and exact edge."""
    binner = SphericalLinear(10.0, 8, 2.0)
    edge = binner.r_limit(3)
    assert _bin_index_for_radius(binner, 0.0) == 0
    assert _bin_index_for_radius(binner, np.nextafter(edge, 0.0)) == 2
    assert _bin_index_for_radius(binner, edge) == 3


@pytest.mark.parametrize("seed", SEEDS)
def test_random_unit_vectors_sphere__have_unit_norm_and_rayleigh_isotropy(
    seed,
) -> None:
    """T1/T4: exact unit norms and the chi-square Rayleigh resultant at 5 sigma."""
    count = 4096
    vectors = _random_unit_vectors_sphere(count, RNG(seed))
    np.testing.assert_allclose(
        np.linalg.norm(vectors, axis=1),
        1.0,
        # Normalization uses one sqrt and three divisions in float64.
        rtol=3e-16,
        atol=3e-16,
    )
    resultant_squared = np.dot(vectors.sum(axis=0), vectors.sum(axis=0))
    # For isotropic unit vectors, 3|sum u|^2/N ~ chi-square_3 asymptotically.
    # 5.0 sigma two-sided normal tail probability is 5.733e-7.
    five_sigma_limit = count * chi2.ppf(1.0 - 5.733e-7, df=3) / 3.0
    assert resultant_squared <= five_sigma_limit


@pytest.mark.parametrize("geometry", ["spherical", "cylindrical"])
def test_create_shells__counts_radii_and_seed_are_deterministic(
    geometry, params_factory
) -> None:
    """T1: counts, shell membership, and same-seed output are invariants."""
    params = params_factory(
        GEOMETRY=geometry,
        BIN_MODE="volume",
        NRBINS=4,
        NSHELL=16,
        R_3D=30.0,
        RCRIT=None,
    )
    binner = create_binner(params)
    if geometry == "spherical":
        first = create_spherical_shells(
            binner, 4, 16, params["RHO_MEAN"], seed=params["SEED"]
        )
        second = create_spherical_shells(
            binner, 4, 16, params["RHO_MEAN"], seed=params["SEED"]
        )
        radii = np.linalg.norm(first[0], axis=1)
    else:
        length = float(np.min(params["LBOX"]))
        first = create_cylindrical_shells(
            binner, 4, 16, length, params["RHO_MEAN"], seed=params["SEED"]
        )
        second = create_cylindrical_shells(
            binner, 4, 16, length, params["RHO_MEAN"], seed=params["SEED"]
        )
        radii = np.linalg.norm(first[0][:, :2], axis=1)
        assert np.all(first[0][:, 2] >= 0.0)
        assert np.all(first[0][:, 2] < length)
    positions, masses = first
    assert positions.shape == (64, 3)
    assert masses.shape == (64,)
    for index in range(4):
        shell_radii = radii[index * 16 : (index + 1) * 16]
        assert np.all(shell_radii >= binner.r_limit(index))
        assert np.all(shell_radii < binner.r_limit(index + 1))
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
