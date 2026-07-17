"""Tests for particle-data transformations."""

from copy import deepcopy

import numpy as np

from stepsic.data import CosmoData
from stepsic.units import UNIT_L, UNIT_M, UNIT_V


def _sample_data():
    return CosmoData(
        pos=np.array([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]]),
        vel=np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]),
        mass=np.array([3.0, 9.0]),
    )


def test_unit_conversion_round_trip():
    """T1: conversion to internal units and back is an algebraic inverse."""
    data = _sample_data()
    original = deepcopy(data)
    params = {
        "UNIT_L_IN_CM": 2 * UNIT_L,
        "UNIT_V_IN_KMPS": 3 * UNIT_V,
        "UNIT_M_IN_G": 4 * UNIT_M,
    }

    data.to_internal_units(params)
    # T2: each explicit array is the input expressed in units 2x/3x/4x larger.
    np.testing.assert_array_equal(
        data.pos,
        np.array([[2.0, 4.0, 6.0], [10.0, 12.0, 14.0]]),
    )
    np.testing.assert_array_equal(
        data.vel,
        np.array([[6.0, 12.0, 18.0], [24.0, 30.0, 36.0]]),
    )
    np.testing.assert_array_equal(data.mass, np.array([12.0, 36.0]))

    data.from_internal_units(params)
    np.testing.assert_allclose(data.pos, original.pos, rtol=0, atol=0)
    np.testing.assert_allclose(data.vel, original.vel, rtol=0, atol=0)
    np.testing.assert_allclose(data.mass, original.mass, rtol=0, atol=0)


def test_rescale_size_matches_geometry_ratio():
    """T2: hand-derived target/current length factors set both geometries."""
    cubical = _sample_data()
    cubical.Lbox = 2.0
    cubical.rescale_snapshot_size(
        {
            "GEOMETRY": "cubical",
            "LBOX": np.array([10.0, 10.0, 10.0]),
        },
    )
    # T2: target/current extent is 10/2=5.
    np.testing.assert_array_equal(
        cubical.pos,
        np.array([[5.0, 10.0, 15.0], [25.0, 30.0, 35.0]]),
    )

    spherical = _sample_data()
    spherical.Lbox = 1.0
    spherical.rescale_snapshot_size(
        {
            "GEOMETRY": "spherical",
            "HINDEPENDENT": False,
            "H": 0.7,
        },
    )
    # T2: values are the inputs converted by h=0.7.
    np.testing.assert_allclose(
        spherical.pos,
        np.array([[0.7, 1.4, 2.1], [3.5, 4.2, 4.9]]),
        # Decimal 0.7 is not binary-exact; 1e-15 covers accumulated roundoff.
        rtol=1e-15,
        atol=0,
    )


def test_rescale_mass_matches_mean_density():
    """T2: total mass is rho_mean times the analytic box volume."""
    data = _sample_data()
    params = {
        "GEOMETRY": "cubical",
        "LBOX": np.array([2.0, 2.0, 2.0]),
        "RHO_CRIT": 3.0,
        "OMEGA_M": 1.0,
        "RHO_MEAN": 3.0,
    }

    data.rescale_snapshot_mass(params)

    # T2: initial omega=12/(8*3)=0.5, so target omega=1 gives factor 2.
    np.testing.assert_allclose(data.mass, np.array([6.0, 18.0]), rtol=0, atol=0)
    np.testing.assert_array_equal(data.mass_list, np.array([6.0, 18.0]))
    # T2: rho_mean=3 and V=8, hence M_box=24.
    assert data.M_box == 24.0


def test_center_then_periodic_shift_respects_axis_policy():
    """T1: centering and modulo wrapping are independently derived transforms."""
    data = CosmoData(
        pos=np.array([[7.0, 5.0, 1.0], [3.0, 1.0, 7.0]]),
        vel=np.zeros((2, 3)),
        mass=np.ones(2),
    )
    params = {
        "LBOX": np.array([8.0, 8.0, 8.0]),
        "COI": np.array([1.0, 2.0, 3.0]),
        "PERIODIC": np.array([True, False, True]),
    }

    data.center_snapshot(params)
    data.periodic_shift(params)

    # T2 hand calculation: subtract half-box/COI, then wrap as (x + L/2) mod L.
    expected = np.array([[6.0, -1.0, 6.0], [2.0, -5.0, 4.0]])
    np.testing.assert_allclose(data.pos, expected, rtol=0, atol=0)
    assert np.all((data.pos[:, [0, 2]] >= 0) & (data.pos[:, [0, 2]] < 8))
    np.testing.assert_array_equal(data.pos[:, 1], expected[:, 1])


def test_deepcopy_is_independent():
    """T1: Python deepcopy provides an independent particle-data snapshot."""
    original = _sample_data()
    copied = deepcopy(original)

    copied.pos[0, 0] = -99
    copied.vel[0, 0] = -98
    copied.mass[0] = -97
    copied.id[0] = 42

    assert original.pos[0, 0] == 1
    assert original.vel[0, 0] == 2
    assert original.mass[0] == 3
    assert original.id[0] == 0
