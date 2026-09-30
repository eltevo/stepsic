from __future__ import annotations

import numpy as np
import pytest

from stepsic.data import CosmoData
from stepsic.geometry import create_binner
from stepsic.particles import create_shell_particles


@pytest.mark.parametrize(
    ("geometry", "bin_mode", "rcrit"),
    [
        ("spherical", "omega", None),
        ("spherical", "volume", None),
        ("cylindrical", "omega", None),
        ("cylindrical", "volume", None),
        ("spherical", "omega", 12.0),
        ("cylindrical", "omega", 12.0),
    ],
)
def test_shell_loads__fill_shells_at_mean_density(geometry, bin_mode, rcrit) -> None:
    """Shell loads realise the mean density shell by shell out to R_3D.

    The mass whose radius falls in [r_i, r_i+1) is rho_mean times the shell
    volume, so the load fills the sphere (4/3 pi R^3) or cylinder
    (pi R^2 L_z) of radius R_3D with uniform mean density. With RCRIT the
    interior up to the last shell edge inside RCRIT is one pool of equal
    masses. The Omega_m normalisation of CosmoData must agree with the load
    and leave its masses unchanged.
    """
    params = {
        "GEOMETRY": geometry, "BIN_MODE": bin_mode, "D_4D": 20.0, "NRBINS": 8,
        "R_3D": 30.0, "RHO_MEAN": 2.0, "NSHELL": 16, "SEED": 42,
        "LBOX": np.array([60.0, 60.0, 20.0]), "RCRIT": rcrit,
    }
    pos, mass = create_shell_particles(params)
    height = 20.0
    if geometry == "spherical":
        radius = np.linalg.norm(pos, axis=1)
        volume = lambda r: 4.0 / 3.0 * np.pi * r**3
    else:
        radius = np.linalg.norm(pos[:, :2], axis=1)
        volume = lambda r: np.pi * r**2 * height
        assert np.all((pos[:, 2] >= 0.0) & (pos[:, 2] < height))

    edges = np.array([create_binner(params).r_limit(i) for i in range(9)])
    # Constant-volume edges come from a root solve with tolerance 1e-6.
    np.testing.assert_allclose(edges[[0, -1]], [0.0, 30.0], rtol=1e-6, atol=0.0)
    if rcrit is not None:
        boundary = edges[edges <= rcrit].max()
        inside = radius < boundary
        np.testing.assert_array_equal(mass[inside], mass[inside][0])
        edges = np.concatenate(([0.0], edges[edges >= boundary]))
    shell_mass = np.histogram(radius, bins=edges, weights=mass)[0]
    # Masses are rho V / count per shell: float64 roundoff only.
    np.testing.assert_allclose(shell_mass, 2.0 * np.diff(volume(edges)), rtol=1e-12, atol=0.0)

    load = CosmoData(pos=pos, mass=mass.copy())
    load.rescale_snapshot_mass(
        {**params, "OMEGA_M": 0.25, "RHO_CRIT": 8.0}
    )
    np.testing.assert_allclose(load.mass, mass, rtol=1e-12, atol=0.0)
