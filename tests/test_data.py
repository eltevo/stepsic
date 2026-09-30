from __future__ import annotations

import numpy as np

from stepsic.data import CosmoData
from stepsic.units import UNIT_L, UNIT_M, UNIT_V


def test_unit_conversion__round_trips() -> None:
    """Glass arrays are read in their own units and written back in them.

    Input units twice/three/four times the internal ones scale the arrays up
    by that factor on the way in, and the inverse conversion restores them.
    """
    pos = np.array([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]])
    vel = np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])
    mass = np.array([3.0, 9.0])
    data = CosmoData(pos=pos.copy(), vel=vel.copy(), mass=mass.copy())
    params = {"UNIT_L_IN_CM": 2 * UNIT_L, "UNIT_V_IN_KMPS": 3 * UNIT_V, "UNIT_M_IN_G": 4 * UNIT_M}

    data.to_internal_units(params)
    np.testing.assert_array_equal(data.pos, 2.0 * pos)
    np.testing.assert_array_equal(data.vel, 3.0 * vel)
    np.testing.assert_array_equal(data.mass, 4.0 * mass)

    data.from_internal_units(params)
    np.testing.assert_array_equal(data.pos, pos)
    np.testing.assert_array_equal(data.vel, vel)
    np.testing.assert_array_equal(data.mass, mass)
