from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from stepsic.geometry import CylindricalConstantVolume, SphericalBinner, SphericalConstantVolume


@pytest.mark.parametrize(
    ("binner", "compact_measure"),
    [
        # Volume of an S^3 cap of angle omega is proportional to 2 omega - sin(2 omega).
        (SphericalConstantVolume(10.0, 8, 30.0), lambda w: 2.0 * w - np.sin(2.0 * w)),
        # Area of an S^2 cap of angle omega is proportional to 1 - cos(omega).
        (CylindricalConstantVolume(10.0, 8, 30.0), lambda w: 1.0 - np.cos(w)),
    ],
    ids=["spherical", "cylindrical"],
)
def test_constant_volume_binners__enclose_equal_compact_volume(binner, compact_measure) -> None:
    """Constant-volume shells hold equal volumes of the compact space.

    The stereographic projection maps radius r to the compact angle
    omega = 2 arctan(r / D_s) with D_s = 2 R_4D = 20 (Racz 2018, eqs. 4-5).
    """
    omega = 2.0 * np.arctan(np.array([binner.r_limit(i) for i in range(9)]) / 20.0)
    increments = np.diff(compact_measure(omega))
    # Root inversion and trigonometric round trips accumulate float64 error.
    np.testing.assert_allclose(increments, increments[0], rtol=2e-8, atol=2e-10)


@pytest.mark.parametrize("y", [0.0, 1e-6, 1.0, np.pi, 2.0 * np.pi - 1e-6, 2.0 * np.pi])
def test_invert_x_minus_sin_x__converges_up_to_2pi(y) -> None:
    """The outermost constant-volume shell needs the inverse right up to 2 pi.

    scipy's bracketed brentq is the independent root oracle; near both ends
    the derivative of x - sin(x) vanishes and a naive starter stalls.
    """
    actual = float(SphericalBinner.invert_x_minus_sin_x(y, tol=1e-10)[0])
    expected = brentq(lambda x: x - np.sin(x) - y, 0.0, 2.0 * np.pi, xtol=1e-14)
    # hybr stops on the residual, brentq on the bracket width.
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-8)
