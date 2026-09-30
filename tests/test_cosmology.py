from __future__ import annotations

import numpy as np
import pytest

from conftest import linear_growth
from stepsic.cosmology import ColossusCosmology, G2_omega


def test_G2_omega__approaches_einstein_de_sitter_value_early() -> None:
    """The second-order growth coefficient is evaluated at the IC epoch.

    D2/D1^2 = -3/7 exactly in an Einstein-de Sitter universe, and at
    a = 1/32 the matter fraction Omega_m(a) of a Lambda CDM model differs
    from 1 by 3e-5. Evaluating with today's Omega_m = 0.31 instead gives
    -3/7 * 0.31^(-1/143), 0.8 % too large.
    """
    np.testing.assert_allclose(G2_omega(1.0 / 32.0, 0.31, 0.69), -3.0 / 7.0, rtol=1e-4, atol=0.0)


@pytest.mark.camb
@pytest.mark.parametrize("z", [9.0, 19.0])
def test_Dzplus0__matches_heath_growth_integral(z) -> None:
    """The backscaling growth factor is the normalised matter + Lambda growing mode.

    The driver removes radiation (Tcmb0 ~ 0) so that the growth factor
    matches the Friedmann model used for H(a); the Heath integral is then
    exact. Colossus tabulates D to ~2e-4 at these redshifts.
    """
    cosmo = ColossusCosmology(H0=70.0, Om0=0.3, Ob0=0.05, Ol0=0.7, Tcmb0=1e-6)
    expected = linear_growth(1.0 / (1.0 + z), 0.3) / linear_growth(1.0, 0.3)
    np.testing.assert_allclose(cosmo.Dzplus0(z), expected, rtol=5e-4, atol=0.0)


def _tophat_sigma8(kh: np.ndarray, pk: np.ndarray) -> float:
    """RMS linear fluctuation in spheres of 8 Mpc/h by direct quadrature."""
    x = 8.0 * kh
    window = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
    integrand = kh**3 * pk * window**2 / (2.0 * np.pi**2)
    return float(np.sqrt(np.trapezoid(integrand, np.log(kh))))


@pytest.mark.camb
@pytest.mark.parametrize("component", [None, "delta_tot", "delta_cdm"])
def test_camb_sigma8_rescale__normalizes_returned_spectrum(
    camb_cosmo, component
) -> None:
    """The spectrum returned after rescaling As must have the requested sigma8."""
    target = 0.8105
    kwargs = {} if component is None else {"component": component}
    kh, pk, _ = camb_cosmo.get_spectrum(
        z=0.0,
        As=2.1064e-9,
        ns=0.96822,
        sigma8_init=target,
        kmin=1e-5,
        kmax=100.0,
        npoints=4096,
        **kwargs,
    )
    np.testing.assert_allclose(
        _tophat_sigma8(kh, pk[0]),
        target,
        # The rescale stops within rtol 1e-4, and CAMB's sigma_R integral
        # differs from this quadrature of the interpolated spectrum by ~1e-4.
        rtol=3e-4,
        atol=0.0,
    )
