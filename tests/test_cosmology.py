from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from conftest import linear_growth
from stepsic.cosmology import ColossusCosmology, F2_omega, F_omega, G2_omega


def test_G2_omega__approaches_einstein_de_sitter_value_early() -> None:
    """The second-order growth coefficient is evaluated at the IC epoch.

    D2/D1^2 = -3/7 exactly in an Einstein-de Sitter universe, and at
    a = 1/32 the matter fraction Omega_m(a) of a Lambda CDM model differs
    from 1 by 3e-5. Evaluating with today's Omega_m = 0.31 instead gives
    -3/7 * 0.31^(-1/143), 0.8 % too large.
    """
    np.testing.assert_allclose(G2_omega(1.0 / 32.0, 0.31, 0.69), -3.0 / 7.0, rtol=1e-4, atol=0.0)


def _integrated_growth(a: float, omega_m: float) -> tuple[float, float, float]:
    """D2/D1^2, dlnD1/dlna and dlnD2/dlna from the growth equations of flat matter + Lambda.

    In x = ln a the two orders obey D1'' + c D1' = s D1 and
    D2'' + c D2' = s (D2 - D1^2), with s = 3/2 Omega_m(a) and c = 2 - s.
    The integration starts deep in the matter era on the growing mode,
    D1 = a and D2 = -3/7 a^2.
    """
    def rhs(x, state):
        source = 1.5 * omega_m / (omega_m + (1.0 - omega_m) * np.exp(3.0 * x))
        d1, d1p, d2, d2p = state
        return [
            d1p, source * d1 - (2.0 - source) * d1p,
            d2p, source * (d2 - d1 * d1) - (2.0 - source) * d2p,
        ]

    a0 = 1e-5
    solution = solve_ivp(
        rhs, (np.log(a0), np.log(a)), [a0, a0, -3.0 * a0**2 / 7.0, -6.0 * a0**2 / 7.0],
        method="DOP853", rtol=1e-11, atol=1e-18,
    )
    d1, d1p, d2, d2p = solution.y[:, -1]
    return d2 / d1**2, d1p / d1, d2p / d2


@pytest.mark.parametrize("z", [31.0, 10.0, 1.0, 0.0])
def test_growth_fits__match_integrated_growth_equations(z) -> None:
    """The 2LPT amplitude and both velocity prefactors come from fits in Omega_m(a).

    stepsic sets D2/D1^2 = -3/7 Omega^(-1/143), f1 = Omega^(5/9) and
    f2 = 2 Omega^(6/11) with Omega = Omega_m(a); the oracle integrates the
    growth equations. The fits are exact for Omega = 1 and drift as Lambda
    grows. Expanding the equations around Omega = 1 gives
    f1 = 1 - 6/11 (1 - Omega), so the 5/9 exponent is off by
    (5/9 - 6/11)(1 - Omega) = (1 - Omega)/99, and f2 = 2 f1 + 3/143 (1 - Omega),
    so 2 Omega^(6/11) is off by (1 - Omega)/95. Both rates are held to
    0.011 (1 - Omega): 8e-7 at z = 31 and 0.8 % today. The D2/D1^2 fit has
    no first-order error; it is held to 5e-4 (1 - Omega)^2, which is 2e-4
    today, plus 1e-9 for the integration. Evaluating any fit with today's
    Omega_m at the IC epoch, or using the f1 exponent in f2, fails here.
    """
    a = 1.0 / (1.0 + z)
    omega_m = 0.31
    lambda_fraction = 1.0 - omega_m / (omega_m + (1.0 - omega_m) * a**3)
    g2, f1, f2 = _integrated_growth(a, omega_m)
    rate_rtol = 0.011 * lambda_fraction
    np.testing.assert_allclose(
        G2_omega(a, omega_m, 1.0 - omega_m), g2, rtol=5e-4 * lambda_fraction**2 + 1e-9, atol=0.0
    )
    np.testing.assert_allclose(F_omega(a, omega_m, 1.0 - omega_m), f1, rtol=rate_rtol, atol=0.0)
    np.testing.assert_allclose(F2_omega(a, omega_m, 1.0 - omega_m), f2, rtol=rate_rtol, atol=0.0)


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
