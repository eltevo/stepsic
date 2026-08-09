from __future__ import annotations

from pathlib import Path

from astropy.cosmology import FlatLambdaCDM
import numpy as np
import pytest

from stepsic.cosmology import ColossusCosmology, F2_omega, F_omega, hubble_a


GOLDEN = Path(__file__).parent / "goldens" / "camb_spectrum.npz"


def test_hubble_a__matches_astropy_and_analytic_limits() -> None:
    """Friedmann EdS/a=1 limits and Astropy FlatLambdaCDM are oracles."""
    scale = np.array([0.2, 0.5, 1.0])
    H0 = 70.0
    astropy_cosmo = FlatLambdaCDM(H0=H0, Om0=0.3, Tcmb0=0.0)
    expected = astropy_cosmo.H(1.0 / scale - 1.0).value
    np.testing.assert_allclose(
        hubble_a(scale, H0, 0.3, 0.7),
        expected,
        # Both libraries evaluate the same flat Friedmann equation in float64.
        rtol=2e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        hubble_a(scale, H0, 1.0, 0.0),
        H0 * scale ** (-1.5),
        # The EdS closed form differs only by sqrt/power evaluation order.
        rtol=2e-15,
        atol=0.0,
    )
    assert hubble_a(1.0, H0, 0.2, 0.5) == H0


@pytest.mark.camb
@pytest.mark.parametrize("scale", [0.25, 0.5, 1.0])
def test_F_omega__matches_colossus_log_growth_derivative(scale) -> None:
    """a centered numerical derivative of Colossus D+ is the oracle."""
    cosmo = ColossusCosmology(H0=70.0, Om0=0.3, Ob0=0.05, Ol0=0.7)
    step = 1e-4
    lower_a = scale * np.exp(-step)
    upper_a = scale * np.exp(step)
    lower = cosmo.Dzplus0(1.0 / lower_a - 1.0)
    upper = cosmo.Dzplus0(1.0 / upper_a - 1.0)
    numerical = (np.log(upper) - np.log(lower)) / (2.0 * step)
    np.testing.assert_allclose(
        F_omega(scale, 0.3, 0.7),
        numerical,
        # Bernardeau et al. (2002), eq. 101a is a 0.3%-level approximation.
        rtol=3e-3,
        atol=0.0,
    )


def test_F2_omega__has_exact_einstein_de_sitter_limit() -> None:
    """Bernardeau et al. (2002), eq. 101b gives F2=2 when Omega_m=1."""
    np.testing.assert_array_equal(
        F2_omega(np.array([0.1, 0.5, 1.0]), 1.0, 0.0),
        np.array([2.0, 2.0, 2.0]),
    )


@pytest.mark.camb
def test_colossus_Dzplus0__is_normalized_and_monotone() -> None:
    """D+(0)/D+(0)=1 exactly and growth decreases with redshift."""
    cosmo = ColossusCosmology()
    growth = np.array([cosmo.Dzplus0(z) for z in (0.0, 1.0, 3.0)])
    assert growth[0] == 1.0
    assert np.all(np.diff(growth) < 0.0)
    assert np.all(growth > 0.0)


@pytest.mark.camb
def test_camb_sigma8_rescale__hits_requested_target(camb_cosmo) -> None:
    """rescaling As must return the requested sigma8 fixed point."""
    target = 0.8105
    camb_cosmo.get_spectrum(
        z=0.0,
        As=2.1064e-9,
        ns=0.96822,
        sigma8_init=target,
        kmin=0.01,
        kmax=3.0,
        npoints=64,
    )
    actual = camb_cosmo.get_sigma8(
        z=0.0,
        As=float(camb_cosmo.params.InitPower.As),
        ns=0.96822,
        kmax=3.0,
    )
    np.testing.assert_allclose(
        actual,
        target,
        # CAMBCosmology's documented rescale convergence criterion is 1e-4.
        rtol=1e-4,
        atol=0.0,
    )


@pytest.mark.camb
def test_camb_spectrum__matches_external_oracle_golden(camb_cosmo) -> None:
    """CAMB solver portability is pinned by the checked-in regeneration artifact."""
    with np.load(GOLDEN, allow_pickle=False) as golden:
        required = {
            "kh",
            "pk",
            "pk3",
            "git_commit",
            "camb_version",
            "numpy_version",
            "generated_date",
            "generator",
            "z",
            "As",
            "ns",
            "kmin",
            "kmax",
            "npoints",
            "component",
        }
        assert required <= set(golden.files)
        assert str(golden["git_commit"])
        assert str(golden["camb_version"])
        assert str(golden["numpy_version"])
        assert str(golden["generated_date"])
        assert str(golden["generator"]) == "tests/regen/regen_camb_spectrum.py"
        kh, pk, pk3 = camb_cosmo.get_spectrum(
            z=golden["z"].tolist(),
            As=float(golden["As"]),
            ns=float(golden["ns"]),
            sigma8_init=None,
            kmin=float(golden["kmin"]),
            kmax=float(golden["kmax"]),
            npoints=int(golden["npoints"]),
            component=str(golden["component"]),
        )
        np.testing.assert_allclose(
            kh,
            golden["kh"],
            # CAMB constructs the requested logarithmic grid to float64 precision.
            rtol=1e-10,
            atol=0.0,
        )
        np.testing.assert_allclose(
            pk,
            golden["pk"],
            # A 1e-6 relative bound catches meaningful CAMB solver/version drift.
            rtol=1e-6,
            atol=0.0,
        )
        np.testing.assert_allclose(
            pk3,
            golden["pk3"],
            # Derived Delta^2 inherits the same CAMB solver/version drift bound.
            rtol=1e-6,
            atol=0.0,
        )
