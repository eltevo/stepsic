from __future__ import annotations

from pathlib import Path

from astropy import constants as const
from astropy import units as u
import numpy as np
import pytest
import toml

from stepsic.parameters import (
    CosmoParameters,
    PType,
    Param,
    _validate_and_cast,
)


TINY_CONFIG = Path(__file__).parent / "data" / "tiny-config.toml"


def test_cosmo_parameters__derives_tiny_cosmology_by_independent_arithmetic(
    tiny_params,
) -> None:
    """T2: Friedmann-density definitions and config literals supply expectations."""
    h = 0.7  # H0=70 from tiny-config.toml, divided by 100 km/s/Mpc.
    unit_l = (1.0 * u.Mpc).to_value(u.cm)
    unit_m = (1e11 * u.Msun).to_value(u.g)
    unit_t_seconds = np.sqrt(unit_l**3 / (const.G.cgs.value * unit_m))
    unit_v = unit_l / unit_t_seconds / 1e5
    rho_crit = 3.0 * (100.0 / unit_v) ** 2 / (8.0 * np.pi)
    omega_nu = 0.06 / (93.14 * h**2)
    expected = {
        "H": h,
        "RHO_CRIT": rho_crit,
        "RHO_MEAN": 0.3 * rho_crit,
        "OMMH2": 0.3 * h**2,
        "OMBH2": 0.05 * h**2,
        "OMEGA_NU": omega_nu,
        "OMEGA_C": 0.3 - 0.05 - omega_nu,
        "OMEGA_K": 0.0,
        "SCALE": 0.1,
    }
    expected["OMCH2"] = expected["OMEGA_C"] * h**2
    for key, value in expected.items():
        np.testing.assert_allclose(
            tiny_params[key],
            value,
            # Astropy unit conversion and production constants are float64.
            rtol=2e-13,
            atol=2e-15,
        )
    assert tiny_params["DTYPE"] is np.float64


def _invalid_config(case: str) -> tuple[dict, str]:
    config = toml.load(TINY_CONFIG)
    if case == "shell_cubical":
        config.update(TYPE="shell", NSHELL=8, BIN_MODE="volume", NRBINS=4)
        return config, "not valid for cubical"
    if case == "grid_spherical":
        config["GEOMETRY"] = "spherical"
        return config, "only valid for cubical"
    if case == "random_spherical":
        config.update(TYPE="random", NPART=16, GEOMETRY="spherical")
        return config, "only valid for cubical"
    if case == "adaptive_grid":
        config["NMESH"] = 0
        return config, "requires TYPE='glass'"
    if case == "negative_redshift":
        config["REDSHIFT"] = -1.0
        return config, "must be >= 0"
    if case == "rcrit_at_edge":
        config.update(
            TYPE="shell",
            GEOMETRY="spherical",
            NSHELL=8,
            BIN_MODE="omega",
            NRBINS=4,
            RCRIT=config["R_3D"],
        )
        return config, "must be smaller than R_3D"
    raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "shell_cubical",
        "grid_spherical",
        "random_spherical",
        "adaptive_grid",
        "negative_redshift",
        "rcrit_at_edge",
    ],
)
def test_cosmo_parameters__each_error_constraint_rejects_invalid_config(
    case, tmp_path
) -> None:
    """T1: every declared error-level Constraint is exercised at its boundary."""
    config, message = _invalid_config(case)
    path = tmp_path / f"{case}.toml"
    path.write_text(toml.dumps(config))
    with pytest.raises(ValueError, match=message):
        CosmoParameters(path)


def test_validate_and_cast__casts_choices_arrays_and_h_scaled_values() -> None:
    """T1: Param descriptor semantics define casting, choices, and h scaling."""
    params = {"HINDEPENDENT": False, "H": 0.7, "METHOD": " CIC ", "L": 10}
    _validate_and_cast(
        Param("METHOD", ptype=PType.STRING, choices=("ngp", "cic", "tsc")),
        params,
    )
    _validate_and_cast(Param("L", ptype=PType.ARRAY, h_scaled=True), params)
    assert params["METHOD"] == "cic"
    np.testing.assert_array_equal(params["L"], (7.0, 7.0, 7.0))


@pytest.mark.parametrize(
    ("param", "value", "error"),
    [
        (Param("X", ptype=PType.STRING, choices=("cic",)), "   ", ValueError),
        (Param("X", ptype=PType.STRING), 3, TypeError),
        (Param("X", ptype=PType.BOOL), 1, TypeError),
        (Param("X", ptype=PType.ARRAY), [], TypeError),
        (Param("X", ptype=PType.ARRAY), [1.0], TypeError),
        (Param("X", ptype=PType.ARRAY), [1.0, 2.0, 3.0, 4.0], TypeError),
    ],
)
def test_validate_and_cast__rejects_boundary_type_and_shape_inputs(
    param, value, error
) -> None:
    """T1: empty/whitespace, singleton, and boundary-length inputs follow Param."""
    with pytest.raises(error):
        _validate_and_cast(param, {"X": value})


def test_load_default_cosmology__preserves_user_keys_and_fills_bundled_values() -> None:
    """T1/T2: setdefault preserves H0=99; Planck table 2.20 supplies Omega_m."""
    params = CosmoParameters.__new__(CosmoParameters)
    params.P = {"H0": 99.0}
    params._load_default_cosmology(cosmology="Planck2018EE+BAO+SN")
    assert params.P["H0"] == 99.0
    assert params.P["OMEGA_M"] == 0.3099
