from __future__ import annotations

import numpy as np

from stepsic.parameters import CosmoParameters


def test_derived_cosmology__camb_densities_add_up_to_omega_m(write_config) -> None:
    """CAMB receives CDM, baryon and neutrino densities that sum to Omega_m.

    Omega_m h^2 = omega_c + omega_b + omega_nu with omega_nu = sum(m_nu) / 93.14 eV
    (Lesgourgues & Pastor 2006); the config sets Omega_m = 0.3, Omega_b = 0.05, h = 0.7.
    """
    P = CosmoParameters(write_config()).get_parameters()
    np.testing.assert_allclose(P["OMBH2"], 0.05 * 0.7**2, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(
        P["OMCH2"] + P["OMBH2"] + P["MNU"] / 93.14, 0.3 * 0.7**2, rtol=1e-12, atol=0.0
    )
