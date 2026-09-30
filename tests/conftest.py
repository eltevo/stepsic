from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import toml
from scipy.integrate import quad

from stepsic.cosmology import CAMBCosmology


TINY_CONFIG = Path(__file__).parent / "data" / "tiny-config.toml"


def linear_growth(a: float, omega_m: float) -> float:
    """Growing mode D(a) = (5/2) Omega_m E(a) int_0^a da' / (a' E)^3, D -> a early.

    Exact for flat matter + Lambda without radiation (Heath 1977).
    """
    E = lambda x: np.sqrt(omega_m / x**3 + 1.0 - omega_m)
    integral = quad(lambda x: (x * E(x)) ** -3, 0.0, a, epsabs=0.0, epsrel=1e-12)[0]
    return 2.5 * omega_m * E(a) * integral


def growth_rate(a: float, omega_m: float) -> float:
    """f = dln D / dln a of `linear_growth`, differentiated analytically."""
    E2 = omega_m / a**3 + 1.0 - omega_m
    integral = linear_growth(a, omega_m) / (2.5 * omega_m * np.sqrt(E2))
    return -1.5 * omega_m / (a**3 * E2) + 1.0 / (a**2 * E2**1.5 * integral)


@pytest.fixture
def write_config(tmp_path):
    """Write the tiny config with overrides; each call gets its own IC_DIR."""
    counter = iter(range(1000))

    def write(**overrides) -> Path:
        index = next(counter)
        config = toml.load(TINY_CONFIG)
        config["IC_DIR"] = str(tmp_path / f"ics-{index}")
        config.update(overrides)
        path = tmp_path / f"config-{index}.toml"
        path.write_text(toml.dumps(config))
        return path

    return write


@pytest.fixture
def tiny_chunks(monkeypatch):
    """Shrink the memory-bounding chunk sizes so every chunked loop runs many times."""
    monkeypatch.setattr("stepsic.field._SPECTRAL_SLAB_BUDGET_BYTES", 1)
    monkeypatch.setattr(
        "stepsic.interpolation.FieldInterpolator._PARTICLES_PER_CHUNK", 7
    )
    monkeypatch.setattr(
        "stepsic.multiresolution._ACCUMULATION_PARTICLES_PER_CHUNK", 5
    )


@pytest.fixture(scope="session")
def camb_cosmo() -> CAMBCosmology:
    return CAMBCosmology()
