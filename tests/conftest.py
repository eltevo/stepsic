from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stepsic.cosmology import CAMBCosmology
from stepsic.field import white_noise
from stepsic.interpolation import GridGeometry
from stepsic.parameters import CosmoParameters


SEEDS = (0, 42, 112358)
TESTS_DIR = Path(__file__).parent


@pytest.fixture(scope="session")
def wn16() -> np.ndarray:
    return white_noise((16, 16, 16), seed=42)


@pytest.fixture(scope="session")
def wn32() -> np.ndarray:
    return white_noise((32, 32, 32), seed=42)


@pytest.fixture
def grid_geom():
    def make(
        *,
        nvox=(8, 8, 8),
        boxsize=(8.0, 8.0, 8.0),
        vox_offset=0.5,
        origin=(0.0, 0.0, 0.0),
        periodic=True,
    ) -> GridGeometry:
        return GridGeometry(
            nvox=np.asarray(nvox, dtype=np.int64),
            boxsize=np.broadcast_to(boxsize, (3,)).astype(np.float64),
            vox_offset=vox_offset,
            origin=np.broadcast_to(origin, (3,)).astype(np.float64),
            periodic=periodic,
        )

    return make


@pytest.fixture(scope="session")
def tiny_params() -> dict:
    return CosmoParameters(TESTS_DIR / "data" / "tiny-config.toml").get_parameters()


@pytest.fixture
def params_factory():
    def make(**overrides) -> dict:
        params = {
            "GEOMETRY": "spherical",
            "BIN_MODE": "volume",
            "D_4D": 20.0,
            "NRBINS": 8,
            "R_3D": 30.0,
            "RHO_MEAN": 2.0,
            "NSHELL": 16,
            "LBOX": np.array([60.0, 60.0, 20.0]),
            "SEED": 42,
            "RCRIT": 0.0,
        }
        params.update(overrides)
        return params

    return make


@pytest.fixture(scope="session")
def camb_cosmo() -> CAMBCosmology:
    return CAMBCosmology()
