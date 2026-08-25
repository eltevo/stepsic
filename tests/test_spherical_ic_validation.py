from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _load_evaluator() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "validation"
        / "spherical-ic"
        / "scripts"
        / "evaluate.py"
    )
    spec = importlib.util.spec_from_file_location("spherical_ic_evaluate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_archive(
    path: Path,
    *,
    rms_dv: float,
    rms_ref: float = 1.0,
    var_geom: tuple[float, float, float] = (0.9, 0.9, 0.9),
) -> None:
    np.savez(
        path,
        rms_dv=np.float64(rms_dv),
        rms_ref=np.float64(rms_ref),
        var_geom=np.asarray(var_geom),
        var_ref=np.ones(3),
        meta_n_geom_core=np.int64(32_127),
        meta_n_ref_core=np.int64(137_525),
        meta_n_cells=np.int64(12_568),
    )


def test_evaluate__gates_matched_grid_residual_not_particle_variances(
    tmp_path: Path,
) -> None:
    evaluator = _load_evaluator()
    archive = tmp_path / "compare.npz"
    _write_archive(archive, rms_dv=0.14)

    result = evaluator.evaluate(archive, figure=tmp_path / "figure.pdf")
    payload = result.as_dict()

    assert payload["status"] == "pass"
    assert payload["metrics"]["velocity_rms_relative_residual"]["value"] == 0.14
    assert payload["metrics"]["component_variance_ratios"]["value"] == [
        0.9,
        0.9,
        0.9,
    ]
    assert payload["checks"][0]["name"] == "core matched velocity-field agreement"


def test_evaluate__rejects_large_matched_grid_residual(tmp_path: Path) -> None:
    evaluator = _load_evaluator()
    archive = tmp_path / "compare.npz"
    _write_archive(archive, rms_dv=0.5, var_geom=(1.0, 1.0, 1.0))

    result = evaluator.evaluate(archive, figure=tmp_path / "figure.pdf")

    assert result.as_dict()["status"] == "fail"


@pytest.mark.parametrize(
    ("rms_dv", "rms_ref"),
    [(np.nan, 1.0), (-0.1, 1.0), (0.1, 0.0)],
)
def test_evaluate__rejects_invalid_velocity_norms(
    tmp_path: Path,
    rms_dv: float,
    rms_ref: float,
) -> None:
    evaluator = _load_evaluator()
    archive = tmp_path / "compare.npz"
    _write_archive(archive, rms_dv=rms_dv, rms_ref=rms_ref)

    with pytest.raises(RuntimeError, match="campaign output is malformed"):
        evaluator.evaluate(archive, figure=tmp_path / "figure.pdf")
