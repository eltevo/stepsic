from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_evaluator():
    path = (
        Path(__file__).parents[1]
        / "validation"
        / "grid-power-recovery"
        / "scripts"
        / "evaluate.py"
    )
    spec = importlib.util.spec_from_file_location("grid_power_evaluate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_archive(path: Path, k: np.ndarray, ratio: np.ndarray) -> None:
    np.savez(
        path,
        meta_nmesh_list=np.array([32]),
        meta_lbox=np.float64(32.0),
        meta_n_total=np.int64(1),
        ref_cic_32_z31_pk=np.ones_like(k),
        lpt2_cic_32_z31_k=k,
        lpt2_cic_32_z31_pk=ratio,
        lpt2_cic_32_z31_nmodes=np.full(k.shape, 1_000_000),
    )


def test_evaluate__checks_only_common_low_k_support(tmp_path: Path) -> None:
    """The Gaussian gate covers k <= 0.1 k_Ny, not the high-k stress plot."""
    archive = tmp_path / "panel.npz"
    # nmesh=32 and Lbox=32 give k_Ny=pi. The first bin is in the common
    # large-scale support; the second deliberately models strong grid transfer.
    _write_archive(
        archive,
        k=np.array([0.25, 2.0]),
        ratio=np.array([1.0, 2.0]),
    )

    result = _load_evaluator().evaluate([archive], figure=tmp_path / "grid.pdf")

    payload = result.as_dict()
    assert payload["status"] == "pass"
    assert payload["parameters"]["comparison_kmax_fraction_nyquist"] == 0.1
    assert payload["parameters"]["comparison_bin_count"] == 1


def test_evaluate__rejects_archive_without_low_k_support(tmp_path: Path) -> None:
    """A gate with no bins at or below 0.1 k_Ny is not a valid comparison."""
    archive = tmp_path / "panel.npz"
    _write_archive(
        archive,
        k=np.array([0.5, 2.0]),
        ratio=np.ones(2),
    )

    with pytest.raises(RuntimeError, match="no LPT power bins in common low-k"):
        _load_evaluator().evaluate([archive], figure=tmp_path / "grid.pdf")
