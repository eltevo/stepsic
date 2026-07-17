"""Tests for deterministic output-directory naming."""

from copy import deepcopy

import pytest

from stepsic._util import _resolution_tag, _run_dirname, ensure_run_dir


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"TYPE": "grid", "NGRID": 4, "NMESH": 8}, "Ng4_Nm8"),
        ({"TYPE": "random", "NPART": 12, "NMESH": 8}, "Np12_Nm8"),
        ({"TYPE": "glass", "NMESH": 8}, "Nm8"),
        (
            {"TYPE": "glass", "NMESH": 0, "NMESHSAMPLES": 27},
            "Ng27",
        ),
        (
            {"TYPE": "shell", "NSHELL": 16, "NRBINS": 8, "NMESH": 8},
            "Nsh16_Nr8_Nm8",
        ),
    ],
)
def test_resolution_tag_matches_string_spec(params, expected):
    """T2: expected tags are literal cases from the naming specification."""
    assert _resolution_tag(params) == expected


def test_run_dirname_matches_string_spec(tiny_params):
    """T2: expected name is assembled independently from tiny-config values."""
    expected = "tiny_cubical_Lx12_Ly12_Lz12_Ng4_Nm8_z9_LPT1_cic"
    assert _run_dirname(tiny_params) == expected


def test_ensure_run_dir_is_scoped_and_idempotent(tmp_path, tiny_params):
    """T1: repeated creation returns the same directory beneath IC_DIR."""
    params = deepcopy(tiny_params)
    params["IC_DIR"] = tmp_path
    params["IC_PREFIX"] = "$(touch-owned)"

    first = ensure_run_dir(params)
    second = ensure_run_dir(params)

    assert first == second
    assert first.parent == tmp_path
    assert first.is_dir()
    # T1: the hostile-looking prefix is a literal path component, not shell input.
    assert not (tmp_path / "touch-owned").exists()
