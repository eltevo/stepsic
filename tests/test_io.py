"""Tests for snapshot path handling and HDF5 persistence."""

import numpy as np
import pytest

from stepsic.data import CosmoData
from stepsic.io import CosmoIO, UnsupportedFormatError


def _snapshot() -> CosmoData:
    return CosmoData(
        pos=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        vel=np.array([[-1.0, -2.0, -3.0], [7.0, 8.0, 9.0]], dtype=np.float32),
        mass=np.array([2.0, 3.0], dtype=np.float32),
        id=np.array([11, 29], dtype=np.uint64),
    )


def test_hdf5_save_load_round_trip_is_bit_identical(tmp_path):
    """lossless float32 HDF5 persistence requires array identity."""
    data = _snapshot()
    path = tmp_path / "snapshot $(touch owned).hdf5"

    CosmoIO.save_snapshot(path, data, "hdf5", dtype=np.float32, BoxSize=12.5)
    particle_ids, positions, velocities, masses = CosmoIO.load_snapshot(path)

    np.testing.assert_array_equal(particle_ids, data.id)
    np.testing.assert_array_equal(positions, data.pos)
    np.testing.assert_array_equal(velocities, data.vel)
    np.testing.assert_array_equal(masses, data.mass)
    # pathlib/HDF5 receive the hostile-looking name as inert data, not shell input.
    assert not (tmp_path / "owned").exists()


def test_collect_files_discovers_documented_multipart_names(tmp_path):
    """expected members follow the documented .N.ext/ext.N grammar."""
    member_names = [
        "snap set.0.hdf5",
        "snap set.1.hdf5",
        "snap set.hdf5.2",
        "snap set.hdf5",
    ]
    for name in member_names:
        (tmp_path / name).touch()
    (tmp_path / "snap set.0.txt").touch()
    (tmp_path / "other.0.hdf5").touch()

    found = CosmoIO._collect_files(tmp_path / "snap set.0.hdf5")

    assert found == sorted(tmp_path / name for name in member_names)


def test_unknown_extension_is_rejected_before_io(tmp_path, monkeypatch):
    """unsupported formats fail before any path-triggered I/O side effect."""
    monkeypatch.setattr("stepsic.io.glio", None)
    path = tmp_path / "untrusted $(touch owned).unknown"

    with pytest.raises(UnsupportedFormatError, match="unsupported extension"):
        CosmoIO.load_snapshot(path)

    assert not path.exists()
    assert not (tmp_path / "owned").exists()


def test_get_box_size_reads_saved_hdf5_header(tmp_path):
    """a written scalar header value must be recovered exactly."""
    path = tmp_path / "box.hdf5"
    CosmoIO.save_snapshot(path, _snapshot(), "hdf5", BoxSize=12.5)

    assert CosmoIO.get_box_size(path) == 12.5
