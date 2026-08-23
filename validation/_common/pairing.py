"""Hash and inspect inputs shared by matched simulation pairs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import h5py
import numpy as np


PAIR_KIND = "stepsic-periodic-matched-pair"
FIELD_HASH_ALGORITHM = "sha256-canonical-hdf5-dataset"


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def snapshot_diagnostics(path: str | Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        header = handle["Header"].attrs
        particles = handle["PartType1"]
        particle_count = len(particles["Coordinates"])
        masses = (
            particles["Masses"][:].astype(np.float64)
            if "Masses" in particles
            else np.full(particle_count, np.asarray(header["MassTable"])[1])
        )
        scale_factor = float(header.get("Time", np.nan))
        redshift = float(header.get("Redshift", 1.0 / scale_factor - 1.0))
    if (
        particle_count == 0 or not np.isfinite(scale_factor)
        or not np.isfinite(redshift) or not np.all(np.isfinite(masses))
        or np.any(masses <= 0.0)
    ):
        raise ValueError(f"{path} has invalid epoch or particle metadata")
    return {
        "final_scale_factor": scale_factor,
        "final_redshift": redshift,
        "particle_count": particle_count,
        "mass_min_1e11_msun_h": float(np.min(masses)),
        "mass_median_1e11_msun_h": float(np.median(masses)),
        "mass_max_1e11_msun_h": float(np.max(masses)),
    }


def canonical_dataset_sha256(path: str | Path, dataset_name: str) -> str:
    with h5py.File(path, "r") as handle:
        if dataset_name not in handle:
            raise ValueError(f"{path} contains no {dataset_name!r} dataset")
        values = np.asarray(handle[dataset_name][...])
    if values.dtype.hasobject:
        raise ValueError("object-valued HDF5 datasets cannot be hashed reproducibly")
    canonical = np.ascontiguousarray(values.astype(values.dtype.newbyteorder("<"), copy=False))
    digest = hashlib.sha256()
    digest.update(FIELD_HASH_ALGORITHM.encode("ascii"))
    digest.update(b"\0" + canonical.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()
