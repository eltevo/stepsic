#!/usr/bin/env python3
"""Create a reproducible random catalog from the complete spherical glass."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

from validation.sphere.evolved import generate_full_selection_randoms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glass", required=True)
    parser.add_argument("--factor", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with h5py.File(args.glass, "r") as handle:
        coordinates = handle["PartType1/Coordinates"][:].astype(np.float64)
        masses = handle["PartType1/Masses"][:].astype(np.float64)
        header = dict(handle["Header"].attrs)
    random_coordinates, random_masses = generate_full_selection_randoms(
        coordinates, masses, factor=args.factor, seed=args.seed,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-", dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as handle:
            header_group = handle.create_group("Header")
            for name, value in header.items():
                if name in ("NumPart_ThisFile", "NumPart_Total"):
                    counts = np.zeros(6, dtype=np.uint32)
                    counts[1] = len(random_masses)
                    header_group.attrs[name] = counts
                else:
                    header_group.attrs[name] = value
            particles = handle.create_group("PartType1")
            particles.create_dataset("Coordinates", data=random_coordinates)
            particles.create_dataset("Masses", data=random_masses)
            particles.create_dataset("Velocities", data=np.zeros_like(random_coordinates))
            particles.create_dataset(
                "ParticleIDs",
                data=np.arange(1, len(random_masses) + 1, dtype=np.uint64),
            )
            handle.flush()
        file_descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
