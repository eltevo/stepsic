#!/usr/bin/env python3
"""Measure matched periodic 1LPT and 2LPT runs evolved with Gadget-4."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile


import h5py
import numpy as np

from stepsic.pk import measure_pk


def load(path: str) -> tuple[np.ndarray, np.ndarray, float]:
    with h5py.File(path, "r") as handle:
        positions = handle["PartType1/Coordinates"][:].astype(np.float64)
        masses = handle["PartType1/Masses"][:].astype(np.float64)
        box_size_mpc_h = float(handle["Header"].attrs["BoxSize"])
    return positions, masses, box_size_mpc_h


def atomic_savez(path: str, **arrays: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-", dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare power from matched periodic 1LPT and 2LPT runs.",
    )
    parser.add_argument("--one-lpt", required=True)
    parser.add_argument("--two-lpt", required=True)
    parser.add_argument("--nmesh", type=int, required=True)
    parser.add_argument("--kmax-frac-ny", type=float, required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    if args.nmesh < 4:
        raise ValueError("--nmesh must be at least 4")
    if not 0.0 < args.kmax_frac_ny <= 1.0:
        raise ValueError("--kmax-frac-ny must lie in (0, 1]")

    pos_1, mass_1, box_1 = load(args.one_lpt)
    pos_2, mass_2, box_2 = load(args.two_lpt)
    if not np.isclose(box_1, box_2, rtol=0.0, atol=1e-12):
        raise ValueError("1LPT and 2LPT snapshots have different box sizes")

    measured = []
    for positions, masses in ((pos_1, mass_1), (pos_2, mass_2)):
        measured.append(measure_pk(
            positions,
            np.array([box_1] * 3),
            nmesh=args.nmesh,
            mass=masses,
            method="cic",
            deconvolve=True,
            interlace=True,
            subtract_shot=True,
        ))
    k_1, pk_1, modes_1 = measured[0]
    k_2, pk_2, modes_2 = measured[1]
    if not np.allclose(k_1, k_2, rtol=1e-13, atol=0.0):
        raise RuntimeError("1LPT and 2LPT estimators produced different k bins")
    if not np.array_equal(modes_1, modes_2):
        raise RuntimeError("1LPT and 2LPT estimators produced different mode counts")

    k_ny_h_mpc = np.pi * args.nmesh / box_1
    keep = (
        (k_1 > 0.0)
        & (k_1 <= args.kmax_frac_ny * k_ny_h_mpc)
        & np.isfinite(pk_1) & np.isfinite(pk_2)
        & (pk_1 > 0.0) & (pk_2 > 0.0)
    )
    if not np.any(keep):
        raise RuntimeError("periodic control has no usable power-spectrum bins")
    atomic_savez(
        args.output,
        k_h_mpc=k_1[keep],
        pk_1lpt_mpc_h3=pk_1[keep],
        pk_2lpt_mpc_h3=pk_2[keep],
        ratio_1lpt_2lpt=pk_1[keep] / pk_2[keep],
        nmodes=modes_1[keep],
        meta_box_size_mpc_h=np.float64(box_1),
        meta_nmesh=np.int64(args.nmesh),
        meta_kmax_frac_ny=np.float64(args.kmax_frac_ny),
    )


if __name__ == "__main__":
    main()
