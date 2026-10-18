#!/usr/bin/env python3
"""Measure matched periodic 1LPT and 2LPT runs evolved with Gadget-4."""

from __future__ import annotations

import argparse

import h5py
import numpy as np

from stepsic.pk import measure_pk
from validation._common.evaluation import archive_provenance, atomic_savez


def load(path: str, final_redshift: float):
    with h5py.File(path, "r") as handle:
        header = handle['Header'].attrs
        redshift, scale = float(header['Redshift']), float(header['Time'])
        if (
            not np.isclose(redshift, final_redshift, rtol=0, atol=1e-8)
            or not np.isclose(scale, 1 / (1 + redshift), rtol=0, atol=1e-8)
        ):
            raise ValueError('snapshot has not reached the requested common epoch')
        ids = handle['PartType1/ParticleIDs'][:]
        positions = handle["PartType1/Coordinates"][:].astype(np.float64)
        masses = handle["PartType1/Masses"][:].astype(np.float64)
        box_size_mpc_h = float(handle["Header"].attrs["BoxSize"])
    if (
        not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(masses))
        or np.any(masses <= 0)
    ):
        raise ValueError('finite positions and positive masses are required')
    order = np.argsort(ids)
    if np.any(np.diff(ids[order]) == 0):
        raise ValueError('particle identities must be unique')
    return positions[order], masses[order], box_size_mpc_h, ids[order]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare power from matched periodic 1LPT and 2LPT runs.",
    )
    parser.add_argument("--one-lpt", required=True)
    parser.add_argument("--two-lpt", required=True)
    parser.add_argument("--final-redshift", type=float, required=True)
    parser.add_argument("--nmesh", type=int, required=True)
    parser.add_argument("--kmax-frac-ny", type=float, required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    if args.nmesh < 4 or args.nmesh % 2:
        raise ValueError("--nmesh must be even and at least 4")
    if not 0.0 < args.kmax_frac_ny <= 1.0:
        raise ValueError("--kmax-frac-ny must lie in (0, 1]")

    pos_1, mass_1, box_1, ids_1 = load(args.one_lpt, args.final_redshift)
    pos_2, mass_2, box_2, ids_2 = load(args.two_lpt, args.final_redshift)
    if not np.array_equal(ids_1, ids_2) or not np.array_equal(mass_1, mass_2):
        raise ValueError("1LPT and 2LPT particle identities and masses must match")
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

    if not np.all(np.isfinite(pk_1)) or not np.all(np.isfinite(pk_2)):
        raise ValueError('measured spectra must be finite')
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
        meta_one_lpt_sha256=np.array(archive_provenance(args.one_lpt)['archive_sha256']),
        meta_two_lpt_sha256=np.array(archive_provenance(args.two_lpt)['archive_sha256']),
        meta_box_size_mpc_h=np.float64(box_1),
        meta_nmesh=np.int64(args.nmesh),
        meta_kmax_frac_ny=np.float64(args.kmax_frac_ny),
    )


if __name__ == "__main__":
    main()
