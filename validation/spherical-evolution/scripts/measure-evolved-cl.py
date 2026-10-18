#!/usr/bin/env python3
"""Measure HEALPix spectra in matching shells from both snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from validation._common.evaluation import atomic_savez
from validation._common.snapshots import load_snapshot
from evolved import (
    measure_angular_spectra,
    minimum_image_displacements,
    parse_shells,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--steps-snapshot", required=True)
    parser.add_argument("--shells", default="")
    parser.add_argument("--radius-mpc-h", type=float, required=True)
    parser.add_argument("--rcrit-mpc-h", type=float, required=True)
    parser.add_argument("--nside", type=int, required=True)
    parser.add_argument("--lmax", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.contract).open(encoding="utf-8") as stream:
        contract = json.load(stream)
    if not contract.get("field_hashes_match"):
        raise ValueError("the two initial density fields do not have matching hashes")
    reference = contract["paths"]["reference"]
    box_size_mpc_h = 2.0 * args.radius_mpc_h
    periodic_origin_mpc_h = np.asarray(
        contract["paths"]["periodic_origin_mpc_h"], dtype=np.float64,
    )
    shells = parse_shells(
        args.shells,
        radius_mpc_h=args.radius_mpc_h,
        rcrit_mpc_h=args.rcrit_mpc_h,
    )
    steps_snapshot = load_snapshot(args.steps_snapshot)
    periodic_snapshot = load_snapshot(reference["snapshot"])
    steps = measure_angular_spectra(
        steps_snapshot.coordinates,
        steps_snapshot.masses,
        shells,
        nside=args.nside,
        lmax=args.lmax,
        omega_m=contract["configuration"]["cosmology"]["omega_m"],
    )
    periodic_displacements = minimum_image_displacements(
        periodic_snapshot.coordinates,
        origin_mpc_h=periodic_origin_mpc_h,
        box_size_mpc_h=box_size_mpc_h,
    )
    periodic = measure_angular_spectra(
        periodic_displacements,
        periodic_snapshot.masses,
        shells,
        nside=args.nside,
        lmax=args.lmax,
        omega_m=contract["configuration"]["cosmology"]["omega_m"],
    )
    arrays = {
        "shell_bounds_mpc_h": shells,
        "ell": steps["ell"],
        "cl_steps": steps["cl"],
        "cl_periodic": periodic["cl"],
        "cl_signed_difference": steps["cl"] - periodic["cl"],
        "meta_nside": np.int64(args.nside),
        "meta_lmax": np.int64(args.lmax),
        "meta_steps_origin_mpc_h": np.zeros(3),
        "meta_periodic_origin_mpc_h": periodic_origin_mpc_h,
    }
    for prefix, values in (("steps", steps), ("periodic", periodic)):
        for name in (
            "particle_count",
            "mass_sum_1e11_msun_h",
            "mass_mean_1e11_msun_h",
            "mass_min_1e11_msun_h",
            "mass_max_1e11_msun_h",
            "pixel_sector_volume_mpc_h3",
        ):
            arrays[f"{prefix}_{name}"] = values[name]
    atomic_savez(args.output, **arrays)


if __name__ == "__main__":
    main()
