#!/usr/bin/env python3
"""Record the periodic run used by the spherical-evolution comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from validation._common.pairing import (
    FIELD_HASH_ALGORITHM,
    PAIR_KIND,
    atomic_write_json,
    canonical_dataset_sha256,
    snapshot_diagnostics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--ic", required=True)
    parser.add_argument("--load", required=True)
    parser.add_argument("--white-noise", required=True)
    parser.add_argument("--delta-k", required=True)
    parser.add_argument("--box-size-mpc-h", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--nmesh", type=int, required=True)
    parser.add_argument("--ngrid", type=int, required=True)
    parser.add_argument("--pm-grid", type=int, required=True)
    parser.add_argument("--lpt-order", type=int, required=True)
    parser.add_argument("--initial-redshift", type=float, required=True)
    parser.add_argument("--final-redshift", type=float, required=True)
    parser.add_argument("--H0-km-s-mpc", type=float, required=True)
    parser.add_argument("--omega-b", type=float, required=True)
    parser.add_argument("--omega-m", type=float, required=True)
    parser.add_argument("--omega-lambda", type=float, required=True)
    parser.add_argument("--spectrum", required=True)
    parser.add_argument("--nonlinear", choices=("true", "false"), required=True)
    parser.add_argument("--halofit", required=True)
    parser.add_argument("--interpolation", required=True)
    parser.add_argument("--compensate", choices=("true", "false"), required=True)
    parser.add_argument("--sphere-mode", choices=("true", "false"), required=True)
    parser.add_argument("--paired", choices=("true", "false"), required=True)
    parser.add_argument("--nmesh-samples", type=int, required=True)
    parser.add_argument("--use-double", choices=("true", "false"), required=True)
    parser.add_argument("--softening-mpc-h", type=float, required=True)
    parser.add_argument("--stepsic-revision", required=True)
    parser.add_argument("--simulation-revision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = {
        name: str(Path(value).resolve())
        for name, value in {
            "snapshot": args.snapshot,
            "ic": args.ic,
            "load": args.load,
            "white_noise": args.white_noise,
            "delta_k": args.delta_k,
        }.items()
    }
    for name, path in paths.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"{name}: {path}")
    evolution = snapshot_diagnostics(paths["snapshot"])
    evolution.update({
        "requested_final_redshift": args.final_redshift,
        "particle_mass_1e11_msun_h": evolution["mass_median_1e11_msun_h"],
        "softening_mpc_h": args.softening_mpc_h,
        "pm_grid": args.pm_grid,
        "positions_64bit": True,
        "double_precision": True,
        "output_double_precision": True,
    })
    half_box_mpc_h = 0.5 * args.box_size_mpc_h
    atomic_write_json(args.output, {
        "kind": PAIR_KIND,
        "paths": paths,
        "extent": {
            "box_size_mpc_h": args.box_size_mpc_h,
            "origin_mpc_h": [half_box_mpc_h] * 3,
        },
        "cosmology": {
            "H0_km_s_mpc": args.H0_km_s_mpc,
            "omega_b": args.omega_b,
            "omega_m": args.omega_m,
            "omega_lambda": args.omega_lambda,
        },
        "initial_conditions": {
            "seed": args.seed,
            "nmesh": args.nmesh,
            "ngrid": args.ngrid,
            "lpt_order": args.lpt_order,
            "initial_redshift": args.initial_redshift,
            "spectrum": args.spectrum,
            "nonlinear": args.nonlinear == "true",
            "halofit": args.halofit,
            "interpolation": args.interpolation,
            "compensate": args.compensate == "true",
            "sphere_mode": args.sphere_mode == "true",
            "paired": args.paired == "true",
            "nmesh_samples": args.nmesh_samples,
            "use_double": args.use_double == "true",
        },
        "fields": {
            "hash_algorithm": FIELD_HASH_ALGORITHM,
            "ic_white_noise": canonical_dataset_sha256(
                paths["white_noise"], "ic_white_noise",
            ),
            "ic_delta_k": canonical_dataset_sha256(
                paths["delta_k"], "ic_delta_k",
            ),
        },
        "evolution": evolution,
        "revisions": {
            "stepsic": args.stepsic_revision,
            "simulation_code": args.simulation_revision,
        },
    })


if __name__ == "__main__":
    main()
