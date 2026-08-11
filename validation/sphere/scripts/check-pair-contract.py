#!/usr/bin/env python3
"""Check and publish the StePS/periodic matched-pair contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from validation.sphere.evolved import (
    atomic_write_json,
    snapshot_diagnostics,
    validate_pair_configuration,
    verify_field_hashes,
)


def _bool(value: str) -> bool:
    if value not in ("true", "false"):
        raise argparse.ArgumentTypeError("expected true or false")
    return value == "true"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("configuration", "fields", "evolution"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--radius-mpc-h", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--nmesh", type=int, required=True)
    parser.add_argument("--ngrid", type=int, required=True)
    parser.add_argument("--lpt-order", type=int, required=True)
    parser.add_argument("--initial-redshift", type=float, required=True)
    parser.add_argument("--final-redshift", type=float, required=True)
    parser.add_argument("--H0-km-s-mpc", type=float, required=True)
    parser.add_argument("--omega-b", type=float, required=True)
    parser.add_argument("--omega-m", type=float, required=True)
    parser.add_argument("--omega-lambda", type=float, required=True)
    parser.add_argument("--spectrum", required=True)
    parser.add_argument("--nonlinear", type=_bool, required=True)
    parser.add_argument("--halofit", required=True)
    parser.add_argument("--interpolation", required=True)
    parser.add_argument("--compensate", type=_bool, required=True)
    parser.add_argument("--sphere-mode", type=_bool, required=True)
    parser.add_argument("--paired", type=_bool, required=True)
    parser.add_argument("--nmesh-samples", type=int, required=True)
    parser.add_argument("--use-double", type=_bool, required=True)
    parser.add_argument("--stepsic-revision", required=True)
    parser.add_argument("--simulation-revision", required=True)
    parser.add_argument("--steps-white-noise")
    parser.add_argument("--steps-delta-k")
    parser.add_argument("--steps-snapshot")
    parser.add_argument("--steps-softening-mpc-h", type=float)
    parser.add_argument("--steps-force-mesh", type=int)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    expected = {
        "radius_mpc_h": args.radius_mpc_h,
        "final_redshift": args.final_redshift,
        "origin_mpc_h": [0.0, 0.0, 0.0],
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
            "nonlinear": args.nonlinear,
            "halofit": args.halofit,
            "interpolation": args.interpolation,
            "compensate": args.compensate,
            "sphere_mode": args.sphere_mode,
            "paired": args.paired,
            "nmesh_samples": args.nmesh_samples,
            "use_double": args.use_double,
        },
    }
    validate_pair_configuration(manifest, expected)
    for name, value in manifest["paths"].items():
        if not Path(value).is_file():
            raise FileNotFoundError(f"reference {name} path does not exist: {value}")

    contract_path = Path(args.output)
    contract = {
        "kind": "stepsic-periodic-matched-pair-contract",
        "reference_manifest": str(manifest_path.resolve()),
        "reference_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "configuration_match": True,
        "field_hashes_match": False,
        "configuration": expected,
        "paths": {
            "steps_origin_mpc_h": [0.0, 0.0, 0.0],
            "periodic_origin_mpc_h": manifest["extent"]["origin_mpc_h"],
            "reference": manifest["paths"],
        },
        "fields": {"reference": manifest["fields"], "steps": {}},
        "evolution": {
            "requested_final_redshift": args.final_redshift,
            "periodic": manifest["evolution"],
        },
        "revisions": {
            "reference": manifest["revisions"],
            "steps": {
                "stepsic": args.stepsic_revision,
                "simulation_code": args.simulation_revision,
            },
        },
    }
    if contract_path.is_file():
        with contract_path.open(encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing.get("reference_manifest_sha256") == contract["reference_manifest_sha256"]:
            contract["fields"] = existing.get("fields", contract["fields"])
            contract["field_hashes_match"] = existing.get("field_hashes_match", False)
            if "steps" in existing.get("evolution", {}):
                contract["evolution"]["steps"] = existing["evolution"]["steps"]

    if args.operation in ("fields", "evolution"):
        if not args.steps_white_noise or not args.steps_delta_k:
            raise ValueError("field and evolution operations require both StePS fields")
        steps_fields = verify_field_hashes(
            manifest,
            args.steps_white_noise,
            "ic_white_noise",
            args.steps_delta_k,
            "ic_delta_k",
        )
        contract["fields"]["steps"] = steps_fields
        contract["field_hashes_match"] = True
        contract["paths"]["steps_white_noise"] = str(
            Path(args.steps_white_noise).resolve()
        )
        contract["paths"]["steps_delta_k"] = str(Path(args.steps_delta_k).resolve())
    if args.operation == "evolution":
        if args.steps_snapshot is None:
            raise ValueError("evolution operation requires --steps-snapshot")
        diagnostics = snapshot_diagnostics(args.steps_snapshot)
        diagnostics.update({
            "softening_mpc_h": args.steps_softening_mpc_h,
            "force_mesh": args.steps_force_mesh,
        })
        contract["paths"]["steps_snapshot"] = str(
            Path(args.steps_snapshot).resolve()
        )
        contract["evolution"]["steps"] = diagnostics
    atomic_write_json(contract_path, contract)


if __name__ == "__main__":
    main()
