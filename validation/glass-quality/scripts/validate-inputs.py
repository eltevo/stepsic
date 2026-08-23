#!/usr/bin/env python3
"""Validate the three explicit pre-generated glass inputs."""

from __future__ import annotations

import argparse

import numpy as np

from validation._common.snapshots import load_snapshot, validate_glass_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cubical", required=True)
    parser.add_argument("--spherical", required=True)
    parser.add_argument("--cylindrical", required=True)
    parser.add_argument("--box-size", type=float, required=True)
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--length", type=float, required=True)
    args = parser.parse_args()
    cubic = load_snapshot(args.cubical)
    if len(cubic.particle_ids) == 0 or args.box_size <= 0.0:
        raise ValueError("cubical glass metadata is invalid")
    if np.any(cubic.coordinates < 0.0) or np.any(cubic.coordinates >= args.box_size):
        raise ValueError("cubical glass coordinates lie outside the periodic box")
    validate_glass_snapshot(args.spherical, geometry="spherical", radius_mpc_h=args.radius)
    validate_glass_snapshot(
        args.cylindrical,
        geometry="cylindrical",
        radius_mpc_h=args.radius,
        length_mpc_h=args.length,
    )


if __name__ == "__main__":
    main()
