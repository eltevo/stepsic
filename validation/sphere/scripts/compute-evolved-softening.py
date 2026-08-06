#!/usr/bin/env python3
"""Compute core mean-spacing / divisor softening for either StePS geometry."""

from __future__ import annotations

import argparse

import numpy as np


from validation._common.catalog import load_snapshot, region_mask  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument(
        "--geometry", choices=("spherical", "cylindrical"), required=True,
    )
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--length", type=float)
    parser.add_argument("--divisor", type=float, default=40.0)
    args = parser.parse_args()
    if not np.isfinite(args.divisor) or args.divisor <= 0.0:
        raise ValueError("--divisor must be finite and positive")

    snapshot = load_snapshot(args.snapshot)
    selection = region_mask(
        snapshot.coordinates, args.geometry, args.radius, args.length,
    )
    if not np.any(selection):
        raise ValueError("softening region contains no particles")
    selected_masses = snapshot.masses[selection]
    minimum_mass = float(np.min(selected_masses))
    constant_resolution = selection & (
        snapshot.masses <= 1.1 * minimum_mass
    )
    n_part = int(np.count_nonzero(constant_resolution))
    if n_part == 0:
        raise ValueError("softening region contains no minimum-mass particles")

    if args.geometry == "spherical":
        volume = 4.0 * np.pi * args.radius**3 / 3.0
    else:
        if args.length is None:
            raise ValueError("cylindrical geometry requires --length")
        volume = np.pi * args.radius**2 * args.length
    softening = np.cbrt(volume / n_part) / args.divisor
    print(f"{softening:.10g}")


if __name__ == "__main__":
    main()
