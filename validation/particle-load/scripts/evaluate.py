#!/usr/bin/env python3
"""Evaluate particle counts and analytic geometry boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from validation._common.evaluation import archive_provenance, write_result
from validation._common.result import Metric, ValidationResult, upper_bound_check


def _coordinates(path: str | Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        group = handle["PartType1"] if "PartType1" in handle else handle
        values = group["Coordinates"][:]
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"{path}: Coordinates must have shape (N, 3)")
    return values


def evaluate(
    *,
    cubic_random: str | Path,
    cubic_grid: str | Path,
    spherical: str | Path,
    cylindrical: str | Path,
    box_size_mpc_h: float,
    grid_size: int,
    random_count: int,
    radius_mpc_h: float,
    cylinder_length_mpc_h: float,
    figures: list[str | Path],
) -> ValidationResult:
    paths = {
        "cubic_random": cubic_random,
        "cubic_grid": cubic_grid,
        "spherical": spherical,
        "cylindrical": cylindrical,
    }
    try:
        if min(box_size_mpc_h, radius_mpc_h, cylinder_length_mpc_h) <= 0:
            raise ValueError("geometry scales must be positive")
        if min(grid_size, random_count) <= 0:
            raise ValueError("particle counts must be positive")
        coordinates = {name: _coordinates(path) for name, path in paths.items()}
        if not all(np.all(np.isfinite(values)) for values in coordinates.values()):
            raise ValueError("particle coordinates must be finite")
        tolerance = (
            8.0
            * max(np.finfo(values.dtype).eps for values in coordinates.values())
            * max(box_size_mpc_h, radius_mpc_h, cylinder_length_mpc_h)
        )
        outside = 0
        for name in ("cubic_random", "cubic_grid"):
            values = coordinates[name]
            outside += int(
                np.count_nonzero(
                    np.any(
                        (values < -tolerance)
                        | (values >= box_size_mpc_h + tolerance),
                        axis=1,
                    )
                )
            )
        radius = np.linalg.norm(coordinates["spherical"], axis=1)
        outside += int(np.count_nonzero(radius > radius_mpc_h + tolerance))
        cylinder = coordinates["cylindrical"]
        radial = np.hypot(cylinder[:, 0], cylinder[:, 1])
        outside += int(
            np.count_nonzero(
                (radial > radius_mpc_h + tolerance)
                | (cylinder[:, 2] < -tolerance)
                | (cylinder[:, 2] >= cylinder_length_mpc_h + tolerance)
            )
        )
        count_error = max(
            abs(len(coordinates["cubic_grid"]) - grid_size**3),
            abs(len(coordinates["cubic_random"]) - random_count),
        )
        return ValidationResult(
            campaign="particle-load",
            parameters={
                "box_size_mpc_h": box_size_mpc_h,
                "grid_size": grid_size,
                "random_count": random_count,
                "radius_mpc_h": radius_mpc_h,
                "cylinder_length_mpc_h": cylinder_length_mpc_h,
            },
            provenance={
                "archives": {
                    name: archive_provenance(path)
                    for name, path in paths.items()
                }
            },
            metrics={
                "outside_particle_count": Metric(outside, "particles"),
                "cubical_particle_count_error": Metric(
                    count_error, "particles"
                ),
            },
            checks=[
                upper_bound_check(
                    name="analytic geometry containment",
                    observed=outside,
                    limit=0,
                    unit="particles",
                    rationale=(
                        "Coordinates must lie in the cube, sphere, or cylinder; "
                        "the membership test includes float32 round-trip error."
                    ),
                    source="analytic geometry definitions",
                ),
                upper_bound_check(
                    name="cubical particle counts",
                    observed=count_error,
                    limit=0,
                    unit="particles",
                    rationale="A grid has NGRID³ points and random mode has NPART.",
                    source="particle-load construction contract",
                ),
            ],
            numerical_archive=Path(cubic_random).parent.parent,
            figures=figures,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        from validation._common.evaluation import malformed_result

        return malformed_result(
            campaign="particle-load",
            archive=cubic_random,
            figures=figures,
            error=error,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cubic-random", required=True)
    parser.add_argument("--cubic-grid", required=True)
    parser.add_argument("--spherical", required=True)
    parser.add_argument("--cylindrical", required=True)
    parser.add_argument("--box-size", type=float, required=True)
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--random-count", type=int, required=True)
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--cylinder-length", type=float, required=True)
    parser.add_argument("--figure", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return write_result(
        evaluate(
            cubic_random=args.cubic_random,
            cubic_grid=args.cubic_grid,
            spherical=args.spherical,
            cylindrical=args.cylindrical,
            box_size_mpc_h=args.box_size,
            grid_size=args.grid_size,
            random_count=args.random_count,
            radius_mpc_h=args.radius,
            cylinder_length_mpc_h=args.cylinder_length,
            figures=args.figure,
        ),
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
