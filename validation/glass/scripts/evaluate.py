#!/usr/bin/env python3
"""Evaluate analytic domain containment for generated glass loads."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import numpy as np

from validation._common.evaluation import archive_provenance, malformed_result, write_result
from validation._common.result import Metric, ValidationResult, upper_bound_check


def _coordinates(path: str | Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        values = handle["PartType1/Coordinates"][:]
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        raise ValueError(f"{path} Coordinates must have non-empty shape (N, 3)")
    return values


def _normalized_excess(
    coordinates: np.ndarray,
    geometry: str,
    *,
    box_size: float,
    radius: float,
    cylinder_length: float,
) -> tuple[float, float]:
    values = np.asarray(coordinates)
    scale = {
        "cubical-random": box_size,
        "cubical-grid": box_size,
        "spherical": radius,
        "cylindrical": max(radius, cylinder_length),
    }[geometry]
    if scale <= 0.0:
        raise ValueError("domain dimensions must be positive")
    if not np.all(np.isfinite(values)):
        return math.inf, 8.0 * np.finfo(values.dtype).eps
    if geometry.startswith("cubical"):
        excess = max(float(-np.min(values)), float(np.max(values) - box_size))
    elif geometry == "spherical":
        excess = float(np.max(np.linalg.norm(values, axis=1)) - radius)
    else:
        radial_excess = float(
            np.max(np.hypot(values[:, 0], values[:, 1])) - radius
        )
        axial_excess = max(
            float(-np.min(values[:, 2])),
            float(np.max(values[:, 2]) - cylinder_length),
        )
        excess = max(radial_excess, axial_excess)
    tolerance = 8.0 * np.finfo(values.dtype).eps
    return max(0.0, excess / scale), tolerance


def evaluate(
    snapshots: dict[str, str | Path],
    *,
    box_size: float,
    radius: float,
    cylinder_length: float,
    figures: list[str | Path],
) -> ValidationResult:
    first_archive: str | Path = next(iter(snapshots.values())) if snapshots else ""
    try:
        if not snapshots:
            raise ValueError("at least one glass snapshot is required")
        metrics: dict[str, Metric] = {}
        checks = []
        provenance = {}
        counts = {}
        for geometry, path in snapshots.items():
            coordinates = _coordinates(path)
            observed, limit = _normalized_excess(
                coordinates,
                geometry,
                box_size=box_size,
                radius=radius,
                cylinder_length=cylinder_length,
            )
            metrics[f"{geometry}_domain_excess"] = Metric(
                observed, "fraction of domain scale"
            )
            checks.append(
                upper_bound_check(
                    name=f"{geometry} domain containment",
                    observed=observed,
                    limit=limit,
                    unit="fraction of domain scale",
                    rationale=(
                        "Every generated position must lie inside the analytic "
                        "domain, allowing only coordinate-dtype rounding."
                    ),
                    source="cubical, spherical, and cylindrical domain definitions",
                )
            )
            provenance[geometry] = archive_provenance(path)
            counts[geometry] = int(len(coordinates))
        return ValidationResult(
            campaign="glass",
            parameters={
                "box_size_mpc_h": box_size,
                "radius_mpc_h": radius,
                "cylinder_length_mpc_h": cylinder_length,
                "particle_counts": counts,
            },
            provenance=provenance,
            metrics=metrics,
            checks=checks,
            numerical_archive=Path(first_archive).parent,
            figures=figures,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        return malformed_result(
            campaign="glass",
            archive=first_archive,
            figures=figures,
            error=error,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cubic-random")
    parser.add_argument("--cubic-grid")
    parser.add_argument("--spherical")
    parser.add_argument("--cylindrical")
    parser.add_argument("--box-size", type=float, required=True)
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--cylinder-length", type=float, required=True)
    parser.add_argument("--figure", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    snapshots = {
        name: path
        for name, path in {
            "cubical-random": args.cubic_random,
            "cubical-grid": args.cubic_grid,
            "spherical": args.spherical,
            "cylindrical": args.cylindrical,
        }.items()
        if path
    }
    return write_result(
        evaluate(
            snapshots,
            box_size=args.box_size,
            radius=args.radius,
            cylinder_length=args.cylinder_length,
            figures=args.figure,
        ),
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
