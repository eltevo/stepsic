#!/usr/bin/env python3
"""Evaluate glass suppression against its matched Poisson twin."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from validation._common.evaluation import archive_provenance, load_npz, malformed_result, write_result
from validation._common.result import Metric, ValidationResult, upper_bound_check


def _zone_ratios(
    glass: dict[str, np.ndarray],
    twin: dict[str, np.ndarray],
) -> list[float]:
    ratios: list[float] = []
    for index, (shot_glass, shot_twin, spacing) in enumerate(
        zip(glass["zone_shot"], twin["zone_shot"], glass["zone_dbar"])
    ):
        k_glass = np.asarray(glass["zone_k"][index], dtype=np.float64)
        p_glass = np.asarray(glass["zone_pk"][index], dtype=np.float64)
        k_twin = np.asarray(twin["zone_k"][index], dtype=np.float64)
        p_twin = np.asarray(twin["zone_pk"][index], dtype=np.float64)
        valid_glass = np.isfinite(k_glass) & np.isfinite(p_glass)
        valid_twin = np.isfinite(k_twin) & np.isfinite(p_twin)
        k_glass, p_glass = k_glass[valid_glass], p_glass[valid_glass]
        k_twin, p_twin = k_twin[valid_twin], p_twin[valid_twin]
        if (
            len(k_glass) == 0
            or len(k_twin) == 0
            or not np.isfinite(shot_glass)
            or not np.isfinite(shot_twin)
            or shot_glass <= 0.0
            or shot_twin <= 0.0
        ):
            raise ValueError("zone contains no finite positive power support")
        particle_k = 2.0 * np.pi / float(spacing)
        x_glass = k_glass / particle_k
        x_twin = k_twin / particle_k
        keep = (
            (x_glass < 1.0)
            & (x_glass >= x_twin[0])
            & (x_glass <= x_twin[-1])
        )
        if not np.any(keep):
            raise ValueError("zone has no common sub-particle-scale modes")
        twin_normalized = np.interp(
            x_glass[keep],
            x_twin,
            p_twin / float(shot_twin),
        )
        ratio = (p_glass[keep] / float(shot_glass)) / twin_normalized
        if np.any(ratio < 0.0) or not np.all(np.isfinite(ratio)):
            raise ValueError("normalized glass/twin power is invalid")
        ratios.append(float(np.median(ratio)))
    return ratios


def evaluate(
    pairs: dict[str, tuple[str | Path, str | Path]],
    *,
    figures: list[str | Path],
) -> ValidationResult:
    first_archive: str | Path = next(iter(pairs.values()))[0] if pairs else ""
    try:
        if not pairs:
            raise ValueError("at least one glass/twin pair is required")
        zone_ratios: dict[str, list[float]] = {}
        provenance: dict[str, object] = {}
        for geometry, (glass_path, twin_path) in pairs.items():
            glass = load_npz(glass_path)
            twin = load_npz(twin_path)
            zone_ratios[geometry] = _zone_ratios(glass, twin)
            provenance[geometry] = {
                "glass": archive_provenance(glass_path),
                "poisson_twin": archive_provenance(twin_path),
            }
        observed = max(max(values) for values in zone_ratios.values())
        check = upper_bound_check(
            name="sub-Poisson glass power",
            observed=observed,
            limit=1.0,
            unit="glass/twin normalized power",
            rationale=(
                "Below the inter-particle wavenumber, a relaxed glass must "
                "have no more power than its matched Poisson twin."
            ),
            source="matched Poisson-twin control",
        )
        return ValidationResult(
            campaign="glass-diagnostics",
            parameters={"geometries": list(pairs), "zone_ratios": zone_ratios},
            provenance=provenance,
            metrics={
                "maximum_sub_particle_power_ratio": Metric(
                    observed, "glass/twin normalized power"
                ),
            },
            checks=[check],
            numerical_archive=Path(first_archive).parent,
            figures=figures,
        )
    except (KeyError, OSError, StopIteration, TypeError, ValueError) as error:
        return malformed_result(
            campaign="glass-diagnostics",
            archive=first_archive,
            figures=figures,
            error=error,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        required=True,
        metavar=("GEOMETRY", "GLASS", "TWIN"),
    )
    parser.add_argument("--figure", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pairs: dict[str, tuple[str, str]] = {}
    for geometry, glass, twin in args.pair:
        pairs[geometry] = (glass, twin)
    return write_result(
        evaluate(pairs, figures=args.figure),
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
