#!/usr/bin/env python3
"""Check relaxed glasses against the corresponding randomized loads."""

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
    rescaled: dict[int, str | Path],
    figures: list[str | Path],
) -> ValidationResult:
    first_archive: str | Path = next(iter(pairs.values()))[0] if pairs else ""
    try:
        if not pairs:
            raise ValueError("at least one glass/twin pair is required")
        zone_ratios: dict[str, list[float]] = {}
        global_force_ratios: dict[str, float] = {}
        zoned_force_ratios: dict[str, list[float]] = {}
        provenance: dict[str, object] = {}
        for geometry, (glass_path, twin_path) in pairs.items():
            glass = load_npz(glass_path)
            twin = load_npz(twin_path)
            zone_ratios[geometry] = _zone_ratios(glass, twin)
            if not bool(glass["meta_has_force"]) or not bool(twin["meta_has_force"]):
                raise ValueError(f"{geometry} matched controls require force data")
            global_force_ratios[geometry] = float(glass["q_all"][0] / twin["q_all"][0])
            zone_force = np.asarray(glass["zone_q"], dtype=np.float64)[:, 0]
            twin_zone_force = np.asarray(twin["zone_q"], dtype=np.float64)[:, 0]
            if zone_force.shape != twin_zone_force.shape or np.any(twin_zone_force <= 0.0):
                raise ValueError(f"{geometry} force zones are not matched")
            zoned_force_ratios[geometry] = (zone_force / twin_zone_force).tolist()
            provenance[geometry] = {
                "glass": archive_provenance(glass_path),
                "poisson_twin": archive_provenance(twin_path),
            }
        power_observed = max(max(values) for values in zone_ratios.values())
        global_force_observed = max(global_force_ratios.values())
        zoned_force_observed = max(max(values) for values in zoned_force_ratios.values())
        cubical_twin = load_npz(pairs["cubical"][1])
        twin_force = float(cubical_twin["q_all"][0])
        rescale_ratios: dict[int, float] = {}
        for aspect, path in rescaled.items():
            archive_data = load_npz(path)
            if not bool(archive_data["meta_has_force"]):
                raise ValueError(f"rescale aspect {aspect} requires force data")
            rescale_ratios[aspect] = float(archive_data["q_all"][0] / twin_force)
            provenance[f"rescale_{aspect}"] = archive_provenance(path)
        if not rescale_ratios:
            raise ValueError("at least one force-measured rescaling is required")
        rescale_observed = max(rescale_ratios.values())
        checks = [
            upper_bound_check(
                name="sub-Poisson glass power", observed=power_observed, limit=1.0,
                unit="glass/twin normalized power",
                rationale="Relaxed glasses suppress sub-particle-scale power.",
                source="comparison with the corresponding randomized load",
            ),
            upper_bound_check(
                name="global residual-force suppression", observed=global_force_observed,
                limit=1.0, unit="glass/twin RMS force",
                rationale="A relaxed glass has lower global residual force than its twin.",
                source="force measured from the corresponding randomized load",
            ),
            upper_bound_check(
                name="zoned residual-force suppression", observed=zoned_force_observed,
                limit=1.0, unit="glass/twin zoned RMS force",
                rationale="Every radial zone should have less residual force than randomized positions.",
                source="zoned force from the corresponding randomized load",
            ),
            upper_bound_check(
                name="qualified aspect-rescaling force", observed=rescale_observed,
                limit=1.0, unit="rescaled/Poisson RMS force",
                rationale="Compression may increase residual force, but it should remain below the force from randomized positions.",
                source="randomized cubical load with the same radial distribution",
            ),
        ]
        return ValidationResult(
            campaign="glass-quality",
            parameters={"geometries": list(pairs), "rescale_aspects": sorted(rescaled)},
            provenance=provenance,
            metrics={
                "maximum_sub_particle_power_ratio": Metric(power_observed, "glass/twin normalized power"),
                "maximum_global_force_ratio": Metric(global_force_observed, "glass/twin RMS force"),
                "maximum_zoned_force_ratio": Metric(zoned_force_observed, "glass/twin zoned RMS force"),
                "maximum_rescaled_force_ratio": Metric(rescale_observed, "rescaled/Poisson RMS force"),
            },
            checks=checks,
            numerical_archive=Path(first_archive).parent,
            figures=figures,
        )
    except (KeyError, OSError, StopIteration, TypeError, ValueError) as error:
        return malformed_result(
            campaign="glass-quality",
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
    parser.add_argument("--rescale", action="append", nargs=2, default=[], metavar=("ASPECT", "ARCHIVE"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pairs: dict[str, tuple[str, str]] = {}
    for geometry, glass, twin in args.pair:
        pairs[geometry] = (glass, twin)
    rescaled = {int(aspect): path for aspect, path in args.rescale}
    return write_result(
        evaluate(pairs, rescaled=rescaled, figures=args.figure),
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
