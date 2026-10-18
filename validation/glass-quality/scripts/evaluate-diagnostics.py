#!/usr/bin/env python3
"""Check relaxed glasses against the corresponding randomized loads."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import norm

from validation._common.evaluation import archive_provenance, load_npz, malformed_result, write_result
from validation._common.result import Metric, ValidationResult, upper_bound_check


def _aperture_ratios(
    glass: dict[str, np.ndarray],
    twin: dict[str, np.ndarray],
) -> list[float]:
    for key in ("power_rmax", "power_boxsize"):
        glass_window = np.asarray(glass[key], dtype=np.float64)
        twin_window = np.asarray(twin[key], dtype=np.float64)
        if glass_window.shape != twin_window.shape or not np.allclose(
            glass_window,
            twin_window,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError("glass and twin power apertures are not matched")
    ratios: list[float] = []
    for index, (shot_glass, shot_twin, spacing) in enumerate(
        zip(
            glass["power_shot"],
            twin["power_shot"],
            glass["power_spacing"],
        )
    ):
        k_glass = np.asarray(glass["power_k"][index], dtype=np.float64)
        p_glass = np.asarray(glass["power_pk"][index], dtype=np.float64)
        k_twin = np.asarray(twin["power_k"][index], dtype=np.float64)
        p_twin = np.asarray(twin["power_pk"][index], dtype=np.float64)
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
            raise ValueError("power aperture has no finite positive support")
        particle_k = 2.0 * np.pi / float(spacing)
        x_glass = k_glass / particle_k
        x_twin = k_twin / particle_k
        keep = (
            (x_glass < 1.0)
            & (x_glass >= x_twin[0])
            & (x_glass <= x_twin[-1])
        )
        if not np.any(keep):
            raise ValueError(
                "power aperture has no common modes above the particle spacing in wavelength"
            )
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


def _tolerance_ratio(
    density: np.ndarray,
    expected: np.ndarray,
    retained: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    # Bonferroni normal quantile for a 1e-3 family false-alarm rate over the retained bins.
    z = float(norm.isf(1.0e-3/(2*np.count_nonzero(retained))))
    deviation = abs(density[retained]-1)
    return z, deviation, deviation/(.01 + z/np.sqrt(expected[retained]))


def evaluate(
    pairs: dict[str, tuple[str | Path, str | Path]],
    *,
    angular: dict[str, str | Path],
    rescaled: dict[int, str | Path],
    figures: list[str | Path],
    density_min_count: float,
    density_rmax_frac: float,
) -> ValidationResult:
    first_archive: str | Path = next(iter(pairs.values()))[0] if pairs else ""
    try:
        if not pairs:
            raise ValueError("at least one glass/twin pair is required")
        zone_ratios: dict[str, list[float]] = {}
        global_force_ratios: dict[str, float] = {}
        zoned_force_ratios: dict[str, list[float]] = {}
        angular_ratios: dict[str, list[float]] = {}
        density_accuracy: dict[str, dict[str, object]] = {}
        usable_zones: dict[str, list[bool]] = {}
        provenance: dict[str, object] = {}
        for geometry, (glass_path, twin_path) in pairs.items():
            glass = load_npz(glass_path)
            twin = load_npz(twin_path)
            zone_ratios[geometry] = _aperture_ratios(glass, twin)
            if not bool(glass["meta_has_force"]) or not bool(twin["meta_has_force"]):
                raise ValueError(f"{geometry} matched controls require force data")
            global_force_ratios[geometry] = float(glass["q_all"][0] / twin["q_all"][0])
            zone_force = np.asarray(glass["zone_q"], dtype=np.float64)[:, 0]
            twin_zone_force = np.asarray(twin["zone_q"], dtype=np.float64)[:, 0]
            if zone_force.shape != twin_zone_force.shape or np.any(twin_zone_force <= 0.0):
                raise ValueError(f"{geometry} force zones are not matched")
            zoned_force_ratios[geometry] = (zone_force / twin_zone_force).tolist()
            if geometry != 'cubical':
                if not np.array_equal(glass['zone_edges'], twin['zone_edges']):
                    raise ValueError('glass and control force zones must have identical physical edges')
                density = np.asarray(glass['prof_rho'], dtype=float)
                if density.size == 0 or not np.all(np.isfinite(density)):
                    raise ValueError('radial density profile must be non-empty and finite')
                # Expected count at the mean density, given the bin's mean particle mass.
                with np.errstate(divide='ignore', invalid='ignore'):
                    expected = np.asarray(glass['prof_count'], dtype=float)/density
                retained = expected >= density_min_count
                if not np.any(retained):
                    raise ValueError(f'{geometry} has no radial bin expecting at least {density_min_count:g} particles')
                _, _, tolerance_ratio = _tolerance_ratio(density, expected, retained)
                edges = glass['prof_edges']
                # The relaxed glass ends inside the configured radius, so the outermost bins are excluded.
                checked = retained & (edges[1:] <= density_rmax_frac*edges[-1])
                qualifying_bins = int(np.count_nonzero(checked))
                if qualifying_bins == 0:
                    raise ValueError(
                        f'{geometry} has no radial bin within {density_rmax_frac:g} of the radius '
                        f'expecting at least {density_min_count:g} particles')
                z, deviation, checked_ratio = _tolerance_ratio(density, expected, checked)
                density_accuracy[geometry] = {
                    'qualifying_bins': qualifying_bins,
                    'rmax_frac': density_rmax_frac,
                    'bonferroni_z': z,
                    'max_standardized_deviation': float(np.max(deviation*np.sqrt(expected[checked]))),
                    'max_tolerance_ratio': float(np.max(checked_ratio)),
                    'bins_outside_tolerance_mpc_h': [
                        [float(lo), float(hi)] for lo, hi, ratio in
                        zip(edges[:-1][checked], edges[1:][checked], checked_ratio) if ratio > 1],
                }
                angular_control = load_npz(angular[geometry])
                if not np.array_equal(glass['zone_edges'], angular_control['zone_edges']):
                    raise ValueError('glass and control force zones must have identical physical edges')
                angular_ratios[geometry] = _aperture_ratios(glass, angular_control)
                centers = (glass['prof_edges'][1:]+glass['prof_edges'][:-1])/2
                usable_zones[geometry] = []
                for i, (lo,hi) in enumerate(zip(glass['zone_edges'][:-1], glass['zone_edges'][1:])):
                    selected = ((centers >= lo) & (centers < hi))[retained]
                    usable_zones[geometry].append(bool(np.any(selected) and
                        np.all(tolerance_ratio[selected] <= 1) and zone_force[i] <= twin_zone_force[i]))
            provenance[geometry] = {
                "glass": archive_provenance(glass_path),
                "poisson_twin": archive_provenance(twin_path),
            }
            if geometry != 'cubical':
                provenance[geometry]["angular_control"] = archive_provenance(angular[geometry])
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
                rationale="Relaxed glasses suppress power at wavelengths larger than the local particle spacing.",
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
        if angular_ratios:
            checks.append(upper_bound_check(
                name="sub-Poisson power beyond radial ordering",
                observed=max(max(values) for values in angular_ratios.values()), limit=1.0,
                unit="glass/angular-control normalized power",
                rationale="A relaxed glass suppresses power at wavelengths larger than the local particle spacing beyond what its radial ordering alone provides.",
                source="comparison with the corresponding load randomized in angle at fixed radii",
            ))
        if density_accuracy:
            checks.append(upper_bound_check(
                name='radial density consistent with intended profile',
                observed=max(entry['max_tolerance_ratio'] for entry in density_accuracy.values()), limit=1.0,
                unit='|density/mean - 1| / (0.01 + z/sqrt(N_expected))',
                rationale=(
                    f'Every radial bin expecting at least {density_min_count:g} particles and ending within {density_rmax_frac:g} of the radius must match the intended density within '
                    'Poisson noise at a Bonferroni-corrected 1e-3 family false-alarm rate; a relaxed glass is sub-Poisson, '
                    'so the Poisson bound is conservative, and the 0.01 term allows a 1% systematic. '
                    'The outer bins are excluded because the relaxed glass ends inside the configured radius.'),
                source='physical radial bins relative to the intended mean density',
            ))
        return ValidationResult(
            campaign="glass-quality",
            parameters={"geometries": list(pairs), "rescale_aspects": sorted(rescaled)},
            provenance=provenance,
            metrics={
                "maximum_resolved_power_ratio": Metric(power_observed, "glass/twin normalized power"),
                "maximum_angular_control_power_ratio": Metric(angular_ratios, "glass/angular-control normalized power"),
                "radial_density_accuracy": Metric(density_accuracy, "standardized deviation in Poisson sigma; Bonferroni z; tolerance ratio; bin edges in Mpc/h"),
                "density_and_force_usable_zones": Metric(usable_zones, "boolean; qualifying density bins within the density-check tolerance and force below the twin"),
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
    parser.add_argument("--angular", action="append", nargs=2, default=[], metavar=("GEOMETRY", "ARCHIVE"))
    parser.add_argument("--rescale", action="append", nargs=2, default=[], metavar=("ASPECT", "ARCHIVE"))
    parser.add_argument("--density-min-count", type=float, required=True)
    parser.add_argument("--density-rmax-frac", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pairs: dict[str, tuple[str, str]] = {}
    for geometry, glass, twin in args.pair:
        pairs[geometry] = (glass, twin)
    rescaled = {int(aspect): path for aspect, path in args.rescale}
    return write_result(
        evaluate(pairs, angular=dict(args.angular), rescaled=rescaled, figures=args.figure,
                 density_min_count=args.density_min_count, density_rmax_frac=args.density_rmax_frac),
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
