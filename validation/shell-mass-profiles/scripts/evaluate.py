#!/usr/bin/env python3
"""Check summed shell mass against the mass of each analytic volume."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from validation._common.evaluation import (
    archive_provenance,
    load_npz,
    malformed_result,
    metadata_parameters,
    write_result,
)
from validation._common.result import Metric, ValidationResult, upper_bound_check


def evaluate(
    archive: str | Path,
    *,
    figure: str | Path,
) -> ValidationResult:
    try:
        data = load_npz(archive)
        radius = float(data["meta_r3d_mpc_h"])
        length = float(data["meta_cylinder_length_mpc_h"])
        density = float(data["meta_rho_mean_internal"])
        d4d = float(data["meta_d4d_mpc_h"])
        shell_count = int(data["meta_particles_per_shell"])
        bin_count = int(data["meta_radial_bin_count"])
        if min(radius, length, density, d4d) <= 0.0:
            raise ValueError("physical scales and density must be positive")
        expected = {
            "spherical": density * (4.0 * math.pi * radius**3 / 3.0),
            "cylindrical": density * (math.pi * radius**2 * length),
        }
        domain_errors: dict[str, float] = {}
        closure_errors: dict[str, float] = {}
        boundary_normalized_errors: dict[str, float] = {}
        rounding_limit = (
            128.0 * np.finfo(np.float64).eps * max(bin_count, 1)
        )
        for geometry in ("spherical", "cylindrical"):
            for mode in ("omega", "volume"):
                prefix = f"{geometry}_{mode}"
                masses = np.asarray(
                    data[f"{prefix}_mass_internal"], dtype=np.float64
                )
                edges = np.asarray(
                    data[f"{prefix}_edges_mpc_h"], dtype=np.float64
                )
                if masses.shape != (bin_count,) or np.any(masses <= 0.0):
                    raise ValueError("mass profiles must be positive bin vectors")
                if (
                    edges.shape != (bin_count + 1,)
                    or np.any(np.diff(edges) <= 0.0)
                ):
                    raise ValueError(
                        "shell edges must be a strictly increasing vector"
                    )
                measured = float(np.sum(masses, dtype=np.float64) * shell_count)
                domain_errors[prefix] = abs(measured / expected[geometry] - 1.0)
                edge_radius = float(edges[-1])
                edge_volume = (
                    4.0 * math.pi * edge_radius**3 / 3.0
                    if geometry == "spherical"
                    else math.pi * edge_radius**2 * length
                )
                closure_errors[prefix] = abs(
                    measured / (density * edge_volume) - 1.0
                )
                boundary_error = abs(edge_radius / radius - 1.0)
                if geometry == "spherical" and mode == "volume":
                    boundary_limit = (
                        1.0e-6
                        * d4d
                        / (4.0 * radius)
                        * (1.0 + (radius / d4d) ** 2)
                    )
                else:
                    boundary_limit = rounding_limit
                boundary_normalized_errors[prefix] = (
                    boundary_error / boundary_limit
                )
        maximum_domain_error = max(domain_errors.values())
        maximum_closure_error = max(closure_errors.values())
        maximum_boundary_normalized_error = max(
            boundary_normalized_errors.values()
        )
        return ValidationResult(
            campaign="shell-mass-profiles",
            parameters=metadata_parameters(data),
            provenance=archive_provenance(archive),
            metrics={
                "maximum_configured_domain_mass_relative_error": Metric(
                    maximum_domain_error, "dimensionless"
                ),
                "maximum_shell_mass_closure_relative_error": Metric(
                    maximum_closure_error, "dimensionless"
                ),
                "maximum_boundary_normalized_error": Metric(
                    maximum_boundary_normalized_error, "dimensionless"
                ),
                **{
                    f"{name}_relative_error": Metric(value, "dimensionless")
                    for name, value in domain_errors.items()
                },
            },
            checks=[
                upper_bound_check(
                    name="shell mass agrees with analytic volume",
                    observed=maximum_closure_error,
                    limit=rounding_limit,
                    unit="dimensionless",
                    rationale=(
                        "Summed particle mass equals mean density times the "
                        "volume at the measured outer edge; the limit covers "
                        "float64 summation."
                    ),
                    source="Euclidean sphere and cylinder volume identities",
                ),
                upper_bound_check(
                    name="configured outer boundary",
                    observed=maximum_boundary_normalized_error,
                    limit=1.0,
                    unit="normalized error",
                    rationale=(
                        "Closed-form edges use float64 rounding; the spherical "
                        "constant-volume edge propagates the documented 1e-6 "
                        "doubled-angle root tolerance through r=D tan(x/4)."
                    ),
                    source="bin-edge formulas and solver stopping tolerance",
                ),
            ],
            numerical_archive=archive,
            figures=[figure],
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        return malformed_result(
            campaign="shell-mass-profiles",
            archive=archive,
            figures=[figure],
            error=error,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return write_result(
        evaluate(args.archive, figure=args.figure),
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
