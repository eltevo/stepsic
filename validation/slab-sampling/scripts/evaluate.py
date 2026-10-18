#!/usr/bin/env python3
"""Combine the slab power-transfer and cube-cut results and check the covariance intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from validation._common.evaluation import (
    archive_provenance,
    load_npz,
    metadata_parameters,
    write_result,
)
from validation._common.result import (
    Check,
    Metric,
    ValidationResult,
    upper_bound_check,
)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or value.get("status") not in ("pass", "fail", "insufficient"):
        raise ValueError(f"{path} is not a valid evaluator result")
    if not isinstance(value.get("checks"), list):
        raise ValueError(f"checks must be a JSON array in {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", type=Path, required=True)
    parser.add_argument("--covariance", required=True)
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--figure", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checks: list[Check] = []
    metrics: dict[str, Metric] = {}
    parameters: dict[str, object] = {}
    provenance: dict[str, object] = {}
    for path in args.result:
        value = _load(path)
        label = value["campaign"]
        parameters[label] = value.get("parameters", {})
        provenance[label] = value.get("provenance", {})
        for name, metric in value.get("metrics", {}).items():
            metrics[f"{label}.{name}"] = Metric(metric["value"], metric["unit"])
        for check in value["checks"]:
            checks.append(Check(**check))

    covariance = load_npz(args.covariance)
    parameters["slab-sampling-covariance"] = metadata_parameters(covariance)
    provenance["slab-sampling-covariance"] = archive_provenance(args.covariance)
    low, mean, high = (
        np.concatenate([
            covariance[f"error_{bound}_{mesh}"]
            for mesh in covariance["meta_meshes"]
        ])
        for bound in ("low", "mean", "high")
    )
    metrics["slab-sampling-covariance.max_mean_variance_error"] = Metric(
        float(np.max(np.abs(mean))), "fraction"
    )
    checks.append(
        upper_bound_check(
            name="covariance intervals contain the mode-sum prediction",
            observed=int(np.count_nonzero((low > 0.0) | (high < 0.0))),
            limit=0,
            unit="intervals",
            rationale=(
                "Each filtered displacement variance, in the slab and in the cut from the cube, should equal the sum over the Fourier modes of its box. The simultaneous 95% Student intervals over seeds cover every mesh, aspect ratio, selection and component; none may exclude the prediction."
            ),
            source="linear displacement variance summed over discrete Fourier modes",
        )
    )
    return write_result(
        ValidationResult(
            campaign="slab-sampling",
            parameters=parameters,
            provenance=provenance,
            metrics=metrics,
            checks=checks,
            numerical_archive=args.data,
            figures=args.figure,
        ),
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
