#!/usr/bin/env python3
"""Combine the slab power-transfer and cube-cut results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validation._common.evaluation import write_result
from validation._common.result import Check, Metric, ValidationResult


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or value.get("status") not in ("pass", "fail"):
        raise ValueError(f"{path} is not a valid evaluator result")
    if not isinstance(value.get("checks"), list):
        raise ValueError(f"checks must be a JSON array in {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", type=Path, required=True)
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
