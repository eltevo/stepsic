"""Machine-readable scientific validation results."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Metric:
    value: Any
    unit: str


@dataclass(frozen=True)
class Check:
    name: str
    observed: Any
    comparison: str
    limit: Any
    unit: str
    rationale: str
    source: str
    passed: bool


@dataclass
class ValidationResult:
    campaign: str
    parameters: dict[str, Any]
    provenance: dict[str, Any]
    metrics: dict[str, Metric]
    checks: list[Check]
    numerical_archive: str | Path | list[str | Path]
    figures: list[str | Path]

    def as_dict(self) -> dict[str, Any]:
        checks = [
            {
                "name": check.name,
                "observed": _json_value(check.observed),
                "comparison": check.comparison,
                "limit": _json_value(check.limit),
                "unit": check.unit,
                "rationale": check.rationale,
                "source": check.source,
                "passed": check.passed,
            }
            for check in self.checks
        ]
        failed = any(not check.passed for check in self.checks)
        data_artifacts = (
            self.numerical_archive
            if isinstance(self.numerical_archive, list)
            else [self.numerical_archive]
        )
        return {
            "campaign": self.campaign,
            "status": "fail" if failed else "pass",
            "profile": {
                "size": os.environ.get("VALIDATION_PROFILE_SIZE", "unknown"),
                "sha256": os.environ.get("VALIDATION_PROFILE_SHA256", "unknown"),
                "resolved": os.environ.get("VALIDATION_RESOLVED_PROFILE", "unknown"),
            },
            "parameters": _json_value(self.parameters),
            "provenance": _json_value(self.provenance),
            "metrics": {
                name: {
                    "value": _json_value(metric.value),
                    "unit": metric.unit,
                }
                for name, metric in self.metrics.items()
            },
            "checks": checks,
            "artifacts": {
                "data": [str(path) for path in data_artifacts],
                "figures": [str(path) for path in self.figures],
            },
        }

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.as_dict()
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    payload,
                    stream,
                    allow_nan=False,
                    indent=2,
                    sort_keys=False,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def upper_bound_check(
    *,
    name: str,
    observed: float,
    limit: float,
    unit: str,
    rationale: str,
    source: str,
) -> Check:
    finite = math.isfinite(float(observed)) and math.isfinite(float(limit))
    return Check(
        name=name,
        observed=observed,
        comparison="<=",
        limit=limit,
        unit=unit,
        rationale=rationale,
        source=source,
        passed=finite and float(observed) <= float(limit),
    )


def _json_value(value: Any) -> Any:
    """Return ``value`` in strict JSON form, mapping non-finite reals to null."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    raise TypeError(
        f"{type(value).__name__} is not a JSON-compatible validation value"
    )


def summarize_results(
    size: str,
    evaluation: str,
    entries: list[tuple[str, int, Path]],
) -> int:
    """Print a compact summary of the completed campaign processes."""
    rows: list[tuple[str, str, str, str, str]] = []
    infrastructure_failed = False
    science_failed = False
    seen: set[str] = set()
    for campaign, run_status, path in entries:
        if not campaign or campaign in seen:
            raise ValueError("summary campaign names must be non-empty and unique")
        if not 0 <= run_status <= 255:
            raise ValueError("summary process status must lie in [0, 255]")
        seen.add(campaign)
        run_label = "pass" if run_status == 0 else f"error {run_status}"
        try:
            with path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
            if not isinstance(payload, dict):
                raise ValueError("result root must be an object")
            result_campaign = payload.get("campaign")
            if not isinstance(result_campaign, str) or not result_campaign:
                raise ValueError("result campaign must be a non-empty string")
            status = payload.get("status")
            if status not in ("pass", "fail"):
                raise ValueError("result status must be pass or fail")
            checks = payload.get("checks")
            if not isinstance(checks, list):
                raise ValueError("result checks must be an array")
            failed_checks = []
            for check in checks:
                if not isinstance(check, dict):
                    raise ValueError("result checks must contain objects")
                if check.get("passed") is False:
                    name = check.get("name")
                    if not isinstance(name, str) or not name:
                        raise ValueError("failed checks must have non-empty names")
                    failed_checks.append(name)
            science_label = status.upper()
            failed_label = "; ".join(failed_checks) or "-"
            science_failed = science_failed or status == "fail"
            if run_status != 0 and not (
                evaluation == "gate" and status == "fail"
            ):
                infrastructure_failed = True
        except (OSError, json.JSONDecodeError, ValueError) as error:
            science_label = "ERROR"
            failed_label = str(error)
            infrastructure_failed = True
        rows.append(
            (campaign, run_label, science_label, failed_label, str(path))
        )

    headers = ("campaign", "run", "science", "failed checks", "result")
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    print(f"Validation summary ({size})")
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    for row in rows:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))

    return int(infrastructure_failed or (science_failed and evaluation == "gate"))


def gate_result(path: Path) -> int:
    """Read a completed result and return its scientific verdict."""
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict) or payload.get("status") not in ("pass", "fail"):
            raise ValueError("result status must be pass or fail")
        checks = payload.get("checks")
        if not isinstance(checks, list) or any(
            not isinstance(check, dict) or not isinstance(check.get("passed"), bool)
            for check in checks
        ):
            raise ValueError("result checks must contain boolean verdicts")
        expected_status = "fail" if any(not check["passed"] for check in checks) else "pass"
        if payload["status"] != expected_status:
            raise ValueError("result status disagrees with its checks")
        return int(payload["status"] == "fail")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"ERROR: malformed result {path}: {error}", file=sys.stderr)
        return 2


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    summary = subparsers.add_parser("summary")
    summary.add_argument("--size", choices=("small", "medium", "large", "custom"), required=True)
    summary.add_argument("--evaluation", choices=("report", "gate"), required=True)
    summary.add_argument(
        "--entry",
        nargs=3,
        action="append",
        metavar=("CAMPAIGN", "PROCESS_STATUS", "RESULT_JSON"),
        required=True,
    )
    gate = subparsers.add_parser("gate")
    gate.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    if args.command == "gate":
        return gate_result(args.path)
    try:
        entries = [
            (campaign, int(run_status), Path(path))
            for campaign, run_status, path in args.entry
        ]
        return summarize_results(args.size, args.evaluation, entries)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_main())
