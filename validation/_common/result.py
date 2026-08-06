"""Machine-readable scientific validation results."""

from __future__ import annotations

import json
import math
import os
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
    required: bool = True


@dataclass
class ValidationResult:
    campaign: str
    parameters: dict[str, Any]
    provenance: dict[str, Any]
    metrics: dict[str, Metric]
    checks: list[Check]
    numerical_archive: str | Path
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
                "required": check.required,
            }
            for check in self.checks
        ]
        failed = any(check.required and not check.passed for check in self.checks)
        return {
            "campaign": self.campaign,
            "status": "fail" if failed else "pass",
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
                "numerical_archive": str(self.numerical_archive),
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
    required: bool = True,
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
        required=required,
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
