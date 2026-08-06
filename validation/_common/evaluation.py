"""Shared I/O and statistical derivations for explicit campaign evaluators."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np

from .result import Check, Metric, ValidationResult, upper_bound_check


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load a typed NumPy archive without permitting pickle payloads."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def atomic_savez(path: str | Path, **arrays: np.ndarray) -> None:
    """Write a NumPy archive atomically in its destination directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **arrays)
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


def archive_provenance(
    archive: str | Path,
    *,
    command: list[str] | None = None,
) -> dict[str, Any]:
    path = Path(archive)
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        archive_sha256: str | None = digest.hexdigest()
    else:
        archive_sha256 = None
    return {
        "archive_sha256": archive_sha256,
        "evaluator": command or list(sys.argv),
        "numpy_version": np.__version__,
    }


def metadata_parameters(
    data: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for name, value in data.items():
        if not name.startswith("meta_"):
            continue
        scalar = value.tolist()
        parameters[name.removeprefix("meta_")] = scalar
    return parameters


def malformed_result(
    *,
    campaign: str,
    archive: str | Path,
    figures: list[str | Path],
    error: Exception,
) -> ValidationResult:
    check = Check(
        name="archive schema",
        observed=None,
        comparison="is",
        limit="well-formed typed data",
        unit="not applicable",
        rationale="Scientific claims require every documented input array.",
        source="campaign archive contract",
        passed=False,
        required=True,
    )
    return ValidationResult(
        campaign=campaign,
        parameters={},
        provenance={
            **archive_provenance(archive),
            "error": f"{type(error).__name__}: {error}",
        },
        metrics={},
        checks=[check],
        numerical_archive=archive,
        figures=figures,
    )


def gaussian_power_check(
    *,
    name: str,
    ratios: np.ndarray,
    mode_counts: np.ndarray,
    sample_count: int,
    source: str,
) -> tuple[Metric, Check]:
    """Return a conservative chi-square check for Gaussian P(k) estimates.

    For two independent Gaussian-field power estimates, the variance of their
    ratio around unity is at most ``4 / (N_modes * N_samples)``. Matched-phase
    comparisons are correlated, so this independent-sample expression is a
    conservative upper variance. The Laurent--Massart bound gives the stated
    chi-square limit with tail probability at most 1e-6.
    """
    ratio = np.asarray(ratios, dtype=np.float64)
    modes = np.asarray(mode_counts, dtype=np.float64)
    if ratio.shape != modes.shape or ratio.size == 0:
        raise ValueError("power ratios and mode counts must be non-empty peers")
    if sample_count <= 0 or np.any(modes <= 0):
        raise ValueError("sample and Fourier-mode counts must be positive")
    if not np.all(np.isfinite(ratio)):
        statistic = math.nan
        dof = int(ratio.size)
    else:
        variance = 4.0 / (modes * sample_count)
        statistic = float(np.sum((ratio - 1.0) ** 2 / variance))
        dof = int(ratio.size)
    log_inverse_tail = math.log(1.0e6)
    limit = (
        dof
        + 2.0 * math.sqrt(dof * log_inverse_tail)
        + 2.0 * log_inverse_tail
    )
    return (
        Metric(statistic, "chi-square"),
        upper_bound_check(
            name=name,
            observed=statistic,
            limit=limit,
            unit="chi-square",
            rationale=(
                "Gaussian power variance 4/(N_modes N_samples), with the "
                "Laurent-Massart 1e-6 upper-tail bound."
            ),
            source=source,
        ),
    )


def write_result(result: ValidationResult, path: str | Path) -> int:
    result.write(path)
    if os.environ.get("VALIDATION_DIAGNOSTIC") == "1":
        return 0
    return 0 if result.as_dict()["status"] == "pass" else 1
