"""File I/O and statistical checks shared by campaign evaluators."""

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


def average_realization_members(
    member_values: np.ndarray,
    *,
    nreal: int,
    paired: bool,
) -> np.ndarray:
    """Average counter-phased members before treating seeds as samples."""
    if isinstance(nreal, bool) or not isinstance(nreal, (int, np.integer)):
        raise TypeError("nreal must be an integer")
    if nreal < 1:
        raise ValueError("nreal must be positive")
    if not isinstance(paired, (bool, np.bool_)):
        raise TypeError("paired must be boolean")

    values = np.asarray(member_values, dtype=np.float64)
    if values.ndim == 0 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("member values must have a non-empty finite sample axis")
    members_per_realization = 2 if paired else 1
    expected_count = nreal * members_per_realization
    if values.shape[0] != expected_count:
        raise ValueError(
            f"expected {expected_count} member values for {nreal} "
            f"realization(s), got {values.shape[0]}"
        )
    grouped_shape = (nreal, members_per_realization, *values.shape[1:])
    return values.reshape(grouped_shape).mean(axis=1)


def ratio_of_means(
    numerator_samples: np.ndarray,
    denominator_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a ratio of means and its paired delta-method standard error."""
    numerator = np.asarray(numerator_samples, dtype=np.float64)
    denominator = np.asarray(denominator_samples, dtype=np.float64)
    if (
        numerator.shape != denominator.shape
        or numerator.ndim == 0
        or numerator.size == 0
    ):
        raise ValueError(
            "numerator and denominator must have the same non-empty shape"
        )
    if numerator.shape[0] < 2:
        raise ValueError("at least two independent realizations are required")
    if not np.all(np.isfinite(numerator)) or not np.all(
        np.isfinite(denominator)
    ):
        raise ValueError("ratio samples must be finite")
    if np.any(denominator <= 0.0):
        raise ValueError("denominator samples must be positive")

    numerator_mean = np.mean(numerator, axis=0, dtype=np.float64)
    denominator_mean = np.mean(denominator, axis=0, dtype=np.float64)
    ratio = numerator_mean / denominator_mean
    linearized_samples = (
        numerator - ratio * denominator
    ) / denominator_mean
    standard_error = np.std(
        linearized_samples, axis=0, ddof=1, dtype=np.float64
    ) / np.sqrt(numerator.shape[0])
    return ratio, standard_error


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
    del campaign, archive, figures
    raise RuntimeError(f"campaign output is malformed: {error}") from error


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
        raise ValueError("power ratios must be finite")
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
    policy = os.environ.get("VALIDATION_EVALUATION", "gate")
    if policy == "report":
        return 0
    if policy == "gate":
        return 0 if result.as_dict()["status"] == "pass" else 1
    raise RuntimeError(f"invalid evaluation policy: {policy}")
