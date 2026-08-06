#!/usr/bin/env python3
"""Evaluate structural validity of the evolved matched-pair evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from validation._common.evaluation import (
    archive_provenance,
    load_npz,
    malformed_result,
    metadata_parameters,
    write_result,
)
from validation._common.result import Check, Metric, ValidationResult


def _artifact(path: str | Path, name: str) -> Path:
    resolved = Path(path)
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValueError(f"required {name} artifact is missing or empty: {resolved}")
    return resolved


def _finite_array(
    data: dict[str, np.ndarray],
    name: str,
    *,
    ndim: int,
) -> np.ndarray:
    values = np.asarray(data[name])
    if values.ndim != ndim or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a nonempty finite {ndim}-D array")
    return values


def evaluate(
    archive: str | Path,
    *,
    contract: str | Path,
    cl_figure: str | Path,
    pk_figure: str | Path,
    cl_native: str | Path,
    steps_pk_native: str | Path,
    periodic_pk_native: str | Path,
) -> ValidationResult:
    figures = [cl_figure, pk_figure]
    try:
        _artifact(archive, "comparison archive")
        contract_path = _artifact(contract, "pair contract")
        for path, name in (
            (cl_figure, "angular-spectrum figure"),
            (pk_figure, "power-spectrum figure"),
            (cl_native, "native angular spectrum"),
            (steps_pk_native, "native StePS P(k)"),
            (periodic_pk_native, "native periodic P(k)"),
        ):
            _artifact(path, name)

        with contract_path.open(encoding="utf-8") as stream:
            pair = json.load(stream)
        if pair.get("kind") != "stepsic-periodic-matched-pair-contract":
            raise ValueError("pair contract has the wrong kind")
        if pair.get("configuration_match") is not True:
            raise ValueError("matched-pair configuration check did not pass")
        if pair.get("field_hashes_match") is not True:
            raise ValueError("matched-pair field hash check did not pass")
        reference_fields = pair["fields"]["reference"]
        steps_fields = pair["fields"]["steps"]
        for name in ("ic_white_noise", "ic_delta_k"):
            if reference_fields[name] != steps_fields[name]:
                raise ValueError(f"pair contract {name} hashes differ")

        evolution = pair["evolution"]
        requested_redshift = float(evolution["requested_final_redshift"])
        for simulation in ("steps", "periodic"):
            actual_redshift = float(evolution[simulation]["final_redshift"])
            actual_scale_factor = float(
                evolution[simulation]["final_scale_factor"]
            )
            expected_scale_factor = 1.0 / (1.0 + requested_redshift)
            if (
                not np.isfinite(actual_redshift)
                or not np.isfinite(actual_scale_factor)
                or not np.isclose(
                    actual_redshift, requested_redshift, rtol=0.0, atol=1.0e-8,
                )
                or not np.isclose(
                    actual_scale_factor,
                    expected_scale_factor,
                    rtol=0.0,
                    atol=1.0e-8,
                )
            ):
                raise ValueError(f"{simulation} snapshot is at the wrong epoch")

        data = load_npz(archive)
        shells = _finite_array(data, "shell_bounds_mpc_h", ndim=2)
        ell = _finite_array(data, "ell", ndim=1)
        cl_steps = _finite_array(data, "cl_steps", ndim=2)
        cl_periodic = _finite_array(data, "cl_periodic", ndim=2)
        steps_counts = _finite_array(
            data, "steps_shell_particle_count", ndim=1,
        )
        periodic_counts = _finite_array(
            data, "periodic_shell_particle_count", ndim=1,
        )
        k_common = _finite_array(data, "k_common_h_mpc", ndim=1)
        pk_steps = _finite_array(data, "pk_steps_common_mpc_h3", ndim=1)
        pk_periodic = _finite_array(
            data, "pk_periodic_common_mpc_h3", ndim=1,
        )
        pk_halofit = _finite_array(
            data, "pk_halofit_common_mpc_h3", ndim=1,
        )
        if shells.shape[1:] != (2,) or np.any(shells[:, 1] <= shells[:, 0]):
            raise ValueError("shell axes are invalid")
        if len(shells) > 1 and np.any(shells[1:, 0] < shells[:-1, 1]):
            raise ValueError("shell axes overlap")
        if np.any(np.diff(ell) <= 0.0) or ell[0] < 0.0:
            raise ValueError("ell axis must be nonnegative and increasing")
        expected_cl_shape = (len(shells), len(ell))
        if cl_steps.shape != expected_cl_shape or cl_periodic.shape != expected_cl_shape:
            raise ValueError("C_ell arrays do not match shell and ell axes")
        if (
            steps_counts.shape != (len(shells),)
            or periodic_counts.shape != (len(shells),)
            or np.any(steps_counts <= 0)
            or np.any(periodic_counts <= 0)
        ):
            raise ValueError("every shell must contain particles in both outputs")
        if np.any(np.diff(k_common) <= 0.0) or np.any(k_common <= 0.0):
            raise ValueError("common k axis must be positive and increasing")
        if not (
            pk_steps.shape == pk_periodic.shape == pk_halofit.shape == k_common.shape
        ):
            raise ValueError("common P(k) arrays must share the k axis")

        cl_difference_rms = float(np.sqrt(np.mean((cl_steps - cl_periodic) ** 2)))
        pk_difference_rms = float(np.sqrt(np.mean((pk_steps - pk_periodic) ** 2)))
        checks = [
            Check(
                name="matched-pair contract",
                observed="matched",
                comparison="is",
                limit="matched",
                unit="not applicable",
                rationale=(
                    "The two simulations must share extent, cosmology, "
                    "field generation, random phases, and generated density."
                ),
                source="pair-contract.json",
                passed=True,
            ),
            Check(
                name="requested evolved epoch",
                observed=requested_redshift,
                comparison="is",
                limit=requested_redshift,
                unit="redshift",
                rationale="Both final snapshots must represent the requested epoch.",
                source="snapshot headers and pair contract",
                passed=True,
            ),
            Check(
                name="finite ordered measurement evidence",
                observed=int(len(ell) + len(k_common)),
                comparison=">=",
                limit=2,
                unit="spectral bins",
                rationale=(
                    "Every configured shell and full-domain spectrum must "
                    "contain finite data on valid axes."
                ),
                source="comparison archive",
                passed=True,
            ),
            Check(
                name="provenance artifacts",
                observed=6,
                comparison="is",
                limit=6,
                unit="required artifacts",
                rationale="The contract, native spectra, and both figures are required.",
                source="campaign artifact contract",
                passed=True,
            ),
        ]
        return ValidationResult(
            campaign="evolved-sphere",
            parameters=metadata_parameters(data),
            provenance={
                **archive_provenance(archive),
                "pair_contract": str(contract_path),
                "cl_native": str(cl_native),
                "steps_pk_native": str(steps_pk_native),
                "periodic_pk_native": str(periodic_pk_native),
            },
            metrics={
                "cl_signed_difference_rms": Metric(cl_difference_rms, "C_ell"),
                "pk_signed_difference_rms": Metric(
                    pk_difference_rms, "(Mpc/h)^3",
                ),
            },
            checks=checks,
            numerical_archive=archive,
            figures=figures,
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return malformed_result(
            campaign="evolved-sphere",
            archive=archive,
            figures=figures,
            error=error,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--cl-figure", required=True)
    parser.add_argument("--pk-figure", required=True)
    parser.add_argument("--cl-native", required=True)
    parser.add_argument("--steps-pk-native", required=True)
    parser.add_argument("--periodic-pk-native", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return write_result(
        evaluate(
            args.archive,
            contract=args.contract,
            cl_figure=args.cl_figure,
            pk_figure=args.pk_figure,
            cl_native=args.cl_native,
            steps_pk_native=args.steps_pk_native,
            periodic_pk_native=args.periodic_pk_native,
        ),
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
