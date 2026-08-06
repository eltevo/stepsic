#!/usr/bin/env python3
"""Assemble the angular and full-domain power comparison archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import camb
import numpy as np

from validation._common.evaluation import atomic_savez, load_npz
from validation.sphere.evolved import physical_pk_to_h_units


def _text_spectrum(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path, comments="#", ndmin=2)
    if values.shape[1] < 2:
        raise ValueError("StePS P(k) must contain k and P(k) columns")
    k = values[:, 0].astype(np.float64)
    power = values[:, 1].astype(np.float64)
    valid = np.isfinite(k) & np.isfinite(power) & (k > 0.0)
    k = k[valid]
    power = power[valid]
    order = np.argsort(k)
    k = k[order]
    power = power[order]
    if len(k) == 0 or np.any(np.diff(k) <= 0.0):
        raise ValueError("StePS P(k) has no finite increasing support")
    return k, power


def _halofit(
    k_h_mpc: np.ndarray,
    *,
    redshift: float,
    cosmology: dict[str, float],
    halofit: str,
) -> np.ndarray:
    h = cosmology["H0_km_s_mpc"] / 100.0
    parameters = camb.CAMBparams()
    parameters.set_cosmology(
        H0=cosmology["H0_km_s_mpc"],
        ombh2=cosmology["omega_b"] * h**2,
        omch2=(cosmology["omega_m"] - cosmology["omega_b"]) * h**2,
        omk=1.0 - cosmology["omega_m"] - cosmology["omega_lambda"],
    )
    parameters.InitPower.set_params(As=2.1e-9, ns=0.9665)
    parameters.NonLinear = camb.model.NonLinear_both
    parameters.NonLinearModel.set_params(halofit_version=halofit)
    interpolator = camb.get_matter_power_interpolator(
        parameters,
        nonlinear=True,
        hubble_units=True,
        k_hunit=True,
        kmax=float(1.05 * k_h_mpc[-1]),
        zmax=max(float(redshift), 0.01),
    )
    return np.asarray(interpolator.P(redshift, k_h_mpc), dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--cl", required=True)
    parser.add_argument("--steps-pk", required=True)
    parser.add_argument("--periodic-pk", required=True)
    parser.add_argument("--random-factor", type=int, required=True)
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument("--p0", type=float, required=True)
    parser.add_argument("--n-fkp-radial-bins", type=int, required=True)
    parser.add_argument("--nmesh", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.contract).open(encoding="utf-8") as stream:
        contract = json.load(stream)
    cl = load_npz(args.cl)
    periodic = load_npz(args.periodic_pk)
    cosmology = contract["configuration"]["cosmology"]
    h = float(cosmology["H0_km_s_mpc"]) / 100.0
    k_steps_per_mpc, pk_steps_mpc3 = _text_spectrum(args.steps_pk)
    k_steps_h_mpc, pk_steps_mpc_h3 = physical_pk_to_h_units(
        k_steps_per_mpc, pk_steps_mpc3, h=h,
    )
    k_periodic_h_mpc = np.asarray(periodic["k_h_mpc"], dtype=np.float64)
    pk_periodic_mpc_h3 = np.asarray(periodic["power_mpc_h3"], dtype=np.float64)
    periodic_valid = (
        np.isfinite(k_periodic_h_mpc)
        & np.isfinite(pk_periodic_mpc_h3)
        & (k_periodic_h_mpc > 0.0)
    )
    k_periodic_h_mpc = k_periodic_h_mpc[periodic_valid]
    pk_periodic_mpc_h3 = pk_periodic_mpc_h3[periodic_valid]
    if len(k_periodic_h_mpc) == 0 or np.any(np.diff(k_periodic_h_mpc) <= 0.0):
        raise ValueError("periodic P(k) has no finite increasing support")
    common = (
        (k_steps_h_mpc >= k_periodic_h_mpc[0])
        & (k_steps_h_mpc <= k_periodic_h_mpc[-1])
        & (pk_steps_mpc_h3 > 0.0)
    )
    k_common_h_mpc = k_steps_h_mpc[common]
    pk_steps_common_mpc_h3 = pk_steps_mpc_h3[common]
    if len(k_common_h_mpc) == 0:
        raise ValueError("native full-domain P(k) results have no common support")
    pk_periodic_common_mpc_h3 = np.interp(
        k_common_h_mpc, k_periodic_h_mpc, pk_periodic_mpc_h3,
    )
    positive_periodic = pk_periodic_common_mpc_h3 > 0.0
    k_common_h_mpc = k_common_h_mpc[positive_periodic]
    pk_steps_common_mpc_h3 = pk_steps_common_mpc_h3[positive_periodic]
    pk_periodic_common_mpc_h3 = pk_periodic_common_mpc_h3[positive_periodic]
    if len(k_common_h_mpc) == 0:
        raise ValueError("common P(k) support contains no positive power")
    redshift = float(contract["evolution"]["requested_final_redshift"])
    pk_halofit_common_mpc_h3 = _halofit(
        k_common_h_mpc,
        redshift=redshift,
        cosmology=cosmology,
        halofit=contract["configuration"]["initial_conditions"]["halofit"],
    )
    if (
        not np.all(np.isfinite(pk_halofit_common_mpc_h3))
        or np.any(pk_halofit_common_mpc_h3 <= 0.0)
    ):
        raise ValueError("CAMB/Halofit prediction must be finite and positive")

    atomic_savez(
        args.output,
        meta_h=np.float64(h),
        meta_steps_pk_estimator=np.array("StePS_Pk.py full spherical selection"),
        meta_periodic_pk_estimator=np.array("stepsic.pk.measure_pk full cube"),
        meta_pk_random_factor=np.int64(args.random_factor),
        meta_pk_random_seed=np.int64(args.random_seed),
        meta_pk_p0=np.float64(args.p0),
        meta_pk_n_fkp_radial_bins=np.int64(args.n_fkp_radial_bins),
        meta_pk_nmesh=np.int64(args.nmesh),
        meta_window_warning=np.array(
            "Direct ratios compare different native estimators and survey windows."
        ),
        shell_bounds_mpc_h=cl["shell_bounds_mpc_h"],
        ell=cl["ell"],
        cl_steps=cl["cl_steps"],
        cl_periodic=cl["cl_periodic"],
        cl_signed_difference=cl["cl_signed_difference"],
        steps_shell_particle_count=cl["steps_particle_count"],
        periodic_shell_particle_count=cl["periodic_particle_count"],
        k_steps_native_h_mpc=k_steps_h_mpc,
        pk_steps_native_mpc_h3=pk_steps_mpc_h3,
        k_periodic_native_h_mpc=k_periodic_h_mpc,
        pk_periodic_native_mpc_h3=pk_periodic_mpc_h3,
        periodic_native_mode_count=periodic["mode_count"][periodic_valid],
        k_common_h_mpc=k_common_h_mpc,
        pk_steps_common_mpc_h3=pk_steps_common_mpc_h3,
        pk_periodic_common_mpc_h3=pk_periodic_common_mpc_h3,
        pk_halofit_common_mpc_h3=pk_halofit_common_mpc_h3,
        ratio_steps_halofit=pk_steps_common_mpc_h3 / pk_halofit_common_mpc_h3,
        ratio_periodic_halofit=(
            pk_periodic_common_mpc_h3 / pk_halofit_common_mpc_h3
        ),
        ratio_steps_periodic=(
            pk_steps_common_mpc_h3 / pk_periodic_common_mpc_h3
        ),
    )


if __name__ == "__main__":
    main()
