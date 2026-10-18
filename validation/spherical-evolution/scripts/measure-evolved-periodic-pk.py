#!/usr/bin/env python3
"""Measure power in the complete periodic cube with stepsic's FFT estimator."""

from __future__ import annotations

import argparse

import numpy as np

from validation._common.evaluation import atomic_savez
from validation._common.snapshots import load_snapshot
from evolved import measure_full_periodic_pk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--nmesh", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    snapshot = load_snapshot(args.snapshot)
    box_size_mpc_h = float(snapshot.header["BoxSize"])
    k_h_mpc, power_mpc_h3, mode_count = measure_full_periodic_pk(
        snapshot.coordinates,
        snapshot.masses,
        box_size_mpc_h=box_size_mpc_h,
        nmesh=args.nmesh,
    )
    atomic_savez(
        args.output,
        k_h_mpc=k_h_mpc,
        power_mpc_h3=power_mpc_h3,
        mode_count=mode_count,
        meta_particle_count=np.int64(len(snapshot.masses)),
        meta_box_size_mpc_h=np.float64(box_size_mpc_h),
        meta_nmesh=np.int64(args.nmesh),
        meta_estimator=np.array("stepsic.pk.measure_pk"),
    )


if __name__ == "__main__":
    main()
