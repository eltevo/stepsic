#!/usr/bin/env python3
"""Measure the complete spherical snapshot with StePS_Pk.py."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--glass", required=True)
    parser.add_argument("--randoms", required=True)
    parser.add_argument("--steps-pk", required=True)
    parser.add_argument("--p0", type=float, required=True)
    parser.add_argument("--n-fkp-radial-bins", type=int, required=True)
    parser.add_argument("--nmesh", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.p0) or args.p0 <= 0.0:
        raise ValueError("--p0 must be finite and positive")
    if min(args.n_fkp_radial_bins, args.nmesh) < 1:
        raise ValueError("FKP bin and mesh counts must be positive")

    steps_pk = Path(args.steps_pk).resolve()
    pk_directory = steps_pk.parent
    steps_ic_source = (pk_directory / "../../StePS_IC/src").resolve()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-", suffix=".txt", dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    with tempfile.TemporaryDirectory(prefix="steps-pk-stubs-") as stub_name:
        stub_directory = Path(stub_name)
        (stub_directory / "pygadgetreader.py").write_text(
            "# unused import stub\n", encoding="utf-8",
        )
        (stub_directory / "glio.py").write_text(
            "# unused import stub\n", encoding="utf-8",
        )
        environment = os.environ.copy()
        python_paths = [
            str(stub_directory), str(pk_directory), str(steps_ic_source),
        ]
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
        command = [
            sys.executable,
            str(steps_pk),
            str(Path(args.snapshot).resolve()),
            str(Path(args.randoms).resolve()),
            str(Path(args.glass).resolve()),
            str(temporary),
            "--Geometry", "spherical",
            "--n_FKP_radial_bins", str(args.n_fkp_radial_bins),
            "--Nmesh", str(args.nmesh),
            "--P0", str(args.p0),
            "--ShotNoise",
        ]
        if args.verbose:
            command.append("--verbose")
        try:
            subprocess.run(command, check=True, env=environment)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("StePS_Pk.py produced no spectrum")
            file_descriptor = os.open(temporary, os.O_RDONLY)
            try:
                os.fsync(file_descriptor)
            finally:
                os.close(file_descriptor)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
