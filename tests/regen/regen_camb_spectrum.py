from __future__ import annotations

from datetime import date
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import camb
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stepsic.cosmology import CAMBCosmology  # noqa: E402


TARGET = ROOT / "tests" / "goldens" / "camb_spectrum.npz"


def main() -> None:
    """Regenerate the T5 spectrum produced by the external CAMB oracle."""
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    z = np.array([9.0, 0.0])
    As = 2.1064e-9
    ns = 0.96822
    kmin = 0.01
    kmax = 3.0
    npoints = 64
    component = "delta_cdm"
    kh, pk, pk3 = CAMBCosmology().get_spectrum(
        z=z.tolist(),
        As=As,
        ns=ns,
        sigma8_init=None,
        kmin=kmin,
        kmax=kmax,
        npoints=npoints,
        component=component,
    )
    payload = {
        "kh": kh,
        "pk": pk,
        "pk3": pk3,
        "git_commit": np.array(commit),
        "camb_version": np.array(camb.__version__),
        "numpy_version": np.array(np.__version__),
        "generated_date": np.array(date.today().isoformat()),
        "generator": np.array("tests/regen/regen_camb_spectrum.py"),
        "z": z,
        "As": np.array(As),
        "ns": np.array(ns),
        "kmin": np.array(kmin),
        "kmax": np.array(kmax),
        "npoints": np.array(npoints, dtype=np.int64),
        "component": np.array(component),
    }
    with tempfile.NamedTemporaryFile(
        dir=TARGET.parent, prefix="tmp-camb-", suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez(temporary, **payload)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(TARGET)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
