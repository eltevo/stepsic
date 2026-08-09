from __future__ import annotations

from datetime import date
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stepsic.field import white_noise  # noqa: E402


TARGET = ROOT / "tests" / "goldens" / "white_noise_portability.npz"


def main() -> None:
    """Regenerate the cross-version white-noise portability fixture."""
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    payload = {
        "w": white_noise((16, 16, 16), seed=42, dtype=np.float64),
        "git_commit": np.array(commit),
        "numpy_version": np.array(np.__version__),
        "scipy_version": np.array(scipy.__version__),
        "generated_date": np.array(date.today().isoformat()),
        "generator": np.array("tests/regen/regen_white_noise_portability.py"),
        "seed": np.array(42, dtype=np.uint64),
        "nvox": np.array([16, 16, 16], dtype=np.int64),
    }
    with tempfile.NamedTemporaryFile(
        dir=TARGET.parent, prefix="tmp-white-noise-", suffix=".npz", delete=False
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
