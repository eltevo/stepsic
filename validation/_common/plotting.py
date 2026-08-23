"""Figure settings shared by validation plots."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


def setup_matplotlib() -> None:
    """Configure the shared A&A figure style."""
    from cycler import cycler
    import matplotlib.pyplot as plt

    palette = [
        "#000000", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
    ]
    settings = {
        "axes.prop_cycle": cycler(color=palette),
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "0.3",
        "axes.linewidth": 1,
        "axes.grid": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.unicode_minus": False,
    }
    for axis, near, far in (("xtick", "bottom", "top"), ("ytick", "left", "right")):
        settings[f"{axis}.direction"] = "in"
        settings[f"{axis}.{near}"] = True
        settings[f"{axis}.{far}"] = True
        settings[f"{axis}.color"] = "0.3"
        for kind, size in (("major", 6), ("minor", 3)):
            settings[f"{axis}.{kind}.width"] = 1
            settings[f"{axis}.{kind}.size"] = size
    plt.rcParams.update(settings)


def atomic_savefig(figure: object, path: str | Path, **kwargs: object) -> None:
    """Save a Matplotlib figure through an adjacent temporary file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.tmp-",
        suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, **kwargs)  # type: ignore[attr-defined]
        file_descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
