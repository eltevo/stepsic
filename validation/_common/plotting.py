"""Figure style and 3D particle views shared by validation plots."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np


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


def subsample(pos, fraction: float, *, seed: int = 42, masses=None):
    """Return a random fraction of the particle positions and, when given, their masses."""
    n_total = pos.shape[0]
    n_keep = max(1, int(n_total * fraction))
    if n_keep >= n_total:
        return pos, masses
    index = np.random.default_rng(seed).choice(n_total, size=n_keep, replace=False)
    return pos[index], (masses[index] if masses is not None else None)


def draw_panel_3d(
    ax,
    pos,
    title: str,
    *,
    sizes,
    colors,
    alpha: float = 0.25,
    elev: float = 22.0,
    azim: float = -42.0,
) -> None:
    """Draw particles as a 3D scatter with unlabelled axes and a panel title."""
    ax.scatter(
        pos[:, 0], pos[:, 1], pos[:, 2],
        s=sizes,
        c=colors,
        alpha=alpha,
        edgecolors='none',
        rasterized=True,
        depthshade=True,
    )
    ax.view_init(elev=elev, azim=azim)

    # The shape of the load is the message; tick labels only clutter the panel.
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(axis='both', which='both', length=0, pad=0)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('0.80')
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo['grid']['color'] = (0.85, 0.85, 0.85, 0.5)
        axis._axinfo['grid']['linewidth'] = 0.4

    ax.text2D(
        0.03, 0.96, title,
        transform=ax.transAxes, fontsize=7,
        va='top', ha='left',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.80, pad=1.5),
    )


def _set_cubic_limits(ax, centre, half_range: float) -> None:
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(centre[0] - half_range, centre[0] + half_range)
    ax.set_ylim(centre[1] - half_range, centre[1] + half_range)
    ax.set_zlim(centre[2] - half_range, centre[2] + half_range)


def set_equal_aspect_3d(ax, pos, pad: float = 0.05) -> None:
    """Give a 3D axis equal scales, centred on the bounding box of the particles."""
    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)
    _set_cubic_limits(ax, 0.5 * (mins + maxs), 0.5 * (1.0 + pad) * (maxs - mins).max())


def set_cylinder_aspect(ax, pos, pad: float = 0.05) -> None:
    """Give a 3D axis equal scales, centred on the axis of a cylinder along z."""
    centre = pos.mean(axis=0)
    r_max = float(np.max(np.hypot(pos[:, 0] - centre[0], pos[:, 1] - centre[1])))
    z_min, z_max = float(pos[:, 2].min()), float(pos[:, 2].max())
    half_range = (1.0 + pad) * max(r_max, 0.5 * (z_max - z_min))
    _set_cubic_limits(ax, (centre[0], centre[1], 0.5 * (z_min + z_max)), half_range)
