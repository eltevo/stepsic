#!/usr/bin/env python3
'''
Plot centred x-z slices through the available glasses.

Each panel shows one glass geometry:

    (a) Cubical glass (from random pre-glass) - X-Z slab of configurable thickness
    (b) Cubical glass (from grid pre-glass)   - single Y-layer of the grid
    (c) Spherical glass                       - X-Z slab
    (d) Cylindrical glass                     - X-Z slab (vertical cut)

Random, spherical, and cylindrical loads use a slab of the requested thickness. A regular cubic load uses only the grid layer nearest the centre so that adjacent layers do not overlap in the projection.

Usage
-----
::

    PYTHONPATH="$PWD" conda run -n stepsic python validation/glass-quality/scripts/plot.py \\
        --cubic-random validation/glass-quality/runs/medium/cache/cubic_random/glass/snapshot_0001.hdf5 \\
        --cubic-grid   validation/glass-quality/runs/medium/cache/cubic_grid/preglass/ic.hdf5 \\
        --spherical    validation/glass-quality/runs/medium/cache/spherical/glass/snapshot_0001.hdf5 \\
        --cylindrical  validation/glass-quality/runs/medium/cache/cylindrical/glass/snapshot_0001.hdf5 \\
        --target-radius 500.0 \\
        --slice-thickness 10.0 \\
        -o validation/glass-quality/runs/medium/output/glass.pdf

Any subset of panels can be omitted; missing panels are left blank.
The ``--target-radius`` value must match ``R_3D`` in ``validation/glass-quality/config.env``
(default 500 Mpc).
'''

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TypeAlias

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from validation._common.plotting import atomic_savefig, setup_matplotlib

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

ArrayF: TypeAlias = NDArray[np.float64]


def load_glass(path: str, part_type: int = 1) -> tuple[ArrayF, ArrayF]:
    '''Read particle positions and masses from a Gadget-format HDF5 file.

    Parameters
    ----------
    path : str
        Path to the HDF5 snapshot.
    part_type : int
        Particle type group index (default 1 = dark matter).

    Returns
    -------
    pos : ndarray, shape (N, 3)
        Particle positions.
    mass : ndarray, shape (N,)
        Particle masses.
    '''
    with h5py.File(path, 'r') as f:
        grp = f[f'PartType{part_type}']
        pos = grp['Coordinates'][:]
        mass = grp['Masses'][:]
    log.info('Loaded %d particles from %s', pos.shape[0], path)
    return pos.astype(np.float64), mass.astype(np.float64)


def get_header(path: str, key: str, default=None):
    '''Read a single header attribute from a Gadget-HDF5 file.'''
    with h5py.File(path, 'r') as f:
        return f['/Header'].attrs.get(key, default)


def rescale_to_radius(
    pos: ArrayF,
    target_radius: float,
    geometry: str,
) -> ArrayF:
    """Rescale expanded glass particles to a displayable extent.

    During reverse-gravity relaxation in StePS, non-periodic axes expand
    beyond the nominal domain. For non-periodic geometries StePS writes
    ``L_BOX = 0`` into the HDF5 header (it is unused at run time), so the
    header cannot be used as a rescaling anchor. Instead this function uses
    the *actual particle extent* as the source scale and ``target_radius``
    (the physical R_3D passed to StePS) as the target.

    This rescaling is purely cosmetic - it restores the intended visual
    shape for the paper figure without modifying the physics.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Particle positions (modified in-place, also returned).
    target_radius : float
        Desired half-extent after rescaling [Mpc]. Pass the same value as
        ``R_3D`` in ``validation/glass-quality/config.env``.
        For spherical: applied to all three axes.
        For cylindrical: applied to x and y only (z is periodic and correct).
    geometry : str
        ``'spherical'`` or ``'cylindrical'``.

    Returns
    -------
    ndarray, shape (N, 3)
        Rescaled positions.
    """
    if geometry == 'spherical':
        axes = [0, 1, 2]
    elif geometry == 'cylindrical':
        axes = [0, 1]
    else:
        raise ValueError(
            f"rescale_to_radius: geometry must be 'spherical' or 'cylindrical', "
            f"got {geometry!r}"
        )

    target_diameter = 2.0 * target_radius

    for ax in axes:
        lo, hi = pos[:, ax].min(), pos[:, ax].max()
        extent = hi - lo
        if extent == 0.0:
            log.warning('Axis %d has zero extent - skipping rescale.', ax)
            continue
        centre = 0.5 * (lo + hi)
        scale = target_diameter / extent
        pos[:, ax] = centre + (pos[:, ax] - centre) * scale
        log.info(
            'Rescaled axis %d: extent %.4f -> %.4f (factor %.6f)',
            ax, extent, target_diameter, scale,
        )

    return pos


def slice_xz(
    pos: ArrayF,
    thickness: float,
    *,
    y_centre: float | None = None,
) -> ArrayF:
    """Select particles within a Y-slab and return their X-Z coordinates.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Full particle positions (columns: x, y, z).
    thickness : float
        Total slab thickness along Y. Particles with
        ``|y - y_centre| <= thickness / 2`` are kept.
    y_centre : float or None
        Centre of the slab. Defaults to the median Y coordinate.

    Returns
    -------
    ndarray, shape (M, 2)
        Columns are (x, z) for the particles inside the slab.
    """
    if y_centre is None:
        y_centre = float(np.median(pos[:, 1]))

    half = 0.5 * thickness
    mask = np.abs(pos[:, 1] - y_centre) <= half
    n_kept = int(np.sum(mask))
    log.info(
        'Y-slab: centre=%.4f, thickness=%.4f -> kept %d / %d particles',
        y_centre, thickness, n_kept, pos.shape[0],
    )
    return pos[mask][:, [0, 2]]  # (x, z)


def pick_grid_layer(pos: ArrayF) -> ArrayF:
    """Select the single Y-layer closest to the centre of a regular grid.

    For a simple-cubic grid, particles sit at a discrete set of Y values.
    This function finds the Y level nearest to the midpoint and returns
    the (x, z) coordinates of all particles on that layer.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Full particle positions.

    Returns
    -------
    ndarray, shape (M, 2)
        Columns are (x, z) for the selected layer.
    """
    y_vals = np.unique(pos[:, 1])
    y_mid = 0.5 * (y_vals.min() + y_vals.max())
    y_layer = y_vals[np.argmin(np.abs(y_vals - y_mid))]

    # Float comparison is safe here: particles sit exactly on lattice nodes.
    mask = pos[:, 1] == y_layer
    n_kept = int(np.sum(mask))
    log.info(
        'Grid layer: y=%.6f (centre=%.6f), %d particles on layer out of '
        '%d unique Y values',
        y_layer, y_mid, n_kept, len(y_vals),
    )
    return pos[mask][:, [0, 2]]


def subsample(
    pos: ArrayF,
    fraction: float,
    *,
    seed: int = 42,
) -> ArrayF:
    '''Return a random subsample of particle positions.

    Parameters
    ----------
    pos : ndarray, shape (N, D)
        Full particle array (works for any number of columns).
    fraction : float
        Fraction of particles to keep, in (0, 1].
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    ndarray, shape (M, D)
        Subsampled positions.
    '''
    n_total = pos.shape[0]
    n_keep = max(1, int(n_total * fraction))
    if n_keep >= n_total:
        return pos
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n_keep, replace=False)
    return pos[idx]


def _draw_panel_2d(
    ax: plt.Axes,
    xz: ArrayF,
    title: str,
    *,
    fraction: float = 1.0,
    point_size: float = 0.15,
    point_color: str = 'k',
    point_alpha: float = 0.35,
) -> int:
    """Draw a 2D X-Z scatter on *ax* and return the number of plotted points."""
    if fraction < 1.0:
        xz = subsample(xz, fraction)

    n_plot = xz.shape[0]

    ax.scatter(
        xz[:, 0], xz[:, 1],
        s=point_size,
        c=point_color,
        alpha=point_alpha,
        edgecolors='none',
        rasterized=True,
    )

    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0, pad=0)

    ax.text(
        0.03, 0.97, title,
        transform=ax.transAxes, fontsize=7,
        va='top', ha='left',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.80, pad=1.5),
    )

    return n_plot


def plot_glasses(
    paths: dict[str, str | None],
    output: str | None = None,
    *,
    fraction: float = 1.0,
    target_radius: float = 250.0,
    slice_thickness: float = 10.0,
) -> None:
    r'''Produce the 4-panel 2D glass validation figure (X-Z slices).

    Parameters
    ----------
    paths : dict
        Mapping ``{'cubic_random': path, 'cubic_grid': path,
        'spherical': path, 'cylindrical': path}``.
        ``None`` values produce a blank panel.
    output : str or None
        Output file path. Interactive display if ``None``.
    fraction : float
        Fraction of *slice* particles to show per panel (random subsample).
        Unlike the 3D version, the slice itself already reduces the count
        substantially, so a value of 1.0 (show all) is a reasonable default.
    target_radius : float
        Physical radius [Mpc] used to rescale spherical and cylindrical
        glass back to their intended domain after StePS expansion.
        Must match ``R_3D`` in ``validation/glass-quality/config.env``.
    slice_thickness : float
        Thickness of the Y-slab [Mpc] for the X-Z cut.
        Ignored for the cubical-grid panel which always picks a single
        Y-layer.
    '''
    setup_matplotlib()

    n_panels = 4
    panel_keys = ['cubic_random', 'cubic_grid', 'spherical', 'cylindrical']
    panel_titles = [
        r'(a) Cubical ($T^3$), random',
        r'(b) Cubical ($T^3$), grid',
        r'(c) Spherical ($\mathbb{R}^3$)',
        r'(d) Cylindrical ($S^1 \!\times\! \mathbb{R}^2$)',
    ]

    # A&A double-column layout: 7.09" wide
    fig_width = 7.09
    panel_size = fig_width / n_panels
    fig_height = panel_size + 0.15

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(fig_width, fig_height),
        dpi=200,
    )

    for i, (key, title) in enumerate(zip(panel_keys, panel_titles)):
        ax = axes[i]
        path = paths.get(key)
        if path is None or not Path(path).exists():
            log.warning('Skipping panel %s (file not provided or missing).', key)
            ax.set_visible(False)
            continue

        pos, _mass = load_glass(path)
        n_total = pos.shape[0]

        # Rescale non-periodic geometries back to their physical domain.
        if key == 'spherical':
            pos = rescale_to_radius(pos, target_radius, 'spherical')
        elif key == 'cylindrical':
            pos = rescale_to_radius(pos, target_radius, 'cylindrical')

        # Extract X-Z slice
        if key == 'cubic_grid':
            xz = pick_grid_layer(pos)
        else:
            xz = slice_xz(pos, slice_thickness)

        n_slice = xz.shape[0]

        n_plotted = _draw_panel_2d(
            ax, xz, title,
            fraction=fraction,
        )
        log.info(
            '%s: N_total=%d, N_slice=%d, N_shown=%d',
            key, n_total, n_slice, n_plotted,
        )

    fig.subplots_adjust(
        left=0.01, right=0.99,
        bottom=0.02, top=0.98,
        wspace=0.08,
    )

    if output:
        atomic_savefig(fig, output, bbox_inches='tight', dpi=300)
        log.info('Figure saved to %s', output)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Plot centred x-z slices through the available glasses.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--cubic-random', type=str, default=None,
        help='HDF5 snapshot of the cubical glass (from random pre-glass).',
    )
    p.add_argument(
        '--cubic-grid', type=str, default=None,
        help='HDF5 snapshot of the cubical glass (from grid pre-glass).',
    )
    p.add_argument(
        '--spherical', type=str, default=None,
        help='HDF5 snapshot of the spherical glass.',
    )
    p.add_argument(
        '--cylindrical', type=str, default=None,
        help='HDF5 snapshot of the cylindrical glass.',
    )
    p.add_argument(
        '--target-radius', type=float, default=500.0,
        help='Physical radius [Mpc] of the sphere/cylinder domain. '
             'Must match R_3D in validation/glass-quality/config.env.',
    )
    p.add_argument(
        '--slice-thickness', type=float, default=10.0,
        help='Thickness of the Y-slab [Mpc] for X-Z cuts. '
             'Ignored for the cubical-grid panel (always a single layer).',
    )
    p.add_argument(
        '--fraction', type=float, default=0.08,
        help='Fraction of slice particles to display per panel.',
    )
    p.add_argument(
        '-o', '--output', type=str, default=None,
        help='Output figure path (e.g. glass.pdf). '
             'Shows interactively if omitted.',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    glass_paths = {
        'cubic_random': args.cubic_random,
        'cubic_grid':   args.cubic_grid,
        'spherical':    args.spherical,
        'cylindrical':  args.cylindrical,
    }

    provided = {k: v for k, v in glass_paths.items() if v is not None}
    if not provided:
        log.error(
            'No glass snapshots provided. Use --cubic-random, --cubic-grid, '
            '--spherical, and/or --cylindrical to specify HDF5 files.'
        )
        sys.exit(1)

    log.info('Panels to plot: %s', ', '.join(provided.keys()))
    plot_glasses(
        glass_paths,
        output=args.output,
        fraction=args.fraction,
        target_radius=args.target_radius,
        slice_thickness=args.slice_thickness,
    )
