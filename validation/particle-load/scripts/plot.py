#!/usr/bin/env python3
'''
Plot particle load distributions for the stepsic paper.

Produces a 4-panel landscape figure (A&A double-column width)
showing 3D scatter views of each particle load type generated
by stepsic with LPTORDER=0:

    (a) Cubical, random (Poisson)
    (b) Cubical, grid (regular lattice)
    (c) Spherical shells
    (d) Cylindrical shells

Each panel renders a random subsample of particles in a 3D
projection, revealing the spatial distribution and distinct
boundary shapes of each particle load strategy.

Usage
-----
::

    python validate-particle-load-plot.py \\
        --cubic-random output/particle-load/cubic_random/.../ic.hdf5 \\
        --cubic-grid   output/particle-load/cubic_grid/.../ic.hdf5 \\
        --spherical    output/particle-load/spherical/.../ic.hdf5 \\
        --cylindrical  output/particle-load/cylindrical/.../ic.hdf5 \\
        -o output/particle-load/validation-particle-load.pdf

Any subset of panels can be omitted; missing panels are left blank.
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (side-effect import)
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from validation import setup_matplotlib

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

ArrayF: TypeAlias = NDArray[np.float64]


def load_particles(path: str, part_type: int = 1) -> ArrayF:
    '''Read particle positions from a Gadget-format HDF5 snapshot.

    Parameters
    ----------
    path : str
        Path to the HDF5 snapshot produced by stepsic.
    part_type : int
        Particle type group index (default 1 = dark matter).

    Returns
    -------
    pos : ndarray, shape (N, 3)
        Particle positions.
    '''
    with h5py.File(path, 'r') as f:
        grp = f[f'PartType{part_type}']
        pos = grp['Coordinates'][:]
    log.info('Loaded %d particles from %s', pos.shape[0], path)
    return pos.astype(np.float64)


def subsample(
    pos: ArrayF,
    fraction: float,
    *,
    seed: int = 42,
) -> ArrayF:
    '''Return a random subsample of particle positions.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Full particle array.
    fraction : float
        Fraction of particles to keep, in (0, 1].
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    ndarray, shape (M, 3)
        Subsampled positions.
    '''
    n_total = pos.shape[0]
    n_keep = max(1, int(n_total * fraction))
    if n_keep >= n_total:
        return pos
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n_keep, replace=False)
    return pos[idx]


def _radial_sizes(
    pos: ArrayF,
    geometry: str,
    *,
    s_min: float = 0.01,
    s_max: float = 2.5,
) -> ArrayF:
    r'''Compute per-particle scatter sizes that grow with radial distance.

    For spherical geometry the radius is measured from the centroid of
    the distribution; for cylindrical it is the distance from the
    z-axis (through the centroid in x-y). Sizes are linearly
    interpolated between ``s_min`` at the centre and ``s_max`` at the
    outermost particle.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Particle positions (already subsampled).
    geometry : ``'spherical'`` or ``'cylindrical'``
        Determines which radius definition to use.
    s_min, s_max : float
        Marker area at the smallest and largest radius respectively
        (matplotlib ``s`` units, i.e. points²).

    Returns
    -------
    sizes : ndarray, shape (N,)
        Per-particle marker sizes.
    '''
    centre = pos.mean(axis=0)
    if geometry == 'spherical':
        r = np.linalg.norm(pos - centre, axis=1)
    elif geometry == 'cylindrical':
        r = np.sqrt(
            (pos[:, 0] - centre[0])**2 + (pos[:, 1] - centre[1])**2
        )
    else:
        raise ValueError(f'Unsupported geometry for radial sizing: {geometry}')

    r_max = r.max()
    if r_max == 0.0:
        return np.full(pos.shape[0], s_min)
    t = r / r_max                       # normalized to [0, 1]
    return s_min + (s_max - s_min) * t   # linear ramp


def _draw_panel_3d(
    ax: Axes3D,
    pos: ArrayF,
    title: str,
    *,
    fraction: float = 0.08,
    point_size: float = 1.5,
    point_color: str = 'k',
    point_alpha: float = 0.25,
    elev: float = 22.0,
    azim: float = -42.0,
    geometry: str | None = None,
) -> int:
    '''Draw a 3D scatter of a subsampled particle distribution.

    Parameters
    ----------
    geometry : str or None
        If ``'spherical'`` or ``'cylindrical'``, marker sizes scale
        with radial distance (bigger dots further from the centre).
        If ``None`` (cubical panels), a uniform ``point_size`` is used.

    Returns the number of particles plotted.
    '''
    sub = subsample(pos, fraction)
    n_plot = sub.shape[0]

    if geometry in ('spherical', 'cylindrical'):
        sizes = _radial_sizes(sub, geometry)
    else:
        sizes = point_size

    ax.scatter(
        sub[:, 0], sub[:, 1], sub[:, 2],
        s=sizes,
        c=point_color,
        alpha=point_alpha,
        edgecolors='none',
        rasterized=True,
        depthshade=True,
    )

    ax.view_init(elev=elev, azim=azim)

    # Clean up axes - for this figure, the *shape* is the message.
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(
        axis='both', which='both',
        length=0, pad=0,
    )

    # Subtle pane styling
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('0.80')

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo['grid']['color'] = (0.85, 0.85, 0.85, 0.5)
        axis._axinfo['grid']['linewidth'] = 0.4

    # Panel title
    ax.text2D(
        0.03, 0.96, title,
        transform=ax.transAxes, fontsize=7,
        va='top', ha='left',
        bbox=dict(
            facecolor='white', edgecolor='none', alpha=0.80, pad=1.5,
        ),
    )

    return n_plot


def _set_equal_aspect_3d(ax: Axes3D, pos: ArrayF, pad: float = 0.05) -> None:
    '''Force equal aspect ratio on a 3D axes by setting matching limits.

    Parameters
    ----------
    ax : Axes3D
        The 3D axes to adjust.
    pos : ndarray, shape (N, 3)
        Particle positions for extent computation.
    pad : float
        Fractional padding around the data extent.
    '''
    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)
    centres = 0.5 * (mins + maxs)
    half_range = 0.5 * (1.0 + pad) * (maxs - mins).max()

    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(centres[0] - half_range, centres[0] + half_range)
    ax.set_ylim(centres[1] - half_range, centres[1] + half_range)
    ax.set_zlim(centres[2] - half_range, centres[2] + half_range)


def _set_cylinder_aspect(
    ax: Axes3D,
    pos: ArrayF,
    r_max: float,
    z_min: float,
    z_max: float,
    xy_centre: ArrayF,
    pad: float = 0.05,
) -> None:
    '''Set 3D limits for a cylindrical geometry, preserving shape fidelity.

    Parameters
    ----------
    ax : Axes3D
        The 3D axes.
    pos : ndarray, shape (N, 3)
        Particle positions (used for fallback only).
    r_max : float
        Maximum cylindrical radius.
    z_min, z_max : float
        Axial extent of the cylinder.
    xy_centre : ndarray, shape (2,)
        Centre of the particle distribution in the x-y plane.
    pad : float
        Fractional padding.
    '''
    r_padded = r_max * (1.0 + pad)
    z_centre = 0.5 * (z_min + z_max)
    z_half = 0.5 * (z_max - z_min) * (1.0 + pad)

    half_range = max(r_padded, z_half)

    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(xy_centre[0] - half_range, xy_centre[0] + half_range)
    ax.set_ylim(xy_centre[1] - half_range, xy_centre[1] + half_range)
    ax.set_zlim(z_centre - half_range, z_centre + half_range)


def plot_particle_loads(
    paths: dict[str, str | None],
    output: str | None = None,
    *,
    fraction: float = 0.08,
    elev: float = 22.0,
    azim: float = -42.0,
) -> None:
    r'''Produce the 4-panel 3D particle load validation figure.

    Parameters
    ----------
    paths : dict
        Mapping ``{'cubic_random': path, 'cubic_grid': path,
        'spherical': path, 'cylindrical': path}``.
        ``None`` values produce a blank panel.
    output : str or None
        Output file path. Interactive display if ``None``.
    fraction : float
        Fraction of particles to show per panel (random subsample).
    elev : float
        3D viewing elevation angle [degrees].
    azim : float
        3D viewing azimuth angle [degrees].
    '''
    setup_matplotlib()

    n_panels = 4
    panel_keys = ['cubic_random', 'cubic_grid', 'spherical', 'cylindrical']
    panel_titles = [
        r'(a) Cubical ($\mathbb{T}^3$), random',
        r'(b) Cubical ($\mathbb{T}^3$), grid',
        r'(c) Spherical ($\mathbb{R}^3$), shells',
        r'(d) Cylindrical ($S^1 \!\times\! \mathbb{R}^2$), shells',
    ]

    # A&A double-column layout: 7.09" wide
    fig_width = 7.09
    panel_size = fig_width / n_panels
    fig_height = panel_size + 0.15

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)

    for i, (key, title) in enumerate(zip(panel_keys, panel_titles)):
        path = paths.get(key)
        if path is None or not Path(path).exists():
            log.warning(
                'Skipping panel %s (file not provided or missing).', key,
            )
            continue

        ax = fig.add_subplot(1, n_panels, i + 1, projection='3d')
        pos = load_particles(path)
        n_total = pos.shape[0]

        # Set axis limits based on geometry
        if key in ('cubic_random', 'cubic_grid'):
            _set_equal_aspect_3d(ax, pos)

        elif key == 'spherical':
            _set_equal_aspect_3d(ax, pos)

        elif key == 'cylindrical':
            centre = pos.mean(axis=0)
            dx = pos[:, 0] - centre[0]
            dy = pos[:, 1] - centre[1]
            r_max = float(np.max(np.sqrt(dx**2 + dy**2)))

            z_min = float(np.min(pos[:, 2]))
            z_max = float(np.max(pos[:, 2]))

            _set_cylinder_aspect(
                ax, pos, r_max,
                z_min=z_min, z_max=z_max,
                xy_centre=centre[:2],
            )

        else:
            raise ValueError(f'Unknown panel key: {key}')

        # Map panel key -> geometry for radial marker sizing.
        # Cubical panels get None (uniform size).
        _geom_map = {
            'spherical': 'spherical',
            'cylindrical': 'cylindrical',
        }

        n_plotted = _draw_panel_3d(
            ax, pos, title,
            fraction=fraction,
            elev=elev,
            azim=azim,
            geometry=_geom_map.get(key),
        )
        log.info(
            '%s: N_total = %d, N_shown = %d (%.1f%%)',
            key, n_total, n_plotted, 100.0 * n_plotted / n_total,
        )

    fig.subplots_adjust(
        left=0.01, right=0.99,
        bottom=0.02, top=0.98,
        wspace=0.02,
    )

    if output:
        fig.savefig(output, bbox_inches='tight', dpi=300)
        log.info('Figure saved to %s', output)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Plot 4-panel 3D particle load validation figure '
                    'for the stepsic paper.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--cubic-random', type=str, default=None,
        help='HDF5 snapshot of the cubical random particle load.',
    )
    p.add_argument(
        '--cubic-grid', type=str, default=None,
        help='HDF5 snapshot of the cubical grid particle load.',
    )
    p.add_argument(
        '--spherical', type=str, default=None,
        help='HDF5 snapshot of the spherical shell particle load.',
    )
    p.add_argument(
        '--cylindrical', type=str, default=None,
        help='HDF5 snapshot of the cylindrical shell particle load.',
    )
    p.add_argument(
        '--fraction', type=float, default=0.08,
        help='Fraction of particles to display per panel.',
    )
    p.add_argument(
        '--elev', type=float, default=22.0,
        help='3D viewing elevation angle [degrees].',
    )
    p.add_argument(
        '--azim', type=float, default=-42.0,
        help='3D viewing azimuth angle [degrees].',
    )
    p.add_argument(
        '-o', '--output', type=str, default=None,
        help='Output figure path (e.g. validation-particle-load.pdf). '
             'Shows interactively if omitted.',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    load_paths = {
        'cubic_random': args.cubic_random,
        'cubic_grid': args.cubic_grid,
        'spherical': args.spherical,
        'cylindrical': args.cylindrical,
    }

    provided = {k: v for k, v in load_paths.items() if v is not None}
    if not provided:
        log.error(
            'No particle load snapshots provided. Use --cubic-random, '
            '--cubic-grid, --spherical, and/or --cylindrical to specify '
            'HDF5 files.'
        )
        sys.exit(1)

    log.info('Panels to plot: %s', ', '.join(provided.keys()))
    plot_particle_loads(
        load_paths,
        output=args.output,
        fraction=args.fraction,
        elev=args.elev,
        azim=args.azim,
    )