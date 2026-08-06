#!/usr/bin/env python3
'''
Plot glass particle distributions for the stepsic / StePS paper.

Produces a 4-panel landscape figure (A&A double-column width)
showing 3D scatter views of each glass geometry:

    (a) Cubical glass (from random pre-glass)
    (b) Cubical glass (from grid pre-glass)
    (c) Spherical glass
    (d) Cylindrical glass

Each panel renders a random subsample of particles in a 3D
projection, revealing the spatial homogeneity and distinct
boundary shapes achieved by the reverse-gravity relaxation
in StePS.

Usage
-----
::

    PYTHONPATH="$PWD" conda run -n stepsic python validation/glass/scripts/plot-3d.py \\
        --cubic-random validation/glass/cubic_random/glass/snapshot_0001.hdf5 \\
        --cubic-grid   validation/glass/cubic_grid/preglass/ic.hdf5 \\
        --spherical    validation/glass/spherical/glass/snapshot_0001.hdf5 \\
        --cylindrical  validation/glass/cylindrical/glass/snapshot_0001.hdf5 \\
        --target-radius 500.0 \\
        -o validation/glass/output/glass.pdf

Any subset of panels can be omitted; missing panels are left blank.
The ``--target-radius`` value must match ``R_3D`` in ``validation/glass/config.env``
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
from matplotlib.colors import LogNorm

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (side-effect import)
from numpy.typing import NDArray

from validation import setup_matplotlib

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

ArrayF: TypeAlias = NDArray[np.float64]


def log_mass_to_colors(
    mass: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap_name: str = 'plasma_r',
) -> np.ndarray:
    """Map positive particle masses to a logarithmic colormap."""
    mass = np.asarray(mass, dtype=np.float64)

    if np.any(mass <= 0.0):
        raise ValueError('Masses must be strictly positive for logarithmic colouring.')

    if vmin is None:
        vmin = float(np.min(mass))
    if vmax is None:
        vmax = float(np.max(mass))

    if vmin <= 0.0:
        raise ValueError(f'vmin must be > 0 for LogNorm, got {vmin}.')
    if vmax <= vmin:
        raise ValueError(
            f'vmax must be > vmin for LogNorm, got vmin={vmin}, vmax={vmax}.'
        )

    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    return cmap(norm(mass))


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
        ``R_3D`` in ``validation/glass/config.env``.
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


def _draw_panel_3d(
    ax: Axes3D,
    pos: ArrayF,
    mass: ArrayF,
    title: str,
    *,
    fraction: float = 0.08,
    point_size: float = 0.5,
    point_color: str | None = None,
    point_vmin: float | None = None,
    point_vmax: float | None = None,
    point_alpha: float = 0.25,
    elev: float = 22.0,
    azim: float = -42.0,
) -> int:
    """Draw a 3D scatter of a subsampled particle distribution."""
    n_total = pos.shape[0]
    n_keep = max(1, int(n_total * fraction))

    if n_keep >= n_total:
        idx = np.arange(n_total)
    else:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_total, size=n_keep, replace=False)

    sub = pos[idx]
    sub_mass = mass[idx]
    n_plot = sub.shape[0]

    if point_color is None:
        if np.allclose(sub_mass, sub_mass[0]):
            point_color = 'k'
        else:
            point_color = log_mass_to_colors(
                sub_mass,
                vmin=point_vmin,
                vmax=point_vmax,
            )

    ax.scatter(
        sub[:, 0], sub[:, 1], sub[:, 2],
        s=point_size,
        c=point_color,
        alpha=point_alpha,
        edgecolors='none',
        rasterized=True,
        depthshade=True,
    )

    ax.view_init(elev=elev, azim=azim)

    # Clean up axes - for this figure, the *shape* is the message.
    # Tick labels clutter 3D panels at publication scale.
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(axis='both', which='both', length=0, pad=0)

    # Subtle pane styling: white panes, light grid
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

    return n_plot


def _set_equal_aspect_3d(ax: Axes3D, pos: ArrayF, pad: float = 0.05) -> None:
    '''Force equal aspect ratio on a 3D axes by setting matching limits.

    Parameters
    ----------
    ax : Axes3D
        The 3D axes to adjust.
    pos : ndarray, shape (N, 3)
        Particle positions (full set, not subsampled) for extent computation.
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
    '''Set 3D limits for a cylindrical geometry.

    Uses the physical aspect ratio of the cylinder so that a squat
    cylinder looks squat. The x-y extent is set by ``r_max`` and the
    z extent by ``[z_min, z_max]``.

    Parameters
    ----------
    ax : Axes3D
        The 3D axes.
    pos : ndarray, shape (N, 3)
        Particle positions (unused; kept for API symmetry with
        ``_set_equal_aspect_3d``).
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


def plot_glasses(
    paths: dict[str, str | None],
    output: str | None = None,
    *,
    fraction: float = 0.08,
    elev: float = 22.0,
    azim: float = -42.0,
    target_radius: float = 250.0,
) -> None:
    r'''Produce the 4-panel 3D glass validation figure.

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
    target_radius : float
        Physical radius [Mpc] used to rescale spherical and cylindrical
        glass back to their intended domain after StePS expansion.
        Must match ``R_3D`` in ``validation/glass/config.env``.
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
    fig_height = panel_size + 0.15  # 3D panels need a bit more vertical room

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=200)

    for i, (key, title) in enumerate(zip(panel_keys, panel_titles)):
        path = paths.get(key)
        if path is None or not Path(path).exists():
            log.warning('Skipping panel %s (file not provided or missing).', key)
            continue

        ax = fig.add_subplot(1, n_panels, i + 1, projection='3d')
        pos, mass = load_glass(path)
        n_total = pos.shape[0]

        # Rescale non-periodic geometries back to their physical domain.
        # StePS writes L_BOX = 0 for non-periodic runs (it is unused), so
        # the header is not a usable anchor - we use target_radius instead.
        if key == 'spherical':
            pos = rescale_to_radius(pos, target_radius, 'spherical')
        elif key == 'cylindrical':
            pos = rescale_to_radius(pos, target_radius, 'cylindrical')

        # Set axis limits based on geometry
        if key in ('cubic_random', 'cubic_grid', 'spherical'):
            _set_equal_aspect_3d(ax, pos)

        elif key == 'cylindrical':
            centre = pos.mean(axis=0)
            dx = pos[:, 0] - centre[0]
            dy = pos[:, 1] - centre[1]
            r_max = float(np.max(np.sqrt(dx**2 + dy**2)))
            _set_cylinder_aspect(
                ax, pos, r_max,
                z_min=float(pos[:, 2].min()),
                z_max=float(pos[:, 2].max()),
                xy_centre=centre[:2],
            )

        else:
            raise ValueError(f'Unknown panel key: {key}')

        n_plotted = _draw_panel_3d(
            ax, pos, mass, title,
            fraction=fraction,
            elev=elev,
            azim=azim,
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
        description='Plot 4-panel 3D glass validation figure for the '
                    'stepsic paper.',
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
             'Must match R_3D in validation/glass/config.env.',
    )
    p.add_argument(
        '--fraction', type=float, default=0.08,
        help='Fraction of particles to display per panel (random subsample).',
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
        elev=args.elev,
        azim=args.azim,
        target_radius=args.target_radius,
    )