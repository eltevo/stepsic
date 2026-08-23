#!/usr/bin/env python3
'''
Plot the particle distributions generated for each geometry.

The four panels show loads generated with ``LPTORDER=0``:

    (a) Cubical, random (Poisson)
    (b) Cubical, grid (regular lattice)
    (c) Spherical shells
    (d) Cylindrical shells

By default, each panel shows a random three-dimensional sample. ``--plot-2d`` instead projects a thin slice. A regular cubic grid uses only the layer nearest the slice centre so adjacent layers do not overlap.

Usage
-----
::

    python plot.py \\
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

from validation._common.plotting import atomic_savefig, setup_matplotlib

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

ArrayF: TypeAlias = NDArray[np.float64]


def load_particles(
    path: str, part_type: int = 1,
) -> tuple[ArrayF, ArrayF | None]:
    '''Read particle positions and masses from a Gadget-format HDF5 snapshot.

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
    masses : ndarray, shape (N,) or None
        Per-particle masses, if present in the snapshot.
    '''
    with h5py.File(path, 'r') as f:
        grp = f[f'PartType{part_type}']
        pos = grp['Coordinates'][:]
        masses = grp['Masses'][:] if 'Masses' in grp else None
    log.info('Loaded %d particles from %s', pos.shape[0], path)
    masses_f = masses.astype(np.float64) if masses is not None else None
    return pos.astype(np.float64), masses_f


def subsample(
    pos: ArrayF,
    fraction: float,
    *,
    seed: int = 42,
    masses: ArrayF | None = None,
) -> tuple[ArrayF, ArrayF | None]:
    '''Return a random subsample of particle positions (and masses).

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Full particle array.
    fraction : float
        Fraction of particles to keep, in (0, 1].
    seed : int
        RNG seed for reproducibility.
    masses : ndarray or None
        Optional per-particle masses, subsampled with the same indices.

    Returns
    -------
    pos_sub : ndarray, shape (M, 3)
        Subsampled positions.
    masses_sub : ndarray or None
        Subsampled masses, or ``None`` if ``masses`` is ``None``.
    '''
    n_total = pos.shape[0]
    n_keep = max(1, int(n_total * fraction))
    if n_keep >= n_total:
        return pos, masses
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n_keep, replace=False)
    return pos[idx], (masses[idx] if masses is not None else None)


def _mass_sizes(
    masses: ArrayF,
    *,
    s_min: float = 0.01,
    s_max: float = 2.5,
) -> ArrayF:
    r'''Compute per-particle scatter sizes that grow with particle mass.

    Sizes are linearly interpolated between ``s_min`` at the smallest
    log-mass and ``s_max`` at the largest log-mass. For shell-based
    geometries with a uniform-resolution interior (the RCRIT zone),
    every interior particle carries the same mass, so this scheme
    renders the interior with a single, uniform marker size while
    outer shells grow logarithmically with the per-shell mass.

    Parameters
    ----------
    masses : ndarray, shape (N,)
        Per-particle masses (already subsampled).
    s_min, s_max : float
        Marker area at the smallest and largest log-mass respectively
        (matplotlib ``s`` units, i.e. points²).

    Returns
    -------
    sizes : ndarray, shape (N,)
        Per-particle marker sizes.
    '''
    log_m = np.log10(masses)
    log_min = log_m.min()
    log_max = log_m.max()
    if log_max == log_min:
        return np.full(masses.shape[0], 0.5 * (s_min + s_max))
    t = (log_m - log_min) / (log_max - log_min)
    return s_min + (s_max - s_min) * t


def _slice_axis_to_int(axis: str) -> int:
    '''Convert a slice-axis label ('x', 'y', 'z') to a coordinate index.'''
    try:
        return {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
    except KeyError as exc:
        raise ValueError(
            f'Unknown slice axis {axis!r}; expected one of x, y, z.'
        ) from exc


def _select_slice(
    pos: ArrayF,
    masses: ArrayF | None,
    *,
    axis: int,
    thickness: float,
    centre: float,
    grid_layer: bool = False,
) -> tuple[ArrayF, ArrayF | None]:
    '''Select particles inside a slab (or single grid layer) of a 3D snapshot.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Particle positions.
    masses : ndarray or None
        Per-particle masses, sliced with the same mask.
    axis : int
        Coordinate index normal to the slice (0=x, 1=y, 2=z).
    thickness : float
        Slab thickness along ``axis`` [Mpc/h]. Ignored if ``grid_layer``.
    centre : float
        Position of the slice centre along ``axis`` [Mpc/h].
    grid_layer : bool
        If ``True``, snap to the unique value of ``pos[:, axis]`` closest
        to ``centre`` and select that layer only.

    Returns
    -------
    pos_sub, masses_sub
        Selected positions and (optional) masses.
    '''
    s = pos[:, axis]
    if grid_layer:
        unique_vals = np.unique(s)
        chosen = unique_vals[np.argmin(np.abs(unique_vals - centre))]
        mask = s == chosen
    else:
        mask = np.abs(s - centre) <= 0.5 * thickness
    return pos[mask], (masses[mask] if masses is not None else None)


def _draw_panel_2d(
    ax: plt.Axes,
    pos_full: ArrayF,
    title: str,
    *,
    slice_axis: int,
    slice_thickness: float,
    grid_layer: bool,
    fraction: float = 1.0,
    point_size: float = 1.2,
    point_color: str = 'k',
    point_alpha: float = 0.55,
    geometry: str | None = None,
    masses: ArrayF | None = None,
    pad: float = 0.02,
) -> int:
    '''Draw a 2D slice of a particle distribution onto a 2D axis.

    Parameters
    ----------
    ax : Axes
        Target 2D axis.
    pos_full : ndarray, shape (N, 3)
        Full particle array (used for both slicing and panel limits).
    slice_axis : int
        Axis normal to the slice (0=x, 1=y, 2=z).
    slice_thickness : float
        Slab thickness [Mpc/h]. Ignored when ``grid_layer`` is ``True``.
    grid_layer : bool
        If ``True`` (cubic grid panel), pick a single grid layer instead
        of a slab so the regular lattice pattern is visible.
    geometry : {'spherical', 'cylindrical'} or None
        Enables mass-based marker sizing for the shell geometries.

    Returns the number of particles drawn.
    '''
    s = pos_full[:, slice_axis]
    centre = 0.5 * (s.min() + s.max())

    sliced, sliced_m = _select_slice(
        pos_full, masses,
        axis=slice_axis,
        thickness=slice_thickness,
        centre=centre,
        grid_layer=grid_layer,
    )

    # Subsample only when not pinned to a single grid layer; otherwise
    # we'd punch holes in the regular lattice we're trying to show.
    if not grid_layer:
        sliced, sliced_m = subsample(sliced, fraction, masses=sliced_m)

    n_plot = sliced.shape[0]

    plot_axes = [a for a in (0, 1, 2) if a != slice_axis]
    x = sliced[:, plot_axes[0]]
    y = sliced[:, plot_axes[1]]

    if geometry in ('spherical', 'cylindrical'):
        if sliced_m is None:
            raise ValueError(
                f'Mass-based sizing requested for {geometry!r} panel '
                'but the snapshot has no Masses dataset.'
            )
        # Larger size range than 3D: 2D slices are sparser, so dots can
        # afford to be bigger without saturating the panel.
        sizes = _mass_sizes(sliced_m, s_min=0.6, s_max=8.0)
    else:
        sizes = point_size

    ax.scatter(
        x, y,
        s=sizes,
        c=point_color,
        alpha=point_alpha,
        edgecolors='none',
        rasterized=True,
    )

    # Square panel from the full particle extent (not just the slab).
    fx = pos_full[:, plot_axes[0]]
    fy = pos_full[:, plot_axes[1]]
    cx = 0.5 * (fx.min() + fx.max())
    cy = 0.5 * (fy.min() + fy.max())
    half = 0.5 * max(fx.max() - fx.min(), fy.max() - fy.min()) * (1.0 + pad)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('0.70')
        spine.set_linewidth(0.5)

    ax.text(
        0.03, 0.97, title,
        transform=ax.transAxes, fontsize=7,
        va='top', ha='left',
        bbox=dict(
            facecolor='white', edgecolor='none', alpha=0.80, pad=1.5,
        ),
    )

    return n_plot


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
    masses: ArrayF | None = None,
) -> int:
    '''Draw a 3D scatter of a subsampled particle distribution.

    Parameters
    ----------
    geometry : str or None
        If ``'spherical'`` or ``'cylindrical'``, marker sizes scale
        with per-particle ``log10(mass)`` (bigger dots for heavier
        outer-shell particles, uniform inside the RCRIT zone).
        If ``None`` (cubical panels), a uniform ``point_size`` is used.
    masses : ndarray or None
        Per-particle masses; required when ``geometry`` enables
        mass-based sizing.

    Returns the number of particles plotted.
    '''
    sub, sub_masses = subsample(pos, fraction, masses=masses)
    n_plot = sub.shape[0]

    if geometry in ('spherical', 'cylindrical'):
        if sub_masses is None:
            raise ValueError(
                f'Mass-based sizing requested for {geometry!r} panel '
                'but the snapshot has no Masses dataset.'
            )
        sizes = _mass_sizes(sub_masses)
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
    '''Set shape-faithful 3D limits for a cylindrical geometry.

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
    plot_2d: bool = False,
    slice_axis: str = 'z',
    slice_thickness: float = 25.0,
) -> None:
    r'''Produce the 4-panel particle load validation figure.

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
        In 2D mode the slab itself already culls most particles, so
        the default is raised to 1.0 (no further subsampling).
    elev, azim : float
        3D viewing elevation and azimuth angles [degrees]. Unused in
        2D mode.
    plot_2d : bool
        If ``True``, render 2D slices instead of 3D scatter views.
    slice_axis : {'x', 'y', 'z'}
        Coordinate normal to the 2D slice. Only used when
        ``plot_2d`` is ``True``.
    slice_thickness : float
        Slab thickness along ``slice_axis`` [Mpc/h]. The ``cubic_grid``
        panel ignores this and snaps to a single grid layer to show
        the regular lattice pattern.
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

    slice_axis_int = _slice_axis_to_int(slice_axis) if plot_2d else None

    # Map panel key -> geometry for radial marker sizing.
    # Cubical panels get None (uniform size).
    _geom_map = {
        'spherical': 'spherical',
        'cylindrical': 'cylindrical',
    }

    for i, (key, title) in enumerate(zip(panel_keys, panel_titles)):
        path = paths.get(key)
        if path is None or not Path(path).exists():
            log.warning(
                'Skipping panel %s (file not provided or missing).', key,
            )
            continue

        pos, masses = load_particles(path)
        n_total = pos.shape[0]

        if plot_2d:
            ax = fig.add_subplot(1, n_panels, i + 1)
            n_plotted = _draw_panel_2d(
                ax, pos, title,
                slice_axis=slice_axis_int,
                slice_thickness=slice_thickness,
                grid_layer=(key == 'cubic_grid'),
                fraction=fraction,
                geometry=_geom_map.get(key),
                masses=masses,
            )
        else:
            ax = fig.add_subplot(1, n_panels, i + 1, projection='3d')

            if key in ('cubic_random', 'cubic_grid', 'spherical'):
                _set_equal_aspect_3d(ax, pos)
            elif key == 'cylindrical':
                centre = pos.mean(axis=0)
                dx = pos[:, 0] - centre[0]
                dy = pos[:, 1] - centre[1]
                r_max = float(np.max(np.sqrt(dx**2 + dy**2)))
                _set_cylinder_aspect(
                    ax, pos, r_max,
                    z_min=float(np.min(pos[:, 2])),
                    z_max=float(np.max(pos[:, 2])),
                    xy_centre=centre[:2],
                )
            else:
                raise ValueError(f'Unknown panel key: {key}')

            n_plotted = _draw_panel_3d(
                ax, pos, title,
                fraction=fraction,
                elev=elev,
                azim=azim,
                geometry=_geom_map.get(key),
                masses=masses,
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
        atomic_savefig(fig, output, bbox_inches='tight', dpi=300)
        log.info('Figure saved to %s', output)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Plot the particle distribution generated for each geometry.',
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
        '--fraction', type=float, default=None,
        help='Fraction of particles to display per panel. '
             'Default: 0.08 in 3D mode, 1.0 in 2D mode (the slab '
             'already culls most particles).',
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
        '--plot-2d', action='store_true',
        help='Render 2D slices instead of 3D scatter views. '
             'The cubic_grid panel snaps to a single grid layer; the '
             'three other panels show a slab of --slice-thickness.',
    )
    p.add_argument(
        '--slice-axis', type=str, default='x', choices=['x', 'y', 'z'],
        help='Coordinate normal to the 2D slice (only used with --plot-2d).',
    )
    p.add_argument(
        '--slice-thickness', type=float, default=25.0,
        help='Slab thickness along --slice-axis [Mpc/h]. '
             'The cubic_grid panel ignores this and uses one grid layer.',
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

    fraction = args.fraction
    if fraction is None:
        fraction = 1.0 if args.plot_2d else 0.08

    plot_particle_loads(
        load_paths,
        output=args.output,
        fraction=fraction,
        elev=args.elev,
        azim=args.azim,
        plot_2d=args.plot_2d,
        slice_axis=args.slice_axis,
        slice_thickness=args.slice_thickness,
    )
