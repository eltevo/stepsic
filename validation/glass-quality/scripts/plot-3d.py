#!/usr/bin/env python3
'''
Plot three-dimensional particle samples from the available glasses.

Each panel shows a random sample from one glass geometry, coloured by particle
mass where the masses differ, so the heavier outer shells stand out:

    (a) Cubical glass (from random pre-glass)
    (b) Cubical glass (from grid pre-glass)
    (c) Spherical glass
    (d) Cylindrical glass

Usage
-----
::

    PYTHONPATH="$PWD" conda run -n stepsic python validation/glass-quality/scripts/plot-3d.py \\
        --cubic-random validation/glass-quality/runs/medium/cache/cubic_random/glass/snapshot_0001.hdf5 \\
        --cubic-grid   validation/glass-quality/runs/medium/cache/cubic_grid/preglass/ic.hdf5 \\
        --spherical    validation/glass-quality/runs/medium/cache/spherical/glass/snapshot_0001.hdf5 \\
        --cylindrical  validation/glass-quality/runs/medium/cache/cylindrical/glass/snapshot_0001.hdf5 \\
        --target-radius 500.0 \\
        -o validation/glass-quality/runs/medium/output/supplementary/glass-generation-3d.pdf

Any subset of panels can be omitted; missing panels are left blank.
The ``--target-radius`` value must match ``R_3D`` in ``validation/glass-quality/config.env``
(default 500 Mpc).
'''

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import numpy as np

from plot import load_glass, rescale_to_radius
from validation._common.plotting import (
    atomic_savefig,
    draw_panel_3d,
    set_cylinder_aspect,
    set_equal_aspect_3d,
    setup_matplotlib,
    subsample,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


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
        Must match ``R_3D`` in ``validation/glass-quality/config.env``.
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

        if key == 'cylindrical':
            set_cylinder_aspect(ax, pos)
        else:
            set_equal_aspect_3d(ax, pos)

        sub, sub_mass = subsample(pos, fraction, masses=mass)
        if np.allclose(sub_mass, sub_mass[0]):
            colors = 'k'
        else:
            norm = LogNorm(vmin=sub_mass.min(), vmax=sub_mass.max())
            colors = plt.get_cmap('plasma_r')(norm(sub_mass))
        draw_panel_3d(ax, sub, title, sizes=0.5, colors=colors, elev=elev, azim=azim)
        n_plotted = sub.shape[0]

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
        description='Plot three-dimensional particle samples from the available glasses.',
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
