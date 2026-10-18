#!/usr/bin/env python3
'''
Plot particle mass across radial shells for spherical and cylindrical geometries.

Reads the archive written by ``measure.py``. Each panel compares
constant-angle (``omega``) and constant-volume (``volume``) bins:

* **Top panel** - spherical geometry (R^3 via stereographic projection
  from S^3): particle mass as a function of radial distance from the
  centre.
* **Bottom panel** - cylindrical geometry (R^2 x R via stereographic
  projection from S^1 x R^2): particle mass as a function of radial
  distance from the central axis.

Usage
-----
::

    python plot.py --archive shell-mass.npz -o shell-mass-profiles.pdf
'''

from __future__ import annotations

import argparse
import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


from validation._common.artifacts import load_archive
from validation._common.plotting import atomic_savefig, setup_matplotlib


AA_COL_WIDTH = 3.5


def _load_profiles(data: dict[str, np.ndarray]) -> dict[str, dict[str, dict]]:
    return {
        "sph": {
            mode: {
                "masses": data[f"spherical_{mode}_mass_internal"],
                "edges": data[f"spherical_{mode}_edges_mpc_h"],
            }
            for mode in ("omega", "volume")
        },
        "cyl": {
            mode: {
                "masses": data[f"cylindrical_{mode}_mass_internal"],
                "edges": data[f"cylindrical_{mode}_edges_mpc_h"],
            }
            for mode in ("omega", "volume")
        },
    }


def plot_shell_mass(*, archive: str, output: str) -> None:
    '''Plot the two-panel shell-mass figure from a measure.py archive.'''
    import matplotlib.pyplot as plt

    setup_matplotlib()

    data = load_archive(archive)
    results = _load_profiles(data)
    r3d = float(data['meta_r3d_mpc_h'])

    # -- Plot -----------------------------------------------------------------
    fig, axes = plt.subplots(
        2, 1,
        figsize=(AA_COL_WIDTH, 2 * 1.1 + 0.35),
        sharex=True,
        gridspec_kw={'hspace': 0.15},
    )

    style = {
        'omega':  {'edgecolor': '#0072B2', 'linestyle': '-',
                   'label': r'constant $\Delta\omega$ binning'},
        'volume': {'edgecolor': '#D55E00', 'linestyle': '--',
                   'label': r'constant volume binning'},
    }

    for ax, geom_tag, title in zip(
        axes,
        ('sph', 'cyl'),
        (r'(a) Spherical',
         r'(b) Cylindrical'),
    ):
        for mode in ('omega', 'volume'):
            d = results[geom_tag][mode]
            ax.stairs(
                d['masses'],
                d['edges'],
                linewidth=1.0,
                fill=False,
                **style[mode],
            )
        ax.set_xlim(0, r3d)
        ax.set_yscale('log')
        ax.set_ylim(1e-1, 1e5)
        if geom_tag == 'sph':
            ax.set_title(
                r'$m_\mathrm{particle}$ [$10^{11}\,M_\odot$]',
                loc='left', fontsize=9
            )
        if geom_tag == 'cyl':
            ax.set_xlabel(r'$r$ [Mpc/$h$]')
        ax.text(
            0.03, 0.91, title,
            transform=ax.transAxes, fontsize=8,
            va='top', ha='left',
        )
        ax.legend(fontsize=7, loc='lower right', frameon=False)

    fig.tight_layout(h_pad=0.6)
    atomic_savefig(fig, output, dpi=300, bbox_inches='tight')
    log.info('Figure saved to %s', output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--archive', type=str, required=True,
        help='Shell-mass archive produced by measure.py.',
    )
    parser.add_argument('-o', '--output', type=str, required=True)
    args = parser.parse_args()
    plot_shell_mass(archive=args.archive, output=args.output)


if __name__ == '__main__':
    main()
