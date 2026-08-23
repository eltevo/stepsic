#!/usr/bin/env python3
r'''
Plot transverse and longitudinal displacement in a slab.

The figure shows the suppression of :math:`|\Psi_z|` relative to :math:`|\Psi_{x,y}|` in a non-cubic box.

Because :math:`L_x = L_y`, the x and y histograms are averaged into one transverse curve. The z histogram gives the longitudinal curve.

Usage
-----
::

    python validate-slab-anisotropy-plot.py \
        -i output/fields_slab.npz \
        -o output/validation-slab-anisotropy.pdf
'''

from __future__ import annotations



import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from validation._common.plotting import atomic_savefig, setup_matplotlib
from validation._common.artifacts import load_archive

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _step_hist(
    ax: plt.Axes,
    centres: NDArray,
    counts: NDArray,
    widths: NDArray,
    **kwargs,
) -> None:
    '''Step-style histogram, trimming zero-count tails for log-y.'''
    edges = np.empty(centres.size + 1, dtype=np.float64)
    edges[:-1] = centres - 0.5 * widths
    edges[-1] = centres[-1] + 0.5 * widths[-1]

    pos = counts > 0.0
    if not np.any(pos):
        return
    idx = np.flatnonzero(pos)
    i0, i1 = int(idx[0]), int(idx[-1]) + 1
    ax.stairs(counts[i0:i1], edges[i0:i1 + 1], **kwargs)


def plot_slab_anisotropy(
    data: dict,
    output: str | None = None,
) -> None:
    r'''Single-panel slab displacement anisotropy figure.

    Parameters
    ----------
    data : dict
        Archive contents from :func:`load_archive`.
    output : str or None
        Save path. Shows interactively if ``None``.
    '''
    setup_matplotlib()

    lbox = data['meta_lbox']
    nvox = data['meta_nvox']
    nmesh = int(data['meta_nmesh'])
    lpt_order = int(data['meta_lpt_order'])
    z = float(data['meta_redshift'])
    method = str(data['meta_method']).upper()
    n_total = int(data['meta_n_total'])

    # -- Build the transverse histogram (average of x and y) --
    # x, y, z share the same bin edges because they belong to the
    # disp_components group in validate-fields-run.py.
    centres = data['hist_disp_x_centres']
    widths = data['hist_disp_x_widths']
    hx = data['hist_disp_x_counts']
    hy = data['hist_disp_y_counts']
    hz = data['hist_disp_z_counts']
    h_perp = 0.5 * (hx + hy)  # exact x <-> y symmetry for L_x = L_y

    # Verify bins match
    assert np.allclose(centres, data['hist_disp_y_centres']), \
        'x and y bin centres differ - archive was built with separate groups?'
    assert np.allclose(centres, data['hist_disp_z_centres']), \
        'x and z bin centres differ - archive was built with separate groups?'

    # -- Variance ratio from the histogram (approximate) --
    # Var(|X|) for half-normal: E[X^2] = Var(X_signed) = sum(c^2 * h) / sum(h)
    def _hist_variance(c, h):
        total = np.sum(h)
        if total == 0:
            return 0.0
        return np.sum(c**2 * h) / total

    var_perp = _hist_variance(centres, h_perp)
    var_z = _hist_variance(centres, hz)
    ratio_var = var_perp / var_z if var_z > 0 else np.inf

    log.info(
        'Histogram variance:  Var(⊥) = %.2f,  Var(∥) = %.2f,  '
        'Var(⊥)/Var(∥) = %.2f',
        var_perp, var_z, ratio_var,
    )

    # -- Figure --
    fig_width = 3.46   # A&A single-column
    fig_height = 2.0
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    _step_hist(
        ax, centres, h_perp, widths,
        color='#0072B2', lw=1.2,
        label=r'$|\Psi_\perp|$ (transverse)',
    )
    _step_hist(
        ax, centres, hz, widths,
        color='#D55E00', lw=1.2,
        label=r'$|\Psi_\parallel|$ (longitudinal)',
    )

    ax.set_yscale('log')
    ax.set_xlabel(
        r'$|\Psi_i|\;[h^{-1}\,\mathrm{kpc}]$',
        fontsize=9,
    )
    ax.set_title(
        r'Mean counts $\langle N \rangle$',
        loc='left', fontsize=9,
    )

    # Annotate geometry info
    ax.legend(fontsize=7, loc='upper right', frameon=False)
    fig.tight_layout()

    if output:
        atomic_savefig(fig, output, bbox_inches='tight')
        log.info('Figure saved to %s', output)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Plot slab displacement anisotropy (⊥ vs ∥).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '-i', '--input', type=str, required=True,
        help='Input .npz archive from validate-fields-run.py.',
    )
    p.add_argument(
        '-o', '--output', type=str, default=None,
        help='Output figure path (e.g. slab_anisotropy.pdf).',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data = load_archive(args.input)

    lbox = data['meta_lbox']
    nvox = data['meta_nvox']
    log.info(
        'Loaded %s: box = %.0fx%.0fx%.0f, mesh = %dx%dx%d, '
        '%dLPT %s z=%g, %d evaluations',
        args.input,
        lbox[0], lbox[1], lbox[2],
        nvox[0], nvox[1], nvox[2],
        int(data['meta_lpt_order']),
        str(data['meta_method']).upper(),
        float(data['meta_redshift']),
        int(data['meta_n_total']),
    )

    plot_slab_anisotropy(data, output=args.output)
