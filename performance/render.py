"""Render the computational-performance figure."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from performance.benchmark import (
    GRID_SIZES, INITIAL_REPEATS, LPT_ORDERS, MAX_TRIALS, is_valid_run,
)
from validation._common.plotting import atomic_savefig, setup_matplotlib


REPOSITORY = Path(__file__).resolve().parents[1]
GIB = 1024**3
METRICS = ('wall_seconds', 'peak_rss_bytes')


def summarize_runs(runs: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    """Return the median and range of each metric for the complete matrix."""
    summary = {}
    for order in LPT_ORDERS:
        for nmesh in GRID_SIZES:
            selected = [
                run for run in runs
                if run['lpt_order'] == order and run['nmesh'] == nmesh
                and is_valid_run(run)
            ]
            if not INITIAL_REPEATS <= len(selected) <= MAX_TRIALS:
                raise ValueError(
                    f'{order}LPT {nmesh}^3 has {len(selected)} valid trials'
                )
            summary[(order, nmesh)] = {'trials': len(selected)}
            for metric in METRICS:
                values = [run[metric] for run in selected]
                summary[(order, nmesh)][metric] = (
                    statistics.median(values), min(values), max(values),
                )
    return summary


def render_figure(
    artifact: dict[str, Any], summary: dict[tuple[int, int], dict[str, Any]], path: Path,
) -> None:
    """Draw wall-time and peak-RSS scaling as a single-column figure."""
    setup_matplotlib()
    plt.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 8,
        'legend.fontsize': 7,
    })
    figure, axes = plt.subplots(2, 1, figsize=(3.46, 3.2), sharex=True)
    colors = {1: '#0072B2', 2: '#D55E00'}
    markers = {1: 'o', 2: 's'}
    nmesh = np.asarray(GRID_SIZES, dtype=np.float64)
    panels = (
        ('wall_seconds', 1, 'Wall time [s]',
         nmesh**3 * np.log2(nmesh), r'$N^3\log_2 N$ reference'),
        ('peak_rss_bytes', GIB, 'Peak RSS [GiB]', nmesh**3, r'$N^3$ reference'),
    )
    for axis, letter, (metric, unit, label, guide, guide_label) in zip(axes, 'ab', panels):
        for order in LPT_ORDERS:
            median, low, high = np.asarray(
                [summary[(order, size)][metric] for size in GRID_SIZES]
            ).T / unit
            axis.errorbar(
                nmesh, median, yerr=np.vstack((median - low, high - median)),
                color=colors[order], marker=markers[order], markersize=3.2,
                linewidth=1.4, elinewidth=.65, capsize=1.5,
                label=f'{order}LPT',
            )
        anchor = GRID_SIZES.index(256)
        axis.plot(
            nmesh, guide * summary[(1, 256)][metric][0] / unit / guide[anchor],
            color='0.45', linestyle='--', linewidth=1.1, label=guide_label,
        )
        axis.set_title(label, loc='left', fontsize=8)
        axis.text(.03, .91, f'({letter})', transform=axis.transAxes, va='top', ha='left')
        axis.legend(
            frameon=False, loc='lower right', handlelength=1.5, handletextpad=.5,
        )
    for axis in axes:
        axis.set_xscale('log', base=2)
        axis.set_yscale('log')
        axis.set_xticks(GRID_SIZES, labels=[str(value) for value in GRID_SIZES])
    axes[1].set_xlabel(r'$N=N_{\rm grid}=N_{\rm mesh}$')
    figure.tight_layout(h_pad=.4)
    recorded = datetime.fromisoformat(
        artifact['provenance']['recorded_at_utc']
    )
    atomic_savefig(
        figure,
        path,
        metadata={
            'Title': 'STEPSIC computational performance and memory footprint',
            'Creator': 'performance.render',
            'CreationDate': recorded,
            'ModDate': recorded,
        },
    )
    plt.close(figure)


def render(results_path: Path, figure_path: Path) -> None:
    with results_path.resolve().open(encoding='utf-8') as stream:
        artifact = json.load(stream)
    summary = summarize_runs(artifact['runs'])
    render_figure(artifact, summary, figure_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Render the STEPSIC computational-performance figure.',
    )
    parser.add_argument(
        '--results', type=Path,
        default=REPOSITORY / 'performance/results/benchmark.json',
    )
    parser.add_argument(
        '--figure', type=Path,
        default=REPOSITORY / 'performance/figures/computational-performance.pdf',
    )
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    render(arguments.results, arguments.figure)


if __name__ == '__main__':
    main()
