"""Measure end-to-end STEPSIC IC generation time and peak memory."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import scipy
import toml


REPOSITORY = Path(__file__).resolve().parents[1]
GRID_SIZES = (64, 128, 256, 512)
LPT_ORDERS = (1, 2)
INITIAL_REPEATS = 3
MAX_TRIALS = 5
SCHEDULE_SEED = 20260918
SPREAD_LIMIT = 0.05
MINIMUM_AVAILABLE_BYTES = 16 * 1024**3
MAXIMUM_BUSY_FRACTION = 0.10
THREAD_VARIABLES = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'BLIS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'NUMEXPR_MAX_THREADS',
)
# Fixed publication workload; NGRID = NMESH and LPTORDER vary per run.
WORKLOAD = {
    'GEOMETRY': 'cubical',
    'LBOX': [1000, 1000, 1000],
    'PERIODIC': [True, True, True],
    'R_3D': 500,
    'D_4D': 75,
    'COI': [0, 0, 0],
    'TYPE': 'grid',
    'USE_DOUBLE': False,
    'SEED': 42,
    'COMOVING': True,
    'HINDEPENDENT': True,
    'ROTATE': 0.0,
    'REDSHIFT': 31,
    'INTERPOLATION': 'cic',
    'COMPENSATE': False,
    'SPHEREMODE': False,
    'PAIRED': False,
    'FIXED': False,
    'IC_PREFIX': 'performance',
    'IC_FORMAT': 'hdf5',
    'SAVE_WHITE_NOISE': False,
    'COSMOLOGY': 'Planck2018EE+BAO',
    'SPECTRUM': 'camb',
    'RESCALE_SIGMA8': True,
    'NONLINEAR': False,
    'UNIT_L_IN_CM': 3.085678e24,
    'UNIT_M_IN_G': 1.989e44,
    'UNIT_V_IN_KMPS': 20.738652969844207,
}


def scheduled_runs() -> list[tuple[int, int, int]]:
    """Return the deterministic shuffled order for the initial trials."""
    schedule = [
        (lpt_order, nmesh, repeat)
        for repeat in range(1, INITIAL_REPEATS + 1)
        for lpt_order in LPT_ORDERS
        for nmesh in GRID_SIZES
    ]
    random.Random(SCHEDULE_SEED).shuffle(schedule)
    return schedule


def parse_time_report(report: str) -> dict[str, int | float]:
    """Parse the fields retained from an English GNU ``time -v`` report."""
    patterns = {
        'elapsed': r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)',
        'user_seconds': r'User time \(seconds\): (\S+)',
        'system_seconds': r'System time \(seconds\): (\S+)',
        'maximum_rss_kib': r'Maximum resident set size \(kbytes\): (\d+)',
        'major_page_faults': r'Major \(requiring I/O\) page faults: (\d+)',
        'swaps': r'Swaps: (\d+)',
        'exit_status': r'Exit status: (\d+)',
    }
    values = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, report)
        if match is None:
            raise ValueError(f'GNU time report lacks {name}')
        values[name] = match.group(1)
    # h:mm:ss or m:ss, read from the seconds field upwards
    wall_seconds = sum(
        float(field) * 60**power
        for power, field in enumerate(reversed(values['elapsed'].split(':')))
    )
    return {
        'wall_seconds': wall_seconds,
        'user_seconds': float(values['user_seconds']),
        'system_seconds': float(values['system_seconds']),
        'peak_rss_bytes': int(values['maximum_rss_kib']) * 1024,
        'major_page_faults': int(values['major_page_faults']),
        'swaps': int(values['swaps']),
        'reported_exit_status': int(values['exit_status']),
    }


def snapshot_arrays(directory: Path) -> list[tuple[tuple[int, ...], str]]:
    """Return shape and dtype of the positions and velocities in the one snapshot."""
    paths = list(directory.glob('*/ic.hdf5'))
    if len(paths) != 1:
        raise ValueError(f'expected one snapshot, found {len(paths)}')
    with h5py.File(paths[0], 'r') as snapshot:
        return [
            (snapshot[name].shape, snapshot[name].dtype.name)
            for name in ('PartType1/Coordinates', 'PartType1/Velocities')
        ]


def is_valid_run(run: dict[str, Any]) -> bool:
    return 'validation_error' not in run


def needs_more_trials(
    runs: list[dict[str, Any]], lpt_order: int, nmesh: int,
) -> bool:
    """Return whether a workload needs another trial under the 5% rule."""
    values = [
        run['wall_seconds'] for run in runs
        if run['lpt_order'] == lpt_order and run['nmesh'] == nmesh
        and is_valid_run(run)
    ]
    if not INITIAL_REPEATS <= len(values) < MAX_TRIALS:
        return False
    return max(values) - min(values) > SPREAD_LIMIT * statistics.median(values)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace a JSON artifact without exposing a partially written file."""
    temporary = path.with_name(f'.{path.name}.tmp')
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2) + '\n', encoding='utf-8',
    )
    os.replace(temporary, path)


def _read_key_values(path: Path, separator: str = ':') -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if separator in line:
            key, value = line.split(separator, 1)
            values[key.strip()] = value.strip().strip('"')
    return values


def _lscpu() -> dict[str, str]:
    completed = subprocess.run(
        ['lscpu'], check=True, capture_output=True, text=True,
    )
    return {
        key.strip(): value.strip()
        for line in completed.stdout.splitlines()
        if ':' in line
        for key, value in [line.split(':', 1)]
    }


def _git(*arguments: str) -> str:
    return subprocess.run(
        ['git', *arguments], cwd=REPOSITORY, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return 'not installed'


def _meminfo() -> dict[str, int]:
    values = {}
    for line in Path('/proc/meminfo').read_text(encoding='ascii').splitlines():
        name, raw = line.split(':', 1)
        fields = raw.split()
        if fields:
            values[name] = int(fields[0]) * 1024
    return values


def _vmstat_swap_pages() -> tuple[int, int]:
    values = _read_key_values(Path('/proc/vmstat'), separator=' ')
    return int(values['pswpin']), int(values['pswpout'])


def _cpu_totals() -> tuple[int, int]:
    fields = Path('/proc/stat').read_text(encoding='ascii').splitlines()[0].split()
    values = [int(value) for value in fields[1:]]
    idle = values[3] + values[4]
    return sum(values), idle


def _cpu_busy_fraction(interval_seconds: float = 1.0) -> float:
    total_before, idle_before = _cpu_totals()
    time.sleep(interval_seconds)
    total_after, idle_after = _cpu_totals()
    return 1.0 - (idle_after - idle_before) / (total_after - total_before)


def _wait_for_idle() -> tuple[int, float]:
    deadline = time.monotonic() + 600
    while True:
        available_bytes = _meminfo()['MemAvailable']
        busy_fraction = _cpu_busy_fraction()
        if (
            available_bytes >= MINIMUM_AVAILABLE_BYTES
            and busy_fraction <= MAXIMUM_BUSY_FRACTION
        ):
            return available_bytes, busy_fraction
        if time.monotonic() >= deadline:
            raise RuntimeError(
                'host did not reach the required idle and available-memory state'
            )
        print(
            f'Waiting for an idle host: {available_bytes / 1024**3:.1f} GiB '
            f'available, {100 * busy_fraction:.1f}% CPU busy',
            flush=True,
        )
        time.sleep(9)


def _collect_provenance() -> dict[str, Any]:
    cpu = _lscpu()
    os_release = _read_key_values(Path('/etc/os-release'), separator='=')
    allowed_cpus = sorted(os.sched_getaffinity(0))
    logical_cpus = int(cpu['CPU(s)'])
    if len(allowed_cpus) != logical_cpus:
        raise RuntimeError(
            f'benchmark requires all {logical_cpus} logical CPUs; '
            f'process affinity contains {len(allowed_cpus)}'
        )
    thread_settings = {name: str(logical_cpus) for name in THREAD_VARIABLES}
    thread_settings['OMP_DYNAMIC'] = 'FALSE'
    return {
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': _git('rev-parse', 'HEAD'),
        'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'generator_source_dirty': bool(
            _git('status', '--porcelain', '--', 'stepsic.py', 'stepsic')
        ),
        'worktree_dirty': bool(_git('status', '--porcelain')),
        'machine': {
            'cpu_model': cpu['Model name'],
            'architecture': cpu['Architecture'],
            'logical_cpus': logical_cpus,
            'sockets': int(cpu['Socket(s)']),
            'cores_per_socket': int(cpu['Core(s) per socket']),
            'threads_per_core': int(cpu['Thread(s) per core']),
            'allowed_cpu_ids': allowed_cpus,
            'memory_bytes': _meminfo()['MemTotal'],
        },
        'operating_system': {
            'name': os_release['PRETTY_NAME'],
            'kernel': platform.release(),
        },
        'software': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'h5py': h5py.__version__,
            'hdf5': h5py.version.hdf5_version,
            'camb': _package_version('camb'),
            'colossus': _package_version('colossus'),
        },
        'thread_settings': thread_settings,
        'measurement_tool': '/usr/bin/time -v',
    }


def _new_artifact(provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        'provenance': provenance,
        'protocol': {
            'workload': WORKLOAD,
            'nmesh_equals_ngrid': list(GRID_SIZES),
            'lpt_orders': list(LPT_ORDERS),
            'initial_repeats': INITIAL_REPEATS,
            'maximum_trials': MAX_TRIALS,
            'retry_wall_spread_fraction': SPREAD_LIMIT,
            'shuffle_seed': SCHEDULE_SEED,
            'minimum_available_memory_bytes': MINIMUM_AVAILABLE_BYTES,
            'maximum_pre_run_cpu_busy_fraction': MAXIMUM_BUSY_FRACTION,
            'measurement_boundary': (
                'fresh process from launch through the written HDF5 snapshot'
            ),
        },
        'smoke_tests': [],
        'runs': [],
    }


@contextmanager
def _artifact_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_suffix(path.suffix + '.lock')
    with lock_path.open('a', encoding='utf-8') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _load_or_create(path: Path) -> dict[str, Any]:
    current = _collect_provenance()
    if current['generator_source_dirty']:
        raise ValueError('publication benchmark requires clean generator sources')
    if not path.exists():
        artifact = _new_artifact(current)
        atomic_write_json(path, artifact)
        return artifact
    with path.open(encoding='utf-8') as stream:
        artifact = json.load(stream)
    for key in (
        'git_commit', 'benchmark_sha256', 'machine', 'software', 'thread_settings',
    ):
        if artifact['provenance'][key] != current[key]:
            raise ValueError(f'cannot resume benchmark after provenance changed: {key}')
    return artifact


def _run_once(
    artifact: dict[str, Any], work_root: Path, *, lpt_order: int, nmesh: int,
    repeat: int, phase: str,
) -> dict[str, Any]:
    available_bytes, busy_fraction = _wait_for_idle()
    load_average_1m = os.getloadavg()[0]
    particle_count = nmesh**3
    before_swap_in, before_swap_out = _vmstat_swap_pages()
    with tempfile.TemporaryDirectory(
        prefix=f'stepsic-{phase}-{lpt_order}lpt-{nmesh}-r{repeat}-',
        dir=work_root,
    ) as temporary_name:
        temporary = Path(temporary_name)
        config_path = temporary / 'config.toml'
        time_path = temporary / 'time.txt'
        log_path = temporary / 'stepsic.log'
        config = WORKLOAD | {
            'NGRID': nmesh, 'NMESH': nmesh, 'LPTORDER': lpt_order,
            'IC_DIR': str(temporary),
        }
        config_path.write_text(toml.dumps(config), encoding='utf-8')
        command = [
            '/usr/bin/time', '-v', '-o', str(time_path),
            sys.executable, 'stepsic.py', str(config_path),
        ]
        environment = os.environ | artifact['provenance']['thread_settings']
        environment['LC_ALL'] = 'C'
        print(
            f'Running {phase}: {lpt_order}LPT, {nmesh}^3, trial {repeat}',
            flush=True,
        )
        started_at = datetime.now(timezone.utc).isoformat()
        with log_path.open('w', encoding='utf-8') as log:
            completed = subprocess.run(
                command, cwd=REPOSITORY, env=environment,
                stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        after_swap_in, after_swap_out = _vmstat_swap_pages()
        metrics = parse_time_report(time_path.read_text(encoding='utf-8'))
        run: dict[str, Any] = {
            'sequence': len(artifact['runs']) + len(artifact['smoke_tests']) + 1,
            'phase': phase,
            'started_at_utc': started_at,
            'lpt_order': lpt_order,
            'nmesh': nmesh,
            'particle_count': particle_count,
            'repeat': repeat,
            **metrics,
            'exit_status': completed.returncode,
            'system_swap_in_pages': after_swap_in - before_swap_in,
            'system_swap_out_pages': after_swap_out - before_swap_out,
            'available_memory_bytes_before': available_bytes,
            'cpu_busy_fraction_before': busy_fraction,
            'load_average_1m_before': load_average_1m,
        }
        try:
            if completed.returncode != metrics['reported_exit_status']:
                raise ValueError('process and GNU time exit statuses disagree')
            if completed.returncode != 0:
                raise ValueError(f'STEPSIC exited with status {completed.returncode}')
            written = snapshot_arrays(temporary)
            if written != [((particle_count, 3), 'float32')] * 2:
                raise ValueError(f'snapshot positions and velocities are {written}')
            if metrics['swaps'] != 0:
                raise ValueError('GNU time reported process swaps')
            if run['system_swap_in_pages'] or run['system_swap_out_pages']:
                raise ValueError('system swap I/O occurred during the benchmark')
        except (KeyError, ValueError) as error:
            run['validation_error'] = str(error)
            log_tail = log_path.read_text(encoding='utf-8').splitlines()[-30:]
            print('\n'.join(log_tail), flush=True)
    return run


def _already_valid(
    runs: list[dict[str, Any]], lpt_order: int, nmesh: int, repeat: int,
) -> bool:
    return any(
        run['lpt_order'] == lpt_order and run['nmesh'] == nmesh
        and run['repeat'] == repeat and is_valid_run(run)
        for run in runs
    )


def run_benchmark(results_path: Path, work_root: Path, *, smoke_only: bool) -> None:
    """Run or resume the smoke test and publication measurement matrix."""
    results_path = results_path.resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    work_root = work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    with _artifact_lock(results_path):
        artifact = _load_or_create(results_path)

        def measure(
            collection: str, phase: str, lpt_order: int, nmesh: int, repeat: int,
        ) -> None:
            if _already_valid(artifact[collection], lpt_order, nmesh, repeat):
                return
            run = _run_once(
                artifact, work_root, lpt_order=lpt_order, nmesh=nmesh,
                repeat=repeat, phase=phase,
            )
            artifact[collection].append(run)
            atomic_write_json(results_path, artifact)
            if not is_valid_run(run):
                raise RuntimeError(run['validation_error'])

        measure('smoke_tests', 'smoke', 2, 64, 1)
        if smoke_only:
            print(f'Smoke test passed; metrics: {results_path}', flush=True)
            return
        for lpt_order, nmesh, repeat in scheduled_runs():
            measure('runs', 'matrix', lpt_order, nmesh, repeat)
        workloads = [(order, nmesh) for order in LPT_ORDERS for nmesh in GRID_SIZES]
        for repeat in range(INITIAL_REPEATS + 1, MAX_TRIALS + 1):
            retry_round = [
                workload for workload in workloads
                if needs_more_trials(artifact['runs'], *workload)
            ]
            random.Random(SCHEDULE_SEED + repeat).shuffle(retry_round)
            for lpt_order, nmesh in retry_round:
                measure('runs', 'retry', lpt_order, nmesh, repeat)
        print(f'Benchmark complete; raw metrics: {results_path}', flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Measure the fixed STEPSIC publication performance matrix.',
    )
    parser.add_argument(
        '--results', type=Path,
        default=REPOSITORY / 'performance/results/benchmark.json',
        help='raw JSON artifact (default: performance/results/benchmark.json)',
    )
    parser.add_argument(
        '--work-root', type=Path,
        default=Path('/tmp/stepsic-performance'),
        help='directory for temporary snapshots, configs, and logs',
    )
    parser.add_argument(
        '--smoke-only', action='store_true',
        help='verify one complete 64^3 2LPT HDF5 run, then stop',
    )
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    run_benchmark(
        arguments.results, arguments.work_root, smoke_only=arguments.smoke_only,
    )


if __name__ == '__main__':
    main()
