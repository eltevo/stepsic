# Computational-performance benchmark

This workflow times the unmodified STEPSIC command line, `python stepsic.py config.toml`, from process launch through the written HDF5 snapshot. The publication workload is fixed in `benchmark.py`: periodic grid particles in a 1000 Mpc/h box at redshift 31, Planck2018EE+BAO, seed 42, single precision, CIC interpolation without compensation, and equal particle and Fourier grids.

Run the preliminary 64³ 2LPT case before starting the matrix:

```bash
conda run --no-capture-output -n stepsic \
    python -m performance.benchmark --preliminary-only
```

The full command resumes the same artifact and runs 1LPT and 2LPT at 64³, 128³, 256³, and 512³. Each trial starts a fresh process once the host is idle, and each workload gets three trials in a deterministic shuffled order. A workload whose wall-time range exceeds 5% of its median receives up to two more trials. Snapshots, configs, and logs go to unique directories under `/tmp/stepsic-performance` and are removed after each trial.

```bash
conda run --no-capture-output -n stepsic \
    python -m performance.benchmark
```

GNU `/usr/bin/time -v` records wall time, user and system CPU time, peak RSS, exit status, major page faults, and process swaps. The runner also reads the system swap-I/O counters around each process. After the process exits, it checks that the snapshot holds N³ single-precision positions and velocities; this check is not timed. A run is invalid if the exit status is nonzero, the snapshot check fails, or either swap indicator changes. An invalid run stops the benchmark and prints the tail of its log. Measurements are stored in `performance/results/benchmark.json`, which is rewritten after every trial. A lock file next to it keeps concurrent runners from interleaving trials.

Generate the figure from the retained JSON:

```bash
conda run --no-capture-output -n stepsic \
    python -m performance.render
```
