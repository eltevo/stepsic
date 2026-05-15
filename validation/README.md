# Validation Suite

This directory contains the reproducible validation runs used to check `stepsic` particle loads, LPT fields, power-spectrum recovery, glass generation, and comparisons against external codes or downstream StePS simulations. Each validation is intended to be runnable on its own, but the directory is organized as one suite with a shared shell library, shared Python helpers, and a common command-line contract.

Most validations generate cached numerical data first, then turn those cached products into publication-style PDF figures. The heavier end-to-end validations also generate `stepsic` ICs, compile StePS binaries, run StePS, and then plot diagnostics.

## Directory layout

Top-level files:

- `_common/lib.sh`: shared Bash library used by every `run.sh` driver.
- `validation.py`: shared Python utilities for cosmology setup, LPT field generation, archive loading, plotting style, and common CLI arguments.
- `hdf5inspect.py`: helper for inspecting HDF5 files during debugging.
- `smoke-test.sh`: small-parameter smoke test that calls all eight main validation pipelines.
- `viz/`: standalone rendering utilities for validation artwork; this is not part of the standard `run.sh` pipeline set.

Each main validation directory follows the same general shape:

- `run.sh`: the top-level controller for that validation run.
- `config.env`: default parameters. Override these by exporting environment variables or by passing `--config=PATH`.
- `scripts/`: Python scripts for numerical work and plotting, when needed.
- Generated directories such as `cache/`, `output/`, `configs/`, `params/`, or `builds/`: intermediate data, figures, generated configuration files, and compiled binaries. These are run products, not hand-maintained source.

## Common run model

The validation drivers all use `validation/_common/lib.sh`. A typical driver:

1. Finds its own base directory.
2. Sources `_common/lib.sh`.
3. Parses common command-line options with `vlib::parse_args`.
4. Sources `config.env` or the file passed with `--config=PATH`.
5. Creates `cache/` and `output/` directories.
6. Checks required conda environments.
7. Runs numbered steps through `vlib::step_check`.
8. Prints a completion banner with `vlib::report_done`.

`vlib::step_check` is the most important control point. It assigns step numbers in the order they appear in `run.sh`, handles `--step`, `--plot-only`, `--force`, `--force-step`, `--list-steps`, and skips cached steps when all declared output paths already exist. This means file ordering is pipeline ordering: read each `run.sh` from top to bottom, and the `vlib::step_check` calls define the run.

Some longer pipelines also use breadcrumbs in `cache/state.env`. Breadcrumbs store discovered file paths, such as the latest StePS snapshot, so later steps and resumed runs can find products created by earlier steps.

## How to run

Run any validation from the root directory:

```bash
bash validation/grid/run.sh
```

List a validation's steps without running anything:

```bash
bash validation/grid/run.sh --list-steps
```

Show the driver's embedded help:

```bash
bash validation/grid/run.sh --help
```

Override parameters with environment variables:

```bash
NMESH=64 NSTEPS=3 bash validation/squish/run.sh
```

Or use an alternate config file:

```bash
bash validation/squish/run.sh --config=/path/to/my-squish.env
```

The standard options are:

- `--step=N`: start at numbered step `N`; earlier steps are skipped.
- `--plot-only`: skip non-plot steps and regenerate figures from cached data.
- `--force`: rerun steps even when their declared outputs already exist.
- `--force-step=A,B`: rerun named steps only.
- `--clean`: delete run products before starting. Exact directories are driver-specific, usually `cache/` and `output/`.
- `--config=PATH`: source a different config file instead of the local `config.env`.
- `--list-steps`: print step names in execution order and exit.
- `-h`, `--help`: print the header help from the selected `run.sh`.

Drivers may add their own extra options. Examples include `glass/run.sh --run=spherical`, `glass/run.sh --no-clean`, `monofonic/run.sh --skip-build`, and `monofonic/run.sh --use-class`.

## What to expect

On startup, each driver prints a banner summarizing the key parameters, cache path, and output path. Each active step then prints a step header. Cached steps print `[cached]`, skipped steps print `[skip]`, and completed steps print an elapsed time.

Common outputs:

- `cache/*.npz`: cached numerical diagnostics used by plotting steps.
- `cache/**/ic.hdf5`: generated initial conditions.
- `output/*.pdf`: final validation figures.
- `configs/*.toml` or `params/*.param`: generated input files for `stepsic` or StePS.
- `builds/*`: compiled StePS binaries for the StePS-backed validations.

The smaller validations usually require only the `STEPSIC_ENV` conda environment. The glass and cylinder validations also need a StePS source tree, a `STEPS_ENV` conda environment, MPI/GPU settings, and a working CUDA-capable StePS build toolchain. The monofonIC validation needs the monofonIC source or repository settings and `MONOFONIC_ENV`.

## Shared python utilities

`validation/validation.py` is the common Python helper module. It provides:

- Planck 2018 cosmological defaults.
- CAMB and Colossus initialization through `init_cosmology`.
- Growth-factor data in `GrowthData`.
- Common LPT field generation through `run_lpt` and `generate_field`.
- Shared parsing for box sizes and common cosmology/LPT CLI flags.
- Matplotlib styling used by validation figures.
- Small archive and histogram helpers.

Most `scripts/run.py` files compute diagnostics and save `.npz` archives. Most `scripts/plot.py` files consume those archives and write PDFs. Prefer the top-level `run.sh` drivers for normal use, because they set paths, environments, cache behavior, and config defaults consistently.

## Validation runs

| Directory | Purpose | Steps |
| --- | --- | --- |
| `field/` | Displacement and velocity histogram validation for 3D and slab boxes. | `run_2lpt`, `run_1lpt`, `run_slab`, `plot_fields`, `plot_comparison`, `plot_slab` |
| `grid/` | Four-panel periodic-cube P(k) recovery test: resolution, redshift, LPT order, and MAS scheme. | `panel_a`, `panel_b`, `panel_c`, `panel_d`, `plot` |
| `squish/` | P(k) recovery as a cubic box is compressed into slab-like aspect ratios. | `run`, `plot` |
| `particle-load/` | Visual validation of cubic random, cubic grid, spherical, and cylindrical particle loads at `LPTORDER=0`. | `cubic_random`, `cubic_grid`, `spherical`, `cylindrical`, `plot`, `plot_2d` |
| `shell-mass/` | Analytic shell-mass distribution comparison for omega and constant-volume radial binning. | `plot` |
| `glass/` | Glass generation controller for cubical, spherical, and cylindrical geometries, followed by a combined figure. | controller: `cubical`, `spherical`, `cylindrical`, `plot`; geometry sub-steps: `preglass`, `param`, `build`, `run` |
| `cylinder/` | End-to-end cylindrical validation: preglass, glass relaxation, 1LPT/2LPT ICs, StePS simulations, and P(k) comparison. | `preglass`, `glass_param`, `build_glass`, `run_glass`, `ic_2lpt`, `ic_1lpt`, `build_sim`, `run_sim_2lpt`, `run_sim_1lpt`, `plot` |
| `monofonic/` | Cross-validation against monofonIC using shared white noise and matched transfer functions. | `run_stepsic`, `build_monofonic`, `run_monofonic`, `compare`, `plot` |

## Smoke test

Run the full lightweight suite from the repository root:

```bash
bash validation/smoke-test.sh
```

The smoke test calls all eight main pipelines with reduced sizes. It is useful for checking that drivers, environments, and plotting paths still work, but it does not replace the full validation settings in each `config.env`.
