# Validation suite

This directory contains reproducible scientific campaigns for particle loads, LPT fields, power-spectrum recovery, glass quality, external-code comparisons, and evolved StePS simulations. Each campaign exposes a `run.sh` entrypoint and declares its flags, configuration variables, cache paths, numerical archives, and principal PDF outputs.

## Scientific pipeline contract

A campaign has three visible responsibilities, in this order:

1. **Measure** scientific quantities into a typed numerical archive (`.npz` or HDF5).
2. **Plot** those archived quantities. Plot code presents evidence; axis limits and shaded visual guides do not decide correctness.
3. **Evaluate** documented claims, after figures exist, and atomically write `output/result.json` (or a campaign-specific output directory for keyed, diagnostic, or evolved runs).

Evaluation writes the result after plotting. A failed required check makes a normal full run exit nonzero, with the result and figures available for diagnosis. A result contains:

- `campaign` and `status`;
- resolved parameters and archive/command provenance;
- named metrics with explicit units;
- checks with the observed value, comparison, derived limit, rationale, source, and pass/fail decision;
- paths to the numerical archive and generated figures.

Evaluators are small Python modules beside each campaign. Their gates use analytic identities, floating-point bounds, sampling distributions, convergence arguments, matched controls, or independent reference ensembles. Campaign output, plot bands, and axes are not gates. A scientific check is required only when its claim and derivation are encoded in the evaluator.

## Completion, judgment, and inspection

These are different outcomes:

- **Smoke completion** proves that the driver, dependencies, and artifact paths execute at reduced settings.
- **Scientific pass/fail** is the machine-readable decision in `result.json` at the resolved settings.
- **Human inspection** reviews the figures and numerical context, including failures and phenomena not represented by a required check.

A successful smoke run does not imply scientific acceptance. A generated PDF does not imply either smoke success or scientific acceptance.

## Driver interface and cache behavior

Every top-level driver provides this interface:

```bash
bash validation/grid/run.sh --help
bash validation/grid/run.sh --list-steps
bash validation/grid/run.sh
bash validation/grid/run.sh --step=3
bash validation/grid/run.sh --plot-only
bash validation/grid/run.sh --force-step=evaluate
```

Standard flags are `--step=N`, `--plot-only`, `--force`, `--force-step=A,B`, `--clean`, `--config=PATH`, `--list-steps`, and `--help`. Each driver header documents its campaign-specific flags.

Step order is declared once in each driver and consumed by listing, execution, and contract tests. A step is cached only when its stored manifest matches its resolved arguments, relevant configuration, declared inputs, implementation files, and outputs. Generated Python bytecode is ignored. Missing or stale manifests cause the step to run; existence alone is not a cache hit. Per-step locks coalesce concurrent runs, manifests and breadcrumbs are atomically replaced, and failed steps are not cached.

`validation/_common/lib.sh` is the shell source facade for the runtime, cosmology, StePS, stepsic-configuration, and orchestration modules. Drivers use `vlib::run_python`, which exposes the repository-root `validation` package without script-local `sys.path` mutation. Direct development invocations use the same package entrypoint explicitly:

```bash
PYTHONPATH="$PWD" conda run -n stepsic python validation/grid/scripts/evaluate.py --help
```

All campaigns inherit the Planck 2018 EE+BAO best-fit parameters from `_common/cosmology/Planck2018EE+BAO.toml`. The shell drivers and Python validation helpers read this same file, and resolved values participate in step manifests. A campaign with a scientifically required deviation overrides only the affected variable in its `config.env`, preserving an exported value:

```bash
COSMO_W0="${COSMO_W0:--0.9}"
```

Unchanged cosmology values are not repeated in campaign configuration files.

## Glass snapshots

The cylinder campaign generates its glass when `GLASS_SNAP` is empty. Set
`GLASS_SNAP` to reuse a pre-generated StePS snapshot; the glass-generation steps
are then omitted.

The spherical evolved run requires `GLASS_SNAP`. The diagnostic campaign accepts
independent `GLASS_SNAP_CUBICAL`, `GLASS_SNAP_SPHERICAL`, and
`GLASS_SNAP_CYLINDRICAL` overrides and otherwise uses snapshots produced by the
corresponding campaigns. Supplied paths must name existing files.

NumPy archives are loaded with `allow_pickle=False`. Glass diagnostic archives store ragged zone spectra as typed, NaN-padded arrays. Object-array archives are rejected.

## Layout and artifacts

Common source files:

- `_common/lib.sh`: shell source facade.
- `_common/runtime.sh`, `cosmology.sh`, `steps.sh`, `stepsic.sh`, and `orchestration.sh`: focused shell behavior.
- `_common/result.py`: result model and atomic JSON serialization.
- `_common/evaluation.py`: typed archive I/O and shared statistical derivations.
- `_common/manifest.py`: content manifests.
- `validation.py`: cosmology, LPT, plotting, and archive helpers exposed through the `validation` package.
- `smoke-test.sh`: reduced local campaign runs.

Campaign directories contain `run.sh`, `config.env`, `scripts/`, and generated `cache/`, `output/`, `configs/`, `params/`, or `builds/` directories. Numerical archives and HDF5 files are measurement evidence; PDFs are presentation; `result.json` is the machine contract.

## Campaigns

| Campaign | Independent claim source |
| --- | --- |
| `shell-mass` | analytic spherical/cylindrical volume and configured boundary solution |
| `field` | component isotropy sampling bound and histogram count conservation |
| `grid`, `squish` | mode-counted Gaussian power variance |
| `padding` | convergence of refined displacement error |
| `slab` | missing-long-mode variance suppression |
| `particle-load`, `glass` | analytic domain containment and exact cubic counts |
| `sphere` IC | Gaussian component-variance sampling bound |
| `glass/diagnose.sh` | matched Poisson-twin sub-particle-scale power |
| `cylinder` | low-wavenumber 1LPT/2LPT convergence |
| `monofonic` | mode-counted matched power comparison |
| `reference-nbody` | periodic mode-counted 1LPT/2LPT control |
| `sphere/evolved.sh` | matched-realization provenance and finite full-output spectra |

`cylinder/reference/` contains two reference spectra for manuscript Fig. 7. Evaluators do not use them as acceptance thresholds.

## Verification

Run the local reduced suite from the repository root:

```bash
bash validation/smoke-test.sh
```

The smoke suite checks that reduced pipelines complete and write inspectable artifacts. It deliberately does not enforce the scientific verdicts produced at reduced settings.

Run the lean pytest calibration and infrastructure contracts with:

```bash
conda run -n stepsic pytest
conda run -n stepsic pytest -m ""
```

These tests do not duplicate campaign science or presentation. They calibrate evaluator pass/fail decisions and protect critical safety, cache, conservation, matching, and fair-sample invariants.

External campaigns require their configured toolchains. Environment-specific verification commands are:

```bash
bash validation/glass/diagnose.sh
bash validation/cylinder/run.sh
bash validation/monofonic/run.sh
bash validation/reference-nbody/run.sh
REFERENCE_MANIFEST=/path/to/pair-manifest.json \
GLASS_SNAP=/path/to/spherical-glass.hdf5 \
bash validation/sphere/evolved.sh
```
