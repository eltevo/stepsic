# Validation campaigns

`validation/` contains the scripts that reproduce the manuscript figures and supporting measurements. Pytest does not run these campaigns.

## Running campaigns

Run one campaign, a named group of campaigns, or every campaign:

```bash
bash validation/run.sh --campaign=particle-loads --size=small
bash validation/run.sh --group=paper --size=medium --evaluation=gate
bash validation/run.sh --all --size=small --evaluation=report
bash validation/run.sh --list-campaigns
```

Groups are `paper`, `supplementary`, and `extended`. A command with no selection prints the usage text and exits without running anything. `--evaluation=gate` is the default at every size. `report` writes the same result but does not make the command fail when a scientific check fails. `skip` does not run the evaluator or write `result.json`.

Each campaign supports `--size`, `--step`, `--plot-only`, `--force`, `--force-step`, `--clean`, `--config`, `--list-steps`, `--evaluation`, and `--help`. `--plot-only` still evaluates unless evaluation is explicitly skipped.

`--config=PATH` selects a TOML file that contains one table for the chosen campaign and values for all of its size-dependent settings. It cannot be combined with `--size`. Custom runs write to `runs/custom/<profile-sha256>/`. Environment variables override the selected profile, and profile values override the defaults in `config.env`. Each run records the final values and where they came from in its `config/` directory and cache manifests.

## Campaign catalog

| Campaign | Role | Evidence |
| --- | --- | --- |
| `particle-loads` | paper | Fig. 1 particle-load views, with supplementary 3D views |
| `shell-mass-profiles` | paper | Fig. 2 comparison with analytic shell masses |
| `field-statistics` | supplementary | cubic LPT displacement and velocity statistics |
| `slab-sampling` | paper | Figs. 3 and 5 slab anisotropy and comparison with a matching cut from a cube |
| `grid-power-recovery` | paper | Fig. 4 recovered power spectrum with Fourier-mode counts |
| `cylindrical-lpt` | paper | Figs. 6 and 7 cylindrical 1LPT and 2LPT convergence |
| `monofonic-agreement` | paper | Fig. 8 comparison with monofonIC from the same white-noise field |
| `periodic-embedding` | supplementary | convergence as padding increases and checks for radial drift |
| `spherical-ic` | supplementary | comparison of spherical and periodic initial conditions |
| `glass-quality` | supplementary | power below the Poisson level, force suppression by radius, and rescaling tests |
| `periodic-lpt` | supplementary | periodic 1LPT and 2LPT comparison evolved with Gadget-4 |
| `spherical-evolution` | extended | optional spectra from evolved spherical and periodic simulations |

`spherical-evolution` does not decide scientific pass or fail. It checks the required files, recorded inputs, array values, and matching epochs before producing spectra for inspection.

## Profiles and outputs

[`small.toml`](_profiles/small.toml) uses the smallest settings that still let every campaign and evaluator run. [`medium.toml`](_profiles/medium.toml) contains the settings for local publication runs and must stay below 48 GiB peak RSS. A campaign is ready for publication only after its medium run passes and someone inspects its figures. [`large.toml`](_profiles/large.toml) contains the approved settings for compute nodes; a campaign that has no large configuration stops before doing any computation. The cylindrical configuration used in the article and the 256³ periodic comparison are large runs.

Every campaign has a `run.sh`, a `config.env`, and a `scripts/` directory. Generated files go under `runs/<profile>/{cache,config,build,output}`. Article figures are written directly to `output/`; other retained figures go to `output/supplementary/`.

Each result records the campaign, selected profile, parameters, input and command information, measurements, checks, and paths to data and figures. Any failed check makes the scientific result fail. A missing or malformed output is an execution error instead.

Before reusing a cached step, the runner compares its selected profile, configuration, direct inputs, implementation files, and earlier outputs with the values recorded in a manifest. It replaces outputs and manifests atomically. A per-step lock lets identical concurrent invocations share the completed work. Cleanup resolves the requested path and refuses to remove anything outside the current campaign's run directory.

`glass-quality`, `cylindrical-lpt`, and `spherical-evolution` use `GLASS_INPUT_MODE=generate|pre-generated`. Pre-generated mode requires explicit snapshot paths. Each campaign checks their geometry and metadata before computation and records their hashes in the manifests for steps that use them.

## Shared implementation

- `_common/cosmology_fields.py`: cosmology and field generation
- `_common/plotting.py`: shared figure style
- `_common/artifacts.py`: NumPy archive helpers
- `_common/result.py`: scientific results and atomic JSON writes
- `_common/profiles.py`: campaign catalog and size profiles
- `_common/snapshots.py`: snapshot I/O and geometry checks
- `_common/runtime.sh` and `_common/orchestration.sh`: command-line options, cached steps, locks, and cleanup
