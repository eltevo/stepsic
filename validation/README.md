# Validation campaigns

`validation/` contains the scripts that reproduce the manuscript figures and supporting measurements. Validation checks are campaign measurements or temporary verification probes; they do not add persistent tests to the repository pytest suite.

## Running campaigns

Run one campaign, a named group of campaigns, or every campaign:

```bash
bash validation/run.sh --campaign=particle-loads --size=small
bash validation/run.sh --group=paper --size=medium --evaluation=gate
bash validation/run.sh --all --size=small --evaluation=report
bash validation/run.sh --list-campaigns
```

Groups are `paper`, `supplementary`, and `extended`. A command with no selection prints the usage text and exits without running anything. `--evaluation=gate` is the default at every size. `report` writes the same result but does not make the command fail when a scientific check fails or evidence is insufficient. `skip` does not run the evaluator or write `result.json`.

Each campaign supports `--size`, `--step`, `--plot-only`, `--force`, `--force-step`, `--clean`, `--config`, `--list-steps`, `--evaluation`, and `--help`. `--plot-only` still evaluates unless evaluation is explicitly skipped.

`--config=PATH` selects a TOML file that contains one table for the chosen campaign and values for all of its size-dependent settings. It cannot be combined with `--size`. Custom runs write to `runs/custom/<profile-sha256>/`. Environment variables override the selected profile, and profile values override the defaults in `config.env`. Each run records the final values and where they came from in its `config/` directory and cache manifests.

## What each campaign shows

Figures used in the article are written to `output/`; the other figures go to `output/supplementary/`. `--list-steps` prints the steps of a campaign in order.

**`particle-loads`** (group `paper`). Builds the four kinds of particle load that stepsic can displace: random points and a regular grid in a cube, radial shells in a sphere, and shells in a cylinder that is periodic along its axis. It draws them in projection and in 3D, so the layout and the growth of particle mass with radius can be seen directly, and checks that each load fills the declared volume with the declared total mass.
Steps: `cubic_random`, `cubic_grid`, `spherical`, `cylindrical`, `plot`, `plot_2d`, `evaluate`.

**`shell-mass-profiles`** (`paper`). Computes the particle masses of the spherical and cylindrical shell layouts from the shell geometry and the mean density alone, without any simulation, and checks that they add up to the analytic volume times the mean density and that the outermost shell ends at the configured radius. The figure compares shells of constant solid angle with shells of constant volume.
Steps: `measure`, `plot`, `evaluate`.

**`field-statistics`** (`supplementary`). Generates many random fields in a cube and compares the histograms of the x, y and z displacements and velocities. No direction is special in a cube, so the three must agree. A second figure shows how far the fitting formulae that stepsic uses for the growth rates are from a direct integration of the growth equations.
Steps: `run_2lpt`, `run_1lpt`, `plot_fields`, `plot_comparison`, `growth`, `evaluate`.

**`slab-sampling`** (`paper`). A thin periodic slab contains no waves longer than its thickness along the short axis. The campaign compares displacements generated in such a slab with those in a layer of the same thickness cut out of a cube, which does contain the long waves, and compares the measured variances with a direct sum over the Fourier modes that each box holds.
Steps: `field_histogram` (displacement histograms along and across the slab), `transfer` (power spectrum as the box is made thinner), `fairness` (slab and cube-cut variances for one thickness), `covariance` (variances for several thicknesses against the mode sum), four plot steps, `evaluate_transfer`, `evaluate_fairness`, `evaluate`.
The notebook `slab-sampling/notebooks/cosmic-variance-slab-displacement.ipynb` explains why the two long axes of a single slab realization can differ: two fundamental modes carry most of the variance.

**`grid-power-recovery`** (`paper`). Displaces a periodic glass, tiled to fill the box, with LPT, measures the power spectrum of the particles, and divides it by the input spectrum averaged over the same Fourier modes. The panels repeat this for several Fourier meshes, redshifts, LPT orders and interpolation schemes, always with the interpolation window compensated. It reads the glass from `GLASS_SNAP` (a periodic cubic Gadget-HDF5 snapshot); the load must be finer than the finest Fourier mesh, and its own power spectrum is stored in each archive.
Steps: `panel_a` (Fourier mesh), `panel_b` (redshift), `panel_c` (1LPT against 2LPT), `panel_d` (interpolation schemes), `plot`, `evaluate`.

**`cylindrical-lpt`** (`paper`). Relaxes a cylindrical glass with StePS, displaces it with first-order and with second-order LPT from the same random field, and evolves both to z=0 with StePS. The figures show the measured power spectrum against the linear input, and the ratio of the two runs, which is the nonlinear power that a first-order start loses.
Steps: `preglass`, `glass_param`, `build_glass`, `run_glass`, `ic_2lpt`, `ic_1lpt`, `build_sim`, `run_sim_2lpt`, `run_sim_1lpt`, `measure`, `plot`, `evaluate`.

**`monofonic-agreement`** (`paper`). Gives stepsic and monofonIC the same white-noise field and the same transfer function, and compares the resulting particle displacements, velocities and power spectrum particle by particle and bin by bin. monofonIC removes aliasing from the quadratic source of the second-order term with the Orszag 3/2 rule, whereas stepsic multiplies the fields on the mesh; the campaign therefore also builds monofonIC with aliased convolutions and compares its second-order term with that of stepsic.
Steps: `run_stepsic`, `build_monofonic`, `run_monofonic`, `compare`, `plot`, `evaluate`, `build_monofonic_aliased`, `run_monofonic_aliased`, `isolated_orders`, `plot_isolated_orders`. `--skip-build` uses an existing monofonIC binary. The last four steps run only in the medium run root and combine six runs made with the profiles `_profiles/experiments/monofonic-{64,128,256}-lpt{1,2}.toml` (see below): `build_monofonic_aliased` builds monofonIC again with aliased convolutions from a copy of the prepared source (`monofonic-aliased/`), `run_monofonic_aliased` runs that build on the three 2LPT profiles, `isolated_orders` compares, shell by shell in Fourier space, the first-order displacement, the second-order correction (2LPT minus 1LPT positions) and the total 2LPT displacement of the two codes, as well as the second-order correction of the aliased monofonIC runs with that of stepsic, for the meshes in `ISOLATED_NMESH` (`config.env`; empty disables all four steps), and `plot_isolated_orders` draws the amplitude ratio and correlation coefficient of the first and second order and the relative difference of the total displacement against k/k_Ny (`output/isolated-orders.pdf`).

**`periodic-embedding`** (`paper` and `supplementary`). stepsic computes the field of a sphere or cylinder inside a periodic Fourier box that encloses it. The campaign builds the same initial conditions in boxes of increasing size and measures how much the particle displacements inside the domain change, and, with the glass steps below, compares each box with the infinite universe: where a box of one diameter gives the same statistics, where it does not, and what a wider box costs.
Steps: `run` (box-size sweep for `GEOMETRY`), `comparison-runs` (the same sweep for the other geometry and at half the mesh), `plot`, `controls` (first-order displacement of a Gaussian dipole against its exact open-space value), `cosmological` (linear structure functions of the padded boxes against open space), `evaluate`. Eleven more steps run on relaxed glasses when their paths are configured; see below.

**`spherical-ic`** (`supplementary`). Builds a spherical (or, with `GEOMETRY=cylindrical`, a cylindrical) initial condition and a periodic-cube one from the same random modes, and checks that their velocities agree in the central region where the resolution is constant. A second figure shows how much the displacement changes when each particle mass gets its own mesh instead of all particles using the finest one.
Steps: `ic_ref`, `ic_geom`, `compare`, `plot`, `multiresolution`, `evaluate`. Two more steps run on relaxed glasses when their paths are configured; see below.

**`glass-quality`** (`supplementary`). Relaxes particle loads into glasses by running StePS with the sign of gravity reversed, then compares each glass with a copy whose particles were placed at random. The glass should have less large-scale power, smaller leftover forces and the intended radial density.
Steps: `cubical`, `spherical`, `cylindrical` (build and relax), `plot`, `plot_3d`, `twins` (randomized copies), `rescale` (compress and tile the cubic glass), `build_force`, `forces`, `diagnose`, `plot`, `evaluate`.

**`periodic-lpt`** (`supplementary`). Evolves two periodic initial conditions made from the same random field, one first-order and one second-order, to z=0 with Gadget-4, and checks that their power spectra agree on large scales within the scatter expected from the number of Fourier modes.
Steps: `ic_load`, `ic_2lpt`, `ic_1lpt`, `build`, `run_2lpt`, `run_1lpt`, `publish` (records the 2LPT run as the reference for `spherical-evolution`), `compare`, `plot`, `evaluate`. An interrupted Gadget run resumes with `--restart` and the same MPI task count.

**`spherical-evolution`** (`extended`). Evolves one spherical StePS simulation and compares it with the periodic `periodic-lpt` run started from the same random field. It measures angular power spectra in two matching radial shells and the power spectrum of each whole volume (FKP for the sphere, FFT for the cube), and plots both against CAMB/Halofit. It checks that the two runs match in inputs and epoch; it does not decide a scientific pass or fail.
Steps: `geom_ic`, `build`, `evolve`, `contract`, `cl`, `pk_randoms`, `pk_steps`, `pk_periodic`, `compare`, `plot_cl`, `plot_pk`. `--prepare-only` stops after the initial condition and the input check.

## Profiles and outputs

[`small.toml`](_profiles/small.toml) uses the smallest settings that still let every campaign and evaluator run. [`medium.toml`](_profiles/medium.toml) contains medium numerical coverage. Heavy commands run serially, remain below 48 GiB process-session RSS and leave at least 8 GiB of available memory. The command wrapper monitors both limits. Publication claims require the stated controls, measured numerical convergence, appropriate uncertainty and inspected figures; successful medium execution alone is insufficient. [`large.toml`](_profiles/large.toml) contains the approved settings for compute nodes; a campaign that has no large configuration stops before doing any computation. The cylindrical configuration used in the article and the 256³ periodic comparison are large runs.

Every campaign has a `run.sh`, a `config.env`, and a `scripts/` directory. Generated files go under `runs/<profile>/{cache,config,build,output}`. Article figures are written directly to `output/`; other retained figures go to `output/supplementary/`.

Each result records the campaign, selected profile, parameters, input and command information, measurements, checks, and paths to data and figures. A false check makes the scientific result fail; a null check or empty check list makes it insufficient unless another check fails. Missing arrays, incompatible inputs and non-finite reported values are execution errors. A paired realization counts as one independent unit; fixed-amplitude ensembles do not have independent Gaussian-mode power errors.

Before reusing a cached step, the runner compares its selected profile, configuration, direct inputs, implementation files, and earlier outputs with the values recorded in a manifest. It replaces outputs and manifests atomically. A per-step lock lets identical concurrent invocations share the completed work. Cleanup resolves the requested path and refuses to remove anything outside the current campaign's run directory.

`glass-quality`, `cylindrical-lpt`, and `spherical-evolution` use `GLASS_INPUT_MODE=generate|pre-generated`. Pre-generated mode requires explicit snapshot paths. Each campaign checks their geometry and metadata before computation and records their hashes in the manifests for steps that use them.

## Shared implementation

- `_common/runtime.sh`, `_common/orchestration.sh`: command-line options, cached steps, locks and cleanup
- `_common/profiles.py`: campaign catalog and size profiles
- `_common/resources.py`: memory monitoring of every command
- `_common/stepsic.sh`, `_common/steps.sh`: writing stepsic configurations; building and running StePS
- `_common/fields.py`: cosmology, field generation and reference power spectra
- `_common/snapshots.py`, `_common/pairing.py`: snapshot I/O, geometry checks and matched-pair manifests
- `_common/result.py`, `_common/evaluation.py`: checks, results and atomic writes
- `_common/precision.py`, `_common/displacement.py`: simultaneous seed intervals; pair statistics of displacements
- `_common/plotting.py`: figure style and 3D particle views

## Steps that need relaxed glass files

Two campaigns have steps that read large relaxed glasses from outside the repository. The steps exist only when the glass paths are set, so the size profiles do not run them. The radii are in the units of the glass files; for the spherical glass this is the `SimulationRadius` header value, not the number in the file name.

`spherical-ic` adds `direct_reference` and `plot_glass_interpolation`. The first evaluates the mesh-based 2LPT displacement at glass positions and compares it with a direct sum over the Fourier modes kept in the field, for `128^3` and `256^3` meshes and six cut-off scales of the input spectrum. The difference should be the error of linear interpolation between mesh cells and nothing else, so the archive also holds that error as predicted from the same modes. The second draws the measured difference on the prediction, next to the `mixing_variance` result of `periodic-embedding` (`output/glass-interpolation.pdf`). It reads `MIXING_VARIANCE_ARCHIVE`, by default the archive of the `periodic-embedding` run of the same profile, so run that step first.

```bash
DIRECT_REFERENCE_SPHERICAL_GLASS=/data/steps/spherical/glasses/Glass_DS19_RC62_R1062_N35M.hdf5 \
DIRECT_REFERENCE_SPHERICAL_RADIUS=531.428571428571 \
DIRECT_REFERENCE_CYLINDRICAL_GLASS=/data/steps/cylindrical/glasses/Glass_Cylindrical_Lz200_R750_DS35_N2M_ordered.hdf5 \
DIRECT_REFERENCE_CYLINDRICAL_RADIUS=750 \
bash validation/spherical-ic/run.sh --size=medium --step=7
```

`periodic-embedding` adds eight steps. They compare boxes one, two and three domain diameters wide with the infinite universe, for the sphere and the cylinder of the two glasses, both stretched to the radius `RELATIVE_MOTION_SPHERICAL_RADIUS` = `RELATIVE_MOTION_CYLINDRICAL_RADIUS` = 750 Mpc.

- `two_point` sums over the Fourier modes of each box, without random fields, to get the RMS relative displacement of two points as a function of their separation, and the correlation of the displacements at opposite points of the boundary. `monopole` generates 200 2LPT realizations in each box and measures, on the boundary of the domain, its mean density contrast, its mean strain and the same opposite-point correlation, next to their predictions from the modes. `plot_two_point` draws both (`output/embedding-two-point.pdf`).
- `glass_prepare`, `glass_run`, `glass_measure` and `plot_relative_motion` displace the complete glasses in each box and show by how many percent the RMS relative displacement of particle pairs changes with box width, with the change expected from the modes drawn on the same axes (`output/supplementary/embedding-relative-motion.pdf`). Twenty seeds scatter by more than the expected changes, so this figure shows that nothing else changes.
- `mixing_variance` checks that adjacent mesh levels differ only by the modes the finer one adds. The `spherical-ic` figure `glass-interpolation.pdf` draws its result.

The settings (`RELATIVE_MOTION_*`, `TWO_POINT_*`, `MONOPOLE_*`, `MIXING_NSEEDS`) are listed in `periodic-embedding/config.env`.

`periodic-embedding` has two more steps, `evolved_pair` and `plot_evolved_pair`, for two StePS simulations of one cylindrical realization evolved to z=0 from boxes of one and two diameters, with the white noise matched inside the domain. They run when `EVOLVED_PAIR_DIR` (the directory of the two runs), `EVOLVED_PAIR_RANDOMS` and `EVOLVED_PAIR_GLASS` are set. `evolved_pair` measures the FKP power spectrum of the region within 157 Mpc of the axis in both runs, in the same volume and with the second run moved back by the displacement common to the region; the difference between the displacements of the same particles; the centre-of-mass shift of the domain against the solution of its equation of motion with a homogeneous exterior; and the mean density contrast of the region. It reads about 2.7 GB of snapshots and peaks near 20 GiB. `plot_evolved_pair` draws the ratio of the two spectra (`output/evolved-pair.pdf`). The two steps come last: they are steps 15 and 16 when the glass steps above are configured and steps 7 and 8 otherwise, as `--list-steps` shows.

```bash
EVOLVED_PAIR_DIR=/data/steps/cylindrical/runs/padding-pair \
EVOLVED_PAIR_RANDOMS=/data/steps/cylindrical/preglass/GlassIC_Cylindrical_Lz200_R750_DS35_N2M.hdf5 \
EVOLVED_PAIR_GLASS=/data/steps/cylindrical/glasses/Glass_Cylindrical_Lz200_R750_DS35_N2M_ordered.hdf5 \
bash validation/periodic-embedding/run.sh --size=medium --step=7
```

```bash
RELATIVE_MOTION_SPHERICAL_GLASS=/data/steps/spherical/glasses/Glass_DS19_RC62_R1062_N3M.hdf5 \
RELATIVE_MOTION_CYLINDRICAL_GLASS=/data/steps/cylindrical/glasses/Glass_Cylindrical_Lz200_R750_DS35_N2M_ordered.hdf5 \
bash validation/periodic-embedding/run.sh --size=medium --step=7
```

## Comparisons across runs

Profiles in `_profiles/experiments/` are passed with `--config`; each run gets its own root, `runs/custom/<sha256 of the profile file>`.

- `monofonic-agreement`: `monofonic-512-lpt2.toml` is the configuration of the article figure. The six profiles `monofonic-{64,128,256}-lpt{1,2}.toml` provide the inputs of `run_monofonic_aliased` and `isolated_orders`; run them first, with one MPI rank so that monofonIC writes a single file. With one rank, Open MPI 5 binds monofonIC to a single core; set `PRTE_MCA_hwloc_default_binding_policy=none`.

```bash
for n in 64 128 256; do for o in 1 2; do
    N_MPI=1 PRTE_MCA_hwloc_default_binding_policy=none \
    bash validation/monofonic-agreement/run.sh --config=validation/_profiles/experiments/monofonic-$n-lpt$o.toml --skip-build --evaluation=report
done; done
PRTE_MCA_hwloc_default_binding_policy=none \
bash validation/monofonic-agreement/run.sh --size=medium --step=7
```
