# Testing

The pytest suite checks the stepsic source: the `stepsic/` package and the `stepsic.py` driver. Nothing else belongs here. Validation campaigns, benchmarks, profiling and plotting code live in their own directories and are not tested by pytest. Physics claims that need large meshes, ensembles, another code or an N-body run belong in `validation/`.

## Running it

```bash
conda run -n stepsic pytest
```

The whole suite takes a few seconds. Tests that call CAMB or Colossus carry the `camb` marker; deselect them with `-m "not camb"` if your environment lacks those packages. Warnings are errors.

## What earns a test

A test has to meet all four conditions before it goes in:

1. **It protects an expectation of the IC generator whose violation would silently corrupt initial conditions.** Examples: the particles carry the requested power spectrum; velocities follow the growth rate at the IC epoch; mesh levels describe the same realisation; masses add up to Omega_m rho_crit V; outputs are in the declared units. A crash is found by running the code; it needs no test.
2. **Its expected value comes from outside stepsic.** Good sources are a physical law or perturbation-theory result (continuity, the F2 kernel, the Heath growth integral), an independent library (astropy, NumPy's Philox, scipy quadrature), or the documented meaning of a config parameter. Never restate the formula under test with inputs the test chose itself; that only shows the code agrees with itself.
3. **It has been seen failing.** Before adding a test, break the behaviour it guards (revert the fix, flip the sign, drop the factor) and watch it fail for that reason. A test that no plausible mistake can fail does not belong here.
4. **It is cheap:** well under a second on the tiny meshes used here. The CAMB tests are the exception.

Tests of config-validation tables, file naming, private helpers already reached through the public path, or external libraries do not qualify.

## How the tests are organised

Each module mirrors the source module whose contract it checks: `test_lpt.py` checks `stepsic/lpt.py`, and `test_stepsic.py` runs the `stepsic.py` driver end to end. Name a test `test_<unit>__<expected behaviour>` and open it with a docstring that states the expectation, where the oracle comes from, and what a failure means.

`conftest.py` provides:

- `linear_growth` and `growth_rate`: the flat matter + Lambda growth oracle.
- `write_config`: the tiny config with overrides, each call in its own output directory.
- `tiny_chunks`: shrinks the memory-bounding chunk sizes so chunked loops run many times.
- `camb_cosmo`

`test_stepsic.py` adds `stepsic_run`, which runs the driver on a config and reads the snapshot with plain h5py.

## Numbers and tolerances

Compare exact quantities (counts, bit patterns, RNG output) exactly. Derive every floating-point tolerance from something you can name, and write it next to the assertion:

- the size of the neglected perturbative order;
- the accuracy of an approximation;
- a solver tolerance;
- the standard error of a statistic.

Never tune a seed or a tolerance until a test passes.
