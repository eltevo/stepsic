# Testing

The pytest suite is the quick check you run while working: deterministic unit and component tests that finish in about a second. Work that needs physics-scale resolution, visual review, a comparison against another code, or a full end-to-end simulation belongs in `validation/`.

## Running it

The default run skips the `slow` marker, so use it while working:

```bash
conda run -n stepsic pytest
```

An empty marker expression runs everything, slow tests included:

```bash
conda run -n stepsic pytest -m ""
```

Tests that call CAMB or Colossus carry the `camb` marker. They run by default because those packages are in the `stepsic` environment; if yours lacks them, use `-m "not camb"`.

Keep an unmarked test under two seconds, and mark it `slow` if it cannot be. Pytest is configured to turn warnings into errors, so a new warning will stop the run; if you have to allowlist one coming from an external package, leave a comment saying what upstream causes it.

## What a good test looks like

Someone reading the test should see four things without reverse-engineering the implementation: the behavior or scientific claim under test, the inputs (including the boundary values that matter), the oracle that decides the expected result, and what a failure would mean.

Choosing that oracle is the hard part. You can use an exact invariant, an analytic result, a published formula, an independent library such as Astropy or Colossus, or a brute-force reference implementation written out in the test. What you cannot do is copy the formula from `stepsic/` into the test, because then you have only shown that the code agrees with itself.

Put a short citation or derivation next to any expectation that is not self-evident; routine assertions do not need explanatory prose. Each test module mirrors one module in `stepsic/` (`tests/test_field.py` covers `stepsic/field.py`, `tests/test_util.py` covers `stepsic/_util.py`), and most tests are named `test_<unit>__<behavior>` so failures read as sentences.

Within a module, start with what the function is mainly supposed to do, then move on to boundaries and error cases, then to numerical or statistical properties. Give unrelated behavior its own test, and parametrize only when the cases really do check the same thing. When you test a private helper you are describing how it behaves today, not promising anything to callers of the public API.

## Numbers and tolerances

Compare integers, bit patterns, and RNG output with exact equality. When you need a floating-point tolerance, work it out from rounding behavior, from a convergence order you can cite, or from the sampling distribution, and write down where the number came from unless it is obvious from context. Decide what should happen to NaN and infinity as well, so that a bad value fails the test instead of quietly vanishing inside something like `np.nanmean`.

`SEEDS = (0, 42, 112358)` in `conftest.py` is there for statistical properties that genuinely need several realizations. Never go looking for the seed or the tolerance that makes one observed output pass.

## Shared configuration

`conftest.py` holds the setup that would otherwise clutter many modules: `wn16` and `wn32` (session-scoped white-noise cubes at seed 42), `grid_geom` (a `GridGeometry` factory with keyword overrides), `tiny_params` (the parsed `data/tiny-config.toml`), `params_factory` (a geometry parameter dict with overrides), and `camb_cosmo` (a session-scoped `CAMBCosmology`). Build the rest inside the test, where a reader can see the inputs that matter.

## Golden files

Some results are supposed to come out the same no matter which version of CAMB or NumPy you happen to have installed, like the matter power spectrum CAMB returns for a fixed set of cosmological parameters, or the white-noise field `stepsic` builds from a given seed. We keep a copy of those under `goldens/`, and the test recomputes the value with your installed libraries and compares it against the stored copy. Only results of that kind belong there; anything you can derive, derive in the test.

The scripts that write those files live in `regen/`. Each one saves the parameters it used along with the numbers, plus a note of where the file came from (git commit, package versions, date), so the test can read the parameters back and confirm the file is documented. When a stored value legitimately has to change, rerun its script and commit the new file.
