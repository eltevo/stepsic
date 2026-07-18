# Test-suite code of conduct

The pytest suite covers fast, deterministic unit and component correctness. Physics-scale, visual, cross-code (including monofonIC), and end-to-end pipeline checks belong in `validation/`, not in pytest.

## Evidence tiers

Every test docstring names its evidence tier and the provenance of every expected value. Lower tiers are preferred.

1. **T1 - exact structural or symmetry invariant.** Examples include conservation, adjointness, determinism, symmetry, and round trips.
2. **T2 - analytic closed-form oracle.** Include the derivation or cite the paper, textbook, or specification that supplies the result.
3. **T3 - differential oracle.** Name the independent library oracle or keep a structurally independent brute-force reference in the test.
4. **T4 - statistical oracle.** State the sampling distribution and derive the tolerance from the sample count at five sigma or more. Parametrize over all canonical seeds.
5. **T5 - regression golden.** This is a last resort. Explain why T1-T4 cannot pin the behavior, store the file in `tests/goldens/`, provide a checked-in script in `tests/regen/`, and assert embedded provenance keys in the test.

An expected number without stated provenance is a test defect. Never copy or transliterate a production formula into a test as its oracle; use an invariant, an analytic result, an external oracle, or a structurally independent method.

## Numerical policy

Use exact equality for integer, bit-level, and RNG behavior. Float64 identities may use `rtol <= 1e-13` only where non-associative arithmetic requires it. Discretization tolerances must follow a cited convergence order. Statistical tolerances must be derived from the number of samples at five sigma or more. Every non-exact tolerance needs a nearby justification comment. Never loosen a tolerance to make a failure green.

The canonical seeds are `SEEDS = (0, 42, 112358)` from `tests/conftest.py`. Statistical properties use at least those three seeds; seeds and tolerances may not be cherry-picked or tuned to one realization.

## Structure and speed

Mirror one `stepsic/` module with one `tests/test_<module>.py` file. Put shared setup only in `tests/conftest.py`, and name tests `test_<unit>__<behavior>`. An unmarked test must take less than two seconds. Mark heavier tests `slow`; mark tests requiring CAMB or Colossus `camb`. The default suite excludes `slow` and must finish in under 30 seconds.

Warnings are errors. Any allowlisted external warning must have a comment that identifies the upstream cause.

## Golden files and AI-agent rules

No golden may be committed without its regeneration script and embedded provenance (git commit, relevant package versions, and generation date). Goldens are never shortcuts for deriving an expected value.

AI agents and human contributors must observe every new test fail once for the intended reason by temporarily mutating the expectation or code under test. Never xfail, skip, delete an assertion, cherry-pick a seed, or weaken a tolerance to get green. Restore every deliberate mutation before committing.
