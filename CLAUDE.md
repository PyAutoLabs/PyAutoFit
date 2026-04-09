# PyAutoFit

**PyAutoFit** is a Python probabilistic programming language for model fitting and Bayesian inference.

- Authors: James Nightingale, Richard Hayes
- Requires Python >= 3.9
- Package name: `autofit`

## Dependency Graph

PyAutoFit depends on **autoconf** (shared configuration and utilities).
PyAutoFit does **NOT** depend on PyAutoArray, PyAutoGalaxy, or PyAutoLens.
Never import from `autoarray`, `autogalaxy`, or `autolens` in this repo.
Shared utilities (e.g. `test_mode`, `jax_wrapper`) belong in autoconf.

## Repository Structure

- `autofit/` - Main package
  - `non_linear/` - Non-linear search algorithms
    - `search/mcmc/` - MCMC (emcee, zeus)
    - `search/mle/` - Maximum likelihood (LBFGS, BFGS, drawer)
    - `search/nest/` - Nested sampling (dynesty, nautilus)
    - `samples/` - Posterior samples handling
    - `paths/` - Output path management
    - `analysis/` - Analysis base classes
  - `mapper/` - Model and prior machinery
    - `prior/` - Prior distributions
    - `prior_model/` - Prior model composition
    - `model.py` - Core model class
  - `graphical/` - Graphical models and expectation propagation
  - `aggregator/` - Results aggregation across runs
  - `database/` - SQLAlchemy-based results database
  - `interpolator/` - Model interpolation
  - `config/` - Default config files packaged with library
- `test_autofit/` - Test suite (pytest)
- `docs/` - Sphinx documentation

## Key Dependencies

- `dynesty==2.1.4` - Nested sampling
- `emcee>=3.1.6` - MCMC
- `scipy<=1.14.0` - Optimisation
- `SQLAlchemy==2.0.32` - Database backend
- `anesthetic==2.8.14` - Posterior analysis/plotting
- Optional: `nautilus-sampler`, `zeus-mcmc`, `getdist`

## Running Tests

```
pytest test_autofit
pytest test_autofit/non_linear
pytest test_autofit/mapper
```

## Codex / sandboxed runs

When running Python from Codex or any restricted environment, set writable cache directories so `numba` and `matplotlib` do not fail on unwritable home or source-tree paths:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib pytest test_autofit
```

This workspace is often imported from `/mnt/c/...` and Codex may not be able to write to module `__pycache__` directories or `/home/jammy/.cache`, which can cause import-time `numba` caching failures without this override.

## Shell Commands

- Prefer simple shell commands
- Avoid chaining with `&&` or pipes; run commands separately

## Related Repos

- **autofit_workspace** (tutorials/examples): `../autofit_workspace`
