# Copilot Coding Agent Instructions

You are working on **PyAutoFit**, a Python probabilistic programming library for model fitting and Bayesian inference.

## Key Rules

- Run tests after every change: `python -m pytest test_autofit/`
- Format code with `black autofit/`
- All files must use Unix line endings (LF, `\n`)
- If changing public API (function signatures, class names, import paths), clearly document what changed in your PR description — downstream packages depend on this

## Architecture

- `autofit/non_linear/search/` — Non-linear search algorithms (MCMC, nested sampling, MLE)
- `autofit/mapper/` — Model composition and prior machinery
- `autofit/graphical/` — Graphical models and expectation propagation
- `autofit/aggregator/` — Results aggregation
- `autofit/database/` — SQLAlchemy results database
- `test_autofit/` — Test suite

## Sandboxed runs

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python -m pytest test_autofit/
```
