# PyAutoFit — Agent Instructions

**PyAutoFit** is a Python probabilistic programming language for model fitting and Bayesian inference.

## Setup

```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
python -m pytest test_autofit/
python -m pytest test_autofit/non_linear/
python -m pytest test_autofit/mapper/
```

### Sandboxed / Codex runs

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python -m pytest test_autofit/
```

## Key Architecture

- **Non-linear searches** (`non_linear/search/`): MCMC (emcee), nested sampling (dynesty, nautilus), MLE (LBFGS, BFGS, drawer)
- **Model composition** (`mapper/`): `af.Model`, `af.Collection`, prior distributions
- **Analysis** (`non_linear/analysis/`): base `af.Analysis` class with `log_likelihood_function`
- **Aggregator** (`aggregator/`): results aggregation across runs
- **Database** (`database/`): SQLAlchemy backend for results storage
- **Graphical models** (`graphical/`): expectation propagation

## Key Rules

- All files must use Unix line endings (LF)
- Format with `black autofit/`

## Working on Issues

1. Read the issue description and any linked plan.
2. Identify affected files and write your changes.
3. Run the full test suite: `python -m pytest test_autofit/`
4. Ensure all tests pass before opening a PR.
5. If changing public API, note the change in your PR description — downstream packages (PyAutoArray, PyAutoGalaxy, PyAutoLens) and workspaces may need updates.
## Never rewrite history

NEVER perform these operations on any repo with a remote:

- `git init` in a directory already tracked by git
- `rm -rf .git && git init`
- Commit with subject "Initial commit", "Fresh start", "Start fresh", "Reset
  for AI workflow", or any equivalent message on a branch with a remote
- `git push --force` to `main` (or any branch tracked as `origin/HEAD`)
- `git filter-repo` / `git filter-branch` on shared branches
- `git rebase -i` rewriting commits already pushed to a shared branch

If the working tree needs a clean state, the **only** correct sequence is:

    git fetch origin
    git reset --hard origin/main
    git clean -fd

This applies equally to humans, local Claude Code, cloud Claude agents, Codex,
and any other agent. The "Initial commit — fresh start for AI workflow" pattern
that appeared independently on origin and local for three workspace repos is
exactly what this rule prevents — it costs ~40 commits of redundant local work
every time it happens.
