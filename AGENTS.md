# PyAutoFit — Agent Instructions

Canonical, agent-agnostic instructions for this repo. `CLAUDE.md` imports this
file; any tool that does not process `@`-imports should read this directly.

## What this repo is

**PyAutoFit** (package `autofit`) is a probabilistic programming language for
model composition, non-linear search, and Bayesian inference: `af.Model` /
`af.Collection`, the `af.Analysis` interface, MCMC / nested-sampling / MLE
searches, samples, aggregator, and a SQLAlchemy results database.

Dependency direction: autofit depends on **autoconf** only. It does **not**
import `autoarray`, `autogalaxy`, or `autolens` — never add such an import.
Shared utilities (e.g. `test_mode`, `jax_wrapper`) belong in autoconf.

## Related repos

- **Source siblings:** PyAutoConf (upstream). PyAutoGalaxy / PyAutoLens are
  downstream consumers (they build `Analysis` subclasses on autofit).
- **autofit_workspace** — runnable tutorials/examples (`../autofit_workspace`).
- **autofit_workspace_test** — integration + JAX/likelihood parity scripts.
- **HowToFit** — the lecture-style tutorial series (`../HowToFit`).
- **docs/** — Sphinx source; published to ReadTheDocs.

## Quick commands

```bash
pip install -e ".[dev]"                       # install with dev/test extras
python -m pytest test_autofit/                # full test suite
python -m pytest test_autofit/non_linear/     # one focused subtree (add -s for output)
black autofit/                                # formatter (advisory — not gated)
```

In a sandboxed / restricted environment, point numba and matplotlib at
writable caches:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python -m pytest test_autofit/
```

## CI / definition of green

PRs must pass `pytest --cov` on the CI matrix (Python 3.12 **and** 3.13). There
is no black/ruff/flake8 gate — formatting is advisory. (`requires-python` in
`pyproject.toml` is `>=3.9`.)

## Configuration & defaults

autoconf supplies the packaged defaults under `autofit/config/`. Workspaces
override them via their own `config/` directory; the test suite pushes a local
config dir via `conf.instance.push(...)` in `test_autofit/conftest.py`. When a
change adds a new config key, mirror it into the packaged defaults so
downstream workspaces inherit it.

## Public API

The public surface is defined authoritatively in `autofit/__init__.py` — read
it rather than trusting a hand-maintained list. Canonical import:

```python
import autofit as af
```

Key subsystems: `non_linear/search/` (MCMC: emcee/zeus; nested: dynesty,
nautilus; MLE: LBFGS/BFGS/drawer), `mapper/` (model + priors),
`non_linear/analysis/` (`af.Analysis.log_likelihood_function`), `aggregator/`,
`database/` (SQLAlchemy), `graphical/` (expectation propagation),
`interpolator/`.

## Key rules / footguns

- Import direction: autoconf only — never `autoarray` / `autogalaxy` /
  `autolens`.
- **The EP seam rule**: `autofit/graphical` is two layers — the inner
  factor-graph/message engine and the `declarative/` user layer. A new
  statistical capability in the inner layer must land **in the same PR**
  with its declarative expression *or* an explicit "not exposed" row in
  the lowering-contract table (`autofit/graphical/README.md` §8), plus a
  seam test where behaviour crosses the boundary
  (`test_autofit/graphical/test_declarative_deterministic.py` is the
  pattern). Capabilities that exist below but are silently absent above
  are the seam's known failure mode (see PyAutoFit#1336/#1337).
- All files use Unix line endings (LF, `\n`) — never `\r\n`.

## Working on issues

1. Read the issue description and any linked plan.
2. Identify affected files and make the change.
3. Run the full suite: `python -m pytest test_autofit/`.
4. If you changed public API, say so explicitly — downstream packages
   (PyAutoGalaxy, PyAutoLens) and the workspaces may need updates.
5. Ensure all tests pass before opening a PR.

<!-- repos_sync:history:begin -->
## Never rewrite history

Never rewrite pushed history on any repo with a remote — no `git init` over a
tracked repo, no force-push to `main`, no fresh-start "Initial commit", no
`filter-repo` / `filter-branch` / `rebase -i` on pushed branches. To get a
clean tree: `git fetch origin && git reset --hard origin/main && git clean -fd`.
<!-- repos_sync:history:end -->
