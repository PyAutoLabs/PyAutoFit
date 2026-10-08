# Design note: the `run(ctx)` search hook and `FitContext`

Status: **frozen** (search-extensibility phase A1, decision D7). Phase A2 ships the
bridge and migrates `Drawer` and `Nautilus` onto it; phase A5 migrates the remaining
searches one PR each. This note fixes the signature and the members of `FitContext`
so that samplers written during the epic can target it before it exists. A change to
anything below is a design change: it needs a new revision of this note, not an
in-passing edit in an implementation PR.

## Why

Today every search implements `_fit(model, analysis) -> (search_internal, fitness)`
and re-implements, inside it, the same chores: building a `Fitness` (11 sites, 4
builders), choosing a pool (4 idioms), drawing start points (8 initializer calls),
resuming (7 idioms), chunking the run between updates, and calling
`perform_update(during_analysis=True)`. A new search therefore costs 400-700 lines,
most of them copied. `run(ctx)` moves the chores into one per-fit context object that
`NonLinearSearch.fit` builds, so a search implements only its backend loop.

## The contract a new search implements

```python
class MySearch(NonLinearSearch):
    # 1. static capabilities (autofit.non_linear.search.capabilities), declared as
    #    class attributes and mirrored in autofit/non_linear/search/registry.py
    jax_use = JaxUse.OPTIONAL
    gradient = Gradient.NONE
    posterior_kind = PosteriorKind.WEIGHTED
    ...

    # 2. the backend loop
    def run(self, ctx: FitContext) -> Any:
        """Run the backend and return its internal state (`search_internal`)."""

    # 3. the samples adapter
    def raw_samples_from(self, model, internal) -> RawSamples:
        """Map the internal state onto parameters, log likelihoods and weights."""

    # 4. optional: extra samples_info entries
    def info_from(self, internal) -> dict:
        ...
```

The frozen signature is:

```python
def run(self, ctx: "FitContext") -> Any
```

- `run` receives exactly one argument and returns the backend's internal state, the
  object `paths.save_search_internal` archives and `samples_via_internal_from` reads.
- `run` never builds a `Fitness`, a pool or an initializer itself, never reads
  `PYAUTO_TEST_MODE`, never calls `perform_update` directly and never writes
  `.completed`: it asks `ctx`.
- `run` may raise; `fit` cleans the context up (pools closed, quick-update threads
  shut down) on every exit path.
- `RawSamples(parameters, log_likelihood | log_posterior, weights, info)` and
  `samples_from_raw` are phase A3's samples adapter; `raw_samples_from` is named here
  so a search written now has the right shape.

## `FitContext` members

`FitContext` is a plain object created by `NonLinearSearch.fit` once per call, after the
test-mode bypass and the fail-fast gates have run.

| Member | Type / shape | What it gives `run` | Replaces |
|---|---|---|---|
| `ctx.model` | `AbstractPriorModel` | The (frozen) model being fitted. | the `model` argument of `_fit` |
| `ctx.paths` | `AbstractPaths` | Output paths; capability flags (`has_timer`, `supports_checkpointing`) rather than `isinstance(NullPaths)` checks. | `self.paths` reads inside `_fit` |
| `ctx.objective(kind)` | `kind in {"scalar", "batched", "value_and_grad", "batched_value_and_grad"}` -> callable | The objective factory handle (phase A2's `Fitness.objective`). `kind` is execution only; what the objective returns, its coordinates and its invalid value are the search's declared `objective_target` and `invalid_value`. On numpy the grad kinds raise; on JAX the callables are lazily jitted. | 11 `Fitness(...)` sites, 4 builders |
| `ctx.fitness` | `Fitness` | The underlying `Fitness`, for searches that need its history or quick-update hooks. | direct `Fitness` construction |
| `ctx.test_mode_level` | `int` | The resolved test-mode level (always `< 2` inside `run`: levels 2 and 3 bypass the backend). Level 1 budgets are already applied from `test_mode_budget`. | `is_test_mode()` / `test_mode_level()` calls |
| `ctx.pool` | `PoolFactory` | `ctx.pool(n_cores)` returns a pool or `None`, applying the one fork rule (JAX analyses and EP factor searches never fork) and recording the effective worker count in `search.summary`. | 4 pool idioms |
| `ctx.start_points(n)` | `(n, n_dim)` arrays of parameters and figures of merit | Initial points from the search's initializer, seeded from `ctx.rng`, including the start-point plot. | 8 initializer calls + manual `plot_start_point` |
| `ctx.rng` | `numpy.random.SeedSequence` fan-out | Stream 0 feeds the initializer, streams 1+ the backend (`ctx.rng.spawn(k)`). One search-level `seed` (phase A4, D4). | per-search seed handling |
| `ctx.resume` | `Optional[resume_state]` | The interrupted run's resume state when one exists, else `None` (phase A3's `resume_state`, separate from the final archive). | 7 resume idioms |
| `ctx.checkpointer` | `Checkpointer` slot | The archive strategy chosen by the search's class attribute (`DillCheckpointer`, `PickleCheckpointer`, `NativeFileCheckpointer`), with `save/load/exists/finalize`. A1 freezes the slot; A3 fills it. | `conf.instance[...]` mutations, ad-hoc pickling |
| `ctx.schedule` | `chunks()` / `next_budget(done)` | How many iterations to run before the next update. | `_steps_until_full_update`, Dynesty/Nautilus `iterations_from` |
| `ctx.update(internal)` | callback | Converts the internal state once, writes samples and visuals, saves the checkpoint (`perform_update(during_analysis=True)` + checkpoint save). Hands the updater a snapshot, never the live object. | `perform_update` calls inside `_fit` |

## Lifecycle rules (D6)

- One `FitContext` per `fit()` call. It is **never** stored on `self`, never pickled or
  dill-dumped with the search, and never copied by `copy_with_paths` (a shallow copy,
  used by grid-search children): each child fit builds its own.
- It is created after the test-mode bypass return and after the fail-fast gates
  (`jax_use='required'` with a numpy analysis, whole-graph backend agreement), so
  `run` only ever sees a runnable configuration.
- It is cleaned up on every exit, including exceptions: pools closed, quick-update
  threads shut down, compiled objectives released.
- Updates receive snapshots, so a background visualizer never races the backend.

## What is not in the context

- **The analysis.** `run` sees the likelihood only through `ctx.objective(kind)` /
  `ctx.fitness`; it never calls `analysis.log_likelihood_function` directly.
  Visualization and result construction stay in `NonLinearSearch`.
- **Configuration.** No `conf.instance` access through the context; search settings
  are constructor arguments, and the test-mode reduction is already applied.
- **Static capabilities.** They are class attributes on the search (`type(self)`), not
  context members, so docs and the registry read them without a fit.
- **Identifier inputs.** Nothing in the context enters the search identifier; the
  identifier is computed from `__identifier_fields__` before the context exists.
- **Grid-search / EP orchestration.** Outer loops (grid search, sensitivity mapping,
  expectation propagation) build one context per inner fit; the context does not know
  it is inside one.

## Migration

- **A2** ships `FitContext`, `PoolFactory`, `start_points`, the objective factory and a
  bridge in `NonLinearSearch.fit`: a search that defines `run` is driven through the
  context, a search that only defines `_fit` keeps working unchanged. `Drawer` and
  `Nautilus` migrate as the proofs.
- **A5** migrates the rest, one PR each: Emcee, Zeus, BFGS; Dynesty, NSS, NUTS, SMC;
  MultiStart last (its `_fit` is split internally first).
- External subclasses that override `_fit` (the developer-tier `nss`, `ultranest` and
  `pyswarms` searches) keep working for at least one release after A5.
