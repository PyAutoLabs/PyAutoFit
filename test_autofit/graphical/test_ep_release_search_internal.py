"""
An EP factor step must not retain its search's internal sampler (#1631).

RAL job `slope_hierarchy_scale` 342410 (25 `AnalysisFactor`s, a JAX likelihood,
Nautilus with `number_of_cores=1`, 64 GB) died with
`LLVM ERROR: Unable to allocate section memory!` after roughly three EP steps
(76 factor searches). The cause is retention, not compile time:

    EPHistory/FactorHistory keeps every (approx, Status) pair
        -> Status.result  (set by `AbstractSearch.optimise`)
            -> Result._search_internal  (the nautilus/dynesty sampler)
                -> the sampler's likelihood callable, a `Fitness`
                    -> `Fitness._vmap` / `._jit` / `._grad`, per-instance
                       `cached_property` JIT caches holding compiled XLA
                       executables

so every executable compiled anywhere in the run stayed alive for the whole
run. Nothing in `autofit/graphical` reads `search_internal` afterwards, and
`Result.search_internal` already falls back to the on-disk dill, so `optimise`
releases the in-memory reference before pinning the result on the `Status`.
"""

import gc
import weakref

import numpy as np
import pytest

import autofit as af
import autofit.graphical as g
from autofit.non_linear.fitness import Fitness
from test_autofit.graphical.gaussian.model import Analysis, Gaussian, make_data

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(name="factor_model")
def make_factor_model():
    x = np.arange(100)
    y = make_data(Gaussian(centre=50.0, normalization=25.0, sigma=10.0), x)

    prior_model = af.Model(
        Gaussian,
        centre=af.GaussianPrior(mean=50, sigma=20),
        normalization=af.GaussianPrior(mean=25, sigma=10),
        sigma=af.GaussianPrior(mean=10, sigma=10),
    )

    return g.AnalysisFactor(prior_model, analysis=Analysis(x=x, y=y))


@pytest.fixture(name="fitness_refs")
def make_fitness_refs(monkeypatch):
    """
    Record a `weakref` to every `Fitness` built while the test runs.

    The search modules import `Fitness` by name, so the class attribute
    `Fitness.__init__` is patched rather than any module-level binding.
    """
    refs = []
    original_init = Fitness.__init__

    def recording_init(self, *args, **kwargs):
        refs.append(weakref.ref(self))
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(Fitness, "__init__", recording_init)

    return refs


def live_fitness_count():
    """
    How many `Fitness` objects are currently alive.

    This counts rather than tracking weakrefs from `Fitness.__init__` because a
    dynesty search that resumes from its checkpoint restores its sampler by
    unpickling it, and the `Fitness` inside the restored sampler is therefore
    built through `__setstate__` and never passes through `__init__`.
    """
    return sum(isinstance(obj, Fitness) for obj in gc.get_objects())


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test__ep_factor_step_does_not_retain_fitness(factor_model, fitness_refs):
    search = af.DynestyStatic(maxcall=5, number_of_cores=1)

    factor_approx = factor_model.mean_field_approximation().factor_approximation(
        factor_model
    )

    gc.collect()
    fitness_before = live_fitness_count()

    # Two EP steps. `optimise` bumps `optimisation_counter` and re-derives
    # `self.paths` as a fresh `SubDirectoryPaths` each call, so the same search
    # object runs both steps without a path collision.
    model_dist, status = search.optimise(factor_approx)
    del model_dist, status

    model_dist, status = search.optimise(factor_approx)

    del factor_approx

    # Defensive: refcounting alone should suffice, but the search and its paths
    # may participate in cycles.
    gc.collect()

    # The searches really ran (they were not bypassed in test mode) ...
    assert len(fitness_refs) >= 2

    # ... and neither step left a `Fitness`, nor therefore its compiled JAX
    # executables, alive behind the retained `Status`.
    assert all(ref() is None for ref in fitness_refs)
    assert live_fitness_count() == fitness_before

    assert status.result._search_internal is None

    assert status.result.projected_model is not None
    assert status.result.samples is not None
    assert status.result.model is not None

    # The on-disk fallback must still be reachable without raising. With the
    # default test config `output.search_internal` is off, so `None` is the
    # expected answer here — the point is that reading it is safe.
    status.result.search_internal


def test__release_search_internal_clears_in_memory_reference():
    result = af.m.MockResult()

    result._search_internal = object()
    child = af.m.MockResult()
    child._search_internal = object()
    result.child_results = [child]

    result.release_search_internal()

    assert result._search_internal is None
    assert child._search_internal is None
