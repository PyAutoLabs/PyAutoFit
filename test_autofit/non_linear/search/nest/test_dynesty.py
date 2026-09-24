import numpy as np
import pytest

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


class MockDynestyResults:
    def __init__(self, samples, logl, logwt, ncall, logz, nlive):
        self.samples = samples
        self.logl = logl
        self.logwt = logwt
        self.ncall = ncall
        self.logz = logz
        self.nlive = nlive


class MockDynestySampler:
    def __init__(self, results):
        self.results = results


def test__explicit_params():

    search = af.DynestyStatic(
        nlive=151,
        dlogz=0.1,
        iterations_per_full_update=501,
        number_of_cores=2,
    )

    assert search.iterations_per_full_update == 501

    assert search.nlive == 151
    assert search.dlogz == 0.1
    assert search.number_of_cores == 2

    search = af.DynestyStatic()

    assert search.nlive == 50
    assert search.dlogz is None
    assert search.number_of_cores == 1

    search = af.DynestyDynamic(
        facc=0.4,
        iterations_per_full_update=501,
        dlogz_init=0.2,
        number_of_cores=3,
    )

    assert search.iterations_per_full_update == 501

    assert search.facc == 0.4
    assert search.dlogz_init == 0.2
    assert search.number_of_cores == 3

    search = af.DynestyDynamic()

    assert search.facc == 0.2
    assert search.dlogz_init == 0.01
    assert search.number_of_cores == 1


@pytest.mark.parametrize("search_cls", [af.DynestyStatic, af.DynestyDynamic])
def test__single_core_builds_no_pool(search_cls, monkeypatch):
    """
    number_of_cores=1 must not construct a multiprocessing pool: dynesty treats
    pool=None as fully serial, whereas a Pool(1) object forces every likelihood
    call into a forked worker — which deadlocks in XLA compilation when the
    likelihood touches JAX, and hangs forever if that single worker dies,
    because the pool never re-issues a dead worker's in-flight task. Nautilus
    received the same fix in #1442/#1443; this is #1630 for dynesty.
    """
    import importlib

    def no_fork_context(*args, **kwargs):
        raise AssertionError("fork_context must not be used when number_of_cores == 1")

    parallel = importlib.import_module("autofit.non_linear.parallel")

    # `_fork_pool_cls` imports `fork_context` from the package at call time, so
    # patching the package attribute is what intercepts the pool build.
    monkeypatch.setattr(parallel, "fork_context", no_fork_context)

    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")

    model = af.Model(af.ex.Gaussian)
    analysis = af.ex.Analysis(
        data=np.full(100, 5.0),
        noise_map=np.full(100, 1.0),
    )

    live_point_kwargs = (
        {"nlive": 10} if search_cls is af.DynestyStatic else {"nlive_init": 10}
    )

    search = search_cls(
        name="dynesty_single_core",
        unique_tag="single_core_no_pool_test",
        number_of_cores=1,
        **live_point_kwargs,
    )

    search.fit(model=model, analysis=analysis)


class _FakeRunSampler:
    """
    A stand-in for a dynesty sampler whose `run_nested` appends a fixed number of
    likelihood calls per run to `results.ncall`, mirroring how dynesty's
    `sum(results.ncall)` grows across successive `run_nested` calls.
    """

    def __init__(self, calls_per_run):
        self.calls_per_run = list(calls_per_run)
        self.results = MockDynestyResults(
            samples=None, logl=None, logwt=None, ncall=[], logz=None, nlive=None
        )
        self.maxcalls = []

    def run_nested(self, maxcall, **kwargs):
        self.maxcalls.append(maxcall)
        self.results.ncall.append(self.calls_per_run.pop(0))


def _search_with_directory_paths(name, **kwargs):
    search = af.DynestyStatic(nlive=20, number_of_cores=1, silence=True, **kwargs)
    search.paths = af.DirectoryPaths(name=name)
    return search


def test__run_search_internal__converged_first_pass_is_finished():
    """
    Under the default `iterations_per_full_update` (1e99) dynesty stops on its own
    convergence criterion, far inside the budget it was handed, so the first pass is
    finished: no second `perform_update` + checkpoint restore + no-op `run_nested`.
    """
    search = _search_with_directory_paths("dynesty_converged_first_pass")
    sampler = _FakeRunSampler(calls_per_run=[1641])

    assert search.run_search_internal(search_internal=sampler) is True
    assert len(sampler.maxcalls) == 1


def test__run_search_internal__chunk_limited_pass_is_not_finished():
    """
    With a finite cadence dynesty's per-run counter stops only once it is strictly
    above `maxcall`, so a pass that consumed more calls than its budget was cut by
    the budget, not by convergence, and the search must loop again.
    """
    search = _search_with_directory_paths(
        "dynesty_chunk_limited", iterations_per_full_update=50
    )
    sampler = _FakeRunSampler(calls_per_run=[51])

    assert search.run_search_internal(search_internal=sampler) is False
    assert sampler.maxcalls == [50]


def test__run_search_internal__global_maxcall_is_finished():
    """
    Reaching the search's global `maxcall` finishes the search even though the pass
    overran its budget.
    """
    search = _search_with_directory_paths("dynesty_global_maxcall", maxcall=100)
    sampler = _FakeRunSampler(calls_per_run=[120])

    assert search.run_search_internal(search_internal=sampler) is True
    assert sampler.maxcalls == [100]


def test__fit__finite_cadence_loops_through_intermediate_updates(monkeypatch):
    """
    A finite `iterations_per_full_update` still chunks the run: two budget-limited
    passes (60 calls each against a budget of 50) followed by a converged pass (30
    calls) give three `run_nested` calls and two intermediate `perform_update`s.
    """
    search = _search_with_directory_paths(
        "dynesty_finite_cadence", iterations_per_full_update=50
    )
    sampler = _FakeRunSampler(calls_per_run=[60, 60, 30])

    monkeypatch.setattr(
        search, "search_internal_from", lambda *args, **kwargs: sampler
    )

    updates = []
    monkeypatch.setattr(
        search,
        "perform_update",
        lambda *args, **kwargs: updates.append(kwargs.get("during_analysis")),
    )

    model = af.Model(af.ex.Gaussian)
    analysis = af.ex.Analysis(data=np.full(10, 1.0), noise_map=np.full(10, 1.0))

    search._fit(model=model, analysis=analysis)

    assert sampler.maxcalls == [50, 50, 50]
    assert updates == [True, True]


@pytest.mark.parametrize(
    "search_cls, kwargs",
    [
        (af.DynestyStatic, {"nlive": 20}),
        (af.DynestyDynamic, {"nlive_init": 20, "maxiter": 400}),
    ],
)
def test__fit__real_dynesty_converges_in_a_single_run_nested(
    search_cls, kwargs, monkeypatch
):
    """
    A real dynesty fit under the default update cadence calls `run_nested` exactly
    once and performs no `during_analysis=True` update: the converged first pass is
    recognised as finished (#1642). Before the fix every search ran a second, no-op
    `run_nested` after an intermediate `perform_update` and a checkpoint restore.
    """
    import dynesty.sampler
    import dynesty.dynamicsampler
    from autofit.non_linear.search.abstract_search import NonLinearSearch

    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)

    calls = []

    for cls in (dynesty.sampler.Sampler, dynesty.dynamicsampler.DynamicSampler):
        original = cls.run_nested

        def counted(self, *args, _original=original, **kw):
            calls.append(type(self).__name__)
            return _original(self, *args, **kw)

        monkeypatch.setattr(cls, "run_nested", counted)

    updates = []
    original_perform_update = NonLinearSearch.perform_update

    def recorded(self, *args, **kw):
        updates.append(kw.get("during_analysis"))
        return original_perform_update(self, *args, **kw)

    monkeypatch.setattr(NonLinearSearch, "perform_update", recorded)

    model = af.Model(af.ex.Gaussian)
    model.normalization = 1.0
    model.sigma = 5.0
    model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)

    xvalues = np.arange(100.0)
    data = af.ex.Gaussian(centre=50.0, normalization=1.0, sigma=5.0).model_data_from(
        xvalues
    )
    analysis = af.ex.Analysis(data=data, noise_map=np.full(100, 0.1))

    search = search_cls(
        name=f"dynesty_single_run_nested_{search_cls.__name__}",
        number_of_cores=1,
        silence=True,
        **kwargs,
    )

    search.fit(model=model, analysis=analysis)

    assert len(calls) == 1
    assert True not in updates
