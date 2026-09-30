import importlib.util

import numpy as np
import pytest

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

# The regression test below runs a real `search.fit`, which imports the
# `nautilus` sampler; it ships via the `[optional]` extras. Envs installed
# without those extras must skip rather than fail with
# `No module named 'nautilus'` — the other tests here only read config.
requires_nautilus = pytest.mark.skipif(
    importlib.util.find_spec("nautilus") is None,
    reason="requires nautilus-sampler (installed via the [optional] extras)",
)


@pytest.mark.parametrize(
    "steps, limit, expected_budgets, expected_updates",
    [
        ([(40, True)], None, [300], 0),
        ([(320, False), (360, True)], None, [300, 620], 1),
        ([(320, False), (640, False)], 600, [300, 600], 1),
        ([(300, False)], 300, [300], 0),
        ([(0, False)], None, [300], 0),
    ],
)
def test__completion_and_likelihood_budgets(
    monkeypatch, steps, limit, expected_budgets, expected_updates
):
    """Convergence ends immediately; incomplete chunks advance by real calls."""
    search = af.Nautilus(
        name="completion_budget", n_live=10,
        iterations_per_full_update=300, n_like_max=limit,
    )
    updates = []
    monkeypatch.setattr(search, "perform_update", lambda **kw: updates.append(kw))

    class Sampler:
        n_like = 0

        def __init__(self):
            self.budgets = []

        def run(self, **kwargs):
            self.budgets.append(kwargs["n_like_max"])
            self.n_like, converged = steps[len(self.budgets) - 1]
            return converged

        def posterior(self):
            raise AssertionError("Posterior sample count is not a call budget")

    sampler = Sampler()
    assert search.call_search(sampler, None, None, None) is sampler
    assert sampler.budgets == expected_budgets
    assert len(updates) == expected_updates
    assert all(update["during_analysis"] for update in updates)


@pytest.mark.parametrize("limit", [None, float("inf"), 100])
def test__null_paths_likelihood_budget(limit):
    from types import SimpleNamespace
    from autofit.non_linear.paths.null import NullPaths

    search = af.Nautilus(n_like_max=limit)
    assert isinstance(search.paths, NullPaths)
    budget, calls = search.iterations_from(SimpleNamespace(n_like=40))
    assert calls == 40
    assert budget == (100 if limit == 100 else int(1e99))


def test__explicit_params():
    search = af.Nautilus(
        n_live=500,
        n_batch=200,
        f_live=0.05,
        n_eff=1000,
        iterations_per_full_update=501,
        number_of_cores=4,
    )

    assert search.iterations_per_full_update == 501

    assert search.n_live == 500
    assert search.n_batch == 200
    assert search.f_live == 0.05
    assert search.n_eff == 1000
    assert search.number_of_cores == 4

    search = af.Nautilus()

    assert search.n_live == 3000
    assert search.n_batch == 100
    assert search.enlarge_per_dim == 1.1
    assert search.split_threshold == 100
    assert search.n_networks == 4
    assert search.vectorized is False
    assert search.f_live == 0.01
    assert search.n_shell == 1
    assert search.n_eff == 500
    assert search.discard_exploration is False
    assert search.number_of_cores == 1


def test__identifier_fields():
    search = af.Nautilus()

    assert "n_live" in search.__identifier_fields__
    assert "n_eff" in search.__identifier_fields__
    assert "n_shell" in search.__identifier_fields__


def test__test_mode():
    search = af.Nautilus()
    search.apply_test_mode()

    assert search.n_like_max == 1


@requires_nautilus
def test__single_core_builds_no_pool(monkeypatch):
    """
    number_of_cores=1 must not construct a multiprocessing pool: nautilus
    treats pool=None as fully serial, whereas a Pool(1) object forces every
    likelihood call into a forked worker — which deadlocks in XLA compilation
    when the likelihood touches JAX (#1442).
    """
    from autofit.non_linear.search.nest.nautilus import search as nautilus_search

    def no_fork_context():
        raise AssertionError(
            "fork_context must not be used when number_of_cores == 1"
        )

    monkeypatch.setattr(nautilus_search, "fork_context", no_fork_context)
    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")

    model = af.Model(af.ex.Gaussian)
    analysis = af.ex.Analysis(
        data=np.full(100, 5.0),
        noise_map=np.full(100, 1.0),
    )

    search = af.Nautilus(
        name="nautilus_single_core",
        unique_tag="single_core_no_pool_test",
        n_live=10,
        number_of_cores=1,
    )

    search.fit(model=model, analysis=analysis)


@requires_nautilus
def test__multi_core_passes_serial_sampler_pool(monkeypatch):
    """
    number_of_cores > 1 must hand nautilus the tuple `(pool, None)`: the fork
    pool for likelihood evaluations (`pool_l`) and no pool for the sampler's
    own calculations (`pool_s`).

    The pool itself is handed over wrapped in `_LikelihoodWorkerPool`, whose
    `map` dispatches nautilus's `likelihood_worker` so the `Fitness` is not
    pickled into every task chunk (#1547).

    Nautilus otherwise trains its neural bounds on the same pool via
    `pool.map`; a worker that dies mid-fit is replaced by
    `multiprocessing.Pool` but its task is never re-issued, so `map` blocks
    forever and the fit hangs (#1547).
    """
    from autofit.non_linear.search.nest.nautilus import search as nautilus_search

    class StopFit(Exception):
        pass

    captured = {}

    class StubPool:
        _processes = 2
        # `_LikelihoodWorkerPool` reads the pool's worker list at construction
        # to capture the workers its watchdog checks; an empty list simply
        # gives it nothing to watch.
        _pool = []

        def close(self):
            captured["closed"] = True

        def terminate(self):
            captured["terminated"] = True

        def join(self):
            captured["joined"] = True

    class StubSampler:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            raise StopFit

    class StubContext:
        def Pool(self, number_of_cores, initializer=None, initargs=()):
            captured["number_of_cores"] = number_of_cores
            captured["initializer"] = initializer
            captured["initargs"] = initargs
            return StubPool()

    monkeypatch.setattr(nautilus_search, "fork_context", lambda: StubContext())
    monkeypatch.setattr(
        af.Nautilus, "sampler_cls", property(lambda self: StubSampler)
    )
    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")

    model = af.Model(af.ex.Gaussian)
    analysis = af.ex.Analysis(
        data=np.full(100, 5.0),
        noise_map=np.full(100, 1.0),
    )

    search = af.Nautilus(
        name="nautilus_multi_core",
        unique_tag="multi_core_serial_sampler_pool_test",
        n_live=10,
        number_of_cores=2,
    )

    with pytest.raises(StopFit):
        search.fit(model=model, analysis=analysis)

    from nautilus.pool import initialize_worker

    assert captured["number_of_cores"] == 2
    assert captured["initializer"] is initialize_worker
    assert len(captured["initargs"]) == 1

    pool = captured["pool"]

    assert isinstance(pool, tuple)
    assert pool[1] is None
    assert isinstance(pool[0], nautilus_search._LikelihoodWorkerPool)
    assert pool[0].size == 2

    captured.clear()

    search = af.Nautilus(
        name="nautilus_single_core_pool_kwarg",
        unique_tag="single_core_serial_sampler_pool_test",
        n_live=10,
        number_of_cores=1,
    )

    with pytest.raises(StopFit):
        search.fit(model=model, analysis=analysis)

    assert captured["pool"] is None


@requires_nautilus
def test__likelihood_pool_does_not_pickle_fitness():
    """
    `_LikelihoodWorkerPool.map` must ignore the function nautilus maps with and
    dispatch `nautilus.pool.likelihood_worker`, which reads the likelihood from
    the worker-global set by `initialize_worker` at fork time.

    If the mapped function were dispatched instead, `multiprocessing` would
    pickle the bound `Fitness.call_wrap` — and with it the analysis, dataset,
    PSF convolver and sparse operator — into every task chunk, for every
    likelihood call (#1547).
    """
    import multiprocessing

    from nautilus.pool import initialize_worker

    from autofit.non_linear.search.nest.nautilus.search import _LikelihoodWorkerPool

    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("requires the fork start method")

    likelihood = _UnpicklableLikelihood()

    context = multiprocessing.get_context("fork")
    pool = context.Pool(
        2,
        initializer=initialize_worker,
        initargs=(likelihood,),
    )

    try:
        with _LikelihoodWorkerPool(pool) as wrapper:
            assert wrapper.size == 2

            result = wrapper.map(
                _must_not_be_dispatched,
                [(1, 2), (3, 4)],
            )

            assert list(result) == [3, 7]
    finally:
        pool.terminate()
        pool.join()


class _UnpicklableLikelihood:
    """
    A stand-in for the `Fitness`: it is inherited by the forked workers, but
    raises if anything tries to pickle it into a task.
    """

    def __getstate__(self):
        raise AssertionError("Fitness must not be pickled")

    def __call__(self, args):
        return sum(args)


def _must_not_be_dispatched(*args):
    raise AssertionError("the mapped function must be ignored")


def _sleepy_likelihood(args):
    """
    A module-level (and therefore picklable) stand-in for the `Fitness`, slow
    enough that a worker can be killed while the map is still in flight.
    """
    import time

    time.sleep(0.2)
    return args


def _ignored(*args):
    raise AssertionError("the mapped function must be ignored")


@requires_nautilus
def test__likelihood_pool_raises_when_worker_dies(monkeypatch):
    """
    A worker that dies mid-`map` must fail the fit, not hang it.

    `multiprocessing.Pool` replaces a dead worker (`_maintain_pool` ->
    `_repopulate_pool`) but never re-issues the task it was running, so a plain
    `Pool.map` blocks to the wall clock — RAL job 342351_0 hung 27 hours this
    way (#1608). `_LikelihoodWorkerPool.map` polls the exit code of every
    worker captured at construction, so the replacement is detectable (the
    count is restored, the original worker stays dead) and raises within
    seconds.
    """
    import multiprocessing
    import os
    import signal
    import threading
    import time

    from nautilus.pool import initialize_worker

    from autofit.non_linear.search.nest.nautilus import search as nautilus_search

    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("requires the fork start method")

    monkeypatch.setattr(nautilus_search, "LIKELIHOOD_POOL_POLL_INTERVAL", 0.1)

    context = multiprocessing.get_context("fork")
    pool = context.Pool(
        2,
        initializer=initialize_worker,
        initargs=(_sleepy_likelihood,),
    )

    try:
        wrapper = nautilus_search._LikelihoodWorkerPool(pool)

        victim_pid = pool._pool[0].pid

        killer = threading.Timer(0.3, os.kill, args=(victim_pid, signal.SIGKILL))
        killer.start()

        start = time.time()

        try:
            with pytest.raises(af.exc.SearchException) as error:
                wrapper.map(_ignored, list(range(40)))
        finally:
            killer.cancel()

        elapsed = time.time() - start

        assert elapsed < 5.0

        message = str(error.value)

        assert str(victim_pid) in message
        assert "-9" in message
        assert "multiprocessing.Pool" in message
    finally:
        pool.terminate()
        pool.join()


@requires_nautilus
def test__likelihood_pool_healthy_map_completes(monkeypatch):
    """
    The `map_async` + polling loop must be transparent when nothing dies: the
    results come back complete and in order, across more tasks than workers.
    """
    import multiprocessing

    from nautilus.pool import initialize_worker

    from autofit.non_linear.search.nest.nautilus import search as nautilus_search

    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("requires the fork start method")

    monkeypatch.setattr(nautilus_search, "LIKELIHOOD_POOL_POLL_INTERVAL", 0.1)

    context = multiprocessing.get_context("fork")
    pool = context.Pool(
        2,
        initializer=initialize_worker,
        initargs=(_sleepy_likelihood,),
    )

    try:
        wrapper = nautilus_search._LikelihoodWorkerPool(pool)

        assert list(wrapper.map(_ignored, list(range(40)))) == list(range(40))
    finally:
        pool.terminate()
        pool.join()
