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


@requires_nautilus
def test__multi_core_prepares_the_fitness_before_forking(monkeypatch):
    """
    `fit_multiprocessing` runs the likelihood once in the parent (`Fitness.prepare_for_fork`)
    before it creates the fork pool, so numba's lazy import / first compile is not done by every
    worker at once (PyAutoArray#654: >= 16 workers segfaulted in llvmlite).
    """
    from autofit.non_linear.fitness import Fitness
    from autofit.non_linear.search.nest.nautilus import search as nautilus_search

    class StopFit(Exception):
        pass

    events = []

    class StubPool:
        _processes = 2
        _pool = []

        def close(self):
            pass

        def terminate(self):
            pass

        def join(self):
            pass

    class StubSampler:
        def __init__(self, **kwargs):
            raise StopFit

    class StubContext:
        def Pool(self, number_of_cores, initializer=None, initargs=()):
            events.append("pool")
            return StubPool()

    original = Fitness.prepare_for_fork

    def recording_prepare_for_fork(self):
        events.append("prepare")
        original(self)

    monkeypatch.setattr(Fitness, "prepare_for_fork", recording_prepare_for_fork)
    monkeypatch.setattr(nautilus_search, "fork_context", lambda: StubContext())
    monkeypatch.setattr(
        af.Nautilus, "sampler_cls", property(lambda self: StubSampler)
    )
    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")

    search = af.Nautilus(
        name="nautilus_prepare_for_fork",
        unique_tag="nautilus_prepare_for_fork_test",
        n_live=10,
        number_of_cores=2,
    )

    with pytest.raises(StopFit):
        search.fit(
            model=af.Model(af.ex.Gaussian),
            analysis=af.ex.Analysis(
                data=np.full(100, 5.0), noise_map=np.full(100, 1.0)
            ),
        )

    assert events[:2] == ["prepare", "pool"]


class _FakeWorker:
    def __init__(self, pid, exitcode):
        self.pid = pid
        self.exitcode = exitcode


class _FakePool:
    """A pool double exposing only what the dead-worker watchdog reads."""

    def __init__(self, workers):
        self._pool = workers
        self._processes = len(workers)
        self.terminated = False

    def terminate(self):
        self.terminated = True


def _watchdog_message(exitcode):
    from autofit.non_linear.search.nest.nautilus import search as nautilus_search

    pool = _FakePool([_FakeWorker(101, None), _FakeWorker(102, None)])
    wrapper = nautilus_search._LikelihoodWorkerPool(pool)

    pool._pool[1].exitcode = exitcode

    with pytest.raises(af.exc.SearchException) as error:
        wrapper._check_workers_alive()

    assert pool.terminated

    return str(error.value)


def test__likelihood_pool__sigsegv_names_native_library_initialisation():
    message = _watchdog_message(exitcode=-11)

    assert "pid 102" in message
    assert "segmentation fault" in message
    assert "numba" in message
    assert "prepare_for_fork" in message
    assert "number_of_cores=1" in message
    assert "report it" in message


def test__likelihood_pool__other_deaths_keep_the_generic_message():
    for exitcode in (-9, 1):
        message = _watchdog_message(exitcode=exitcode)

        assert "pid 102" in message
        assert "multiprocessing.Pool" in message
        assert "segmentation fault" not in message


# --- D >= 91: the live-point guard and the HDF5 checkpoint format (#1697) ---


class _VectorGaussianAnalysis(af.Analysis):
    """
    An isotropic Gaussian over every free parameter, counting likelihood calls.
    """

    def __init__(self, n_dim):
        super().__init__()
        self.names = [f"x{index}" for index in range(n_dim)]
        self.calls = 0

    def log_likelihood_function(self, instance):
        self.calls += 1
        vector = np.array([getattr(instance, name) for name in self.names])
        return -0.5 * float(np.sum(((vector - 0.5) / 0.1) ** 2))


def _vector_model(n_dim):
    return af.Collection(
        **{f"x{index}": af.UniformPrior(lower_limit=0.0, upper_limit=1.0) for index in range(n_dim)}
    )


def _ellipsoid_d100():
    from nautilus.bounds.basic import Ellipsoid

    rng = np.random.default_rng(1)
    return Ellipsoid.compute(rng.random((300, 100)))


@requires_nautilus
def test__d100_ellipsoid_round_trips_through_the_proxied_file(tmp_path):
    """
    A D = 100 ellipsoid's ``n_dim x n_dim`` attributes write and read back when the
    file is opened through the proxy's ``File`` (HDF5 1.8 format).
    """
    import h5py
    from nautilus.bounds.basic import Ellipsoid

    from autofit.non_linear.search.nest.nautilus.search import _LargeAttributeH5py

    bound = _ellipsoid_d100()
    filepath = tmp_path / "checkpoint.hdf5"

    with _LargeAttributeH5py(h5py).File(filepath, "x") as fstream:
        bound.write(fstream.create_group("bound"))

    with h5py.File(filepath, "r+") as fstream:
        read = Ellipsoid.read(fstream["bound"])

    assert np.array_equal(read.B, bound.B)
    assert np.array_equal(read.B_inv, bound.B_inv)


@requires_nautilus
@pytest.mark.xfail(
    strict=True,
    raises=OSError,
    reason="nautilus 1.0.5 writes D x D attributes in HDF5's earliest format; "
    "XPASSes (and fails this strict xfail) once nautilus fixes it upstream, "
    "the cue to remove _large_attribute_hdf5",
)
def test__d100_ellipsoid_write_fails_in_nautilus_default_format(tmp_path):
    """
    The control: nautilus's own ``h5py.File(filepath, 'x')`` cannot hold the bound.
    """
    import h5py

    bound = _ellipsoid_d100()

    with h5py.File(tmp_path / "checkpoint.hdf5", "x") as fstream:
        bound.write(fstream.create_group("bound"))


def test__large_attribute_hdf5_restores_nautilus_h5py():
    pytest.importorskip("nautilus")
    import nautilus.sampler as nautilus_sampler

    from autofit.non_linear.search.nest.nautilus import search as nautilus_search

    original = nautilus_sampler.h5py

    with pytest.raises(RuntimeError):
        with nautilus_search._large_attribute_hdf5():
            assert isinstance(nautilus_sampler.h5py, nautilus_search._LargeAttributeH5py)
            raise RuntimeError

    assert nautilus_sampler.h5py is original


class _StopAtFirstUpdate(Exception):
    pass


def _fast_ellipsoid_bound_cls():
    """
    A stand-in for nautilus's ``NautilusBound`` that keeps its D x D ellipsoid
    matrices but none of its cost.

    With ~100 live points at D = 100, nautilus's real bound rejection-samples an
    enlarged ellipsoid against the unit cube (``Union.sample`` /
    ``NeuralBound.sample``) with a near-zero acceptance rate, which takes minutes, so
    the test cannot afford it. This bound is a real ``Ellipsoid`` fitted to the live
    set (written through ``Ellipsoid.write``: the ``A``, ``B``, ``B_inv`` attributes
    that overflow nautilus's default HDF5 format), sampled from the unit cube and
    declared smaller than it so the sampler accepts it and writes the checkpoint.
    """
    from nautilus.bounds.basic import Ellipsoid

    class FastEllipsoidBound(Ellipsoid):
        @classmethod
        def compute(cls, points, log_l, log_l_min, log_v_target, enlarge_per_dim=1.1, rng=None, **kwargs):
            bound = super().compute(
                points[log_l >= log_l_min], enlarge_per_dim=enlarge_per_dim, rng=rng
            )
            return bound

        @property
        def log_v(self):
            return -1.0

        def sample(self, n_points=100, return_points=True, pool=None):
            points = self.rng.random((n_points, self.n_dim))
            return points if return_points else None

        def update(self, group):
            pass

    return FastEllipsoidBound


@requires_nautilus
def test__d100_fit_writes_a_checkpoint_that_nautilus_resumes(tmp_path, monkeypatch):
    """
    A D = 100 fit reaches its first checkpoint write after an ellipsoid bound
    (nautilus-sampler 1.0.5 raised ``object header message is too large`` there),
    and a fresh nautilus ``Sampler`` on the same file resumes it.

    The fit is stopped at its first full update, as an interrupted run would be:
    a completed fit deletes its resume state. nautilus's bound class is replaced by
    ``_fast_ellipsoid_bound_cls`` (same ellipsoid serialisation, no rejection
    sampling) to keep the test to seconds; everything else is the search's real
    ``fit`` -> ``run(ctx)`` -> ``fit_x1_cpu`` path and nautilus's real write and
    resume.
    """
    from nautilus import Sampler

    from autofit.conf import instance

    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)
    monkeypatch.setattr(instance, "output_path", str(tmp_path))

    n_dim = 100
    model = _vector_model(n_dim)
    analysis = _VectorGaussianAnalysis(n_dim)

    search = af.Nautilus(
        name="nautilus_d100_checkpoint",
        n_live=101,
        n_networks=1,
        n_batch=101,
        n_like_max=404,
        seed=1,
        verbose=False,
    )

    import nautilus.sampler as nautilus_sampler

    monkeypatch.setattr(nautilus_sampler, "NautilusBound", _fast_ellipsoid_bound_cls())

    stopped = {}

    def stop(search_internal=None, **kwargs):
        stopped["n_like"] = search_internal.n_like
        stopped["bounds"] = len(search_internal.bounds)
        raise _StopAtFirstUpdate

    monkeypatch.setattr(search, "perform_update", stop)

    with pytest.raises(_StopAtFirstUpdate):
        search.fit(model=model, analysis=analysis)

    checkpoint_file = search.checkpoint_file

    assert str(checkpoint_file).startswith(str(tmp_path))
    assert checkpoint_file.exists()
    assert stopped["bounds"] >= 2

    resumed = Sampler(
        prior=lambda x: x,
        likelihood=lambda x: 0.0,
        n_dim=n_dim,
        n_live=101,
        filepath=checkpoint_file,
        resume=True,
    )

    assert resumed.n_like == stopped["n_like"]
    assert len(resumed.bounds) >= 2


def test__checkpoint_false_passes_no_filepath():
    search = af.Nautilus(name="nautilus_no_checkpoint", checkpoint=False)

    assert search.checkpoint is False
    assert search.checkpoint_file is None
    assert "checkpoint" not in search.__identifier_fields__

    checkpointed = af.Nautilus(name="nautilus_no_checkpoint")

    assert checkpointed.checkpoint_file is not None


def _guard_fit(search, n_dim, monkeypatch):
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)
    analysis = _VectorGaussianAnalysis(n_dim)

    def backend_must_not_run(*args, **kwargs):
        raise AssertionError("the backend ran before the live-point guard")

    monkeypatch.setattr(search, "_fit", backend_must_not_run)

    with pytest.raises(af.exc.SearchException) as error:
        search.fit(model=_vector_model(n_dim), analysis=analysis)

    assert analysis.calls == 0
    return str(error.value)


def test__n_live_at_prior_count_raises_before_any_likelihood_call(monkeypatch):
    message = _guard_fit(af.Nautilus(n_live=100), 100, monkeypatch)

    assert message.startswith(
        "Nautilus cannot fit this 100-parameter model with n_live=100"
    )
    assert "at least 101" in message


def test__n_points_min_below_floor_raises(monkeypatch):
    message = _guard_fit(af.Nautilus(n_live=200, n_points_min=50), 100, monkeypatch)

    assert "n_points_min=50" in message
    assert "at least 101" in message


def test__live_point_floor_passes_the_gate():
    model = _vector_model(100)

    af.Nautilus(n_live=101).check_model(model)
    af.Nautilus(n_live=101, n_points_min=101).check_model(model)

    with pytest.raises(af.exc.SearchException):
        af.Nautilus(n_live=101, n_points_min=100).check_model(model)


def test__run_ctx_repeats_the_guard():
    from types import SimpleNamespace

    with pytest.raises(af.exc.SearchException, match="n_live=10"):
        af.Nautilus(n_live=10).run(SimpleNamespace(model=_vector_model(20)))
