"""
`Fitness.prepare_for_fork`: the one likelihood call a search makes in the parent before it forks
likelihood workers, so lazily initialised native libraries (numba's import and first LLVM compile
since PyAutoArray made numba lazy) are initialised once rather than in every forked worker at once,
which segfaulted >= 16-worker Nautilus fits on RAL (PyAutoArray#654).
"""
import logging
import pickle

import numpy as np
import pytest

import autofit as af
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.parallel import PoolFactory
from autofit.non_linear.parallel import pool as pool_module
from autofit.non_linear.parallel import sneaky as sneaky_module

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


class CountingAnalysis(af.ex.Analysis):
    def __init__(self, *args, raises=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0
        self.raises = raises

    def log_likelihood_function(self, instance, **kwargs):
        self.calls += 1
        if self.raises:
            raise ValueError("broken likelihood")
        return super().log_likelihood_function(instance=instance, **kwargs)


def _fitness(use_jax=False, raises=False, **kwargs):
    analysis = CountingAnalysis(
        data=np.ones(20), noise_map=np.ones(20) * 0.1, use_jax=use_jax, raises=raises
    )
    return Fitness(model=af.Model(af.ex.Gaussian), analysis=analysis, **kwargs)


def test__numpy__calls_the_likelihood_exactly_once_and_records_nothing():
    fitness = _fitness(store_history=True, iterations_per_quick_update=1)

    fitness.prepare_for_fork()
    fitness.prepare_for_fork()

    assert fitness.analysis.calls == 1

    # The warm-up bypasses `call_wrap`: no history and no quick-update progress.
    assert fitness.parameters_history_list == []
    assert fitness.log_likelihood_history_list == []
    assert fitness.quick_update_count == 0


def test__flag_survives_pickling_so_a_worker_never_repeats_it():
    fitness = _fitness()
    fitness.prepare_for_fork()

    restored = pickle.loads(pickle.dumps(fitness))
    restored.prepare_for_fork()

    assert restored.analysis.calls == 1

    # A fitness pickled before the flag existed simply warms up once.
    del fitness.__dict__["_prepared_for_fork"]
    legacy = pickle.loads(pickle.dumps(fitness))
    legacy.prepare_for_fork()

    assert legacy.analysis.calls == 2


def test__a_raising_likelihood_only_warns(caplog):
    fitness = _fitness(raises=True)

    with caplog.at_level(logging.WARNING):
        fitness.prepare_for_fork()

    assert fitness.analysis.calls == 1
    assert "warm-up before forking workers failed" in caplog.text
    assert "broken likelihood" in caplog.text


def test__jax_analysis_is_a_no_op():
    pytest.importorskip("jax")

    fitness = _fitness(use_jax=True)
    calls_after_init = fitness.analysis.calls

    fitness.prepare_for_fork()

    assert fitness.analysis.calls == calls_after_init


class _Analysis:
    def __init__(self, is_jax):
        self.is_jax = is_jax


class _RecordingFitness:
    def __init__(self, events):
        self.events = events

    def prepare_for_fork(self):
        self.events.append("prepare")


def test__pool_factory_call_prepares_the_fitness_before_forking(monkeypatch):
    events = []

    class StubContext:
        def Pool(self, processes, **kwargs):
            events.append("pool")
            return object()

    monkeypatch.setattr(pool_module, "fork_context", lambda: StubContext())

    factory = PoolFactory(af.Emcee(number_of_cores=2), _Analysis(is_jax=False))
    factory.fitness = _RecordingFitness(events)

    factory()

    assert events == ["prepare", "pool"]

    # One core forks nothing, so nothing is warmed up.
    events.clear()
    factory = PoolFactory(af.Emcee(number_of_cores=1), _Analysis(is_jax=False))
    factory.fitness = _RecordingFitness(events)

    assert factory() is None
    factory.prepare_for_fork()
    assert events == []


def test__sneaky_pool_prepares_the_fitness_before_starting_processes(monkeypatch):
    events = []

    class StubProcess:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            events.append("start")

    monkeypatch.setattr(sneaky_module, "SneakyProcess", StubProcess)

    sneaky_module.SneakyPool(
        processes=2, fitness=_RecordingFitness(events), paths=None
    )

    assert events == ["prepare", "start", "start"]


def test__fit_hands_the_fitness_to_the_pool_factory(monkeypatch):
    """
    Searches that fork via `ctx.pool()` (Zeus, Dynesty) rely on the fit's `PoolFactory`
    knowing the fitness, so its `__call__` can warm it up.
    """
    captured = {}

    class StopFit(Exception):
        pass

    def run(self, ctx):
        captured["ctx"] = ctx
        raise StopFit

    monkeypatch.setattr(af.Emcee, "run", run)
    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")

    search = af.Emcee(
        name="prepare_for_fork_pool_factory",
        unique_tag="prepare_for_fork_pool_factory_test",
        number_of_cores=2,
    )

    with pytest.raises(StopFit):
        search.fit(
            model=af.Model(af.ex.Gaussian),
            analysis=af.ex.Analysis(data=np.ones(20), noise_map=np.ones(20)),
        )

    ctx = captured["ctx"]

    assert ctx.pool.fitness is ctx.fitness
