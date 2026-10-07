import multiprocessing
import sys

import numpy as np
import pytest

import autofit as af
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.parallel import fork_context
from autofit.non_linear.parallel.process import Process
from autofit.non_linear.search.nest.dynesty.search.abstract import (
    _fork_pool_cls,
    prior_transform,
)

# pytest's own machinery keeps a background thread alive, so CPython 3.12+
# flags every os.fork() here as fork-in-a-multi-threaded-process, and JAX
# (itself multithreaded once imported) registers an equivalent RuntimeWarning
# hook. The tests exercise the fork-pinned pool deliberately; the deadlock
# caveat does not apply to these short-lived, likelihood-only workers.
pytestmark = [
    pytest.mark.filterwarnings("ignore:This process:DeprecationWarning"),
    pytest.mark.filterwarnings(r"ignore:os\.fork\(\) was called:RuntimeWarning"),
]

pins_fork = (
    sys.platform != "darwin"
    and "fork" in multiprocessing.get_all_start_methods()
)

requires_fork = pytest.mark.skipif(
    not pins_fork,
    reason="fork start method is not pinned on this platform",
)


@requires_fork
def test_fork_context_is_fork():
    assert fork_context().get_start_method() == "fork"


def test_fork_context_valid_everywhere():
    assert fork_context().get_start_method() in multiprocessing.get_all_start_methods()


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_fork_context_non_fork_platform_returns_module(monkeypatch, platform):
    """
    On macOS (and on platforms without "fork") ``fork_context()`` returns the
    ``multiprocessing`` module itself, which exposes the same Process / Queue /
    Pool API and defers the start-method choice until a process starts.
    """
    monkeypatch.setattr(sys, "platform", platform)
    if platform == "win32":
        monkeypatch.setattr(multiprocessing, "get_all_start_methods", lambda: ["spawn"])

    context = fork_context()

    assert context is multiprocessing
    for name in ("Process", "Queue", "Pool"):
        assert hasattr(context, name)


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_fork_context_non_fork_platform_does_not_fix_start_method(
    monkeypatch, platform
):
    """
    Regression test for PyAutoFit#1657: ``fork_context()`` runs at import time
    (``class Process(fork_context().Process)``), and the no-argument
    ``multiprocessing.get_context()`` call it used to make on macOS/Windows
    fixed the interpreter's default start method, so a later
    ``multiprocessing.set_start_method(...)`` raised "context has already been
    set". The non-fork branch must never call ``get_context()`` without a
    method.
    """
    monkeypatch.setattr(sys, "platform", platform)
    if platform == "win32":
        monkeypatch.setattr(multiprocessing, "get_all_start_methods", lambda: ["spawn"])

    calls = []
    real_get_context = multiprocessing.get_context

    def spy_get_context(*args, **kwargs):
        calls.append((args, kwargs))
        return real_get_context(*args, **kwargs)

    monkeypatch.setattr(multiprocessing, "get_context", spy_get_context)

    fork_context()

    assert ((), {}) not in calls
    assert ((None,), {}) not in calls


@requires_fork
def test_process_class_is_fork_bound():
    assert Process._start_method == "fork"


@pytest.fixture(name="factor_graph_and_fitness")
def make_factor_graph_and_fitness():
    data = np.full(100, 5.0)
    noise_map = np.full(100, 1.0)

    model = af.Collection(
        gaussian=af.Model(af.ex.Gaussian),
        exponential=af.Model(af.ex.Exponential),
    )

    analysis_factor_list = [
        af.AnalysisFactor(
            prior_model=model.copy(),
            analysis=af.ex.Analysis(data=data, noise_map=noise_map),
        )
        for _ in range(2)
    ]

    factor_graph = af.FactorGraphModel(*analysis_factor_list)
    global_prior_model = factor_graph.global_prior_model

    fitness = Fitness(
        model=global_prior_model,
        analysis=factor_graph,
        paths=None,
        fom_is_log_likelihood=True,
        resample_figure_of_merit=-1.0e99,
    )

    return factor_graph, global_prior_model, fitness


@requires_fork
def test_factor_graph_instance_shape_in_pool_worker(factor_graph_and_fitness):
    """
    A pool worker must see the same nested instance structure as the main
    process. On Python 3.14 the default start method became "forkserver",
    under which workers received a flattened instance (a bare Gaussian in
    place of the per-factor Collection) — the failure documented in issue
    #1437. The fork-pinned dynesty pool preserves the structure.
    """
    _, global_prior_model, fitness = factor_graph_and_fitness

    unit_vector = np.full(global_prior_model.prior_count, 0.5)
    in_process_vector = global_prior_model.vector_from_unit_vector(
        unit_vector=unit_vector
    )
    in_process_likelihood = fitness(in_process_vector)

    ForkPool = _fork_pool_cls()

    with ForkPool(
        njobs=2,
        loglike=fitness,
        prior_transform=prior_transform,
        logl_args=(global_prior_model, fitness),
        ptform_args=(global_prior_model,),
    ) as pool:
        worker_vector = list(pool.map(pool.prior_transform, [unit_vector]))[0]
        worker_likelihood = list(pool.map(pool.loglike, [worker_vector]))[0]

    assert worker_likelihood == pytest.approx(in_process_likelihood, rel=1e-10)
