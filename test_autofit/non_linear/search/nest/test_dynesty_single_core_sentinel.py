"""
Dynesty's single-core control flow (search-extensibility A3b; survey 01 §8, survey 02
§3.7).

``AbstractDynesty._fit`` used ``raise RuntimeError`` inside a ``try`` to select its
single-core path, so its ``except RuntimeError`` also swallowed any ``RuntimeError``
raised while a pooled run executed (a genuine dynesty error, or JAX's
``XlaRuntimeError``, which subclasses it) and silently restarted the run single-core.
The control flow is now the private ``_SingleCoreRun`` sentinel: only a failure to
*create* the pool still falls back.

The backend is stubbed (``search_internal_from`` / ``run_search_internal``), so no
sampler runs.
"""

import pytest

import autofit as af
from autofit.non_linear.search.nest.dynesty.search import abstract as dynesty_abstract

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


class XlaRuntimeErrorLike(RuntimeError):
    """
    Stands in for ``jaxlib.xla_extension.XlaRuntimeError``, a ``RuntimeError`` subclass.
    """


class _Analysis(af.Analysis):
    def log_likelihood_function(self, instance):
        return 0.0


class _RecordingPool:
    """
    A context-manager pool whose ``__enter__`` can be made to fail like a platform
    without working multiprocessing.
    """

    fail_on_enter = False
    entered = 0

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        if self.fail_on_enter:
            raise RuntimeError("multiprocessing is not available here")
        type(self).entered += 1
        return self

    def __exit__(self, *args):
        return False


def _search(monkeypatch, pool_cls, run_search_internal):
    search = af.DynestyStatic(number_of_cores=2)

    pools = []

    def search_internal_from(model, fitness, checkpoint_exists, pool, queue_size):
        pools.append(pool)
        return object()

    monkeypatch.setattr(dynesty_abstract, "_fork_pool_cls", lambda: pool_cls)
    monkeypatch.setattr(search, "search_internal_from", search_internal_from)
    monkeypatch.setattr(search, "run_search_internal", run_search_internal)
    return search, pools


def test_runtime_error_inside_a_pooled_run_propagates(monkeypatch):
    calls = []

    def run_search_internal(search_internal):
        calls.append(search_internal)
        raise XlaRuntimeErrorLike("RESOURCE_EXHAUSTED: out of memory")

    class Pool(_RecordingPool):
        entered = 0

    search, pools = _search(monkeypatch, Pool, run_search_internal)

    with pytest.raises(XlaRuntimeErrorLike, match="RESOURCE_EXHAUSTED"):
        search._fit(model=af.Model(af.ex.Gaussian), analysis=_Analysis())

    # One pooled attempt, no silent single-core rerun.
    assert len(calls) == 1
    assert len(pools) == 1 and pools[0] is not None
    assert Pool.entered == 1


def test_pool_creation_failure_still_falls_back_to_single_core(monkeypatch, caplog):
    def run_search_internal(search_internal):
        return True

    class Pool(_RecordingPool):
        fail_on_enter = True

    search, pools = _search(monkeypatch, Pool, run_search_internal)

    with caplog.at_level("INFO"):
        search._fit(model=af.Model(af.ex.Gaussian), analysis=_Analysis())

    assert pools == [None]
    assert "multiprocessing pool could not be created" in caplog.text
    assert "multiprocessing is not available here" in caplog.text


def test_single_core_is_selected_without_a_runtime_error(monkeypatch):
    def run_search_internal(search_internal):
        return True

    def pool_must_not_be_built():
        raise AssertionError("a single-core fit built a pool")

    search = af.DynestyStatic(number_of_cores=1)
    pools = []

    monkeypatch.setattr(dynesty_abstract, "_fork_pool_cls", pool_must_not_be_built)
    monkeypatch.setattr(
        search,
        "search_internal_from",
        lambda model, fitness, checkpoint_exists, pool, queue_size: pools.append(pool),
    )
    monkeypatch.setattr(search, "run_search_internal", run_search_internal)

    search._fit(model=af.Model(af.ex.Gaussian), analysis=_Analysis())

    assert pools == [None]
    assert not issubclass(dynesty_abstract._SingleCoreRun, RuntimeError)


def test_runtime_error_in_a_single_core_run_propagates(monkeypatch):
    def run_search_internal(search_internal):
        raise XlaRuntimeErrorLike("INTERNAL")

    search = af.DynestyStatic(number_of_cores=1)
    monkeypatch.setattr(
        search,
        "search_internal_from",
        lambda model, fitness, checkpoint_exists, pool, queue_size: object(),
    )
    monkeypatch.setattr(search, "run_search_internal", run_search_internal)

    with pytest.raises(XlaRuntimeErrorLike):
        search._fit(model=af.Model(af.ex.Gaussian), analysis=_Analysis())
