"""
The minimal `run(ctx)` bridge (search-extensibility phase A2, `docs/design/run_ctx.md`).

`Drawer` and `Nautilus` run through `run(ctx)`; a search that overrides `_fit` keeps
working; a toy search written only against the frozen contract fits end to end.
"""
import numpy as np
import pytest

import autofit as af
from autofit.non_linear.search import capabilities as cap
from autofit.non_linear.search.abstract_search import NonLinearSearch
from autofit.non_linear.search.fit_context import FitContext, RawSamples

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

FROZEN_MEMBERS = (
    "model",
    "paths",
    "objective",
    "fitness",
    "test_mode_level",
    "pool",
    "start_points",
    "rng",
    "resume",
    "checkpointer",
    "schedule",
    "update",
)


def _analysis():
    return af.ex.Analysis(
        data=np.ones(20), noise_map=np.ones(20) * 0.1, use_jax=False
    )


@pytest.mark.parametrize("search_cls", [af.Drawer, af.Nautilus])
def test_proofs_are_on_the_bridge(search_cls):
    assert search_cls.run is not NonLinearSearch.run
    assert search_cls.raw_samples_from is not NonLinearSearch.raw_samples_from
    assert search_cls._fit is NonLinearSearch._fit


def test_drawer_fit_runs_through_run_ctx(monkeypatch):
    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")

    seen = []
    original = af.Drawer.run

    def spy(self, ctx):
        seen.append(ctx)
        return original(self, ctx)

    monkeypatch.setattr(af.Drawer, "run", spy)

    search = af.Drawer(total_draws=5)
    result = search.fit(model=af.Model(af.ex.Gaussian), analysis=_analysis())

    assert len(seen) == 1
    ctx = seen[0]
    assert isinstance(ctx, FitContext)
    for member in FROZEN_MEMBERS:
        assert hasattr(ctx, member), member
    assert ctx.test_mode_level == 1
    assert ctx.resume is None and ctx.checkpointer is None

    # Never stored on the search.
    assert not any(isinstance(value, FitContext) for value in vars(search).values())

    assert len(result.samples.sample_list) == 5


class ToySearch(NonLinearSearch):
    """
    A search written only against the frozen `run(ctx)` contract: draw points, evaluate
    them through the batched objective, keep them all.
    """

    posterior_kind = cap.PosteriorKind.POINT
    objective_target = cap.ObjectiveTarget(
        cap.ObjectiveQuantity.LOG_LIKELIHOOD, cap.CoordinateSpace.PHYSICAL
    )
    invalid_value = -float("inf")

    def run(self, ctx):
        parameters, _ = ctx.start_points(6)
        log_likelihoods = np.asarray(ctx.objective("batched")(parameters))
        return {"parameters": parameters, "log_likelihoods": log_likelihoods}

    def raw_samples_from(self, model, internal):
        return RawSamples(
            parameters=internal["parameters"].tolist(),
            log_likelihood=internal["log_likelihoods"].tolist(),
            info={"total_samples": len(internal["log_likelihoods"])},
        )

    def plot_results(self, samples):
        pass


def test_a_new_search_needs_only_run_and_raw_samples_from(monkeypatch):
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)

    search = ToySearch(initializer=af.InitializerPrior())
    model = af.Model(af.ex.Gaussian)
    result = search.fit(model=model, analysis=_analysis())

    assert len(result.samples.sample_list) == 6
    assert result.samples.max_log_likelihood() is not None


def test_failure_in_run_closes_the_context(monkeypatch):
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)

    closed = []

    def failing_run(self, ctx):
        ctx.objective("scalar")
        original_close = ctx.close

        def close(failed):
            closed.append(failed)
            return original_close(failed)

        ctx.close = close
        raise RuntimeError("backend failed")

    monkeypatch.setattr(ToySearch, "run", failing_run)

    with pytest.raises(RuntimeError, match="backend failed"):
        ToySearch().fit(model=af.Model(af.ex.Gaussian), analysis=_analysis())

    assert closed == [True]


def test_a_search_with_neither_hook_is_abstract():
    import inspect

    class Empty(NonLinearSearch):
        pass

    assert inspect.isabstract(NonLinearSearch)
    assert inspect.isabstract(Empty)
    assert not inspect.isabstract(ToySearch)
    assert not inspect.isabstract(af.Drawer)

    with pytest.raises(TypeError):
        Empty()


def test_schedule_chunks_cover_the_budget():
    search = af.Drawer(iterations_per_full_update=4)
    from autofit.non_linear.search.fit_context import UpdateSchedule

    assert list(UpdateSchedule(search).chunks(10)) == [4, 4, 2]
