"""
Nautilus as the EP ``default_optimiser`` over a graph that includes ``PriorFactor``s.

A ``PriorFactor`` is an ``Analysis`` that never calls ``Analysis.__init__``, so it has
no ``_use_jax`` attribute. Nautilus once read ``analysis._use_jax`` unguarded while
setting up a fit, so an EP run with Nautilus as the default optimiser raised
``AttributeError`` on the first ``PriorFactor`` it reached. ``Analysis.is_jax`` now
reads it through ``getattr(self, "_use_jax", False)``; these tests stop the unguarded
read coming back.

``PYAUTO_TEST_MODE=1`` caps Nautilus at one likelihood call, so one EP step takes
seconds while still running the real search setup (preflight, fitness, sampler).

Nautilus refuses models with fewer than two parameters, so each one-parameter
``PriorFactor`` step fails with a ``ValueError`` that EP records and skips. That is a
separate limitation and is not asserted here. ``AttributeError`` is not one of the
exceptions EP catches per factor, so a regression surfaces as a raised exception.
"""

import pytest

import autofit as af
import autofit.graphical as g
from autofit.graphical.declarative.factor.prior import PriorFactor
from autofit.graphical.utils import StatusFlag
from autofit.non_linear.analysis.analysis import Analysis

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


class CentreAnalysis(af.Analysis):
    def __init__(self, centre: float):
        super().__init__()
        self.centre = centre

    def log_likelihood_function(self, instance) -> float:
        return float(-0.5 * (instance.centre - self.centre) ** 2)


def _run_one_ep_step(monkeypatch):
    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")

    factor = g.AnalysisFactor(
        af.Model(af.ex.Gaussian), analysis=CentreAnalysis(centre=50.0)
    )
    factor_graph_model = g.FactorGraphModel(factor, include_prior_factors=True)

    optimiser = g.EPOptimiser(
        factor_graph_model.graph,
        default_optimiser=af.Nautilus(number_of_cores=1),
    )
    optimiser.run(factor_graph_model.mean_field_approximation(), max_steps=1)

    return factor_graph_model, factor, optimiser


def test_nautilus_ep_step_over_prior_factors_runs(monkeypatch):
    factor_graph_model, factor, optimiser = _run_one_ep_step(monkeypatch)

    prior_factors = factor_graph_model.prior_factors
    assert len(prior_factors) == 3
    # The premise of the guard: a PriorFactor carries no `_use_jax` of its own.
    assert all("_use_jax" not in vars(prior_factor) for prior_factor in prior_factors)
    assert all(prior_factor.is_jax is False for prior_factor in prior_factors)

    histories = dict(optimiser.ep_history.items())

    # Every factor, the PriorFactors included, was stepped by Nautilus.
    assert set(histories) == {factor, *prior_factors}

    # The AnalysisFactor's search ran to the end without raising. Whether its one-call
    # sample projects to a valid message is sample-dependent, so `success` is not
    # asserted.
    _, analysis_status = histories[factor].history[-1]
    assert analysis_status.flag is not StatusFlag.EXCEPTION

    for history in histories.values():
        for _, status in history.history:
            assert not any("AttributeError" in m for m in status.messages)
            assert not any("_use_jax" in m for m in status.messages)


def test_unguarded_use_jax_read_raises(monkeypatch):
    """
    Control: with the pre-fix unguarded read restored, the same EP step raises
    ``AttributeError`` on a ``PriorFactor``, so the test above would catch the
    regression.
    """
    missing = []

    def unguarded_is_jax(self):
        try:
            return self._use_jax
        except AttributeError:
            missing.append(type(self))
            raise

    monkeypatch.setattr(Analysis, "is_jax", property(unguarded_is_jax))

    with pytest.raises(AttributeError):
        _run_one_ep_step(monkeypatch)

    assert missing
    assert all(issubclass(cls, PriorFactor) for cls in missing)
