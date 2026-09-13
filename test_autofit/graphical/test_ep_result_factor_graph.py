"""
``FactorGraphModel.optimise`` returns the graph it swept (PyAutoFit #1620).

The EP state figure is drawn from the *optimiser's* factor graph, because the
``EPHistory`` is keyed by that graph's factor objects, while
``AbstractDeclarativeFactor.graph`` builds a fresh graph -- renaming every
``PriorFactor`` -- on every access. The optimiser is built inside ``optimise``,
so before ``EPResult.factor_graph`` existed a caller of the high-level API had
no way to reach it.
"""

import numpy as np
import pytest

import autofit as af
import autofit.graphical as g
from autofit.graphical.declarative.factor.prior import PriorFactor
from test_autofit.graphical.gaussian.model import Analysis, Gaussian, make_data


@pytest.fixture(name="factor_graph_model")
def make_factor_graph_model():
    """A one factor graph, small enough that two sweeps cost milliseconds."""
    x = np.arange(50)
    y = make_data(Gaussian(centre=25.0, normalization=25.0, sigma=10.0), x)

    prior_model = af.Model(
        Gaussian,
        centre=af.GaussianPrior(mean=25, sigma=20),
        normalization=af.GaussianPrior(mean=25, sigma=10),
        sigma=af.GaussianPrior(mean=10, sigma=10),
    )

    return g.FactorGraphModel(
        g.AnalysisFactor(prior_model, analysis=Analysis(x=x, y=y))
    )


@pytest.fixture(name="ep_result")
def make_ep_result(factor_graph_model):
    return factor_graph_model.optimise(
        af.LaplaceOptimiser(),
        paths=False,
        # kl_tol=None disables the convergence check so both sweeps run.
        ep_history=af.EPHistory(kl_tol=None),
        max_steps=2,
    )


def test_result_carries_the_swept_graph(ep_result):
    """
    The returned graph is the one the history is keyed by -- not a rebuild.
    """
    assert ep_result.factor_graph is not None
    assert isinstance(ep_result.factor_graph, g.FactorGraph)

    history_factors = set(ep_result.ep_history.history)

    assert history_factors
    assert history_factors.issubset(set(ep_result.factor_graph.factors))


def test_rebuilt_graph_is_not_the_swept_one(factor_graph_model, ep_result):
    """
    Why the attribute has to exist: the declarative ``graph`` property builds a
    new graph, renaming every ``PriorFactor``, on every access -- so it is not
    the graph the history was keyed by and must not be handed to the plotter.
    """
    rebuilt = factor_graph_model.graph

    assert rebuilt is not ep_result.factor_graph
    assert factor_graph_model.graph is not rebuilt

    def prior_factor_names(graph):
        return sorted(
            factor.name for factor in graph.factors if isinstance(factor, PriorFactor)
        )

    assert prior_factor_names(rebuilt) != prior_factor_names(ep_result.factor_graph)


def test_state_figure_from_the_result(ep_result, tmp_path):
    """
    The high-level form the docs give: plot straight off the result.
    """
    plotter = af.EPPlotter(
        ep_result.factor_graph,
        ep_history=ep_result.ep_history,
    )

    plotter.figure(kind="state", path=tmp_path, format="png")

    assert (tmp_path / "graph_state.png").exists()

    state = plotter.state()

    assert state.factors
    assert [
        factor.status for factor in state.factors if factor.status == "absent"
    ] == []


def test_factor_graph_defaults_to_none():
    """The argument is additive: the old three argument call still works."""
    from autofit.graphical.declarative.result import EPResult

    result = EPResult(
        ep_history=af.EPHistory(),
        declarative_factor=None,
        updated_ep_mean_field=None,
    )

    assert result.factor_graph is None
