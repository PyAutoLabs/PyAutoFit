"""
An `InitializerException` in one factor should not kill the whole EP fit.

A factor's own optimiser can fail to find a start point — most often because EP
has driven that factor to a state where every drawn point has the same figure of
merit. That is a failure of one sweep's update for one factor, not of the graph
fit, so the sweep should continue on that factor's previous message with the
failure recorded.

These tests use a **shared-variable, non-hierarchical** graph, which is the shape
that reproduced this on the release leg (PyAutoFit#1405): two factors connected by
one shared variable, no `HierarchicalFactor` involved.
"""

import numpy as np
import pytest

from autofit import exc
from autofit import graphical as graph
from autofit.graphical.expectation_propagation.factor_optimiser import (
    AbstractFactorOptimiser,
    ExactFactorFit,
)
from autofit.graphical.expectation_propagation.history import EPHistory
from autofit.graphical.utils import StatusFlag
from autofit.mapper.variable import Variable
from autofit.messages.normal import NormalMessage


def make_shared_variable_approx():
    """
    Two factors joined by one shared variable `x` — the minimal form of the
    graph that failed on the release leg (a shared prior across several
    `AnalysisFactor`s), small enough to converge in a few sweeps.
    """
    x = Variable("x")
    prior = NormalMessage(1.0, 2.0).as_factor(x, name="prior_x")
    likelihood = NormalMessage(3.0, 0.5).as_factor(x, name="like_x")
    factor_graph = graph.FactorGraph([prior, likelihood])
    model_approx = graph.EPMeanField.from_approx_dists(
        factor_graph, {x: NormalMessage(0.0, 10.0)}
    )
    return model_approx, factor_graph, prior, likelihood


class InitializerFailingOptimiser(AbstractFactorOptimiser):
    """
    Stands in for a per-factor search whose initializer cannot find a start
    point. Fails its first `n_failures` calls, then defers to an exact fit — so
    a test can model either an intermittent failure (the observed case, ~23% of
    runs) or a factor that never initialises.
    """

    def __init__(self, n_failures=1):
        super().__init__()
        self.n_failures = n_failures
        self.call_count = 0

    def optimise(self, factor_approx, status=graph.Status()):
        self.call_count += 1
        if self.call_count <= self.n_failures:
            raise exc.InitializerException(
                "The initial samples all have the same figure of merit"
            )
        return self.exact_fit(factor_approx, status)


def test_initializer_exception_does_not_abort_the_fit():
    """
    The headline behaviour: one factor failing to initialise on one sweep leaves
    the graph fit running, and it still returns a usable mean field.
    """
    model_approx, factor_graph, prior, likelihood = make_shared_variable_approx()

    failing = InitializerFailingOptimiser(n_failures=1)
    optimiser = graph.EPOptimiser(
        factor_graph,
        factor_optimisers={prior: failing, likelihood: ExactFactorFit()},
        paths=False,
    )

    result = optimiser.run(model_approx, max_steps=4)

    assert failing.call_count > 1, "the failing factor was never retried"
    (x,) = [v for v in result.mean_field if v.name == "x"]
    assert np.isfinite(result.mean_field[x].mean)


def test_failure_is_recorded_as_a_failure_not_a_success():
    """
    The failure must stay loud. Degrading the crash to a skipped update is only
    acceptable because it is still visible — a failed step recorded as a success
    is the silent-failure mode this fix exists to avoid.
    """
    model_approx, factor_graph, prior, likelihood = make_shared_variable_approx()

    optimiser = graph.EPOptimiser(
        factor_graph,
        factor_optimisers={
            prior: InitializerFailingOptimiser(n_failures=1),
            likelihood: ExactFactorFit(),
        },
        paths=False,
    )
    optimiser.run(model_approx, max_steps=4)

    flags = [
        row["flag"]
        for row in optimiser.diagnostics.factor_rows
        if row["factor"] == prior.name
    ]
    assert StatusFlag.EXCEPTION.name in flags, (
        "the failed factor update was not recorded as a raise in the "
        f"diagnostics rows: {flags}"
    )


def test_persistent_failure_aborts_naming_the_factor():
    """
    A factor that fails *every* sweep must not be tolerated indefinitely: EP
    would converge on its stale message and report success.
    """
    model_approx, factor_graph, prior, likelihood = make_shared_variable_approx()

    optimiser = graph.EPOptimiser(
        factor_graph,
        factor_optimisers={
            prior: InitializerFailingOptimiser(n_failures=1000),
            likelihood: ExactFactorFit(),
        },
        # `kl_tol=None` disables the convergence check: this graph is exact and
        # would otherwise be declared converged after one sweep, before the
        # failure count could build up.
        ep_history=EPHistory(kl_tol=None),
        paths=False,
    )

    with pytest.raises(exc.FactorOptimisationException) as exc_info:
        optimiser.run(model_approx, max_steps=20, max_consecutive_failures=3)

    assert prior.name in str(exc_info.value), "the abort message does not name the factor"


def test_consecutive_failure_count_resets_on_success():
    """
    Counting is per-factor and consecutive, so an intermittent failure — the
    common case — never trips the abort even over many sweeps.
    """
    model_approx, factor_graph, prior, likelihood = make_shared_variable_approx()

    class IntermittentOptimiser(InitializerFailingOptimiser):
        def optimise(self, factor_approx, status=graph.Status()):
            self.call_count += 1
            if self.call_count % 2 == 1:
                raise exc.InitializerException("degenerate start point")
            return self.exact_fit(factor_approx, status)

    intermittent = IntermittentOptimiser()
    optimiser = graph.EPOptimiser(
        factor_graph,
        factor_optimisers={prior: intermittent, likelihood: ExactFactorFit()},
        ep_history=EPHistory(kl_tol=None),
        paths=False,
    )

    # Alternating failure/success over many sweeps: never two failures in a row,
    # so this must not abort even with a threshold of 2.
    optimiser.run(model_approx, max_steps=8, max_consecutive_failures=2)

    assert intermittent.call_count > 2


def test_returned_failure_status_does_not_trip_the_abort():
    """
    Only a *raise* counts toward the abort. Optimisers return
    `StatusFlag.FAILURE` routinely — the Laplace optimiser does so every time
    its line search fails — and EP is designed to absorb that. Counting returned
    failures aborts healthy fits, which is exactly what an earlier revision of
    this guard did to `test_full_hierachical`.
    """
    model_approx, factor_graph, prior, likelihood = make_shared_variable_approx()

    class AlwaysReturnsFailure(AbstractFactorOptimiser):
        def optimise(self, factor_approx, status=graph.Status()):
            return (
                factor_approx.model_dist,
                graph.Status(
                    success=False,
                    messages=("Line search failed",),
                    updated=False,
                    flag=StatusFlag.FAILURE,
                ),
            )

    optimiser = graph.EPOptimiser(
        factor_graph,
        factor_optimisers={prior: AlwaysReturnsFailure(), likelihood: ExactFactorFit()},
        ep_history=EPHistory(kl_tol=None),
        paths=False,
    )

    # Would raise if returned failures were counted: 10 sweeps, threshold 2.
    optimiser.run(model_approx, max_steps=10, max_consecutive_failures=2)


def test_nan_likelihoods_cannot_raise_the_identical_merit_exception():
    """
    Guards the diagnostic wording. The exception's message used to offer "always
    returning `nan`" as a possible cause, which sent one investigation chasing a
    nan that cannot occur: the check is `np.allclose`, which is False for `nan`,
    and nan draws are discarded before it anyway.
    """
    from autofit.non_linear.initializer import IDENTICAL_FIGURES_OF_MERIT_MESSAGE

    assert not np.allclose(np.nan, [np.nan, np.nan])
    assert "always returning `nan`" not in IDENTICAL_FIGURES_OF_MERIT_MESSAGE
