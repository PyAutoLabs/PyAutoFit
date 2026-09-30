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

import csv
import logging

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
from autofit.non_linear.paths.directory import DirectoryPaths


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


def test_persistent_failure_stops_sweeping_early():
    """
    A factor that fails *every* sweep is not going to start working, so EP stops
    rather than burning the full `max_steps` on it — but it still returns.
    """
    model_approx, factor_graph, prior, likelihood = make_shared_variable_approx()

    failing = InitializerFailingOptimiser(n_failures=1000)
    optimiser = graph.EPOptimiser(
        factor_graph,
        factor_optimisers={prior: failing, likelihood: ExactFactorFit()},
        # `kl_tol=None` disables the convergence check: this graph is exact and
        # would otherwise be declared converged after one sweep, before the
        # failure count could build up.
        ep_history=EPHistory(kl_tol=None),
        paths=False,
    )

    optimiser.run(model_approx, max_steps=20, max_consecutive_failures=3)

    assert failing.call_count == 3, (
        "expected the run to stop after 3 consecutive raises, not sweep on to "
        f"max_steps; got {failing.call_count} attempts"
    )


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


def test_never_updating_factor_is_warned_about_loudly(caplog):
    """
    The result is returned even when no factor ever updated — but it must not be
    returned quietly.

    When every factor raises, nothing in the mean field changes, so the KL step
    between sweeps is zero and `EPHistory` declares convergence — in practice
    within two sweeps, before any per-factor count reaches its threshold. The
    mean field then holds the starting priors, and a caller reading it without
    the warning would take priors for a posterior.

    The threshold here is deliberately higher than the number of sweeps that
    will run, so this can only pass via the end-of-run check.
    """
    model_approx, factor_graph, prior, likelihood = make_shared_variable_approx()

    optimiser = graph.EPOptimiser(
        factor_graph,
        factor_optimisers={
            prior: InitializerFailingOptimiser(n_failures=1000),
            likelihood: InitializerFailingOptimiser(n_failures=1000),
        },
        paths=False,
    )

    with caplog.at_level(logging.WARNING):
        result = optimiser.run(model_approx, max_steps=2, max_consecutive_failures=100)

    assert result is not None, "the result should still be returned"

    warnings = optimiser._stale_factor_warnings()
    assert len(warnings) == 1
    assert "never completed a single update" in warnings[0]
    assert prior.name in warnings[0] and likelihood.name in warnings[0]

    logged = caplog.text
    assert "STALE FACTORS" in logged, "the stale-factor warning was not logged"


def test_stale_factor_warning_is_written_to_the_diagnostics_file(tmp_path):
    """
    The warning has to survive the run, not just scroll past in a log — it goes
    into `ep_diagnostics.results` beside the sigma-collapse warnings.
    """
    model_approx, factor_graph, prior, likelihood = make_shared_variable_approx()

    optimiser = graph.EPOptimiser(
        factor_graph,
        factor_optimisers={
            prior: InitializerFailingOptimiser(n_failures=1000),
            likelihood: InitializerFailingOptimiser(n_failures=1000),
        },
        paths=DirectoryPaths(name="stale_factors", path_prefix=str(tmp_path)),
    )

    optimiser.run(model_approx, max_steps=2, max_consecutive_failures=100)

    written = (optimiser.output_path / "ep_diagnostics.results").read_text()
    assert "STALE FACTORS" in written
    assert prior.name in written and likelihood.name in written


def test_partially_updating_factor_is_not_treated_as_stale():
    """
    The end-of-run check must stay narrow: a factor that failed at some point
    but landed at least one update has a real message, and its fit is returned.
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

    result = optimiser.run(model_approx, max_steps=4)

    (x,) = [v for v in result.mean_field if v.name == "x"]
    assert np.isfinite(result.mean_field[x].mean)


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


class OverWideFit(AbstractFactorOptimiser):
    """
    A factor fit that comes back wider than its cavity in every variable — the
    shape a near-singular or noisy finite-difference Hessian produces. The
    quotient `q* / cavity` then has negative precision, so `update_invalid`
    reverts every parameter and the factor's message never moves.
    """

    def optimise(self, factor_approx, status=graph.Status()):
        model_dist = graph.MeanField(
            {
                v: NormalMessage(float(m.mean), float(m.sigma) * 3.0)
                for v, m in factor_approx.cavity_dist.items()
            }
        )
        return model_dist, graph.Status(success=True, messages=(), updated=True)


def test_fully_reverted_projection_reports_no_update():
    """
    `check_valid` after `update_invalid` is true by construction — the reverted
    parameters come from a valid message — so it measured validity, not change,
    and a projection that reverted everything claimed `updated=True`
    (PyAutoFit#1571).
    """
    x, y = Variable("x"), Variable("y")
    q_star = graph.MeanField({x: NormalMessage(0.0, 2.0), y: NormalMessage(1.0, 4.0)})
    cavity = graph.MeanField({x: NormalMessage(0.0, 1.0), y: NormalMessage(1.0, 1.0)})
    last = graph.MeanField({x: NormalMessage(0.5, 3.0), y: NormalMessage(2.0, 5.0)})

    new, status = q_star.update_factor_mean_field(
        cavity_dist=cavity,
        last_dist=last,
        delta=1.0,
        status=graph.Status(success=True, flag=StatusFlag.SUCCESS),
    )

    assert status.flag is StatusFlag.BAD_PROJECTION
    assert status.updated is False
    assert any("every parameter reverted" in m for m in status.messages)
    for v in (x, y):
        assert tuple(new[v].parameters) == tuple(last[v].parameters)


def test_partially_reverted_projection_still_reports_an_update():
    """
    The narrow half: only a *full* revert is a skipped update. A projection that
    moves one variable and reverts another has really updated.
    """
    x, y = Variable("x"), Variable("y")
    # x's projection is valid (tighter than the cavity), y's is not
    q_star = graph.MeanField({x: NormalMessage(0.0, 0.5), y: NormalMessage(1.0, 4.0)})
    cavity = graph.MeanField({x: NormalMessage(0.0, 1.0), y: NormalMessage(1.0, 1.0)})
    last = graph.MeanField({x: NormalMessage(0.5, 3.0), y: NormalMessage(2.0, 5.0)})

    new, status = q_star.update_factor_mean_field(
        cavity_dist=cavity,
        last_dist=last,
        delta=1.0,
        status=graph.Status(success=True, flag=StatusFlag.SUCCESS),
    )

    assert status.updated is True
    assert tuple(new[x].parameters) != tuple(last[x].parameters)
    assert tuple(new[y].parameters) == tuple(last[y].parameters)


def test_always_reverting_factor_is_counted_as_skipped_and_warned_about(tmp_path):
    """
    End to end: a factor whose projection is rejected every sweep returns its
    starting message, so it belongs in `factors skipped` and must be named by
    the STALE FACTORS warning — which it was not while `status.updated` came
    back True.
    """
    model_approx, factor_graph, prior, likelihood = make_shared_variable_approx()
    (x,) = [v for v in model_approx.mean_field if v.name == "x"]
    start = tuple(model_approx.factor_mean_field[likelihood][x].parameters)

    optimiser = graph.EPOptimiser(
        factor_graph,
        factor_optimisers={prior: ExactFactorFit(), likelihood: OverWideFit()},
        ep_history=EPHistory(kl_tol=None),
        paths=DirectoryPaths(name="full_revert", path_prefix=str(tmp_path)),
    )

    result = optimiser.run(model_approx, max_steps=4, max_consecutive_failures=100)

    # the factor's message never moved off the one it started with
    assert tuple(result.factor_mean_field[likelihood][x].parameters) == start

    assert likelihood in optimiser._factors_skipped
    assert likelihood not in optimiser._factors_updated

    warnings = optimiser._stale_factor_warnings()
    assert len(warnings) == 1 and likelihood.name in warnings[0]

    written = (optimiser.output_path / "ep_diagnostics.results").read_text()
    assert "STALE FACTORS" in written and likelihood.name in written


def make_two_variable_approx():
    """
    Two variables joined by one factor, each with its own prior — the smallest
    graph on which a factor can revert *one* of its variables on every
    projection while the other updates (the partial revert of PyAutoFit#1575).
    """
    x, y = Variable("x"), Variable("y")

    def joint(x, y):
        return -0.5 * (np.sum((x - 3.0) ** 2) + np.sum((y - 2.0) ** 2))

    prior_x = NormalMessage(1.0, 2.0).as_factor(x, name="prior_x")
    prior_y = NormalMessage(1.0, 2.0).as_factor(y, name="prior_y")
    likelihood = graph.Factor(joint, x, y, name="like_xy")

    factor_graph = graph.FactorGraph([prior_x, prior_y, likelihood])
    model_approx = graph.EPMeanField.from_approx_dists(
        factor_graph,
        {x: NormalMessage(0.0, 10.0), y: NormalMessage(0.0, 10.0)},
    )
    return model_approx, factor_graph, prior_x, prior_y, likelihood


class PartialRevertFit(AbstractFactorOptimiser):
    """
    A factor fit that is valid in one variable and over-wide in another, on
    every sweep: the quotient `q* / cavity` has positive precision for the
    first (so it updates) and negative precision for the second (so
    `update_invalid` reverts every one of its parameters and its message never
    moves). This is the hierarchical scatter's shape — the factor updates, one
    of its variables never does.
    """

    def __init__(self, reverting: str):
        super().__init__()
        self.reverting = reverting

    def optimise(self, factor_approx, status=graph.Status()):
        model_dist = graph.MeanField(
            {
                v: NormalMessage(
                    float(m.mean),
                    float(m.sigma) * (3.0 if v.name == self.reverting else 0.5),
                )
                for v, m in factor_approx.cavity_dist.items()
            }
        )
        return model_dist, graph.Status(success=True, messages=(), updated=True)


def test_partial_revert_names_the_stale_variable_not_the_factor(tmp_path):
    """
    The gap #1574 left. A factor that reverts one variable on every projection
    and updates the others is `updated`, so no factor-level STALE FACTORS line
    is emitted — yet the reverted variable's reported posterior is the message
    it started with. The warning must name the (factor, variable) pair.
    """
    model_approx, factor_graph, prior_x, prior_y, likelihood = (
        make_two_variable_approx()
    )
    (y,) = [v for v in model_approx.mean_field if v.name == "y"]
    start = tuple(model_approx.factor_mean_field[likelihood][y].parameters)

    optimiser = graph.EPOptimiser(
        factor_graph,
        factor_optimisers={
            prior_x: ExactFactorFit(),
            prior_y: ExactFactorFit(),
            likelihood: PartialRevertFit(reverting="y"),
        },
        ep_history=EPHistory(kl_tol=None),
        paths=DirectoryPaths(name="partial_revert", path_prefix=str(tmp_path)),
    )

    result = optimiser.run(model_approx, max_steps=4, max_consecutive_failures=100)

    # the factor did update -- so the factor-level test cannot see this
    assert likelihood in optimiser._factors_updated

    # ... but y's message never moved off the one it started with
    assert tuple(result.factor_mean_field[likelihood][y].parameters) == start

    assert optimiser._variables_seen[likelihood.name] == {
        v for v in likelihood.variables
    }
    assert y not in optimiser._variables_changed[likelihood.name]

    warnings = optimiser._stale_factor_warnings()
    y_lines = [w for w in warnings if "variable 'y'" in w]
    assert len(y_lines) == 1
    assert "STALE FACTORS" in y_lines[0]
    assert likelihood.name in y_lines[0]
    assert not [w for w in warnings if "variable 'x'" in w]

    diagnostics = (optimiser.output_path / "ep_diagnostics.results").read_text()
    assert "WARNINGS" in diagnostics
    assert y_lines[0] in diagnostics


def test_partial_revert_is_recorded_in_ep_history_csv(tmp_path):
    """
    `ep_history.csv` gains a `reverted_variables` column so a workspace referee
    can tally the per-variable signal without parsing the warning text.

    The column records the variables whose projection each update *rejected*
    -- the ones `update_invalid` reverted -- not the ones whose message
    happened not to move: a valid projection accepts every variable, so a
    healthy factor records nothing even once it has settled. A variable listed
    on *every* row for a factor was therefore never accepted, which is what
    `_stale_factor_warnings` reports.
    """
    model_approx, factor_graph, prior_x, prior_y, likelihood = (
        make_two_variable_approx()
    )

    optimiser = graph.EPOptimiser(
        factor_graph,
        factor_optimisers={
            prior_x: ExactFactorFit(),
            prior_y: ExactFactorFit(),
            likelihood: PartialRevertFit(reverting="y"),
        },
        ep_history=EPHistory(kl_tol=None),
        paths=DirectoryPaths(name="partial_revert_csv", path_prefix=str(tmp_path)),
    )
    optimiser.run(model_approx, max_steps=4, max_consecutive_failures=100)

    with open(optimiser.output_path / "ep_history.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    assert "reverted_variables" in rows[0]

    reverting_rows = [row for row in rows if row["factor"] == likelihood.name]
    assert reverting_rows
    # y is reverted on every one of the factor's projections ...
    assert all(
        "y" in row["reverted_variables"].split(";") for row in reverting_rows
    )
    # ... and on the first sweep, before the graph settles, it is the only one:
    # x moved on that same projection, which is the partial revert the
    # factor-level `updated` flag cannot express.
    assert reverting_rows[0]["reverted_variables"] == "y"

    exact_rows = [row for row in rows if row["factor"] == prior_x.name]
    assert exact_rows
    # an exact fit that moves its message records nothing as unmoved
    assert exact_rows[0]["reverted_variables"] == ""
    # and x is not stale: it moved at least once, so no warning names it
    assert not [
        w for w in optimiser._stale_factor_warnings() if "variable 'x'" in w
    ]


def test_restart_from_fixed_point_is_not_stale(tmp_path):
    """
    EP restarted from its own converged `EPMeanField` reproduces every message
    exactly: nothing moves, but nothing is rejected either. A fixed point is
    not a rejection (PyAutoFit#1579), so no factor may be reported stale and
    no history row may name a reverted variable.
    """
    model_approx, factor_graph, prior, likelihood = make_shared_variable_approx()

    def run(approx, name, steps):
        optimiser = graph.EPOptimiser(
            factor_graph,
            factor_optimisers={
                prior: ExactFactorFit(),
                likelihood: ExactFactorFit(),
            },
            ep_history=EPHistory(kl_tol=None),
            paths=DirectoryPaths(name=name, path_prefix=str(tmp_path)),
        )
        return optimiser, optimiser.run(
            approx, max_steps=steps, max_consecutive_failures=100
        )

    _, converged = run(model_approx, "burn_in", 6)
    optimiser, _ = run(converged, "fixed_point", 4)

    assert optimiser._stale_factor_warnings() == []

    with open(optimiser.output_path / "ep_history.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows
    assert all(row["flag"] == StatusFlag.SUCCESS.name for row in rows)
    assert all(row["reverted_variables"] == "" for row in rows)


def test_factor_step_preserves_the_changed_mask_on_the_status():
    """
    `factor_step` rebuilds the `Status` it returns (to fold in caught
    warnings), so every field it does not name is silently dropped. The
    per-variable mask has to survive that reconstruction or the optimiser's
    bookkeeping never sees it.
    """
    from autofit.graphical.expectation_propagation.optimiser import factor_step
    from autofit.mapper.variable import VariableData

    model_approx, _, prior_x, _, _ = make_two_variable_approx()
    (x,) = [v for v in model_approx.mean_field if v.name == "x"]
    mask = VariableData({x: False})

    class MaskCarryingFit(AbstractFactorOptimiser):
        def optimise(self, factor_approx, status=graph.Status()):
            return factor_approx.model_dist, graph.Status(
                success=True, updated=True, changed=mask
            )

    _, status = factor_step(
        model_approx.factor_approximation(prior_x), MaskCarryingFit()
    )

    assert status.changed is mask


def _grouped_optimiser(tmp_path, name, reverting_first, reverting_second):
    """
    A graph with two factors that share one name, each joined to the same
    shared variable `s` and to its own drawn variable — the shape a
    `HierarchicalFactor` takes once it decomposes into one factor per drawn
    variable (each member built with `name=distribution_model.name`, `s`
    standing in for the parent scatter). Each member gets a
    `PartialRevertFit`, so the test controls exactly which variable it moves;
    a `reverting` name matching nothing means the member moves everything.
    """
    s, x1, x2 = Variable("s"), Variable("x1"), Variable("x2")

    def first_density(s, x1):
        return -0.5 * (np.sum((s - 2.0) ** 2) + np.sum((x1 - 3.0) ** 2))

    def second_density(s, x2):
        return -0.5 * (np.sum((s - 2.0) ** 2) + np.sum((x2 - 3.0) ** 2))

    priors = [
        NormalMessage(1.0, 2.0).as_factor(v, name=f"prior_{v.name}")
        for v in (s, x1, x2)
    ]
    # distinct functions, so the two factors are distinct objects (`Factor`
    # equality is on the function and its arguments) that nonetheless share a
    # name, exactly as a decomposed hierarchical factor's members do
    first = graph.Factor(first_density, s, x1, name=name)
    second = graph.Factor(second_density, s, x2, name=name)

    factor_graph = graph.FactorGraph(priors + [first, second])
    model_approx = graph.EPMeanField.from_approx_dists(
        factor_graph,
        {v: NormalMessage(0.0, 10.0) for v in (s, x1, x2)},
    )

    optimisers = {prior: ExactFactorFit() for prior in priors}
    optimisers[first] = PartialRevertFit(reverting=reverting_first)
    optimisers[second] = PartialRevertFit(reverting=reverting_second)

    optimiser = graph.EPOptimiser(
        factor_graph,
        factor_optimisers=optimisers,
        ep_history=EPHistory(kl_tol=None),
        paths=DirectoryPaths(name=f"group_{name}", path_prefix=str(tmp_path)),
    )
    optimiser.run(model_approx, max_steps=4, max_consecutive_failures=100)
    return optimiser


def test_one_member_of_a_group_moving_a_variable_clears_the_group(tmp_path):
    """
    A `HierarchicalFactor` decomposes into one factor per drawn variable, all
    sharing its name, and the parent scatter's message is the product of every
    member's. One member moving it is enough for the reported value to be a
    posterior, so the pair must be tracked per group, not per member —
    otherwise every healthy hierarchical run is flagged.
    """
    optimiser = _grouped_optimiser(
        tmp_path, "group_mixed", reverting_first="s", reverting_second="none"
    )

    # the first member reverts the shared variable on every projection; the
    # second moves it, so the group's message for it is a posterior
    warnings = optimiser._stale_factor_warnings()
    assert warnings == []


def test_a_group_no_member_of_which_moves_a_variable_is_named_once(tmp_path):
    """
    The other half: when *no* member of the group ever moves the variable, its
    posterior really is the message it started with — one line, naming the
    group, not one line per member.
    """
    optimiser = _grouped_optimiser(
        tmp_path, "group_stale", reverting_first="s", reverting_second="s"
    )

    warnings = optimiser._stale_factor_warnings()
    assert len(warnings) == 1
    assert "STALE FACTORS" in warnings[0]
    assert "variable 's' of group_stale" in warnings[0]
    assert "updates (reverted on every projection of that factor)" in warnings[0]


class NonFiniteProjectionOptimiser(AbstractFactorOptimiser):
    """
    Stands in for a per-factor search whose weighted samples project to
    non-finite sufficient statistics (RAL 342411, PyAutoFit#1653): the first
    `n_failures` calls project a sample set containing a nan, then it defers
    to an exact fit.
    """

    def __init__(self, n_failures=1):
        super().__init__()
        self.n_failures = n_failures
        self.call_count = 0

    def optimise(self, factor_approx, status=graph.Status()):
        self.call_count += 1
        if self.call_count <= self.n_failures:
            with np.errstate(all="ignore"):
                NormalMessage.project(np.array([1.0, 2.0, np.nan]), np.zeros(3))
        return self.exact_fit(factor_approx, status)


def test_nonfinite_projection_degrades_factor_step_to_previous_message():
    """
    A non-finite projection of one factor's samples is a failed sweep update,
    not a failed graph fit: `factor_step` keeps the factor's previous message
    and records the failure, rather than letting the error kill the run.
    """
    from autofit.graphical.expectation_propagation.optimiser import factor_step

    model_approx, _, prior, _ = make_shared_variable_approx()
    factor_approx = model_approx.factor_approximation(prior)

    new_dist, status = factor_step(factor_approx, NonFiniteProjectionOptimiser(1))

    assert status.flag is StatusFlag.EXCEPTION
    assert status.updated is False
    assert new_dist is factor_approx.model_dist
    assert "non-finite" in status.messages[0]
    assert "samples" in status.messages[0]


def test_nonfinite_projection_does_not_abort_the_ep_run():
    """
    End to end: one bad projection on the first sweep leaves the EP run going,
    it returns a finite mean field, and the failure stays loud in the
    diagnostics rows.
    """
    model_approx, factor_graph, prior, likelihood = make_shared_variable_approx()

    failing = NonFiniteProjectionOptimiser(n_failures=1)
    optimiser = graph.EPOptimiser(
        factor_graph,
        factor_optimisers={prior: failing, likelihood: ExactFactorFit()},
        paths=False,
    )

    result = optimiser.run(model_approx, max_steps=4)

    assert failing.call_count > 1, "the failing factor was never retried"
    (x,) = [v for v in result.mean_field if v.name == "x"]
    assert np.isfinite(result.mean_field[x].mean)

    flags = [
        row["flag"]
        for row in optimiser.diagnostics.factor_rows
        if row["factor"] == prior.name
    ]
    assert StatusFlag.EXCEPTION.name in flags
