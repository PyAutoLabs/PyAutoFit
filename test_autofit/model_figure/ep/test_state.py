"""
Layer 2 of the EP figure: the run's state, read off a real ``EPHistory``.

Every fixture here drives a real ``EPOptimiser`` loop over the doubles that
``test_factor_failure_recovery`` wrote for the reversion bugs themselves
(PyAutoFit #1571, #1575, #1579), so the overlay is asserted against the same
behaviour those tests pin rather than against a hand-built history. They are
tiny graphs with ``max_steps=4``.
"""

import csv
import json

import pytest

from autofit import graphical as graph
from autofit.graphical.expectation_propagation.factor_optimiser import ExactFactorFit
from autofit.graphical.expectation_propagation.history import EPHistory
from autofit.model_figure.ep.spec import EPGraphSpec, factor_by_key
from autofit.model_figure.ep.state import EPState, _reverted_names
from autofit.non_linear.paths.directory import DirectoryPaths
from test_autofit.graphical.functionality.test_factor_failure_recovery import (
    OverWideFit,
    PartialRevertFit,
    make_shared_variable_approx,
    make_two_variable_approx,
)


def _state(optimiser, factor_graph):
    spec = EPGraphSpec.from_factor_graph(factor_graph)
    return spec, EPState.from_history(
        spec, optimiser.ep_history, factor_by_key(factor_graph)
    )


def _by_name(spec, state, name):
    (node,) = [factor for factor in spec.factors if factor.name == name]
    (factor_state,) = [factor for factor in state.factors if factor.key == node.key]
    return factor_state


def _variable_key(spec, label):
    (variable,) = [variable for variable in spec.variables if variable.label == label]
    return variable.key


@pytest.fixture
def partial_revert(tmp_path):
    """
    One factor that reverts ``y`` on every projection while moving ``x`` -- the
    partial revert of PyAutoFit #1575. The factor itself updates, so nothing at
    factor level can see it.
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
        paths=DirectoryPaths(name="partial_revert", path_prefix=str(tmp_path)),
    )
    optimiser.run(model_approx, max_steps=4, max_consecutive_failures=100)
    return optimiser, factor_graph, likelihood


def test_a_partial_revert_names_the_reverted_variable_and_only_it(partial_revert):
    optimiser, factor_graph, _ = partial_revert
    spec, state = _state(optimiser, factor_graph)

    joint = _by_name(spec, state, "like_xy")

    assert joint.status == "reverting"
    assert joint.reverted_variable_keys == (_variable_key(spec, "prior_y.y"),)
    assert _variable_key(spec, "prior_x.x") not in joint.reverted_variable_keys
    assert joint.last_flag == "BAD_PROJECTION"

    # ... and the factor really did update, which is why nothing at factor
    # level could see this
    assert joint.updates >= 1


def test_the_reverted_variable_carries_the_factor_that_reverted_it(partial_revert):
    optimiser, factor_graph, _ = partial_revert
    spec, state = _state(optimiser, factor_graph)

    joint = _by_name(spec, state, "like_xy")
    by_key = {variable.key: variable for variable in state.variables}

    assert by_key[_variable_key(spec, "prior_y.y")].reverted_by == (joint.key,)
    assert by_key[_variable_key(spec, "prior_x.x")].reverted_by == ()


def test_the_exact_factors_are_not_reverting(partial_revert):
    optimiser, factor_graph, _ = partial_revert
    spec, state = _state(optimiser, factor_graph)

    for name in ("prior_x", "prior_y"):
        factor_state = _by_name(spec, state, name)
        assert factor_state.reverted_variable_keys == ()
        assert factor_state.status != "reverting"


def test_reverted_names_matches_the_ep_history_csv_column(partial_revert):
    """
    ``_reverted_names`` is a private copy of the ``reverted_variables`` column
    logic, so it is pinned to a real CSV written by a real run rather than to
    a reading of ``diagnostics.py``.
    """
    optimiser, _, likelihood = partial_revert

    with open(optimiser.output_path / "ep_history.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    column = [
        row["reverted_variables"] for row in rows if row["factor"] == likelihood.name
    ]
    helper = [
        ";".join(_reverted_names(status))
        for _, status in optimiser.ep_history.history[likelihood].history
    ]

    assert column == helper
    assert column == ["y", "y", "y", "y"]


def test_a_factor_that_never_updates_is_stale_and_as_old_as_the_run(tmp_path):
    """
    ``OverWideFit`` reverts every parameter on every projection, so the factor
    is both stale and reverting -- and the stronger claim wins, because its
    reported posterior is the message it started with.
    """
    model_approx, factor_graph, prior, likelihood = make_shared_variable_approx()

    optimiser = graph.EPOptimiser(
        factor_graph,
        factor_optimisers={prior: ExactFactorFit(), likelihood: OverWideFit()},
        ep_history=EPHistory(kl_tol=None),
        paths=False,
    )
    optimiser.run(model_approx, max_steps=4, max_consecutive_failures=100)

    spec, state = _state(optimiser, factor_graph)
    over_wide = _by_name(spec, state, "like_x")

    assert over_wide.status == "stale"
    assert over_wide.updates == 0
    assert over_wide.sweeps == 4
    assert over_wide.age == over_wide.sweeps
    assert over_wide.reverted_variable_keys  # it is reverting too -- stale wins

    # the optimiser's own end-of-run check agrees
    assert likelihood in optimiser._factors_skipped


def test_a_fixed_point_restart_is_neither_stale_nor_reverting(tmp_path):
    """
    EP restarted from its own converged mean field reproduces every message
    exactly: nothing moves, but nothing is rejected either. A fixed point is
    not a rejection (PyAutoFit #1579).
    """
    model_approx, factor_graph, prior, likelihood = make_shared_variable_approx()

    def run(approx, name, steps):
        optimiser = graph.EPOptimiser(
            factor_graph,
            factor_optimisers={prior: ExactFactorFit(), likelihood: ExactFactorFit()},
            ep_history=EPHistory(kl_tol=None),
            paths=DirectoryPaths(name=name, path_prefix=str(tmp_path)),
        )
        return optimiser, optimiser.run(
            approx, max_steps=steps, max_consecutive_failures=100
        )

    _, converged = run(model_approx, "burn_in", 6)
    optimiser, _ = run(converged, "fixed_point", 4)

    spec, state = _state(optimiser, factor_graph)

    assert optimiser._stale_factor_warnings() == []
    for factor_state in state.factors:
        assert factor_state.status not in ("stale", "reverting")
        assert factor_state.reverted_variable_keys == ()
    assert all(variable.reverted_by == () for variable in state.variables)


def test_a_converged_factor_is_reported_as_converged():
    """
    ``converged`` is ``EPHistory``'s own test, not a reimplementation of it --
    the same predicate the optimiser stops on.
    """
    model_approx, factor_graph, prior, likelihood = make_shared_variable_approx()

    optimiser = graph.EPOptimiser(
        factor_graph,
        factor_optimisers={prior: ExactFactorFit(), likelihood: ExactFactorFit()},
        paths=False,
    )
    optimiser.run(model_approx, max_steps=6, max_consecutive_failures=100)

    spec, state = _state(optimiser, factor_graph)

    assert optimiser.ep_history.is_converged(likelihood) is True
    assert _by_name(spec, state, "like_x").status == "converged"


def test_a_factor_the_sweep_has_not_reached_is_absent():
    """
    ``EPHistory[factor]`` *creates* an empty history for any factor it is asked
    about, so a state read through it would report a never-visited factor as
    visited zero times instead of as absent. It is read through
    ``EPHistory.history`` for exactly that reason.
    """
    _, factor_graph, _, _ = make_shared_variable_approx()

    spec = EPGraphSpec.from_factor_graph(factor_graph)
    state = EPState.from_history(spec, EPHistory(), factor_by_key(factor_graph))

    assert {factor.status for factor in state.factors} == {"absent"}
    assert all(factor.sweeps == 0 and factor.age == 0 for factor in state.factors)
    assert state.step == 0


def test_a_missing_factor_key_is_absent_rather_than_an_error(partial_revert):
    """
    A collapsed ``PriorFactor`` stub stands for a group and has no single
    history, so the mapping will not always cover every spec node.
    """
    optimiser, factor_graph, _ = partial_revert
    spec = EPGraphSpec.from_factor_graph(factor_graph)

    mapping = factor_by_key(factor_graph)
    del mapping["factor-2"]

    state = EPState.from_history(spec, optimiser.ep_history, mapping)

    (missing,) = [factor for factor in state.factors if factor.key == "factor-2"]
    assert missing.status == "absent"


def test_step_is_the_longest_run_of_sweeps(partial_revert):
    optimiser, factor_graph, _ = partial_revert
    _, state = _state(optimiser, factor_graph)

    assert state.step == max(factor.sweeps for factor in state.factors) == 4


def test_model_approx_is_accepted_and_ignored(partial_revert):
    """
    Posterior values are a later overlay; the argument is in the signature so
    that it will not have to change when they arrive.
    """
    optimiser, factor_graph, _ = partial_revert
    spec = EPGraphSpec.from_factor_graph(factor_graph)
    mapping = factor_by_key(factor_graph)

    without = EPState.from_history(spec, optimiser.ep_history, mapping)
    with_approx = EPState.from_history(
        spec, optimiser.ep_history, mapping, model_approx=object()
    )

    assert without == with_approx


def test_to_dict_is_json_serialisable_and_stable(partial_revert):
    optimiser, factor_graph, _ = partial_revert
    spec = EPGraphSpec.from_factor_graph(factor_graph)
    mapping = factor_by_key(factor_graph)

    first = EPState.from_history(spec, optimiser.ep_history, mapping).to_dict()
    second = EPState.from_history(spec, optimiser.ep_history, mapping).to_dict()

    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
