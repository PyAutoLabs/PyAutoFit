"""
The phase-5 acceptance: the four things the EP figure exists to show.

Each bullet of the prompt is asserted twice -- once on the presentation and
state dataclasses, which is where the claim actually lives, and once as a PNG
smoke, because a claim that cannot survive being drawn is not shipped. The PNG
check is deliberately shallow (it exists, it is a PNG, it is not an empty
canvas); the encoding itself is pinned by ``test_presentation`` and
``test_layout``, which do not have to render anything to do it.

The doubles are the ones the reversion bugs themselves were fixed against
(``test_factor_failure_recovery``, PyAutoFit #1571, #1575, #1579), so an
acceptance failure and a graphical-behaviour failure are the same failure.
"""

import pytest

import autofit as af
from autofit import graphical as graph
from autofit.graphical.expectation_propagation.factor_optimiser import ExactFactorFit
from autofit.graphical.expectation_propagation.history import EPHistory
from test_autofit.graphical.functionality.test_factor_failure_recovery import (
    OverWideFit,
    PartialRevertFit,
    make_shared_variable_approx,
    make_two_variable_approx,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _written(path, filename: str):
    """The figure was really drawn: a PNG, with something on it."""
    target = path / f"{filename}.png"

    assert target.exists(), f"{target} was not written"
    contents = target.read_bytes()
    assert contents[:8] == PNG_MAGIC
    assert len(contents) > 1024, f"{target} is {len(contents)} bytes -- an empty canvas"
    return target


def _plotter(factor_graph, optimiser=None):
    return af.EPPlotter(
        factor_graph,
        ep_history=None if optimiser is None else optimiser.ep_history,
    )


@pytest.fixture
def partial_revert():
    """
    One factor that reverts ``y`` on every projection while moving ``x``: the
    partial revert of PyAutoFit #1575, run for real.
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
        paths=False,
    )
    optimiser.run(model_approx, max_steps=4, max_consecutive_failures=100)
    return optimiser, factor_graph, likelihood


@pytest.fixture
def over_wide():
    """A factor that reverts everything on every projection, run for real."""
    model_approx, factor_graph, prior, likelihood = make_shared_variable_approx()
    optimiser = graph.EPOptimiser(
        factor_graph,
        factor_optimisers={prior: ExactFactorFit(), likelihood: OverWideFit()},
        ep_history=EPHistory(kl_tol=None),
        paths=False,
    )
    optimiser.run(model_approx, max_steps=4, max_consecutive_failures=100)
    return optimiser, factor_graph, likelihood


# ----------------------------------------------------------------------------
# 1. both views of a hierarchical run are drawn
# ----------------------------------------------------------------------------


def test_both_views_of_a_hierarchical_graph_are_written(stalled_plate, tmp_path):
    """
    The headline: one call draws the graph, another draws the run on it.

    The history here is built from the same ``Status`` objects a run records
    rather than by running one, because an ``AnalysisFactor`` needs a real
    non-linear search per factor per sweep -- minutes, for a picture whose
    content is entirely determined by the statuses. The real-run legs are the
    fixtures below, on graphs small enough to sweep in milliseconds.
    """
    factor_graph, history, _ = stalled_plate

    af.EPPlotter(factor_graph).figure(path=tmp_path, format="png", kind="model")
    af.EPPlotter(factor_graph, ep_history=history).figure(
        path=tmp_path, format="png", kind="state"
    )

    _written(tmp_path, "graph_model")
    _written(tmp_path, "graph_state")


def test_the_state_view_refuses_to_draw_without_a_history(stalled_plate, tmp_path):
    """
    A diagnostic figure that silently contains no diagnostics is worse than an
    error, so ``kind="state"`` without a history says so.
    """
    factor_graph, _, _ = stalled_plate

    with pytest.raises(ValueError, match="ep_history"):
        af.EPPlotter(factor_graph).figure(kind="state", format=None)


def test_the_model_view_is_the_same_figure_without_the_run(stalled_plate):
    factor_graph, history, _ = stalled_plate
    plotter = af.EPPlotter(factor_graph, ep_history=history)

    model = plotter.presentation(kind="model")
    state = plotter.presentation(kind="state")

    assert plotter.state() is not None
    assert af.EPPlotter(factor_graph).state() is None
    assert all(node.state is None for node in model.nodes)
    assert {node.key for node in model.nodes} <= {node.key for node in state.nodes}


# ----------------------------------------------------------------------------
# 2. a stalled factor is drawn as stalled
# ----------------------------------------------------------------------------


def test_a_stalled_factor_is_a_stale_node_badged_with_its_zero_updates(
    over_wide, tmp_path
):
    optimiser, factor_graph, likelihood = over_wide
    presentation = _plotter(factor_graph, optimiser).presentation(kind="state")

    (node,) = [
        node for node in presentation.factor_nodes() if node.title == likelihood.name
    ]

    assert node.state == "stale"
    assert any("0 updates" in badge for badge in node.badges)
    assert likelihood in optimiser._factors_skipped  # the optimiser agrees

    _plotter(factor_graph, optimiser).figure(
        path=tmp_path, filename="stale", format="png", kind="state"
    )
    _written(tmp_path, "stale")


# ----------------------------------------------------------------------------
# 3. a reverted update is drawn on the variable it happened to
# ----------------------------------------------------------------------------


def test_a_partial_revert_marks_one_edge_and_only_one(partial_revert, tmp_path):
    optimiser, factor_graph, likelihood = partial_revert
    presentation = _plotter(factor_graph, optimiser).presentation(kind="state")

    reverted = [edge for edge in presentation.edges if edge.kind == "reverted"]

    assert [edge.label for edge in reverted] == ["like_xy.y"]
    assert not [
        edge
        for edge in presentation.edges
        if edge.kind == "reverted" and edge.label.endswith(".x")
    ]

    _plotter(factor_graph, optimiser).figure(
        path=tmp_path, filename="reverted", format="png", kind="state"
    )
    _written(tmp_path, "reverted")


def test_a_fixed_point_restart_marks_no_edge_at_all(tmp_path):
    """
    EP restarted from its own converged mean field reproduces every message
    exactly: nothing moves, and nothing is rejected. A fixed point is not a
    rejection (PyAutoFit #1579), and the figure must not paint one.
    """
    model_approx, factor_graph, prior, likelihood = make_shared_variable_approx()

    def run(approx, steps):
        optimiser = graph.EPOptimiser(
            factor_graph,
            factor_optimisers={prior: ExactFactorFit(), likelihood: ExactFactorFit()},
            ep_history=EPHistory(kl_tol=None),
            paths=False,
        )
        return optimiser, optimiser.run(
            approx, max_steps=steps, max_consecutive_failures=100
        )

    _, converged = run(model_approx, 6)
    optimiser, _ = run(converged, 4)

    presentation = _plotter(factor_graph, optimiser).presentation(kind="state")

    assert [edge for edge in presentation.edges if edge.kind == "reverted"] == []
    assert [edge for edge in presentation.edges if edge.kind == "stale"] == []
    assert optimiser._stale_factor_warnings() == []

    _plotter(factor_graph, optimiser).figure(
        path=tmp_path, filename="fixed_point", format="png", kind="state"
    )
    _written(tmp_path, "fixed_point")


# ----------------------------------------------------------------------------
# 4. a plate never hides the member that failed
# ----------------------------------------------------------------------------


def test_a_plate_names_and_draws_the_member_that_failed(stalled_plate, tmp_path):
    """
    The acceptance the epic review asked for by name: three datasets, one of
    which never updates, must not be reported as "3 datasets, working".
    """
    factor_graph, history, stalled = stalled_plate
    model = af.EPPlotter(factor_graph).presentation(kind="model")
    assert model.node("plate-0").note is None  # ... not in the model view

    presentation = af.EPPlotter(factor_graph, ep_history=history).presentation(
        kind="state"
    )
    plate = presentation.node("plate-0")

    assert plate.members == ("factor-0", "factor-1", "factor-2")
    assert stalled.name in plate.note

    (expanded,) = [
        node
        for node in presentation.factor_nodes()
        if node.title == stalled.name and node.key != plate.key
    ]
    assert expanded.state == "stale"
    assert expanded.plate_key == plate.key

    af.EPPlotter(factor_graph, ep_history=history).figure(
        path=tmp_path, filename="plate", format="png", kind="state"
    )
    _written(tmp_path, "plate")


def test_the_expanded_member_is_drawn_beside_the_plate_not_on_top_of_it(stalled_plate):
    factor_graph, history, _ = stalled_plate
    layout = af.EPPlotter(factor_graph, ep_history=history).layout(kind="state")

    plate = layout.box("plate-0")
    expanded = layout.box("factor-2")

    assert plate.rank == expanded.rank
    assert plate.x + plate.width < expanded.x or expanded.x + expanded.width < plate.x


# ----------------------------------------------------------------------------
# the plotter's own contract
# ----------------------------------------------------------------------------


def test_the_filename_defaults_by_kind_and_the_formats_are_the_usual_four(
    stalled_plate, tmp_path
):
    factor_graph, history, _ = stalled_plate
    plotter = af.EPPlotter(factor_graph, ep_history=history)

    plotter.figure(path=tmp_path, format="png", kind="model")
    plotter.figure(path=tmp_path, format="svg", kind="state")

    assert (tmp_path / "graph_model.png").exists()
    assert (tmp_path / "graph_state.svg").exists()

    with pytest.raises(ValueError):
        plotter.figure(path=tmp_path, format="gif")


def test_format_none_builds_the_figure_and_writes_nothing(stalled_plate, tmp_path):
    factor_graph, _, _ = stalled_plate

    figure = af.EPPlotter(factor_graph).figure(path=tmp_path, format=None)

    assert figure is not None
    assert list(tmp_path.iterdir()) == []

    import matplotlib.pyplot as plt

    plt.close(figure)


def test_an_unknown_kind_is_refused(stalled_plate):
    factor_graph, history, _ = stalled_plate

    with pytest.raises(ValueError, match="kind"):
        af.EPPlotter(factor_graph, ep_history=history).presentation(kind="diagnostic")
