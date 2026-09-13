"""
Layer 3 of the EP figure: what a reader is shown.

The assertions here are the review's, not the implementation's: a plate is
drawn once, an aggregate is never shown without the members that contradict it,
a label names something the reader can look up, and an edge carries the
sharpest claim that is true of it.
"""

import pytest

from autofit import graphical as graph
from autofit.graphical.expectation_propagation.factor_optimiser import ExactFactorFit
from autofit.graphical.expectation_propagation.history import EPHistory
from autofit.model_figure.ep.presentation import (
    EPPresentation,
    build_ep_presentation,
)
from autofit.model_figure.ep.spec import EPGraphSpec, factor_by_key
from autofit.model_figure.ep.state import EPState
from test_autofit.graph_spec.graphical_doubles import variable_graph
from test_autofit.graphical.functionality.test_factor_failure_recovery import (
    OverWideFit,
    PartialRevertFit,
    make_shared_variable_approx,
    make_two_variable_approx,
)


def _presentation(factor_graph, ep_history=None) -> EPPresentation:
    spec = EPGraphSpec.from_factor_graph(factor_graph)
    state = (
        None
        if ep_history is None
        else EPState.from_history(spec, ep_history, factor_by_key(factor_graph))
    )
    return build_ep_presentation(spec, state)


@pytest.fixture
def model_view(hierarchical_factor_graph):
    return _presentation(hierarchical_factor_graph)


# ----------------------------------------------------------------------------
# the model view
# ----------------------------------------------------------------------------


def test_a_plate_is_one_node_badged_with_its_count(model_view):
    """
    Six factors and three datasets collapse to two factor nodes: the figure
    scales with the *shape* of the graph, not with the number of datasets.
    """
    factors = model_view.factor_nodes()

    assert [node.key for node in factors] == ["plate-0", "plate-3"]
    assert [node.title for node in factors] == ["AnalysisFactor", "HierarchicalFactor0"]
    assert factors[0].badges == ("3 datasets",)
    assert factors[1].badges == ("3 members",)
    assert factors[0].members == ("factor-0", "factor-1", "factor-2")


def test_a_plate_variable_is_badged_with_how_many_it_stands_for(model_view):
    inside = [node for node in model_view.variable_nodes() if node.plate_key]

    assert [node.title for node in inside] == ["centre", "normalization", "sigma"]
    assert all(node.badges == ("x3",) for node in inside)


def test_a_hyper_variable_keeps_its_qualified_name(model_view):
    """
    There is one ``mean``, and ``HierarchicalFactor0.mean`` is what
    ``graph.info`` calls it -- stripping that prefix would name nothing.
    """
    hyper = [node for node in model_view.variable_nodes() if node.kind == "hyper"]

    assert [node.title for node in hyper] == [
        "HierarchicalFactor0.mean",
        "HierarchicalFactor0.sigma",
    ]
    assert all(node.badges == ("hyper",) for node in hyper)


def test_a_shared_variable_loses_the_one_member_its_label_named():
    """
    ``variable_graph`` shares ``centre`` across all three datasets, so the spec
    has to label it ``AnalysisFactor0.centre`` -- it sits outside the plate and
    the spec strips prefixes only inside one. Naming one of three members is
    worse than naming none, so the presentation strips it and says how many
    share it instead.
    """
    presentation = _presentation(variable_graph().graph)

    (shared,) = [node for node in presentation.variable_nodes() if not node.plate_key]

    assert shared.title == "centre"
    assert shared.badges == ("shared x 3",)


def test_the_model_view_paints_no_state(model_view):
    assert all(node.state is None for node in model_view.nodes)
    assert all(node.note is None for node in model_view.nodes)
    assert {edge.kind for edge in model_view.edges} == {"incidence"}
    assert model_view.kind == "model"


def test_every_incidence_becomes_exactly_one_edge_after_the_collapse(model_view):
    """
    Eighteen incidences, six edges: three from the dataset plate and three from
    the hierarchical group, each drawn once however many members it stands for.
    """
    assert len(model_view.edges) == 6
    assert len({(edge.source_key, edge.target_key) for edge in model_view.edges}) == 6


def test_the_footer_counts_the_graph_not_the_picture(model_view):
    assert model_view.footer == "6 factors   5 variables   2 plates"


def test_prior_stubs_are_drawn_beside_their_variable_when_asked_for(
    hierarchical_factor_graph,
):
    spec = EPGraphSpec.from_factor_graph(
        hierarchical_factor_graph, show_prior_factors=True
    )
    presentation = build_ep_presentation(spec)

    stubs = [node for node in presentation.nodes if node.kind == "prior"]

    assert len(stubs) == 5
    assert all(stub.title == "prior" for stub in stubs)
    # the stub on a plate variable stands for three priors, not one
    assert sorted(stub.badges for stub in stubs) == [(), (), ("x3",), ("x3",), ("x3",)]
    # ... and the footer still counts the six real factors
    assert presentation.footer.startswith("6 factors")


# ----------------------------------------------------------------------------
# the state view
# ----------------------------------------------------------------------------


def test_a_stalled_member_is_named_in_the_plates_note(stalled_plate):
    """
    The rule the diagnostic figure exists for: the plate may not report "3
    datasets, working" while one of them has never landed an update.
    """
    factor_graph, history, stalled = stalled_plate
    presentation = _presentation(factor_graph, history)

    plate = presentation.node("plate-0")

    assert plate.state == "working"
    assert plate.note == f"1 of 3 stale: {stalled.name}"


def test_the_stalled_member_is_also_drawn_as_a_node_of_its_own(stalled_plate):
    factor_graph, history, stalled = stalled_plate
    presentation = _presentation(factor_graph, history)

    expanded = presentation.node("factor-2")

    assert expanded.title == stalled.name
    assert expanded.state == "stale"
    assert expanded.subtitle == "member 3 of 3"
    assert expanded.plate_key == "plate-0"
    # the badges say why it is exceptional without the reader opening a log
    assert expanded.badges == ("0 updates / 4 sweeps", "age 4", "BAD_PROJECTION")


def test_the_stalled_members_edges_are_stale_and_the_plates_are_not(stalled_plate):
    """
    A plate's edges speak for the members it still stands for. The stale
    member's own edges carry its claim -- greying the aggregate too would say
    three datasets had stalled when one had.
    """
    factor_graph, history, _ = stalled_plate
    presentation = _presentation(factor_graph, history)

    own = [edge for edge in presentation.edges if edge.source_key == "factor-2"]
    plate = [edge for edge in presentation.edges if edge.source_key == "plate-0"]

    assert len(own) == 3
    assert {edge.kind for edge in own} == {"stale"}
    assert {edge.kind for edge in plate} == {"incidence"}


def test_the_footer_counts_every_status(stalled_plate):
    factor_graph, history, _ = stalled_plate
    presentation = _presentation(factor_graph, history)

    assert presentation.footer == (
        "6 factors   5 variables   2 plates   "
        "1 stale   0 reverting   0 converged   5 working   sweep 4"
    )
    assert presentation.kind == "state"
    assert "red dashed = reverted update" in presentation.legend


def test_a_reverted_update_is_a_reverted_edge_on_that_variable_alone(tmp_path):
    """
    ``PartialRevertFit`` reverts ``y`` on every projection and moves ``x``. The
    figure has to separate them: the factor updated, so nothing at factor level
    can see this, and an edge is the only place the claim fits.
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

    presentation = _presentation(factor_graph, optimiser.ep_history)
    (joint,) = [
        node for node in presentation.factor_nodes() if node.title == likelihood.name
    ]
    edges = {
        edge.label: edge.kind
        for edge in presentation.edges
        if edge.source_key == joint.key
    }

    assert joint.state == "reverting"
    assert edges["like_xy.y"] == "reverted"
    assert edges["like_xy.x"] == "incidence"


def test_a_stale_factors_edges_are_stale_even_though_it_reverted(tmp_path):
    """
    ``OverWideFit`` reverts everything on every projection, so the factor is
    both stale and reverting -- and the node reports the stronger claim, so its
    edges must not quietly report the weaker one.
    """
    model_approx, factor_graph, prior, likelihood = make_shared_variable_approx()
    optimiser = graph.EPOptimiser(
        factor_graph,
        factor_optimisers={prior: ExactFactorFit(), likelihood: OverWideFit()},
        ep_history=EPHistory(kl_tol=None),
        paths=False,
    )
    optimiser.run(model_approx, max_steps=4, max_consecutive_failures=100)

    presentation = _presentation(factor_graph, optimiser.ep_history)
    (over_wide,) = [
        node for node in presentation.factor_nodes() if node.title == likelihood.name
    ]
    kinds = {
        edge.kind for edge in presentation.edges if edge.source_key == over_wide.key
    }

    assert over_wide.state == "stale"
    assert over_wide.badges[0] == "0 updates / 4 sweeps"
    assert kinds == {"stale"}


def test_a_plate_with_nothing_exceptional_carries_no_note(hierarchical_factor_graph):
    """
    The exception list is not decoration: a plate whose members agree says
    nothing extra, or the one that does would not stand out.
    """
    from autofit.graphical.expectation_propagation.history import FactorHistory
    from autofit.graphical.utils import Status

    history = EPHistory(kl_tol=None, evidence_tol=None)
    for factor in hierarchical_factor_graph.factors:
        entry = FactorHistory(factor)
        entry.history.append((None, Status(success=True, updated=True)))
        history.history[factor] = entry

    presentation = _presentation(hierarchical_factor_graph, history)

    assert presentation.node("plate-0").note is None
    assert [node.key for node in presentation.factor_nodes()] == ["plate-0", "plate-3"]


def test_to_dict_is_stable_and_serialisable(stalled_plate):
    import json

    factor_graph, history, _ = stalled_plate

    first = _presentation(factor_graph, history).to_dict()
    second = _presentation(factor_graph, history).to_dict()

    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
