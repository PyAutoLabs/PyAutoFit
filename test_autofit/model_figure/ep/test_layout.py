"""
Layer 4 of the EP figure: where every box goes.

The measure of a factor-graph figure is whether it can be read, and these are
the four properties that decide that: ranks that tell the generative story top
to bottom, declaration order kept inside each rank, no edge crossing another,
and no box overlapping another. Each is asserted on the phase-4 hierarchical
double, so a layout regression and a figure regression are the same failure.
"""

import pytest

from autofit.model_figure.ep.layout import (
    NODE_GAP,
    _bipartite_fallback,
    _count_crossings,
    build_ep_layout,
    rank_of,
)
from autofit.model_figure.ep.presentation import build_ep_presentation
from autofit.model_figure.ep.spec import EPGraphSpec


@pytest.fixture
def presentation(hierarchical_factor_graph):
    return build_ep_presentation(
        EPGraphSpec.from_factor_graph(hierarchical_factor_graph)
    )


@pytest.fixture
def layout(presentation):
    return build_ep_layout(presentation)


def _overlap(first, second) -> bool:
    return (
        first.x < second.x + second.width
        and second.x < first.x + first.width
        and first.y < second.y + second.height
        and second.y < first.y + first.height
    )


def test_the_ranks_read_top_to_bottom_as_the_model_generates(layout):
    """
    Hyper parameters, then the distribution, then the variables it draws, then
    the analyses that observe them -- the generative story, in reading order.
    """
    ranks = {box.key: box.rank for box in layout.nodes}

    assert ranks["factor-3/var-1"] == 1  # HierarchicalFactor0.mean
    assert ranks["factor-3/var-2"] == 1  # HierarchicalFactor0.sigma
    assert ranks["plate-3"] == 2  # the hierarchical group
    assert ranks["plate-0/var-0"] == 3  # centre, drawn from it
    assert ranks["plate-0"] == 4  # the dataset plate

    # ... and a rank really is a row: same rank, same top edge.
    tops = {}
    for box in layout.nodes:
        tops.setdefault(box.rank, set()).add(box.y)
    assert all(len(values) == 1 for values in tops.values())


def test_a_prior_stub_ranks_by_the_variable_it_wraps(hierarchical_factor_graph):
    """
    A stub on a distribution's own parameter is a *hyper factor* and belongs
    above it; every other stub belongs at the bottom, under its variable.
    """
    spec = EPGraphSpec.from_factor_graph(
        hierarchical_factor_graph, show_prior_factors=True
    )
    layout = build_ep_layout(build_ep_presentation(spec))

    stubs = {box.key: box.rank for box in layout.nodes if box.kind == "prior"}

    assert stubs["factor-3/var-1/prior"] == 0
    assert stubs["plate-0/var-0/prior"] == 5


def test_declaration_order_is_kept_within_a_rank(layout):
    """
    Nothing is reordered to fill space. The three plate variables are laid out
    in the order the spec emitted them, so adding a fourth dataset moves
    nothing that was already on the figure.
    """
    rank_three = [box for box in layout.nodes if box.rank == 3]

    assert [box.key for box in rank_three] == [
        "plate-0/var-0",
        "plate-0/var-1",
        "plate-0/var-2",
    ]
    assert [box.title for box in rank_three] == ["centre", "normalization", "sigma"]
    assert sorted(box.x for box in rank_three) == [box.x for box in rank_three]


def test_no_edge_crosses_another(layout):
    assert _count_crossings(layout) == 0


def test_no_box_overlaps_another(layout):
    boxes = list(layout.nodes)

    for index, first in enumerate(boxes):
        for second in boxes[index + 1 :]:
            assert not _overlap(first, second), f"{first.key} overlaps {second.key}"


def test_boxes_on_a_rank_are_separated_by_at_least_the_node_gap(layout):
    by_rank = {}
    for box in layout.nodes:
        by_rank.setdefault(box.rank, []).append(box)

    for boxes in by_rank.values():
        ordered = sorted(boxes, key=lambda box: box.x)
        for left, right in zip(ordered, ordered[1:]):
            assert right.x - (left.x + left.width) >= NODE_GAP - 1e-6


def test_a_plate_frame_encloses_its_plate_node_and_its_variables(layout):
    (frame,) = [frame for frame in layout.plates if frame.key == "plate-0"]

    for key in ("plate-0", "plate-0/var-0", "plate-0/var-1", "plate-0/var-2"):
        box = layout.box(key)
        assert frame.x <= box.x
        assert box.x + box.width <= frame.x + frame.width
        assert frame.y <= box.y
        assert box.y + box.height <= frame.y + frame.height


def test_the_figure_is_big_enough_for_its_legend_and_footer(layout):
    """
    A figure sized only to its boxes clips the two lines that explain it.
    """
    assert layout.height > max(box.y + box.height for box in layout.nodes)
    assert layout.legend_y < layout.footer_y < layout.height


def test_an_edge_between_touching_ranks_is_a_straight_line(layout):
    for edge in layout.edges:
        source = layout.box(edge.source_key)
        target = layout.box(edge.target_key)
        if abs(source.rank - target.rank) <= 1:
            assert len(edge.points) == 2


def test_an_edge_across_a_rank_is_an_orthogonal_elbow(hierarchical_factor_graph):
    """
    A long edge drawn diagonally reads as an edge *to* the nodes it passes, so
    it turns instead.
    """
    spec = EPGraphSpec.from_factor_graph(
        hierarchical_factor_graph, show_prior_factors=True
    )
    layout = build_ep_layout(build_ep_presentation(spec))

    long_edges = [
        edge
        for edge in layout.edges
        if abs(layout.box(edge.source_key).rank - layout.box(edge.target_key).rank) > 1
    ]

    assert long_edges
    for edge in long_edges:
        assert len(edge.points) == 4
        # the two middle points share a y: the elbow is orthogonal
        assert edge.points[1][1] == edge.points[2][1]


def test_the_fallback_layout_is_used_when_networkx_cannot_place_the_graph(
    presentation, monkeypatch
):
    """
    ``multipartite_layout`` does not raise on any graph autofit builds -- but a
    figure must never be the reason a fit fails, so the two-column layout
    ``graphical.factor_graphs.graph.bipartite_layout`` has always drawn is
    there if it ever does.
    """
    import networkx

    def explode(*args, **kwargs):
        raise ValueError("no layout for you")

    monkeypatch.setattr(networkx, "multipartite_layout", explode)

    layout = build_ep_layout(presentation)

    assert layout.fallback is True
    assert len(layout.nodes) == len(presentation.nodes)
    boxes = list(layout.nodes)
    for index, first in enumerate(boxes):
        for second in boxes[index + 1 :]:
            assert not _overlap(first, second)


def test_the_fallback_spreads_every_rank_around_its_centre():
    ranks = {0: ["a"], 1: ["b", "c", "d"]}

    positions = _bipartite_fallback(ranks)

    assert positions["a"] == 0
    assert positions["b"] < positions["c"] < positions["d"]
    assert positions["c"] == 0


def test_rank_of_never_invents_a_rank(presentation):
    for node in presentation.nodes:
        assert 0 <= rank_of(node) <= 5
