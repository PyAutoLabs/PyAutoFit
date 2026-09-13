"""
Layer 4 of the EP figure -- measurement and placement.

Everything here is in **inches**, as in the model figure's own
:mod:`autofit.model_figure.layout`, so the drawing layer is a straight
transcription of the numbers this module computes. The two rules of that module
hold here too: text is **measured** (``measure_text``, a scratch Agg canvas),
never estimated, and font sizes never change with graph size.

What is different is the shape of the problem. The model figure is a
*containment* diagram, so it nests boxes. A factor graph is a **layered
bipartite** diagram, so it ranks them:

===== ==========================================================
rank  what sits on it
===== ==========================================================
0     hyper factors -- the priors on a distribution's parameters
1     hyper variables -- the distribution's own parameters
2     hierarchical factors (one node per group)
3     the model's variables -- drawn and free
4     analysis factors (one node per plate)
5     prior stubs, in the column of the variable they wrap
===== ==========================================================

Reading top to bottom is then reading the generative story: the hyper priors
feed the distribution, the distribution draws the per-dataset variables, and
the analyses observe them.

Within a rank the order is **declaration order**, adjusted by one stable
barycentre pass -- a node with neighbours on the rank above slides toward their
average position, a node without stays exactly where it was declared. One pass,
not the usual sweep to convergence: the graphs are small, and a figure that
reorders itself when a factor is added is not a figure anyone can diff.

``networkx.multipartite_layout`` supplies the base coordinates (the same
networkx that is already a hard dependency, ``pyproject.toml``; graphviz's
``dot`` is absent locally, on the GitHub runner and on Colab, so it is not an
option). Its coordinates are normalised, so they are scaled by the measured
width of the widest rank and then packed left to right, which is what
guarantees no two boxes overlap. If it raises,
:func:`_bipartite_fallback` draws the two-column layout
``autofit.graphical.factor_graphs.graph.bipartite_layout`` has always drawn.

:meth:`EPLayout.to_dict` is the determinism artefact: every coordinate rounded
to 4 dp, so ``json.dumps`` of it is byte-stable for a given graph and state.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from autofit.model_figure.layout import Style, _line_height, measure_text
from autofit.model_figure.ep.presentation import (
    EPPresentation,
    FACTOR_NODE_KINDS,
)

__all__ = [
    "EPNodeBox",
    "EPEdgeRoute",
    "EPPlateFrame",
    "EPLayout",
    "build_ep_layout",
    "rank_of",
]

#: Vertical space between two ranks, in inches.
RANK_GAP = 0.42

#: Horizontal space between two boxes on the same rank, in inches.
NODE_GAP = 0.26

#: Padding inside a node box.
NODE_PAD = 0.1

#: Padding between a plate's frame and the boxes it encloses.
PLATE_PAD = 0.16

#: The number of ranks. Kept explicit so :func:`rank_of` cannot silently
#: invent one.
RANKS = 6


def rank_of(node, neighbour_kinds: Tuple[str, ...] = ()) -> int:
    """
    The rank an :class:`~autofit.model_figure.ep.presentation.EPNode` sits on.

    A prior stub is a *hyper factor* when it wraps a distribution's own
    parameter -- it belongs above that parameter, at the top of the generative
    story -- and an ordinary stub otherwise, on the bottom rank in its
    variable's column. A stub carries no kind of its own that says which, so
    the kinds of the variables it is joined to are passed in.
    """
    kind = node.kind
    if kind == "prior":
        return 0 if "hyper" in neighbour_kinds else RANKS - 1
    if kind == "hyper":
        return 1
    if kind == "hierarchical":
        return 2
    if kind in ("drawn", "free"):
        return 3
    if kind in FACTOR_NODE_KINDS:
        return 4
    return 3


@dataclass(frozen=True)
class EPNodeBox:
    """One node's box, in inches, with the origin at the top left."""

    key: str
    kind: str
    title: str
    subtitle: Optional[str] = None
    badges: Tuple[str, ...] = ()
    state: Optional[str] = None
    note: Optional[str] = None
    rank: int = 0
    plate_key: Optional[str] = None
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    pad: float = NODE_PAD
    title_y: float = 0.0
    badge_offset: float = 0.0
    subtitle_y: Optional[float] = None
    note_y: Optional[float] = None

    @property
    def centre_x(self) -> float:
        return self.x + self.width / 2

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "title": self.title,
            "subtitle": self.subtitle,
            "badges": list(self.badges),
            "state": self.state,
            "note": self.note,
            "rank": self.rank,
            "plate_key": self.plate_key,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "width": round(self.width, 4),
            "height": round(self.height, 4),
        }


@dataclass(frozen=True)
class EPEdgeRoute:
    """One edge, routed: two points when the ranks touch, four when they do not."""

    source_key: str
    target_key: str
    kind: str
    points: Tuple[Tuple[float, float], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_key": self.source_key,
            "target_key": self.target_key,
            "kind": self.kind,
            "points": [[round(x, 4), round(y, 4)] for x, y in self.points],
        }


@dataclass(frozen=True)
class EPPlateFrame:
    """The dashed rectangle drawn around a plate's nodes."""

    key: str
    title: str
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "width": round(self.width, 4),
            "height": round(self.height, 4),
        }


@dataclass(frozen=True)
class EPLayout:
    """Every box, line and frame of the EP figure, in inches."""

    width: float
    height: float
    nodes: Tuple[EPNodeBox, ...] = ()
    edges: Tuple[EPEdgeRoute, ...] = ()
    plates: Tuple[EPPlateFrame, ...] = ()
    legend: str = ""
    footer: str = ""
    legend_y: float = 0.0
    footer_y: float = 0.0
    fallback: bool = False
    style: Style = field(default_factory=Style)

    def box(self, key: str) -> EPNodeBox:
        for node in self.nodes:
            if node.key == key:
                return node
        raise KeyError(key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": round(self.width, 4),
            "height": round(self.height, 4),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "plates": [plate.to_dict() for plate in self.plates],
            "legend": self.legend,
            "footer": self.footer,
            "legend_y": round(self.legend_y, 4),
            "footer_y": round(self.footer_y, 4),
            "fallback": self.fallback,
            "style": self.style.to_dict(),
        }


# ----------------------------------------------------------------------------
# measurement
# ----------------------------------------------------------------------------


def _badge_width(badge: str, style: Style) -> float:
    return round(measure_text(badge, style.badge_size)[0] + 0.12, 4)


def _measure(node, style: Style) -> Tuple[float, float, float, float]:
    """
    ``(width, height, badge_offset, title_height)`` of one node's box.

    A variable is a pill -- smaller type, tighter padding -- and a factor is a
    card, exactly as in the model figure.
    """
    title_size = style.title_size if node.is_factor else style.pill_size
    pad = NODE_PAD if node.is_factor else NODE_PAD * 0.75

    title_w = measure_text(
        node.title, title_size, "bold" if node.is_factor else "normal"
    )[0]
    badge_offset = title_w + (style.inner_gap if node.badges else 0.0)
    head_w = badge_offset + sum(
        _badge_width(badge, style) + style.inner_gap for badge in node.badges
    )
    if node.badges:
        head_w -= style.inner_gap

    widths = [head_w]
    if node.subtitle:
        widths.append(measure_text(node.subtitle, style.badge_size)[0])
    if node.note:
        widths.append(measure_text(node.note, style.note_size)[0])

    title_h = _line_height(title_size)
    height = title_h
    if node.subtitle:
        height += _line_height(style.badge_size)
    if node.note:
        height += _line_height(style.note_size)

    return (
        round(max(widths) + 2 * pad, 4),
        round(height + 2 * pad, 4),
        round(pad + badge_offset, 4),
        title_h,
    )


# ----------------------------------------------------------------------------
# ordering
# ----------------------------------------------------------------------------


def _ranked(presentation: EPPresentation, neighbours) -> Dict[int, List[str]]:
    """Node keys per rank, in declaration order."""
    kinds = {node.key: node.kind for node in presentation.nodes}
    ranks: Dict[int, List[str]] = {}
    for node in presentation.nodes:
        neighbour_kinds = tuple(
            kinds[other] for other in neighbours.get(node.key, ()) if other in kinds
        )
        ranks.setdefault(rank_of(node, neighbour_kinds), []).append(node.key)
    return ranks


def _barycentre(ranks: Dict[int, List[str]], neighbours: Dict[str, List[str]]):
    """
    One stable pass down the ranks.

    A node with neighbours on the rank above moves to their average position;
    a node without neighbours there does not move at all -- it keeps the slot
    declaration order gave it, and the movable nodes are re-dealt into the
    slots they already occupied. Ties keep declaration order, so the pass is
    idempotent and a graph that needs no reordering gets none.
    """
    for rank in sorted(ranks)[1:]:
        above = ranks.get(rank - 1)
        if not above:
            continue
        index_above = {key: index for index, key in enumerate(above)}
        order = ranks[rank]
        declared = {key: index for index, key in enumerate(order)}

        centres = {}
        for key in order:
            positions = [
                index_above[other]
                for other in neighbours.get(key, ())
                if other in index_above
            ]
            if positions:
                centres[key] = sum(positions) / len(positions)

        slots = [index for index, key in enumerate(order) if key in centres]
        if len(slots) < 2:
            continue
        movable = sorted(centres, key=lambda key: (centres[key], declared[key]))
        for slot, key in zip(slots, movable):
            order[slot] = key
    return ranks


def _base_positions(
    ranks: Dict[int, List[str]], edges
) -> Tuple[Dict[str, float], bool]:
    """
    Normalised horizontal positions, from ``networkx.multipartite_layout``.

    The layout's own within-rank order is not used -- it reverses insertion
    order -- so only its *slots* are taken: the sorted coordinates of a rank
    are dealt out to that rank's nodes in the order
    :func:`_barycentre` settled on.
    """
    try:
        import networkx as nx

        graph = nx.Graph()
        for rank in sorted(ranks):
            for key in ranks[rank]:
                graph.add_node(key, rank=rank)
        for edge in edges:
            if edge.source_key in graph and edge.target_key in graph:
                graph.add_edge(edge.source_key, edge.target_key)

        raw = nx.multipartite_layout(graph, subset_key="rank", align="horizontal")
        positions = {}
        for rank in sorted(ranks):
            slots = sorted(float(raw[key][0]) for key in ranks[rank])
            for key, slot in zip(ranks[rank], slots):
                positions[key] = slot
        return positions, False
    except Exception:  # pragma: no cover - exercised by monkeypatching
        return _bipartite_fallback(ranks), True


def _bipartite_fallback(ranks: Dict[int, List[str]]) -> Dict[str, float]:
    """
    The two-column layout of
    ``autofit.graphical.factor_graphs.graph.bipartite_layout``, flattened onto
    one axis: every rank simply spreads its nodes evenly.

    Only reached if ``multipartite_layout`` raises, which it does not for any
    graph autofit builds -- but a figure must never be the reason a fit fails.
    """
    widest = max((len(keys) for keys in ranks.values()), default=1)
    positions = {}
    for keys in ranks.values():
        offset = (len(keys) - 1) / 2
        for index, key in enumerate(keys):
            positions[key] = (index - offset) * widest / max(len(keys), 1)
    return positions


# ----------------------------------------------------------------------------
# routing
# ----------------------------------------------------------------------------


def _anchor(source: EPNodeBox, target: EPNodeBox):
    """Where an edge leaves one box and lands on the other."""
    if source.bottom <= target.y:
        return (source.centre_x, source.bottom), (target.centre_x, target.y)
    if target.bottom <= source.y:
        return (source.centre_x, source.y), (target.centre_x, target.bottom)
    # Same rank (only a fallback layout can do this): leave and land sideways.
    if source.centre_x <= target.centre_x:
        return (
            (source.x + source.width, source.y + source.height / 2),
            (target.x, target.y + target.height / 2),
        )
    return (
        (source.x, source.y + source.height / 2),
        (target.x + target.width, target.y + target.height / 2),
    )


def _route(edge, source: EPNodeBox, target: EPNodeBox) -> EPEdgeRoute:
    start, end = _anchor(source, target)
    if abs(source.rank - target.rank) <= 1 or start[0] == end[0]:
        points = (start, end)
    else:
        # An orthogonal elbow: a long edge that cut diagonally across a rank
        # would read as an edge *to* the nodes it passes.
        middle = round((start[1] + end[1]) / 2, 4)
        points = (start, (start[0], middle), (end[0], middle), end)
    return EPEdgeRoute(
        source_key=edge.source_key,
        target_key=edge.target_key,
        kind=edge.kind,
        points=points,
    )


def _segments(route: EPEdgeRoute):
    return list(zip(route.points, route.points[1:]))


def _crosses(first, second) -> bool:
    """Whether two segments properly intersect (touching at an end does not)."""

    def side(a, b, c):
        value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(value) < 1e-9:
            return 0
        return 1 if value > 0 else -1

    (p1, p2), (p3, p4) = first, second
    d1, d2 = side(p3, p4, p1), side(p3, p4, p2)
    d3, d4 = side(p1, p2, p3), side(p1, p2, p4)
    return d1 * d2 < 0 and d3 * d4 < 0


def _count_crossings(layout: EPLayout) -> int:
    """
    How many pairs of edges cross.

    Edges that share a node are not counted: three edges leaving one factor
    meet at that factor by construction, and calling that a crossing would make
    every bipartite figure look bad. Used by the layout tests as the quality
    measure the ranking exists to keep at zero.
    """
    crossings = 0
    edges = layout.edges
    for index, first in enumerate(edges):
        for second in edges[index + 1 :]:
            if {first.source_key, first.target_key} & {
                second.source_key,
                second.target_key,
            }:
                continue
            if any(
                _crosses(one, other)
                for one in _segments(first)
                for other in _segments(second)
            ):
                crossings += 1
    return crossings


# ----------------------------------------------------------------------------
# the build
# ----------------------------------------------------------------------------


def build_ep_layout(
    presentation: EPPresentation, style: Optional[Style] = None
) -> EPLayout:
    """
    Measure and place an
    :class:`~autofit.model_figure.ep.presentation.EPPresentation`.

    Parameters
    ----------
    presentation
        What is drawn, from
        :func:`~autofit.model_figure.ep.presentation.build_ep_presentation`.
    style
        Geometry and typography. The model figure's :class:`Style` is reused
        unchanged, so the two figures are visibly the same family.

    Returns
    -------
    An :class:`EPLayout`: every box in inches, ready to draw verbatim.
    """
    style = style or Style()

    nodes = {node.key: node for node in presentation.nodes}
    neighbours: Dict[str, List[str]] = {key: [] for key in nodes}
    for edge in presentation.edges:
        if edge.source_key in neighbours and edge.target_key in neighbours:
            neighbours[edge.source_key].append(edge.target_key)
            neighbours[edge.target_key].append(edge.source_key)

    ranks = _barycentre(_ranked(presentation, neighbours), neighbours)
    slots, fallback = _base_positions(ranks, presentation.edges)

    measured = {key: _measure(node, style) for key, node in nodes.items()}

    # -- place each rank: its normalised slots, scaled so the rank is exactly as
    # -- wide as its measured content needs, then packed left to right so that
    # -- nothing can overlap whatever the base coordinates said.
    placed: Dict[str, Tuple[float, float]] = {}  # key -> (left, top)
    blocks: Dict[int, Tuple[float, float]] = {}
    top = style.margin
    for rank in sorted(ranks):
        keys = ranks[rank]
        row_height = max(measured[key][1] for key in keys)
        span = max(slots[key] for key in keys) - min(slots[key] for key in keys)
        needed = sum(measured[key][0] for key in keys) + NODE_GAP * (len(keys) - 1)
        # Per rank, not per figure: one rank of long hyper-parameter names must
        # not stretch every other rank across a figure of empty inches.
        scale = needed / span if span > 0 else 1.0
        cursor = None
        for key in keys:
            width = measured[key][0]
            left = slots[key] * scale - width / 2
            if cursor is not None and left < cursor + NODE_GAP:
                left = cursor + NODE_GAP
            placed[key] = (left, top)
            cursor = left + width
        blocks[rank] = (
            min(placed[key][0] for key in keys),
            max(placed[key][0] + measured[key][0] for key in keys),
        )
        top += row_height + RANK_GAP

    content = max(right - left for left, right in blocks.values())
    boxes: Dict[str, EPNodeBox] = {}
    for rank in sorted(ranks):
        left, right = blocks[rank]
        offset = style.margin + (content - (right - left)) / 2 - left
        for key in ranks[rank]:
            node = nodes[key]
            width, height, badge_offset, title_h = measured[key]
            x, y = placed[key]
            pad = NODE_PAD if node.is_factor else NODE_PAD * 0.75
            subtitle_y = y + pad + title_h if node.subtitle else None
            note_y = (
                y + height - pad - _line_height(style.note_size) if node.note else None
            )
            boxes[key] = EPNodeBox(
                key=key,
                kind=node.kind,
                title=node.title,
                subtitle=node.subtitle,
                badges=node.badges,
                state=node.state,
                note=node.note,
                rank=rank,
                plate_key=node.plate_key,
                x=round(x + offset, 4),
                y=round(y, 4),
                width=width,
                height=height,
                pad=pad,
                title_y=round(y + pad, 4),
                badge_offset=badge_offset,
                subtitle_y=None if subtitle_y is None else round(subtitle_y, 4),
                note_y=None if note_y is None else round(note_y, 4),
            )

    ordered = tuple(boxes[node.key] for node in presentation.nodes)
    edges = tuple(
        _route(edge, boxes[edge.source_key], boxes[edge.target_key])
        for edge in presentation.edges
        if edge.source_key in boxes and edge.target_key in boxes
    )

    frames = []
    for plate in presentation.plates:
        enclosed = [boxes[key] for key in plate.node_keys if key in boxes]
        if len(enclosed) < 2:
            # A frame around the plate node alone says nothing its own
            # ``N members`` badge has not already said.
            continue
        left = min(box.x for box in enclosed) - PLATE_PAD
        right = max(box.x + box.width for box in enclosed) + PLATE_PAD
        top_edge = min(box.y for box in enclosed) - PLATE_PAD
        bottom = max(box.bottom for box in enclosed) + PLATE_PAD
        frames.append(
            EPPlateFrame(
                key=plate.key,
                title=plate.badge,
                x=round(left, 4),
                y=round(top_edge, 4),
                width=round(right - left, 4),
                height=round(bottom - top_edge, 4),
            )
        )

    bottom = max(
        (box.bottom for box in ordered),
        default=style.margin,
    )
    bottom = max(bottom, max((frame.y + frame.height for frame in frames), default=0.0))
    bottom += style.legend_gap
    legend_y = bottom
    bottom += _line_height(style.footer_size) + 0.04
    footer_y = bottom
    bottom += _line_height(style.footer_size) + style.margin

    text_right = style.margin + max(
        (
            measure_text(presentation.legend, style.footer_size)[0]
            if presentation.legend
            else 0.0
        ),
        (
            measure_text(presentation.footer, style.footer_size)[0]
            if presentation.footer
            else 0.0
        ),
    )
    right_edge = max(
        (box.x + box.width for box in ordered),
        default=style.margin,
    )
    right_edge = max(
        right_edge, max((frame.x + frame.width for frame in frames), default=0.0)
    )
    width = round(max(right_edge, text_right) + style.margin, 4)

    return EPLayout(
        width=width,
        height=round(bottom, 4),
        nodes=ordered,
        edges=edges,
        plates=tuple(frames),
        legend=presentation.legend,
        footer=presentation.footer,
        legend_y=round(legend_y, 4),
        footer_y=round(footer_y, 4),
        fallback=fallback,
        style=style,
    )
