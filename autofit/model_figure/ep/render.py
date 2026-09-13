"""
Layer 5 of the EP figure -- drawing.

Pure matplotlib, and pure *re-use*: the palette, the rounded box, the text
helper, the badge colours, the link colours and :func:`save` all come from
:mod:`autofit.model_figure.render`, so the two figures cannot drift apart. What
this module adds is the encoding the diagnostic view needs and the model figure
has no use for -- a factor's status painted on its node, and a rejected update
painted on the edge that carries it.

The encoding
------------

======================= ====================================================
what                    how
======================= ====================================================
analysis factor         a tinted rounded card, as a model card is
hierarchical factor     a violet card, as a hyper card is
variable                a white pill
plate                   a dashed frame, as plate notation has it
stale factor            grey fill, muted text, and its edges greyed
reverting factor        a red outline; the rejected edge is dashed red with
                        an arrow head landing on the variable that was put
                        back
converged factor        a green outline
working / absent        the default outline
======================= ====================================================

There is no *observed data* node: the EP spec is read from the factor graph,
which carries no notion of the data an ``AnalysisFactor``'s analysis holds. The
model figure draws observed rows because its spec has them; inventing one here
would be a claim the figure cannot support.

matplotlib is imported inside the functions, as everywhere else in
``model_figure``, so ``import autofit`` stays drawing-free.
"""

from autofit.model_figure.render import (
    PALETTE,
    _LINK_COLOURS,
    _badge_colours,
    _box,
    _text,
    save,
)

__all__ = ["draw", "save", "PALETTE"]

#: ``state -> (face, edge, text, line width)`` for a factor node.
_STATE_STYLE = {
    "stale": (PALETTE["grey_fill"], PALETTE["plate_border"], PALETTE["muted"], 0.9),
    "reverting": (None, PALETTE["red"], PALETTE["text"], 1.1),
    "converged": (None, PALETTE["green"], PALETTE["text"], 1.0),
    "working": (None, PALETTE["border"], PALETTE["text"], 0.9),
    "absent": (None, PALETTE["pill_border"], PALETTE["muted"], 0.8),
}

#: ``edge kind -> (colour, dashed, arrow head)``.
_EDGE_STYLE = {
    "incidence": (PALETTE["muted"], False, False),
    "stale": (PALETTE["pill_border"], False, False),
    "reverted": (PALETTE["red"], True, True),
}

#: The face colour of a factor node, before its state has a say.
_FACE = {
    "analysis": PALETTE["tints"][1],
    "hierarchical": PALETTE["violet_fill"],
    "prior": PALETTE["tints"][2],
    "factor": PALETTE["tints"][1],
}

#: The edge colour of a factor node, before its state has a say.
_EDGE = {
    "analysis": PALETTE["border"],
    "hierarchical": PALETTE["violet"],
    "prior": PALETTE["pill_border"],
    "factor": PALETTE["border"],
}


def _node_colours(box):
    """
    ``(face, edge, text, line width)`` for one node.

    The kind decides what the node *is*; the state decides what the run made of
    it, and where the two disagree the state wins -- a stale analysis factor is
    grey, not tinted, because "this never updated" is the thing the reader has
    to see first.
    """
    if box.kind in ("free", "drawn", "hyper"):
        face = PALETTE["violet_fill"] if box.kind == "drawn" else "#ffffff"
        edge = PALETTE["violet"] if box.kind == "drawn" else PALETTE["pill_border"]
        return face, edge, PALETTE["text"], 0.8

    face = _FACE.get(box.kind, _FACE["factor"])
    edge = _EDGE.get(box.kind, _EDGE["factor"])
    text = PALETTE["text"]
    width = 0.9

    state = _STATE_STYLE.get(box.state)
    if state is not None:
        state_face, state_edge, state_text, state_width = state
        face = state_face or face
        edge = state_edge
        text = state_text
        width = state_width
    return face, edge, text, width


def _draw_node(ax, box, style):
    from autofit.model_figure.layout import _line_height, measure_text

    face, edge, colour, width = _node_colours(box)
    variable = box.kind in ("free", "drawn", "hyper")
    title_size = style.pill_size if variable else style.title_size

    _box(
        ax,
        box.x,
        box.y,
        box.width,
        box.height,
        face=face,
        edge=edge,
        lw=width,
        radius=0.06 if variable else 0.04,
        z=20,
    )

    title_middle = box.title_y + _line_height(title_size) / 2
    _text(
        ax,
        box.x + box.pad,
        title_middle,
        box.title,
        size=title_size,
        colour=colour,
        weight="normal" if variable else "bold",
        z=40,
    )

    offset = box.badge_offset
    for badge in box.badges:
        fill, badge_colour = _badge_colours(
            badge, "drawn" if box.kind == "drawn" else "free"
        )
        badge_width = measure_text(badge, style.badge_size)[0] + 0.12
        _box(
            ax,
            box.x + offset,
            box.y + box.pad + 0.02,
            badge_width,
            _line_height(title_size) - 0.04,
            face=fill,
            edge=fill,
            radius=0.02,
            z=30,
        )
        _text(
            ax,
            box.x + offset + 0.05,
            title_middle,
            badge,
            size=style.badge_size,
            colour=badge_colour,
            z=40,
        )
        offset += badge_width + style.inner_gap

    if box.subtitle:
        _text(
            ax,
            box.x + box.pad,
            box.subtitle_y + _line_height(style.badge_size) / 2,
            box.subtitle,
            size=style.badge_size,
            colour=PALETTE["muted"],
            z=40,
        )
    if box.note:
        # The exception list is the one thing on the figure a reader must not
        # skim past, so it is drawn in the reverting colour rather than muted.
        _text(
            ax,
            box.x + box.pad,
            box.note_y + _line_height(style.note_size) / 2,
            box.note,
            size=style.note_size,
            colour=PALETTE["red"] if box.state != "stale" else PALETTE["muted"],
            z=40,
        )


def _draw_edge(ax, route, boxes):
    """
    One edge.

    A plain incidence from a hierarchical factor onto the variable it draws is
    coloured with the model figure's own ``draw`` colour: it is the same claim
    that figure's violet arrow makes, and the two must not say it in different
    colours.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    colour, dashed, arrow = _EDGE_STYLE.get(route.kind, _EDGE_STYLE["incidence"])
    source = boxes.get(route.source_key)
    target = boxes.get(route.target_key)
    if (
        route.kind == "incidence"
        and source is not None
        and target is not None
        and source.kind == "hierarchical"
        and target.kind == "drawn"
    ):
        colour = _LINK_COLOURS["draw"]
    xs = [x for x, _ in route.points]
    ys = [y for _, y in route.points]

    ax.add_line(
        Line2D(
            xs[:-1] if arrow else xs,
            ys[:-1] if arrow else ys,
            color=colour,
            linewidth=1.0 if route.kind == "reverted" else 0.8,
            linestyle=(0, (2.6, 1.8)) if dashed else "solid",
            solid_capstyle="round",
            zorder=10,
        )
    )
    if arrow:
        # A reverted update is *directed*: the message was put back onto this
        # variable, so the head lands on the variable's pill.
        ax.add_patch(
            FancyArrowPatch(
                (xs[-2], ys[-2]),
                (xs[-1], ys[-1]),
                arrowstyle="-|>",
                mutation_scale=7.0,
                linewidth=1.0,
                linestyle="dashed",
                color=colour,
                shrinkA=0.0,
                shrinkB=0.0,
                zorder=11,
            )
        )


def _draw_plate(ax, frame, style):
    from autofit.model_figure.layout import _line_height, measure_text

    _box(
        ax,
        frame.x,
        frame.y,
        frame.width,
        frame.height,
        face="none",
        edge=PALETTE["plate_border"],
        dashed=True,
        lw=0.9,
        radius=0.05,
        z=5,
    )
    if frame.title:
        width = measure_text(frame.title, style.badge_size)[0] + 0.1
        height = _line_height(style.badge_size)
        _text(
            ax,
            frame.x + frame.width - width,
            frame.y + frame.height - height / 2 - 0.02,
            frame.title,
            size=style.badge_size,
            colour=PALETTE["muted"],
            z=6,
        )


def draw(layout, style=None):
    """
    Draw a measured :class:`~autofit.model_figure.ep.layout.EPLayout`.

    One axes, one inch per data unit, y inverted and the axis off -- the same
    canvas the model figure uses, so every coordinate the layout computed is
    used verbatim.
    """
    import matplotlib.pyplot as plt

    from autofit.model_figure.layout import _line_height

    style = style or layout.style

    figure = plt.figure(figsize=(layout.width, layout.height), dpi=style.dpi)
    ax = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, layout.width)
    ax.set_ylim(layout.height, 0.0)
    ax.set_aspect("equal")
    ax.axis("off")
    figure.patch.set_facecolor("#ffffff")

    for frame in layout.plates:
        _draw_plate(ax, frame, style)
    boxes = {box.key: box for box in layout.nodes}
    for route in layout.edges:
        _draw_edge(ax, route, boxes)
    for box in layout.nodes:
        _draw_node(ax, box, style)

    if layout.legend:
        _text(
            ax,
            style.margin,
            layout.legend_y + _line_height(style.footer_size) / 2,
            layout.legend,
            size=style.footer_size,
            colour=PALETTE["muted"],
            z=90,
        )
    if layout.footer:
        _text(
            ax,
            style.margin,
            layout.footer_y + _line_height(style.footer_size) / 2,
            layout.footer,
            size=style.footer_size,
            colour=PALETTE["text"],
            z=90,
        )
    return figure
