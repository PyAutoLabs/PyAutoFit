"""
Layer 3b of the model figure -- drawing.

Pure matplotlib: rounded ``FancyBboxPatch`` cards and pills, ``ax.text`` labels
and ``Line2D`` cross links.  **Zero new dependencies** -- graphviz is
deliberately not adopted (the ``dot`` binary is absent locally, on the GitHub
``ubuntu-24.04`` runner and often on Colab, and its rank layout is wasted on a
containment figure).

The layout has already decided every coordinate, in inches, so the axes are set
up as one inch per data unit with the y axis inverted: what :mod:`layout` calls
``y`` grows downward, as reading order does.

matplotlib is imported inside the functions, following
``autofit/non_linear/plot/plot_util.py``, so importing :mod:`autofit` never
drags a drawing library in.

Palette: near monochrome, **three card tints only** (review: "let boundaries,
indentation and labels carry hierarchy; tint should assist").  Blue is sharing
and nothing else; orange is relations and constraints and nothing else; red is
missing configuration.
"""

import os
from pathlib import Path

__all__ = ["draw", "save", "PALETTE"]


PALETTE = {
    "tints": ("#ffffff", "#f5f5f9", "#ebebf1"),
    "border": "#9c9ca8",
    "plate_border": "#6f6f7d",
    "text": "#17171d",
    "muted": "#4c4c57",
    "blue": "#15539e",
    "blue_fill": "#e2ecfb",
    "orange": "#9c4a06",
    "cream": "#fdf3e1",
    "grey_fill": "#e3e3e9",
    "pill_border": "#b4b4c0",
    "red": "#a81f16",
    "red_fill": "#fceceb",
    "violet": "#6a4b8c",
    "violet_fill": "#f1eaf8",
    "green": "#3c6b3c",
    "tag_fill": "#ececf1",
}

#: ``state -> (face colour, edge colour, text colour, dashed)``
_PILL_STYLE = {
    "free": ("#ffffff", PALETTE["pill_border"], PALETTE["text"], False),
    "fixed": (PALETTE["grey_fill"], PALETTE["pill_border"], PALETTE["muted"], False),
    "fixed-varies": (
        PALETTE["grey_fill"],
        PALETTE["plate_border"],
        PALETTE["muted"],
        False,
    ),
    "relation": (PALETTE["cream"], PALETTE["orange"], PALETTE["text"], False),
    "solved": ("#ffffff", PALETTE["muted"], PALETTE["text"], True),
    "missing": (PALETTE["red_fill"], PALETTE["red"], PALETTE["red"], False),
    "observed": ("#e9f1e9", PALETTE["green"], PALETTE["text"], False),
    "drawn": (PALETTE["violet_fill"], PALETTE["violet"], PALETTE["text"], False),
    "folded": ("#f4f4f8", PALETTE["pill_border"], PALETTE["muted"], True),
}


def _box(
    ax, x, y, width, height, *, face, edge, dashed=False, lw=0.8, radius=0.045, z=1
):
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw,
        facecolor=face,
        edgecolor=edge,
        linestyle=(0, (2.4, 1.6)) if dashed else "solid",
        mutation_aspect=1.0,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def _text(ax, x, y, text, *, size, colour, weight="normal", z=3):
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=colour,
        fontweight=weight,
        ha="left",
        va="center",
        zorder=z,
    )


def _badge_colours(badge: str, state: str = "free"):
    """
    Blue for sharing; **violet** for a hierarchical draw; neutral for everything
    else (repetition is not sharing, and a draw is not sharing either -- the two
    say opposite things, so they never share a colour).
    """
    if state == "drawn":
        return PALETTE["violet_fill"], PALETTE["violet"]
    if badge.startswith("shared") or badge.startswith("↗"):
        return PALETTE["blue_fill"], PALETTE["blue"]
    return PALETTE["tag_fill"], PALETTE["muted"]


def _draw_pill(ax, pill, style, card_level=1):
    from .layout import TAG_2D

    face, edge, colour, dashed = _PILL_STYLE.get(pill.state, _PILL_STYLE["free"])
    _box(
        ax,
        pill.x,
        pill.y,
        pill.width,
        pill.height,
        face=face,
        edge=edge,
        dashed=dashed,
        lw=1.1 if pill.state == "missing" else 0.8,
        radius=0.035,
        z=40 + card_level,
    )
    middle = pill.y + pill.height / 2
    _text(
        ax,
        pill.x + pill.text_offset,
        middle,
        pill.text,
        size=style.pill_size,
        colour=colour,
        z=60 + card_level,
    )
    if pill.tag_offset is not None:
        tag_w = pill.width - pill.tag_offset - style.pill_pad_x
        if pill.badge_offset is not None:
            tag_w = pill.badge_offset - pill.tag_offset - style.inner_gap
        _box(
            ax,
            pill.x + pill.tag_offset,
            pill.y + 0.035,
            max(tag_w, 0.05),
            pill.height - 0.07,
            face=PALETTE["tag_fill"],
            edge=PALETTE["tag_fill"],
            radius=0.02,
            z=50 + card_level,
        )
        _text(
            ax,
            pill.x + pill.tag_offset + 0.03,
            middle,
            TAG_2D,
            size=style.badge_size,
            colour=PALETTE["muted"],
            z=60 + card_level,
        )
    if pill.badge_offset is not None:
        fill, colour = _badge_colours(pill.badge, pill.state)
        _box(
            ax,
            pill.x + pill.badge_offset,
            pill.y + 0.035,
            pill.width - pill.badge_offset - style.pill_pad_x,
            pill.height - 0.07,
            face=fill,
            edge=fill,
            radius=0.02,
            z=50 + card_level,
        )
        _text(
            ax,
            pill.x + pill.badge_offset + 0.04,
            middle,
            pill.badge,
            size=style.badge_size,
            colour=colour,
            z=60 + card_level,
        )


def _draw_card(ax, card, style):
    from .layout import _line_height, measure_text

    tint = PALETTE["tints"][min(card.level - 1, len(PALETTE["tints"]) - 1)]
    plate = card.kind == "plate"
    # A hyper card is the source of every violet draw arrow, and a hoisted card
    # holds what is shared: each is outlined in its own edge's colour so the
    # reader can follow the arrow back to the box it comes from.
    edge = PALETTE["border"]
    if plate:
        edge = PALETTE["plate_border"]
    elif card.kind == "hyper":
        edge = PALETTE["violet"]
    elif card.kind == "hoist":
        edge = PALETTE["blue"]
    _box(
        ax,
        card.x,
        card.y,
        card.width,
        card.height,
        face=tint,
        edge=edge,
        dashed=plate,
        lw=1.1 if card.level == 1 or plate else 0.8,
        z=card.level,
    )

    title_middle = card.title_y + _line_height(style.title_size) / 2
    _text(
        ax,
        card.x + card.pad,
        title_middle,
        card.title,
        size=style.title_size,
        colour=PALETTE["text"],
        weight="bold",
        z=card.level + 0.5,
    )
    if card.badge:
        width = measure_text(card.title, style.title_size, "bold")[0]
        badge_w = measure_text(card.badge, style.badge_size)[0] + 0.1
        badge_x = card.x + card.pad + width + style.inner_gap
        _box(
            ax,
            badge_x,
            card.title_y + 0.03,
            badge_w,
            _line_height(style.title_size) - 0.06,
            face=PALETTE["tag_fill"],
            edge=PALETTE["plate_border"],
            radius=0.02,
            lw=0.6,
            z=card.level + 0.4,
        )
        _text(
            ax,
            badge_x + 0.05,
            title_middle,
            card.badge,
            size=style.badge_size,
            colour=PALETTE["muted"],
            z=card.level + 0.5,
        )
    if card.subtitle:
        _text(
            ax,
            card.x + card.pad,
            card.subtitle_y + _line_height(style.badge_size) / 2,
            card.subtitle,
            size=style.badge_size,
            colour=PALETTE["muted"],
            z=card.level + 0.5,
        )

    for pill in card.pills:
        _draw_pill(ax, pill, style, card.level)

    for box in card.constraints:
        # Dashed outline, no fill: a constraint is not a relation, and the solid
        # cream relation pill must stay distinguishable from it.
        _box(
            ax,
            box.x,
            box.y,
            box.width,
            box.height,
            face="none",
            edge=PALETTE["orange"],
            dashed=True,
            radius=0.025,
            lw=0.9,
            z=70,
        )
        _text(
            ax,
            box.x + 0.08,
            box.y + box.height / 2,
            box.text,
            size=style.badge_size,
            colour=PALETTE["orange"],
            z=71,
        )

    if card.note:
        _text(
            ax,
            card.x + card.pad,
            card.note_y + _line_height(style.note_size) / 2,
            card.note,
            size=style.note_size,
            colour=PALETTE["muted"],
            z=card.level + 0.5,
        )

    for child in card.children:
        _draw_card(ax, child, style)


_LINK_COLOURS = {
    "shared": PALETTE["blue"],
    "draw": PALETTE["violet"],
    "relation": PALETTE["orange"],
}


def _draw_link(ax, route):
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    draw = route.kind == "draw"
    colour = _LINK_COLOURS.get(route.kind, PALETTE["orange"])
    xs = [x for x, _ in route.points]
    ys = [y for _, y in route.points]
    # A draw is **directed**: the last segment carries an arrowhead that lands
    # on the drawn pill, because "centre_i is drawn from this distribution" is a
    # claim with a direction, unlike sharing.
    ax.add_line(
        Line2D(
            xs[:-1] if draw else xs,
            ys[:-1] if draw else ys,
            color=colour,
            linewidth=0.9 if route.kind in ("shared", "draw") else 0.7,
            linestyle="solid" if route.kind in ("shared", "draw") else (0, (3, 1.6)),
            solid_capstyle="round",
            zorder=80,
        )
    )
    if draw:
        ax.add_patch(
            FancyArrowPatch(
                (xs[-2], ys[-2]),
                (xs[-1], ys[-1]),
                arrowstyle="-|>",
                mutation_scale=7.0,
                linewidth=0.9,
                color=colour,
                shrinkA=0.0,
                shrinkB=0.0,
                zorder=81,
            )
        )
    # A filled dot where each end touches the card it belongs to.
    ax.add_line(
        Line2D(
            [xs[0]] if draw else [xs[0], xs[-1]],
            [ys[0]] if draw else [ys[0], ys[-1]],
            linestyle="none",
            marker="o",
            markersize=2.6,
            markerfacecolor=colour,
            markeredgecolor=colour,
            color=colour,
            zorder=81,
        )
    )


def draw(layout, style=None):
    """
    Draw a measured :class:`~autofit.model_figure.layout.LayoutTree`.

    One axes, one inch per data unit, y inverted and the axis off, so every
    coordinate the layout computed is used verbatim.
    """
    import matplotlib.pyplot as plt
    from .layout import _line_height

    style = style or layout.style

    figure = plt.figure(figsize=(layout.width, layout.height), dpi=style.dpi)
    ax = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, layout.width)
    ax.set_ylim(layout.height, 0.0)
    ax.set_aspect("equal")
    ax.axis("off")
    figure.patch.set_facecolor("#ffffff")

    for card in layout.cards:
        _draw_card(ax, card, style)
    for route in layout.links:
        _draw_link(ax, route)

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


def save(figure, path=None, filename="model", format="show"):
    """
    Show or write the figure, mirroring
    ``autofit/non_linear/plot/plot_util.py:output_figure`` with ``svg`` added.

    ``format="show"`` returns ``None``, as ``output_figure`` does: the figure has
    already been drawn, and handing a ``Figure`` back to a notebook makes IPython
    render it a *second* time as the cell's ``execute_result`` beside the one
    ``plt.show()`` published.

    ``format=None`` builds the figure and neither shows nor writes it; any other
    unrecognised format raises rather than silently doing nothing.

    ``metadata={"Software": None}`` keeps PNG bytes identical between two saves
    of the same model, which the determinism acceptance needs.
    """
    import matplotlib.pyplot as plt

    if format is None:
        # Built, neither shown nor written -- the caller owns the figure and is
        # responsible for closing it.
        return figure
    if format == "show":
        plt.show()
        plt.close(figure)
        return None
    if format not in ("png", "pdf", "svg"):
        raise ValueError(
            f"format must be one of None, 'show', 'png', 'svg', 'pdf', not {format!r}"
        )
    if path is not None:
        os.makedirs(path, exist_ok=True)
        target = Path(path) / f"{filename}.{format}"
        metadata = {"Software": None} if format == "png" else None
        figure.savefig(target, format=format, metadata=metadata)
    plt.close(figure)
    return figure
