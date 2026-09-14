"""
The public entry point -- ``af.ModelPlotter(model).figure()``.

The plotter owns the three layers and nothing else: it builds the
:class:`~autofit.graph_spec.GraphSpec` once (lazily, cached per ``collapse``
setting), transforms it into a
:class:`~autofit.model_figure.presentation.Presentation`, measures it into a
:class:`~autofit.model_figure.layout.LayoutTree` and draws it.  Each step is
also exposed on its own so tests can assert the vocabulary and the geometry
without rendering a pixel.

The name mirrors the ``*Plotter`` convention used across the organism, and the
signature mirrors ``autofit``'s own plotting convention -- ``path`` is a
directory and ``format`` is one of ``show | png | svg | pdf``, exactly as
``autofit/non_linear/plot/plot_util.py:output_figure`` has it.
"""

from typing import Optional, Sequence

__all__ = ["ModelPlotter"]


class ModelPlotter:
    """
    Draw a model's structure -- the map of which ``model.info`` is the legend.

    Parameters
    ----------
    model
        Any ``AbstractPriorModel``: ``af.Model``, ``af.Collection``, ``af.Array``
        or a ``FactorGraphModel``'s ``global_prior_model``.
    analysis
        An optional ``af.Analysis``; its latent catalogue is drawn as ``solved``
        pills marked absent from ``model.info``.
    solved_paths
        Paths whose rows are re-stated as *solved during fitting*.  Phase 3
        supplies the domain rules that populate this.

    Examples
    --------
    .. code-block:: python

        af.ModelPlotter(model).figure(path="output", format="png")
        af.ModelPlotter(model).figure(detail="priors", show_fixed=False)
    """

    def __init__(self, model, analysis=None, solved_paths: Sequence = ()):
        self.model = model
        self.analysis = analysis
        self.solved_paths = tuple(solved_paths)
        self._specs = {}
        self._priors = None

    # -- the layers ---------------------------------------------------------

    def spec(self, collapse: bool = True):
        """The (cached) semantic tree of the model."""
        from autofit.graph_spec import GraphSpec

        if collapse not in self._specs:
            self._specs[collapse] = GraphSpec.from_model(
                self.model,
                analysis=self.analysis,
                collapse=collapse,
                solved_paths=self.solved_paths,
            )
        return self._specs[collapse]

    def prior_summaries(self):
        """``{prior id: "U(0, 100)"}`` -- what ``detail="priors"`` prints."""
        from .presentation import prior_summaries

        if self._priors is None:
            self._priors = prior_summaries(self.model)
        return self._priors

    def presentation(
        self,
        detail: str = "names",
        collapse: bool = True,
        max_depth: Optional[int] = None,
        show_fixed: bool = True,
    ):
        """Layer 2 -- cards, pills, links, constraints, legend and footer."""
        from .presentation import build_presentation

        return build_presentation(
            self.spec(collapse=collapse),
            detail=detail,
            max_depth=max_depth,
            show_fixed=show_fixed,
            priors=self.prior_summaries() if detail == "priors" else None,
        )

    def layout(
        self,
        detail: str = "names",
        collapse: bool = True,
        max_depth: Optional[int] = None,
        show_fixed: bool = True,
        width: float = 14.0,
        style=None,
    ):
        """Layer 3a -- every box of the figure, in inches."""
        from .layout import Style, build_layout

        style = style or Style(width=width)
        return build_layout(
            self.presentation(
                detail=detail,
                collapse=collapse,
                max_depth=max_depth,
                show_fixed=show_fixed,
            ),
            style=style,
        )

    # -- the figure ---------------------------------------------------------

    def figure(
        self,
        path=None,
        filename: str = "model",
        format: str = "show",
        detail: str = "names",
        collapse: bool = True,
        max_depth: Optional[int] = None,
        show_fixed: bool = True,
        width: float = 14.0,
        style=None,
    ):
        """
        Draw the model figure and show or save it.

        Parameters
        ----------
        path
            The **directory** the file is written to (as everywhere else in
            autofit).  ``None`` writes nothing.
        filename
            The file's stem; the extension comes from ``format``.
        format
            ``show`` | ``png`` | ``svg`` | ``pdf``, or ``None`` to build the
            figure without showing or writing it.
        detail
            ``names`` (the map) or ``priors`` (the map with the legend's numbers
            on it).
        collapse
            Whether repeated sibling components collapse into plates.
        max_depth
            Fold anything deeper into one ``… N components / M priors`` row.
        show_fixed
            Fixed parameters are shown by default; hiding them prints a visible
            hidden count in the footer.
        width
            The width budget in inches.  Top-level cards wrap within it; text is
            never scaled down to fit.

        Returns
        -------
        The ``matplotlib`` ``Figure``, or ``None`` when ``format="show"``: the
        figure has been shown and closed, and returning it would make a notebook
        draw it a second time as the cell's own output.
        """
        from .render import draw, save

        layout = self.layout(
            detail=detail,
            collapse=collapse,
            max_depth=max_depth,
            show_fixed=show_fixed,
            width=width,
            style=style,
        )
        figure = draw(layout, layout.style)
        return save(figure, path=path, filename=filename, format=format)
