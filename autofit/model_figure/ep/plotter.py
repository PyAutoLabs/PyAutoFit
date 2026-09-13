"""
The public entry point -- ``af.EPPlotter(factor_graph).figure()``.

``af.ModelPlotter`` draws a model's containment structure; ``af.EPPlotter``
draws the **factor graph** an :class:`~autofit.graphical.EPOptimiser` sweeps,
and -- given the optimiser's history -- what the run has done to it. The two
plotters are deliberately the same shape: the plotter owns the layers and
nothing else, each layer is exposed on its own so a test can assert the
vocabulary and the geometry without rendering a pixel, and
:meth:`EPPlotter.figure` takes ``path`` as a *directory* and ``format`` as one
of ``show | png | svg | pdf``, exactly as
``autofit/non_linear/plot/plot_util.py:output_figure`` has it.

The two views
-------------

``kind="model"``
    Structure only. Drawable before the first sweep, and written once per run.
``kind="state"``
    The same structure with the run painted on -- factor status, update age,
    the (factor, variable) pairs whose projection was rejected. Needs an
    ``ep_history``; asking for it without one raises rather than quietly
    drawing the model view, because a diagnostic figure that silently contains
    no diagnostics is worse than an error.

Examples
--------
.. code-block:: python

    optimiser = af.graphical.EPOptimiser(model.graph)
    result = optimiser.run(model_approx)

    af.EPPlotter(optimiser.factor_graph).figure(path="output", format="png")
    af.EPPlotter(
        optimiser.factor_graph, ep_history=optimiser.ep_history
    ).figure(path="output", format="png", kind="state")
"""

from typing import Optional

__all__ = ["EPPlotter"]

#: ``kind -> default filename``. The names the EP hook writes into a fit's
#: output directory, so a plotter call and a fit produce the same files.
FILENAMES = {"model": "graph_model", "state": "graph_state"}


class EPPlotter:
    """
    Draw the factor graph an EP fit sweeps, with the fit's state on it.

    Parameters
    ----------
    factor_graph
        ``EPOptimiser.factor_graph`` -- the graph the optimiser sweeps and the
        one its ``EPHistory`` is keyed by. Never a rebuilt
        ``AbstractDeclarativeFactor.graph``: that property constructs a new
        graph and renames every ``PriorFactor`` on each access, and the history
        would then match nothing.
    ep_history
        ``EPOptimiser.ep_history``. Optional: without it only the model view
        can be drawn.
    model_approx
        The latest ``EPMeanField``. Accepted and passed through; posterior
        values (mean, std, precision, KL) are a later overlay.
    """

    def __init__(self, factor_graph, ep_history=None, model_approx=None):
        self.factor_graph = factor_graph
        self.ep_history = ep_history
        self.model_approx = model_approx
        self._specs = {}
        self._state = None

    # -- the layers ---------------------------------------------------------

    def spec(self, show_prior_factors: bool = False):
        """Layer 1 -- the (cached) structure of the factor graph."""
        from .spec import EPGraphSpec

        if show_prior_factors not in self._specs:
            self._specs[show_prior_factors] = EPGraphSpec.from_factor_graph(
                self.factor_graph, show_prior_factors=show_prior_factors
            )
        return self._specs[show_prior_factors]

    def state(self, show_prior_factors: bool = False):
        """
        Layer 2 -- what the run did, or ``None`` when there is no history.

        Not cached across ``show_prior_factors`` settings: the overlay is
        painted onto a spec's keys, and the two specs have different ones.
        """
        if self.ep_history is None:
            return None

        from .spec import factor_by_key
        from .state import EPState

        return EPState.from_history(
            self.spec(show_prior_factors=show_prior_factors),
            self.ep_history,
            factor_by_key(self.factor_graph),
            model_approx=self.model_approx,
        )

    def presentation(self, kind: str = "model", show_prior_factors: bool = False):
        """Layer 3 -- nodes, edges, plates, legend and footer."""
        from .presentation import build_ep_presentation

        return build_ep_presentation(
            self.spec(show_prior_factors=show_prior_factors),
            self._state_for(kind, show_prior_factors),
        )

    def layout(
        self,
        kind: str = "model",
        show_prior_factors: bool = False,
        width: float = 14.0,
        style=None,
    ):
        """Layer 4 -- every box of the figure, in inches."""
        from autofit.model_figure.layout import Style

        from .layout import build_ep_layout

        return build_ep_layout(
            self.presentation(kind=kind, show_prior_factors=show_prior_factors),
            style=style or Style(width=width),
        )

    # -- the figure ---------------------------------------------------------

    def figure(
        self,
        path=None,
        filename: Optional[str] = None,
        format: str = "show",
        kind: str = "model",
        show_prior_factors: bool = False,
        width: float = 14.0,
        style=None,
    ):
        """
        Draw the EP figure and show or save it.

        Parameters
        ----------
        path
            The **directory** the file is written to (as everywhere else in
            autofit). ``None`` writes nothing.
        filename
            The file's stem; the extension comes from ``format``. Defaults to
            ``graph_model`` or ``graph_state`` by ``kind``, which are the names
            a fit's own output carries.
        format
            ``show`` | ``png`` | ``svg`` | ``pdf``, or ``None`` to build the
            figure without showing or writing it.
        kind
            ``model`` (the graph) or ``state`` (the graph with the run on it).
        show_prior_factors
            Draw the graph's ``PriorFactor``s as nodes of their own. Off by
            default: there is one per prior, and they treble the node count to
            say what a stub on the variable already says.
        width
            The width budget in inches. Text is never scaled down to fit.

        Returns
        -------
        The ``matplotlib`` ``Figure``.
        """
        from .render import draw, save

        layout = self.layout(
            kind=kind,
            show_prior_factors=show_prior_factors,
            width=width,
            style=style,
        )
        figure = draw(layout, layout.style)
        return save(
            figure,
            path=path,
            filename=filename or FILENAMES[kind],
            format=format,
        )

    # -- internals ----------------------------------------------------------

    def _state_for(self, kind: str, show_prior_factors: bool):
        if kind == "model":
            return None
        if kind != "state":
            raise ValueError(
                f"kind must be 'model' or 'state', not {kind!r}",
            )
        if self.ep_history is None:
            raise ValueError(
                "kind='state' draws what an EP run has done to the graph, so it "
                "needs the optimiser's history: "
                "EPPlotter(optimiser.factor_graph, ep_history=optimiser.ep_history). "
                "Without one, only kind='model' can be drawn."
            )
        return self.state(show_prior_factors=show_prior_factors)
