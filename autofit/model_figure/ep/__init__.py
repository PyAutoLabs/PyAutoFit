"""
The EP factor-graph figure -- the *diagnostic* view of a graphical fit.

``autofit.model_figure`` draws a model's containment structure; this package
draws the **factor graph** an :class:`~autofit.graphical.EPOptimiser` actually
sweeps, with the run's state painted onto it.  It follows the same three-layer
split, each layer importable and testable on its own:

* :mod:`autofit.model_figure.ep.spec` -- structure: which factor nodes, which
  variable nodes, which edges, which plates.
* :mod:`autofit.model_figure.ep.state` -- overlay: what the EP run did to each
  factor (updates, sweeps, age, reverted variables).
* :mod:`autofit.model_figure.ep.presentation` -- what is drawn and what it
  says.
* :mod:`autofit.model_figure.ep.layout` -- where it goes, measured in inches.
* :mod:`autofit.model_figure.ep.render` -- matplotlib.

:class:`~autofit.model_figure.ep.plotter.EPPlotter` is the public entry point
and is re-exported as ``af.EPPlotter``.  Nothing here imports a drawing library
at module level, so ``import autofit`` stays drawing-free.
"""

from .plotter import EPPlotter

__all__ = ["EPPlotter"]
