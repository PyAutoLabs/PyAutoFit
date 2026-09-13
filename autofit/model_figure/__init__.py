"""
The model figure -- a containment picture of a model's structure.

``model.info`` is the legend; this package draws the map.  Three layers, each
importable and testable on its own:

* :mod:`autofit.model_figure.presentation` -- what is drawn and what it says.
* :mod:`autofit.model_figure.layout` -- where it goes, measured in inches.
* :mod:`autofit.model_figure.render` -- matplotlib.

:class:`~autofit.model_figure.plotter.ModelPlotter` is the public entry point
and is re-exported as ``af.ModelPlotter``.  Importing this package does **not**
import matplotlib: every drawing import is inside a function, so
``import autofit`` stays drawing-free.

The :mod:`autofit.model_figure.ep` sub-package draws the other picture of a
graphical fit -- the factor graph an ``EPOptimiser`` sweeps, with the run's
state on it -- through :class:`~autofit.model_figure.ep.plotter.EPPlotter`.
"""

from .plotter import ModelPlotter
from .ep import EPPlotter

__all__ = ["ModelPlotter", "EPPlotter"]
