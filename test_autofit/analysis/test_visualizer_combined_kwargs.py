"""
Regression test: ``Visualizer.visualize_combined`` must accept the
``quick_update`` kwarg that ``AnalysisFactor.visualize_combined``
forwards. The original bug was that the base signature did not list
``quick_update``, so any user Analysis without an explicit override
crashed mid-search with::

    TypeError: Visualizer.visualize_combined() got an unexpected
    keyword argument 'quick_update'

This only fired on graphical / joint-fit search paths whose
periodic-update cadence calls ``visualize_combined`` with
``quick_update=True`` partway through a long Dynesty chain.
"""
import inspect

import autofit as af
from autofit.non_linear.analysis.visualize import Visualizer


def test_base_visualize_combined_accepts_quick_update_kwarg():
    sig = inspect.signature(Visualizer.visualize_combined)
    assert "quick_update" in sig.parameters, (
        "Visualizer.visualize_combined must accept ``quick_update`` to match "
        "the call site in AnalysisFactor.visualize_combined."
    )
    # And the kwarg must be optional so existing call sites that don't pass it
    # continue to work.
    assert sig.parameters["quick_update"].default is False


def test_base_visualize_combined_callable_with_quick_update():
    # The default impl is a no-op; the test is just that calling it with
    # ``quick_update`` does not raise. Pass the minimum positional args.
    Visualizer.visualize_combined(
        analyses=[],
        paths=None,
        instance=None,
        during_analysis=False,
        quick_update=True,
    )


def test_analysis_factor_visualize_combined_default_visualizer():
    """
    End-to-end: ``AnalysisFactor.visualize_combined(quick_update=True)``
    on a default analysis (whose ``Visualizer`` is the base class) must
    not raise. Reproduces the original crash path.
    """
    factor = af.AnalysisFactor(
        prior_model=af.Model(af.ex.Gaussian),
        analysis=af.Analysis(),
    )
    # ``analyses`` must be a list of factors so the unwrap step
    # (``getattr(factor, "analysis", factor)``) yields valid Analysis
    # objects when the visualizer is invoked.
    factor.visualize_combined(
        analyses=[factor],
        paths=None,
        instance=None,
        during_analysis=False,
        quick_update=True,
    )
