"""
Tests for EPAnalysisFactor cavity-message injection.

EPAnalysisFactor is a thin AnalysisFactor subclass that exposes the EP
cavity distribution to its Analysis on each fit, via the hook in
``autofit.graphical.expectation_propagation.optimiser.factor_step``.
The Analysis can then read per-variable cavity messages inside
``log_likelihood_function`` — the canonical use case is a "global"
Analysis that compares predictions to per-dataset posterior summaries
produced by upstream local fits.
"""
from unittest.mock import MagicMock

import autofit as af
from autofit.graphical.expectation_propagation.optimiser import factor_step
from autofit.graphical.utils import Status, StatusFlag


class _RecordingAnalysis(af.Analysis):
    """Records every log_likelihood_function call's cavity state."""

    def __init__(self):
        super().__init__()
        self.observed_cavities = []

    def log_likelihood_function(self, instance):
        self.observed_cavities.append(getattr(self, "_cavity_mean_field", None))
        return 0.0


def test_set_cavity_dist_attaches_to_analysis():
    """``set_cavity_dist`` should populate ``analysis._cavity_mean_field``."""
    model = af.Model(af.ex.Gaussian)
    analysis = _RecordingAnalysis()
    factor = af.EPAnalysisFactor(prior_model=model, analysis=analysis)

    sentinel = object()
    factor.set_cavity_dist(sentinel)

    assert analysis._cavity_mean_field is sentinel


def test_plain_analysis_factor_has_no_set_cavity_dist():
    """Plain AnalysisFactor must remain untouched by the hook."""
    model = af.Model(af.ex.Gaussian)
    analysis = _RecordingAnalysis()
    factor = af.AnalysisFactor(prior_model=model, analysis=analysis)

    assert not hasattr(factor, "set_cavity_dist")


def test_factor_step_invokes_set_cavity_dist():
    """
    ``factor_step`` should call ``set_cavity_dist`` before optimisation
    so the Analysis sees the cavity during every likelihood evaluation.
    """
    model = af.Model(af.ex.Gaussian)
    analysis = _RecordingAnalysis()
    factor = af.EPAnalysisFactor(prior_model=model, analysis=analysis)

    cavity_sentinel = object()

    factor_approx = MagicMock()
    factor_approx.factor = factor
    factor_approx.cavity_dist = cavity_sentinel
    factor_approx.model_dist = MagicMock()

    optimiser = MagicMock()
    optimiser.optimise.return_value = (
        MagicMock(),
        Status(success=True, messages=(), flag=StatusFlag.SUCCESS),
    )

    factor_step(factor_approx, optimiser)

    assert analysis._cavity_mean_field is cavity_sentinel
    optimiser.optimise.assert_called_once_with(factor_approx)


def test_factor_step_no_op_for_plain_analysis_factor():
    """No exception should be raised for plain ``AnalysisFactor``."""
    model = af.Model(af.ex.Gaussian)
    analysis = _RecordingAnalysis()
    factor = af.AnalysisFactor(prior_model=model, analysis=analysis)

    factor_approx = MagicMock()
    factor_approx.factor = factor
    factor_approx.cavity_dist = object()
    factor_approx.model_dist = MagicMock()

    optimiser = MagicMock()
    optimiser.optimise.return_value = (
        MagicMock(),
        Status(success=True, messages=(), flag=StatusFlag.SUCCESS),
    )

    factor_step(factor_approx, optimiser)

    assert not hasattr(analysis, "_cavity_mean_field")
