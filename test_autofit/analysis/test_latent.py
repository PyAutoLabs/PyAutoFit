"""
Tests for the first-class ``Latent`` extension class and the
``latent_samples_from`` engine (``autofit/non_linear/analysis/latent.py``).

These cover the NEW class-based path (subclass ``af.Latent``, declare
``Analysis.Latent = ...``). The legacy method-override path
(``compute_latent_variables`` / ``LATENT_KEYS``) is covered by
``test_latent_variables.py``, which also exercises the back-compat shim.
"""
import numpy as np
import pytest

import autofit as af
from autofit import SamplesPDF
from autofit.non_linear.analysis.latent import latent_samples_from

# fwhm for a Gaussian with sigma=3.0 (2 * sqrt(2 ln2) * 3).
FWHM_SIGMA_3 = 7.0644601350928475


def _samples(sample_list):
    return SamplesPDF(model=af.Model(af.ex.Gaussian), sample_list=sample_list)


def _sample(centre=1.0, sigma=3.0, weight=1.0, log_likelihood=1.0):
    return af.Sample(
        log_likelihood=log_likelihood,
        log_prior=0.0,
        weight=weight,
        kwargs={"centre": centre, "normalization": 2.0, "sigma": sigma},
    )


class FwhmLatent(af.Latent):
    @staticmethod
    def keys(analysis):
        return ["fwhm"]

    @staticmethod
    def variables(analysis, parameters, model):
        instance = model.instance_from_vector(vector=parameters)
        return (instance.fwhm,)


class FwhmAnalysis(af.Analysis):
    Latent = FwhmLatent

    def log_likelihood_function(self, instance):
        return 1.0


def test_latent_class_path_matches_method_override():
    """Declaring `Latent = FwhmLatent` produces the same latent samples the
    legacy `compute_latent_variables` override would (see test_latent_variables)."""
    latent = FwhmAnalysis().compute_latent_samples(_samples([_sample()]))
    assert latent.sample_list[0].kwargs == {("fwhm",): FWHM_SIGMA_3}
    assert latent.model.instance_from_vector([1.0]).fwhm == 1.0


def test_latent_samples_from_callable_directly():
    """The engine is usable as a free function, decoupled from the Analysis."""
    latent = latent_samples_from(FwhmAnalysis(), _samples([_sample(), _sample(centre=2.0)]))
    assert latent is not None
    assert len(latent.sample_list) == 2


class LegacyAndLatentAnalysis(af.Analysis):
    """A `Latent` subclass must win over the legacy attrs when both are present."""

    Latent = FwhmLatent
    LATENT_KEYS = ["should_not_be_used"]

    def compute_latent_variables(self, parameters, model):
        raise AssertionError(
            "legacy compute_latent_variables must not be called when Latent overrides variables()"
        )

    def log_likelihood_function(self, instance):
        return 1.0


def test_latent_class_overrides_legacy_attrs():
    latent = LegacyAndLatentAnalysis().compute_latent_samples(_samples([_sample()]))
    assert set(latent.sample_list[0].kwargs) == {("fwhm",)}


def test_default_latent_shim_reads_legacy_attrs():
    """The default base `Latent` delegates to the legacy attrs (back-compat)."""

    class LegacyAnalysis(af.Analysis):
        LATENT_KEYS = ["fwhm"]

        def compute_latent_variables(self, parameters, model):
            instance = model.instance_from_vector(vector=parameters)
            return (instance.fwhm,)

        def log_likelihood_function(self, instance):
            return 1.0

    analysis = LegacyAnalysis()
    assert af.Analysis.Latent.keys(analysis) == ["fwhm"]
    latent = analysis.compute_latent_samples(_samples([_sample()]))
    assert latent.sample_list[0].kwargs == {("fwhm",): FWHM_SIGMA_3}


def test_default_latent_batch_mode_is_none_fallback():
    """`BATCH_MODE = None` is the sentinel meaning 'defer to analysis.LATENT_BATCH_MODE'."""
    assert af.Latent.BATCH_MODE is None
    assert af.Analysis.LATENT_BATCH_MODE == "vmap"


class RaisingLatent(af.Latent):
    @staticmethod
    def keys(analysis):
        return ["fwhm"]

    @staticmethod
    def variables(analysis, parameters, model):
        if parameters[0] < 0:
            raise ValueError("boom from latent function")
        instance = model.instance_from_vector(vector=parameters)
        return (instance.fwhm,)


class RaisingAnalysis(af.Analysis):
    Latent = RaisingLatent

    def log_likelihood_function(self, instance):
        return 1.0


def test_new_path_skips_arbitrary_exception_samples():
    latent = RaisingAnalysis().compute_latent_samples(
        _samples([_sample(centre=1.0), _sample(centre=-1.0, weight=0.0)])
    )
    assert len(latent.sample_list) == 1
    assert latent.sample_list[0].kwargs == {("fwhm",): FWHM_SIGMA_3}


class AntiCorrelatedNaNLatent(af.Latent):
    @staticmethod
    def keys(analysis):
        return ["a", "b"]

    @staticmethod
    def variables(analysis, parameters, model):
        c = parameters[0]
        return (np.nan if c < 0 else 1.0, np.nan if c >= 0 else 2.0)


class AntiCorrelatedNaNAnalysis(af.Analysis):
    Latent = AntiCorrelatedNaNLatent

    def log_likelihood_function(self, instance):
        return 1.0


def test_new_path_salvages_anti_correlated_nans():
    latent = AntiCorrelatedNaNAnalysis().compute_latent_samples(
        _samples([_sample(centre=cv) for cv in (1.0, -1.0, 2.0, -2.0)])
    )
    assert latent is not None
    surviving = set(latent.sample_list[0].kwargs)
    assert surviving == {("b",)}
    assert all(set(s.kwargs) == surviving for s in latent.sample_list)


class OneSurvivorLatent(af.Latent):
    @staticmethod
    def keys(analysis):
        return ["fwhm"]

    @staticmethod
    def variables(analysis, parameters, model):
        if parameters[0] != 1.0:
            return (np.nan,)
        instance = model.instance_from_vector(vector=parameters)
        return (instance.fwhm,)


class OneSurvivorAnalysis(af.Analysis):
    Latent = OneSurvivorLatent

    def log_likelihood_function(self, instance):
        return 1.0


def test_new_path_single_survivor_summary_does_not_crash():
    latent = OneSurvivorAnalysis().compute_latent_samples(
        _samples([
            _sample(centre=1.0, weight=0.5),
            _sample(centre=2.0, weight=0.3),
            _sample(centre=3.0, weight=0.2),
        ])
    )
    assert latent is not None
    assert len(latent.sample_list) == 1
    assert latent.pdf_converged  # weight 0.5 <= 0.99 => median_pdf routes to quantile (n=1)
    _ = latent.summary()
    instance = latent.median_pdf()
    assert instance.fwhm == pytest.approx(FWHM_SIGMA_3)
