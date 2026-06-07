import numpy as np
import pytest

import autofit as af
from autoconf.conf import with_config
from autofit import DirectoryPaths, SamplesPDF
from autofit.text.text_util import result_info_from


class Analysis(af.Analysis):

    LATENT_KEYS = ["fwhm"]

    def log_likelihood_function(self, instance):
        return 1.0

    def compute_latent_variables(self, parameters, model):

        instance = model.instance_from_vector(vector=parameters)

        return (instance.fwhm,)


@with_config(
    "general",
    "output",
    "samples_to_csv",
    value=True,
)
def test_set_directory_paths(output_directory, latent_samples):
    directory_paths = DirectoryPaths()
    directory_paths.save_latent_samples(
        latent_samples=latent_samples,
    )
    loaded = directory_paths.load_latent_samples()
    assert len(loaded) == 1


class MockSamples:
    @property
    def max_log_likelihood_index(self):
        return 0


def test_set_database_paths(session, latent_samples):
    database_paths = af.DatabasePaths(session)
    database_paths.save_latent_samples(
        latent_samples=latent_samples,
    )
    loaded = database_paths.load_latent_samples()
    assert loaded.max_log_likelihood_sample.kwargs == {("fwhm",): 7.0644601350928475}


@pytest.fixture(name="latent_samples")
def make_latent_samples():
    analysis = Analysis()
    return analysis.compute_latent_samples(
        SamplesPDF(
            model=af.Model(af.ex.Gaussian),
            sample_list=[
                af.Sample(
                    log_likelihood=1.0,
                    log_prior=0.0,
                    weight=1.0,
                    kwargs={
                        "centre": 1.0,
                        "normalization": 2.0,
                        "sigma": 3.0,
                    },
                )
            ],
        ),
    )


def test_compute_latent_samples(latent_samples):
    assert latent_samples.sample_list[0].kwargs == {("fwhm",): 7.0644601350928475}
    assert latent_samples.model.instance_from_vector([1.0]).fwhm == 1.0


class AssertionAnalysis(af.Analysis):

    LATENT_KEYS = ["fwhm"]

    def log_likelihood_function(self, instance):
        return 1.0

    def compute_latent_variables(self, parameters, model):
        if parameters[0] < 0:
            raise af.exc.FitException("assertion violated")
        instance = model.instance_from_vector(vector=parameters)
        return (instance.fwhm,)


def test_latent_batch_mode_default_is_vmap():
    """
    Subclasses that don't override `LATENT_BATCH_MODE` must inherit "vmap" so
    `compute_latent_samples` keeps its existing parallel-batched behaviour
    on the JAX path. PyAutoGalaxy's `AnalysisDataset` overrides this to "jit"
    for vmap-incompatible inner calls — that override is what unlocks
    lensing latents, but it must stay opt-in here.
    """
    assert af.Analysis.LATENT_BATCH_MODE == "vmap"
    assert Analysis.LATENT_BATCH_MODE == "vmap"


def test_latent_batch_mode_invalid_value_raises_clear_error():
    """
    Misspelt `LATENT_BATCH_MODE` values should fail with a clear ValueError
    rather than silently falling through. This guards against subclasses
    setting e.g. `LATENT_BATCH_MODE = "JIT"` (case sensitivity) and getting
    surprising behaviour.
    """
    class BadAnalysis(Analysis):
        LATENT_BATCH_MODE = "JIT"  # wrong case — should be lowercase

    samples = SamplesPDF(
        model=af.Model(af.ex.Gaussian),
        sample_list=[
            af.Sample(
                log_likelihood=1.0,
                log_prior=0.0,
                weight=1.0,
                kwargs={"centre": 1.0, "normalization": 2.0, "sigma": 3.0},
            )
        ],
    )
    bad = BadAnalysis(use_jax=True)
    with pytest.raises(ValueError, match="LATENT_BATCH_MODE"):
        bad.compute_latent_samples(samples)


def test_compute_latent_samples_skips_fit_exception_samples():
    analysis = AssertionAnalysis()
    latent_samples = analysis.compute_latent_samples(
        SamplesPDF(
            model=af.Model(af.ex.Gaussian),
            sample_list=[
                af.Sample(
                    log_likelihood=1.0,
                    log_prior=0.0,
                    weight=1.0,
                    kwargs={"centre": 1.0, "normalization": 2.0, "sigma": 3.0},
                ),
                af.Sample(
                    log_likelihood=-1.0,
                    log_prior=0.0,
                    weight=0.0,
                    kwargs={"centre": -1.0, "normalization": 2.0, "sigma": 3.0},
                ),
            ],
        ),
    )
    assert len(latent_samples.sample_list) == 1
    assert latent_samples.sample_list[0].kwargs == {("fwhm",): 7.0644601350928475}


def test_info(latent_samples):
    info = result_info_from(latent_samples)
    assert (
        info
        == """Maximum Log Likelihood                                                          1.00000000

model                                                                           Collection (N=1)

Maximum Log Likelihood Model:

fwhm                                                                            7.064

 WARNING: The samples have not converged enough to compute a PDF and model errors. 
 The model below over estimates errors. 



Summary (1.0 sigma limits):

fwhm                                                                            7.0645 (7.0645, 7.0645)

instances

"""
    )


class ComplexAnalysis(af.Analysis):

    LATENT_KEYS = ["lens.mass", "lens.brightness", "source.brightness"]

    def log_likelihood_function(self, instance):
        return 1.0

    def compute_latent_variables(self, parameters, model):

        instance = model.instance_from_vector(vector=parameters)

        return (1.0, 2.0, 3.0)


def test_complex_model():
    analysis = ComplexAnalysis()
    latent_samples = analysis.compute_latent_samples(
        SamplesPDF(
            model=af.Model(af.ex.Gaussian),
            sample_list=[
                af.Sample(
                    log_likelihood=1.0,
                    log_prior=0.0,
                    weight=1.0,
                    kwargs={
                        "centre": 1.0,
                        "normalization": 2.0,
                        "sigma": 3.0,
                    },
                )
            ],
        ),
    )

    instance = latent_samples.model.instance_from_prior_medians()

    lens = instance.lens

    assert lens.mass == 1.0
    assert lens.brightness == 2.0

    assert instance.source.brightness == 3.0


class RaisingAnalysis(af.Analysis):
    """Latent that raises a non-FitException (e.g. an unexpected solver error)
    for some samples."""

    LATENT_KEYS = ["fwhm"]

    def log_likelihood_function(self, instance):
        return 1.0

    def compute_latent_variables(self, parameters, model):
        if parameters[0] < 0:
            raise ValueError("boom from latent function")
        instance = model.instance_from_vector(vector=parameters)
        return (instance.fwhm,)


def test_compute_latent_samples_skips_arbitrary_exception_samples():
    """
    A latent function that raises ANY exception (not just FitException) must
    drop that sample rather than crashing the whole latent pass.
    """
    analysis = RaisingAnalysis()
    latent_samples = analysis.compute_latent_samples(
        SamplesPDF(
            model=af.Model(af.ex.Gaussian),
            sample_list=[
                af.Sample(log_likelihood=1.0, log_prior=0.0, weight=1.0,
                          kwargs={"centre": 1.0, "normalization": 2.0, "sigma": 3.0}),
                af.Sample(log_likelihood=-1.0, log_prior=0.0, weight=0.0,
                          kwargs={"centre": -1.0, "normalization": 2.0, "sigma": 3.0}),
            ],
        ),
    )
    assert len(latent_samples.sample_list) == 1
    assert latent_samples.sample_list[0].kwargs == {("fwhm",): 7.0644601350928475}


class AntiCorrelatedNaNAnalysis(af.Analysis):
    """Two latents whose NaNs are anti-correlated across samples: latent ``a`` is
    NaN where ``centre < 0`` and latent ``b`` is NaN where ``centre >= 0``. No
    single sample is finite in both."""

    LATENT_KEYS = ["a", "b"]

    def log_likelihood_function(self, instance):
        return 1.0

    def compute_latent_variables(self, parameters, model):
        c = parameters[0]
        a = np.nan if c < 0 else 1.0
        b = np.nan if c >= 0 else 2.0
        return (a, b)


def test_compute_latent_samples_salvages_anti_correlated_nans():
    """
    When NaNs are anti-correlated so the rectangular (samples x latents) block is
    empty, the masking must NOT discard all latent output. It sacrifices the
    worst-coverage latent and retains the maximal-coverage one with its finite
    samples — rather than returning None.
    """
    analysis = AntiCorrelatedNaNAnalysis()
    latent_samples = analysis.compute_latent_samples(
        SamplesPDF(
            model=af.Model(af.ex.Gaussian),
            sample_list=[
                af.Sample(log_likelihood=1.0, log_prior=0.0, weight=1.0,
                          kwargs={"centre": cv, "normalization": 2.0, "sigma": 3.0})
                for cv in (1.0, -1.0, 2.0, -2.0)
            ],
        ),
    )
    assert latent_samples is not None
    # The two latents tie on NaN count; the first (a) is sacrificed, b is kept.
    surviving_keys = set(latent_samples.sample_list[0].kwargs)
    assert surviving_keys == {("b",)}
    # All retained samples share the same single-key set (no KeyError downstream).
    assert all(set(s.kwargs) == surviving_keys for s in latent_samples.sample_list)


class OneSurvivorAnalysis(af.Analysis):
    """Only the ``centre == 1.0`` sample yields a finite latent; all others NaN."""

    LATENT_KEYS = ["fwhm"]

    def log_likelihood_function(self, instance):
        return 1.0

    def compute_latent_variables(self, parameters, model):
        if parameters[0] != 1.0:
            return (np.nan,)
        instance = model.instance_from_vector(vector=parameters)
        return (instance.fwhm,)


def test_compute_latent_samples_single_survivor_summary_does_not_crash():
    """
    When masking leaves exactly one finite latent sample whose weight is < 0.99
    (so `pdf_converged` is True and `median_pdf` uses `quantile`), `summary()` and
    `median_pdf()` must succeed. Regression for the `quantile` n=1 IndexError.
    """
    analysis = OneSurvivorAnalysis()
    latent_samples = analysis.compute_latent_samples(
        SamplesPDF(
            model=af.Model(af.ex.Gaussian),
            sample_list=[
                af.Sample(log_likelihood=3.0, log_prior=0.0, weight=0.5,
                          kwargs={"centre": 1.0, "normalization": 2.0, "sigma": 3.0}),
                af.Sample(log_likelihood=2.0, log_prior=0.0, weight=0.3,
                          kwargs={"centre": 2.0, "normalization": 2.0, "sigma": 3.0}),
                af.Sample(log_likelihood=1.0, log_prior=0.0, weight=0.2,
                          kwargs={"centre": 3.0, "normalization": 2.0, "sigma": 3.0}),
            ],
        ),
    )
    assert latent_samples is not None
    assert len(latent_samples.sample_list) == 1
    # weight 0.5 <= 0.99 => pdf_converged True => median_pdf routes to quantile (n=1).
    assert latent_samples.pdf_converged
    # The real regression: these must not raise.
    _ = latent_samples.summary()
    instance = latent_samples.median_pdf()
    assert instance.fwhm == pytest.approx(7.0644601350928475)
