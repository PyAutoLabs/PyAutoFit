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
    assert loaded.max_log_likelihood_sample.kwargs == {"fwhm": 7.0644601350928475}


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
    assert latent_samples.sample_list[0].kwargs == {"fwhm": 7.0644601350928475}
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
    assert latent_samples.sample_list[0].kwargs == {"fwhm": 7.0644601350928475}


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
