import autofit as af
from autofit.database import Object
from autofit.non_linear.samples.efficient import EfficientSamples


import pytest


@pytest.fixture(name="samples")
def make_samples():
    return af.SamplesPDF(
        model=af.Model(af.ex.Gaussian),
        sample_list=[
            af.Sample(
                log_likelihood=1.0,
                log_prior=2.0,
                weight=4.0,
                kwargs={"centre": 1.0, "normalization": 2.0, "sigma": 3.0},
            ),
            af.Sample(
                log_likelihood=1.0,
                log_prior=2.0,
                weight=4.0,
                kwargs={"centre": 4.0, "normalization": 5.0, "sigma": 6.0},
            ),
        ],
    )


@pytest.fixture(name="efficient")
def make_efficient(samples):
    return EfficientSamples(samples=samples)


def test_conversion(efficient):
    samples = efficient.samples

    sample = samples.sample_list[0]
    assert sample.log_likelihood == 1.0
    assert sample.log_prior == 2.0
    assert sample.weight == 4.0


def test_values(efficient):
    samples = efficient.samples

    sample_1, sample_2 = samples.sample_list

    assert sample_1.kwargs[("centre",)] == 1.0
    assert sample_1.kwargs[("normalization",)] == 2.0
    assert sample_1.kwargs[("sigma",)] == 3.0

    assert sample_2.kwargs[("centre",)] == 4.0
    assert sample_2.kwargs[("normalization",)] == 5.0
    assert sample_2.kwargs[("sigma",)] == 6.0


def test_database(efficient, session):
    recovered = Object.from_object(efficient)()
    samples = recovered.samples

    assert samples.model.cls is af.ex.Gaussian
