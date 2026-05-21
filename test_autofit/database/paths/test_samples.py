import pytest

import autofit as af
from autofit import database as m


@pytest.fixture(
    name="save_samples",
    autouse=True
)
def save_samples(
        paths,
        sample
):
    samples =af.m.MockSamples(
        model=af.Model(
            af.ex.Gaussian
        ),

    )
    samples.sample_list = [sample]
    paths.save_samples(
        samples
    )


@pytest.fixture(
    name="sample"
)
def make_sample():
    return af.Sample(
        log_likelihood=1.0,
        log_prior=1.0,
        weight=0.5,
        kwargs=dict(
            centre=1.0,
            normalization=2.0,
            sigma=3.0
        )
    )


def test_serialise_sample(sample):
    assert isinstance(
        sample.kwargs,
        dict
    )

    sample = m.Object.from_object(
        sample
    )()
    assert ("centre",) in sample.kwargs


def test_load_samples(
        paths
):
    samples = paths._load_samples()

    assert samples.model.cls is af.ex.Gaussian

    sample, = samples.sample_list
    assert sample.weight == 0.5
