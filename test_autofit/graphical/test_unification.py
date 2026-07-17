from random import random

import dill
import numpy as np
import pytest

import autofit as af
from autofit import graphical as g
from autofit.messages.composed_transform import TransformedMessage
from autofit.messages.normal import UniformNormalMessage, NormalMessage
from autofit.messages.transform import log_10_transform, LinearShiftTransform


@pytest.fixture(name="prior")
def make_prior():
    return af.GaussianPrior(
        mean=1,
        sigma=2,
    ).message


def test():
    mean_field = g.MeanField({})
    mean_field.instance_for_arguments({})


def test_retain_id(prior):
    new_message = prior * prior
    assert new_message.id == prior.id


@pytest.fixture(name="x")
def make_x():
    return np.linspace(0, 1, 100)


def test_projected_model():
    model = af.Model(af.ex.Gaussian, centre=af.UniformPrior())
    samples = af.Samples(
        model,
        [
            af.Sample(
                -1.0,
                -1.0,
                weight=random(),
                kwargs={
                    ("centre",): random(),
                    ("normalization",): random(),
                    ("sigma",): random(),
                },
            )
            for _ in range(100)
        ],
    )
    result = af.mock.MockResult(samples=samples)
    projected_model = result.projected_model

    assert projected_model.prior_count == 3
    assert projected_model.centre is not model.centre
    assert projected_model.centre.id == model.centre.id
    assert isinstance(projected_model.centre, af.UniformPrior)


def test_projected_model_moments():
    """
    Regression test for PyAutoFit#1382: `projected_model` must convert the
    samples' linear importance weights to log weights before the message
    projection's weighted moment match.

    The sample set mimics a nested-sampling run: a broad prior-exploration
    phase with negligible weights plus a concentrated posterior with almost
    all the weight. The projected prior's moments must match the posterior,
    not the prior-phase spread — with linear weights fed in as log weights,
    every sample counts almost equally and the projection lands near a
    prior bound.
    """
    rng = np.random.default_rng(1)

    posterior_mean = 2.05
    posterior_sigma = 0.03

    prior_phase = rng.uniform(1.5, 3.0, size=2000)
    posterior_phase = rng.normal(posterior_mean, posterior_sigma, size=500)
    values = np.concatenate([prior_phase, posterior_phase])

    weights = np.concatenate(
        [
            np.full(prior_phase.size, 1.0e-9),
            np.full(posterior_phase.size, (1.0 - prior_phase.size * 1.0e-9) / 500),
        ]
    )

    model = af.Model(
        af.ex.Gaussian,
        centre=af.UniformPrior(lower_limit=1.5, upper_limit=3.0),
    )
    # normalization and sigma have UniformPrior(0, 1) under the test config, so
    # they get their own in-support draws; the moment assertions are on centre.
    unit_values = rng.uniform(0.05, 0.95, size=values.size)
    samples = af.Samples(
        model,
        [
            af.Sample(
                -1.0,
                -1.0,
                weight=weight,
                kwargs={
                    ("centre",): value,
                    ("normalization",): unit_value,
                    ("sigma",): unit_value,
                },
            )
            for value, unit_value, weight in zip(values, unit_values, weights)
        ],
    )
    result = af.mock.MockResult(samples=samples)

    projected_centre = result.projected_model.centre

    assert projected_centre.mean == pytest.approx(posterior_mean, abs=0.02)
    assert float(np.sqrt(projected_centre.variance)) == pytest.approx(
        posterior_sigma, rel=0.5
    )


def test_uniform_normal(x):
    message = TransformedMessage(
        UniformNormalMessage,
        LinearShiftTransform(shift=1, scale=2.1),
    )

    assert message.pdf(0.9) == pytest.approx(0)
    assert message.pdf(3.2) == pytest.approx(0)
    assert message.pdf(1.5) > 0


@pytest.mark.parametrize(
    "lower_limit, upper_limit, unit_value, physical_value",
    [
        (0.0, 1.0, 0.5, 0.5),
        (0.0, 1.0, 1.0, 1.0),
        (0.0, 1.0, 0.0, 0.0),
        (1.0, 2.0, 0.5, 1.5),
        (1.0, 2.0, 1.0, 2.0),
        (1.0, 2.0, 0.0, 1.0),
        (0.0, 2.0, 0.5, 1.0),
        (0.0, 2.0, 1.0, 2.0),
        (0.0, 2.0, 0.0, 0.0),
    ],
)
def test_uniform_prior(lower_limit, upper_limit, unit_value, physical_value):
    assert af.UniformPrior(
        lower_limit=lower_limit,
        upper_limit=upper_limit,
    ).value_for(
        unit_value
    ) == pytest.approx(physical_value)


def test_uniform_odd_result():
    prior = af.UniformPrior(90.0, 100.0)
    assert prior.value_for(0.0) == pytest.approx(90.0)


@pytest.mark.parametrize("lower_limit", [1, 90])
@pytest.mark.parametrize("upper_limit", [110, 200])
@pytest.mark.parametrize("unit", [0.00001, 0.5, 0.9])
def test_log10(lower_limit, upper_limit, unit):
    prior = af.LogUniformPrior(lower_limit=lower_limit, upper_limit=upper_limit)

    assert 10.0 ** (
        np.log10(lower_limit) + unit * (np.log10(upper_limit) - np.log10(lower_limit))
    ) == pytest.approx(prior.value_for(unit), abs=0.001)


@pytest.fixture(name="uniform_prior")
def make_uniform_prior():
    return af.UniformPrior(
        lower_limit=10,
        upper_limit=20,
        id_=1,
    ).message


def test_prior_arithmetic(uniform_prior):
    multiplied = uniform_prior * uniform_prior
    divided = multiplied / uniform_prior

    multiplied_value = multiplied.value_for(0.3)
    divided_value = divided.value_for(0.3)
    uniform_prior_value = uniform_prior.value_for(0.3)

    assert multiplied_value != uniform_prior_value
    assert multiplied_value != divided_value
    assert divided_value == uniform_prior_value


def test_pickle_uniform_prior(uniform_prior):
    pickled_prior = dill.loads(dill.dumps(uniform_prior))
    assert pickled_prior == uniform_prior
    assert pickled_prior.id == uniform_prior.id


def test_pickle_log_uniform_prior():
    log_uniform_prior = af.LogUniformPrior()
    pickled_prior = dill.loads(dill.dumps(log_uniform_prior))
    assert pickled_prior == log_uniform_prior


@pytest.fixture(name="LogMessage")
def make_log_message():
    return TransformedMessage(
        UniformNormalMessage,
        log_10_transform,
        LinearShiftTransform(
            shift=1,
            scale=2,
        ),
    )


def test_double_transform():
    transformed = TransformedMessage(NormalMessage(0, 1), log_10_transform)
    assert isinstance(transformed.base_message, NormalMessage)

    transformed = TransformedMessage(transformed, log_10_transform)
    assert isinstance(transformed.base_message, NormalMessage)


def test_pickle_transformed(LogMessage):
    dill.loads(dill.dumps(LogMessage))


def test_pickle_transformed_instantiated(LogMessage):
    instance = LogMessage(mean=1, sigma=2)
    dill.loads(dill.dumps(instance))
