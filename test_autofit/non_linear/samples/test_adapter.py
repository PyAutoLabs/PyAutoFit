import numpy as np
import pytest

import autofit as af
from autofit import exc
from autofit.non_linear.samples.adapter import (
    ChainPosterior,
    RawSamples,
    invalid_count,
    is_invalid,
    log_prior_list_from,
    samples_from_raw,
)
from autofit.non_linear.search.mcmc.auto_correlations import AutoCorrelations


@pytest.fixture(name="model")
def make_model():
    model = af.Model(af.ex.Gaussian)
    model.centre = af.GaussianPrior(mean=1.0, sigma=2.0)
    model.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
    model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
    return model


PARAMETERS = [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]


def test_log_likelihood_from_log_posterior(model):
    log_prior = log_prior_list_from(model, PARAMETERS)

    samples = samples_from_raw(
        model,
        RawSamples(parameters=PARAMETERS, log_posterior=[-1.0, -2.0]),
        af.SamplesMCMC,
    )

    assert samples.log_prior_list == log_prior
    assert samples.log_likelihood_list == [
        -1.0 - log_prior[0],
        -2.0 - log_prior[1],
    ]
    assert samples.weight_list == [1.0, 1.0]


def test_weights_are_never_renormalised(model):
    samples = samples_from_raw(
        model,
        RawSamples(
            parameters=PARAMETERS, log_likelihood=[-1.0, -2.0], weights=[3.0, 5.0]
        ),
        af.SamplesNest,
    )

    assert samples.weight_list == [3.0, 5.0]


def test_time_and_class_path_are_injected(model):
    samples = samples_from_raw(
        model,
        RawSamples(parameters=PARAMETERS, log_likelihood=[-1.0, -2.0], info={"a": 1}),
        af.Samples,
        time=12.5,
    )

    assert samples.samples_info["a"] == 1
    assert samples.samples_info["time"] == 12.5
    assert samples.samples_info["class_path"].endswith(".Samples")


def test_a_search_time_wins_over_the_injected_one(model):
    samples = samples_from_raw(
        model,
        RawSamples(
            parameters=PARAMETERS, log_likelihood=[-1.0, -2.0], info={"time": 3.0}
        ),
        af.Samples,
        time=12.5,
    )

    assert samples.samples_info["time"] == 3.0


@pytest.mark.parametrize(
    "field, name",
    [
        ("log_posterior", "log posterior"),
        ("log_likelihood", "log likelihood"),
    ],
)
def test_length_mismatch_raises(model, field, name):
    with pytest.raises(exc.SamplesException, match=f"number of {name} values"):
        samples_from_raw(
            model,
            RawSamples(parameters=PARAMETERS, label="Test", **{field: [-1.0]}),
            af.Samples,
        )


def test_weight_length_mismatch_raises(model):
    with pytest.raises(exc.SamplesException, match="weight"):
        samples_from_raw(
            model,
            RawSamples(
                parameters=PARAMETERS, log_likelihood=[-1.0, -2.0], weights=[1.0]
            ),
            af.Samples,
        )


def test_exactly_one_of_log_likelihood_and_log_posterior():
    with pytest.raises(exc.SamplesException):
        RawSamples(parameters=PARAMETERS)
    with pytest.raises(exc.SamplesException):
        RawSamples(parameters=PARAMETERS, log_likelihood=[1.0], log_posterior=[1.0])


@pytest.mark.parametrize("value", [-1.0e99, -1.0e30, -np.inf, -1.0e99 - 3.0])
def test_sentinels_are_recognised(value):
    assert is_invalid(value)


@pytest.mark.parametrize("value", [0.0, -1.0e10, np.nan, "x"])
def test_ordinary_values_are_not_sentinels(value):
    assert not is_invalid(value)


def test_sentinels_are_kept_as_stored(model):
    samples = samples_from_raw(
        model,
        RawSamples(parameters=PARAMETERS, log_likelihood=[-1.0e99, -np.inf]),
        af.Samples,
    )

    assert invalid_count(samples.log_likelihood_list) == 2
    assert samples.log_likelihood_list[0] == -1.0e99
    assert samples.log_likelihood_list[1] == -np.inf


def _chain(n_steps=6, n_chains=2, n_dim=3):
    chain = np.arange(n_steps * n_chains * n_dim, dtype=float).reshape(
        n_steps, n_chains, n_dim
    )
    log_prob = chain[:, :, 0] * 10.0
    return chain, log_prob


def _emcee_backend(chain, log_prob):
    emcee = pytest.importorskip("emcee")

    backend = emcee.backends.Backend()
    backend.reset(chain.shape[1], chain.shape[2])
    backend.chain = chain
    backend.log_prob = log_prob
    backend.iteration = chain.shape[0]
    backend.accepted = np.ones(chain.shape[1])
    return backend


def test_from_arrays_thins_like_emcee():
    chain, log_prob = _chain()
    backend = _emcee_backend(chain, log_prob)

    for discard, thin in [(0, 1), (1, 2), (2, 3)]:
        expected = ChainPosterior.from_sampler(backend).thin(discard, thin)
        result = ChainPosterior.from_arrays(chain, log_prob).thin(discard, thin)

        assert np.array_equal(result[0], expected[0])
        assert np.array_equal(result[1], expected[1])


def test_from_arrays_chain_major():
    chain, log_prob = _chain()

    parameters, log_values = ChainPosterior.from_arrays(
        chain, log_prob, chain_major=True
    ).thin(0, 1)

    assert np.array_equal(parameters, np.moveaxis(chain, 0, 1).reshape(-1, 3))
    assert np.array_equal(log_values, np.moveaxis(log_prob, 0, 1).reshape(-1))


def test_empty_chain_falls_back_to_the_whole_chain():
    chain, log_prob = _chain()

    parameters, log_values = ChainPosterior.from_arrays(chain, log_prob).thin(
        discard=100, thin=5
    )

    assert len(parameters) == chain.shape[0] * chain.shape[1]
    assert len(log_values) == len(parameters)


def test_emcee_gets_the_empty_chain_fallback(model, monkeypatch):
    """
    Emcee shares Zeus's fallback: burn-in removal that leaves nothing builds the
    samples from the whole chain instead of failing.
    """
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)

    rng = np.random.default_rng(1)
    chain = rng.uniform(1.0, 3.0, size=(4, 6, 3))
    backend = _emcee_backend(chain, -np.sum(chain**2, axis=2))

    search = af.Emcee()

    times = np.array([10.0, 10.0, 10.0])
    monkeypatch.setattr(
        type(search),
        "auto_correlations_from",
        lambda self, search_internal=None: AutoCorrelations(
            check_size=2,
            required_length=1,
            change_threshold=0.01,
            times=times,
            previous_times=times,
        ),
    )

    samples = search.samples_via_internal_from(model=model, search_internal=backend)

    assert len(samples.sample_list) == 24
