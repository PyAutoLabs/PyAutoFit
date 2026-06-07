import itertools

import numpy as np
import pytest

import autofit as af
import autofit.graphical as g


@pytest.fixture(autouse=True)
def reset_ids():
    af.Prior._ids = itertools.count()


class CountingAnalysis(af.ex.Analysis):
    """
    An example `Analysis` that counts how many times the (notionally expensive) model
    data computation runs, so a test can prove the `FactorGraphModel` shared-state
    mechanism computes it once per evaluation rather than once per factor.
    """

    def __init__(self, data, noise_map, share_model_data=True):
        super().__init__(
            data=data, noise_map=noise_map, share_model_data=share_model_data
        )
        self.model_data_calls = 0

    def model_data_1d_from(self, instance):
        self.model_data_calls += 1
        return super().model_data_1d_from(instance=instance)


def _shared_gaussian_graph(analyses):
    """
    Build a `FactorGraphModel` whose factors share the *entire* Gaussian model via
    shared prior objects, so the model data is identical for every factor.
    """
    centre = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
    normalization = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
    sigma = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

    factors = []
    for analysis in analyses:
        gaussian = af.Model(af.ex.Gaussian)
        gaussian.centre = centre
        gaussian.normalization = normalization
        gaussian.sigma = sigma
        factors.append(af.AnalysisFactor(gaussian, analysis))

    return g.FactorGraphModel(*factors)


def _datasets(n=3, size=10):
    """`n` distinct 1D datasets sharing a common noise map of ones."""
    return [
        (np.arange(size, dtype=float) + float(i), np.ones(size))
        for i in range(n)
    ]


def _instance(collection):
    prior_count = collection.global_prior_model.prior_count
    return collection.global_prior_model.instance_from_unit_vector(
        [0.5] * prior_count
    )


def _reference_log_likelihood(collection, instance):
    """Sum each factor's likelihood with no sharing (each computes its own model data)."""
    return sum(
        factor.analysis.log_likelihood_function(instance_)
        for factor, instance_ in zip(collection.model_factors, instance)
    )


def test_shared_state_computed_once_per_evaluation():
    analyses = [
        CountingAnalysis(data, noise_map) for data, noise_map in _datasets(n=3)
    ]
    collection = _shared_gaussian_graph(analyses)
    instance = _instance(collection)

    collection.log_likelihood_function(instance)

    total_calls = sum(analysis.model_data_calls for analysis in analyses)
    assert total_calls == 1


def test_shared_likelihood_equals_unshared_sum():
    analyses = [
        CountingAnalysis(data, noise_map) for data, noise_map in _datasets(n=3)
    ]
    collection = _shared_gaussian_graph(analyses)
    instance = _instance(collection)

    shared_log_likelihood = collection.log_likelihood_function(instance)
    reference_log_likelihood = _reference_log_likelihood(collection, instance)

    assert shared_log_likelihood == pytest.approx(reference_log_likelihood)


def test_no_provider_graph_is_unchanged():
    """
    With `share_model_data=False` no factor opts in, so no state is shared: each factor
    computes its own model data (N calls) and the summed likelihood is unchanged.
    """
    analyses = [
        CountingAnalysis(data, noise_map, share_model_data=False)
        for data, noise_map in _datasets(n=3)
    ]
    collection = _shared_gaussian_graph(analyses)
    instance = _instance(collection)

    log_likelihood = collection.log_likelihood_function(instance)
    reference_log_likelihood = _reference_log_likelihood(collection, instance)

    total_calls = sum(analysis.model_data_calls for analysis in analyses)
    # one call per factor from the graph evaluation, plus one per factor from the
    # reference sum — the graph did not share, so it computed all three itself.
    assert total_calls == 2 * len(analyses)
    assert log_likelihood == pytest.approx(reference_log_likelihood)


def test_shared_state_from_default_returns_none():
    analysis = af.ex.Analysis(
        data=np.arange(10, dtype=float), noise_map=np.ones(10)
    )
    model = af.Model(af.ex.Gaussian)
    instance = model.instance_from_unit_vector([0.5] * model.prior_count)

    assert analysis.shared_state_from(instance) is None
