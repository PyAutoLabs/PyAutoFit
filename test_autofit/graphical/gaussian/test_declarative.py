import numpy as np
import pytest

import autofit as af
import autofit.graphical as ep
from test_autofit.graphical.gaussian.model import Gaussian, make_data, Analysis


@pytest.fixture(name="make_model_factor")
def make_make_model_factor(normalization, normalization_prior, x):
    def make_factor_model(
        centre: float, sigma: float, optimiser=None
    ) -> ep.AnalysisFactor:
        """
        We'll make a LikelihoodModel for each Gaussian we're fitting.

        First we'll make the actual data to be fit.

        Note that the normalization value is shared.
        """
        y = make_data(
            Gaussian(centre=centre, normalization=normalization, sigma=sigma), x
        )

        """
        Next we need a prior model.
    
        Note that the normalization prior is shared.
        """
        prior_model = af.Model(
            Gaussian,
            centre=af.GaussianPrior(mean=50, sigma=20),
            normalization=normalization_prior,
            sigma=af.GaussianPrior(mean=10, sigma=10),
        )

        """
        Finally we combine the likelihood function with the prior model to produce a likelihood
        factor - this will be converted into a ModelFactor which is like any other factor in the
        factor graph.
        
        We can also pass a custom optimiser in here that will be used to fit the factor instead
        of the default optimiser.
        """
        return ep.AnalysisFactor(
            prior_model, analysis=Analysis(x=x, y=y), optimiser=optimiser
        )

    return make_factor_model


@pytest.fixture(name="normalization")
def make_normalization():
    return 25.0


@pytest.fixture(name="normalization_prior")
def make_normalization_prior():
    return af.GaussianPrior(mean=25, sigma=10)


@pytest.fixture(name="factor_model")
def make_factor_model_collection(make_model_factor):
    """
    Here's a good example in which we have two Gaussians fit with a shared variable

    We have a shared normalization value and a shared normalization prior

    Multiplying together multiple LikelihoodModels gives us a factor model.

    The factor model can compute all the variables and messages required as well as construct
    a factor graph representing a fit on the ensemble.
    """
    return ep.FactorGraphModel(
        make_model_factor(centre=40, sigma=10), make_model_factor(centre=60, sigma=15)
    )


def test_custom_optimiser(make_model_factor):
    other_optimiser = ep.LaplaceOptimiser()

    factor_1 = make_model_factor(centre=40, sigma=10, optimiser=other_optimiser)
    factor_2 = make_model_factor(centre=60, sigma=15)

    factor_model = ep.FactorGraphModel(factor_1, factor_2)

    default_optimiser = ep.LaplaceOptimiser()
    ep_optimiser = factor_model._make_ep_optimiser(default_optimiser)

    factor_optimisers = ep_optimiser.factor_optimisers
    assert factor_optimisers[factor_1] is other_optimiser
    assert factor_optimisers[factor_2] is default_optimiser


def test_custom_updater_reaches_ep_optimiser(factor_model):
    updater = af.SimplerUpdater(delta=0.5)

    ep_optimiser = factor_model._make_ep_optimiser(
        ep.LaplaceOptimiser(),
        updater=updater,
    )

    assert ep_optimiser.updater is updater


def test_default_updater_is_undamped(factor_model):
    ep_optimiser = factor_model._make_ep_optimiser(ep.LaplaceOptimiser())

    assert isinstance(ep_optimiser.updater, af.SimplerUpdater)
    assert ep_optimiser.updater.delta(factor=None, model_approx=None) == 1.0


@pytest.mark.parametrize(
    "updater",
    [None, af.SimplerUpdater(delta=0.5)],
)
def test_optimise_forwards_optional_updater(factor_model, updater, monkeypatch):
    captured = {}

    class StubEPOptimiser:
        ep_history = None

        @staticmethod
        def run(model_approx, **kwargs):
            return model_approx

    def make_ep_optimiser(
        optimiser,
        paths=None,
        ep_history=None,
        updater=None,
    ):
        captured["updater"] = updater
        return StubEPOptimiser()

    monkeypatch.setattr(factor_model, "_make_ep_optimiser", make_ep_optimiser)

    factor_model.optimise(ep.LaplaceOptimiser(), updater=updater)

    assert captured["updater"] is updater


def test_factor_model_attributes(factor_model):
    """
    There are:
    - 5 messages - one for each prior
    - 7 factors - one for each prior plus one for each likelihood
    """
    assert len(factor_model.message_dict) == 5
    assert len(factor_model.graph.factors) == 7


def _test_optimise_factor_model(factor_model):
    """
    We optimise the model
    """
    laplace = ep.LaplaceOptimiser()

    collection = factor_model.optimise(
        laplace,
    )

    """
    And what we get back is actually a PriorModelCollection
    """
    assert 25.0 == pytest.approx(collection[0].normalization.mean, rel=0.1)
    assert collection[0].normalization is collection[1].normalization


def test_gaussian():
    n_observations = 100
    x = np.arange(n_observations)
    y = make_data(Gaussian(centre=50.0, normalization=25.0, sigma=10.0), x)

    prior_model = af.Model(
        Gaussian,
        centre=af.GaussianPrior(mean=50, sigma=20),
        normalization=af.GaussianPrior(mean=25, sigma=10),
        sigma=af.GaussianPrior(mean=10, sigma=10),
    )

    factor_model = ep.AnalysisFactor(prior_model, analysis=Analysis(x=x, y=y))

    laplace = ep.LaplaceOptimiser()
    model = factor_model.optimise(laplace).model[0]

    assert model.centre.mean == pytest.approx(50, rel=0.1)
    assert model.normalization.mean == pytest.approx(25, rel=0.1)
    assert model.sigma.mean == pytest.approx(10, rel=0.1)


@pytest.fixture(name="prior_model")
def make_prior_model():
    return af.Model(Gaussian)


@pytest.fixture(name="likelihood_model")
def make_factor_model(prior_model):
    class MockAnalysis(af.Analysis):
        @staticmethod
        def log_likelihood_function(*_):
            return 1

    return ep.AnalysisFactor(prior_model, analysis=af.m.MockAnalysis())


def test_messages(likelihood_model):
    assert len(likelihood_model.message_dict) == 3


def test_graph(likelihood_model):
    graph = likelihood_model.graph
    assert len(graph.factors) == 4


def test_prior_model_node(likelihood_model):
    prior_model_node = likelihood_model.graph

    result = prior_model_node(
        {variable: np.array([0.5]) for variable in prior_model_node.variables}
    )

    assert isinstance(result, ep.FactorValue)


# def test_pytrees(
#     recreate,
#     factor_model,
#     make_model_factor,
# ):
#     recreate(factor_model)
#
#     model_factor = make_model_factor(centre=60, sigma=15)
#     recreate(model_factor)
