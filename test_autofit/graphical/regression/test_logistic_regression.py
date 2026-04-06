import numpy as np
import pytest

from autofit import graphical as graph
from autofit.messages.fixed import FixedMessage
from autofit.messages.normal import NormalMessage


def likelihood(z, y):
    expz = np.exp(-z)
    logp = -np.log1p(expz)
    log1p = -np.log1p(1 / expz)
    loglike = y * logp + (1 - y) * log1p
    return loglike.sum()


def likelihood_jacobian(z, y):
    expz = np.exp(-z)
    logp = -np.log1p(expz)
    log1p = -np.log1p(1 / expz)
    loglike = y * logp + (1 - y) * log1p

    jac_z = y - 1 / (1 + expz)
    jac_y = logp - log1p

    return loglike.sum(), (jac_z, jac_y)


@pytest.fixture(name="likelihood_factor")
def make_likelihood_factor(z_, y_, obs, dims):
    factor = graph.Factor(likelihood, z_, y_)
    return factor


@pytest.fixture(name="likelihood_factor_jac")
def make_likelihood_factor_jac(z_, y_, obs, dims):
    factor = graph.Factor(likelihood, z_, y_, factor_jacobian=likelihood_jacobian)
    return factor


@pytest.fixture(name="model")
def make_model(prior_a, prior_b, likelihood_factor_jac, linear_factor_jac):
    return likelihood_factor_jac * linear_factor_jac * prior_a * prior_b


@pytest.fixture(name="start_approx")
def make_start_approx(
        a_,
        b_,
        z_,
        x_,
        y_,
):
    a = np.array([[-1.3], [0.7]])
    b = np.array([-0.5])
    n_obs = 100
    n_features, n_dims = a.shape
    x = 2 * np.random.randn(n_obs, n_features)
    z = x.dot(a) + b
    p = 1 / (1 + np.exp(-z))
    y = np.random.binomial(1, p)

    return {
        a_: NormalMessage.from_mode(np.zeros((n_features, n_dims)), 10),
        b_: NormalMessage.from_mode(np.zeros(n_dims), 10),
        z_: NormalMessage.from_mode(np.zeros((n_obs, n_dims)), 100),
        x_: FixedMessage(x),
        y_: FixedMessage(y),
    }


def test_laplace(
        model,
        start_approx,
        y_,
        z_,
):
    model_approx = graph.EPMeanField.from_approx_dists(model, start_approx)
    laplace = graph.LaplaceOptimiser()
    opt = graph.EPOptimiser(model_approx.factor_graph, default_optimiser=laplace)
    new_approx = opt.run(model_approx, max_steps=5)

    y = new_approx.mean_field[y_].mean
    z_pred = new_approx(new_approx.mean_field.mean)[z_]
    y_pred = z_pred > 0
    (tpr, fpr), (fnr, tnr) = np.dot(
        np.array([y, 1 - y]).reshape(2, -1),
        np.array([y_pred, 1 - y_pred]).reshape(2, -1).T,
    )

    accuracy = (tpr + tnr) / (tpr + fpr + fnr + tnr)
    assert 0.95 > accuracy > 0.75