import pytest

import numpy as np
from scipy import stats

from autofit import messages, graphical as graph
from autofit.graphical.expectation_propagation.optimiser import SimplerUpdater

np.random.seed(1)

error_std = 1.0
prior_std = 10.0
a = np.array([[-1.3], [0.7]])
b = np.array([-0.5])

n_obs = 50
n_features, n_dims = a.shape

x = 5 * np.random.randn(n_obs, n_features)
y = x.dot(a) + b + np.random.randn(n_obs, n_dims)

obs = graph.Plate(name="obs")
features = graph.Plate(name="features")
dims = graph.Plate(name="dims")

x_ = graph.Variable("x", obs, features)
a_ = graph.Variable("a", features, dims)
b_ = graph.Variable("b", dims)
y_ = graph.Variable("y", obs, dims)
z_ = graph.Variable("z", obs, dims)


def make_model():
    prior_norm = stats.norm(loc=0, scale=prior_std)

    def prior(x):
        return prior_norm.logpdf(x).sum()

    def linear(x, a, b):
        return x.dot(a) + b

    linear_factor = graph.Factor(
        linear,
        x_,
        a_,
        b_,
        factor_out=z_,
        vjp=False,
    )

    likelihood_factor = messages.NormalMessage(y, np.full_like(y, error_std)).as_factor(
        z_
    )
    prior_a = graph.Factor(prior, a_)
    prior_b = graph.Factor(prior, b_)

    model = likelihood_factor * linear_factor * prior_a * prior_b
    return model


def make_model_approx():
    mean_field0 = {
        a_: messages.NormalMessage.from_mode(np.zeros((n_features, n_dims)), 100),
        b_: messages.NormalMessage.from_mode(np.zeros(n_dims), 100),
        z_: messages.NormalMessage.from_mode(np.zeros((n_obs, n_dims)), 100),
        x_: messages.FixedMessage(x),
    }

    model_approx = graph.EPMeanField.from_approx_dists(make_model(), mean_field0)
    return model_approx


def test_stochastic_linear_regression():
    np.random.seed(2)
    params = [
        (30, 15, False, 1),
        (5, 15, False, 0.5),
    ]
    for n_batch, n_iters, inplace, delta in params:
        model_approx = make_model_approx()
        ep_opt = graph.StochasticEPOptimiser(
            model_approx.factor_graph,
            default_optimiser=graph.LaplaceOptimiser(),
            updater=SimplerUpdater(delta=delta),
        )
        rng = np.random.default_rng(1)
        batches = graph.utils.gen_dict(
            {obs: graph.utils.gen_subsets(n_batch, n_obs, n_iters=n_iters, rng=rng)}
        )
        new_approx = ep_opt.run(model_approx, batches, inplace=inplace)
        mean_field = new_approx.mean_field

        X = np.c_[x, np.ones(n_obs)]
        XTX = X.T.dot(X) + np.eye(3) / prior_std
        cov = np.linalg.inv(XTX)

        cov_a = cov[:2, :]
        cov_b = cov[2, :]
        mean_a = cov_a.dot(X.T.dot(y))
        mean_b = cov_b.dot(X.T.dot(y))

        a_std = cov_a.diagonal()[:, None] ** 0.5
        b_std = cov_b[[-1]] ** 0.5

        assert mean_field[a_].mean == pytest.approx(mean_a, rel=5e-1), n_batch
        assert mean_field[b_].mean == pytest.approx(mean_b, rel=5e-1), n_batch
        assert mean_field[a_].sigma == pytest.approx(a_std, rel=2.0), n_batch
        assert mean_field[b_].sigma == pytest.approx(b_std, rel=2.0), n_batch
