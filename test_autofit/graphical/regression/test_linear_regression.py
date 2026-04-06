import numpy as np
import pytest

from autofit.graphical import (
    EPMeanField,
    LaplaceOptimiser,
    EPOptimiser,
    Factor,
)
from autofit.graphical.expectation_propagation.optimiser import ParallelEPOptimiser
from autofit.messages import FixedMessage, NormalMessage

np.random.seed(1)
prior_std = 10.0
error_std = 1.0
a = np.array([[-1.3], [0.7]])
b = np.array([-0.5])

n_obs = 50
n_features, n_dims = a.shape
x = 5 * np.random.randn(n_obs, n_features)
y = x.dot(a) + b + np.random.randn(n_obs, n_dims)


class Likelihood:
    def __init__(self, norm):
        self.norm = norm

    def __call__(self, z, y):
        return self.norm.logpdf(z - y).sum()

    __name__ = "likelihood"


@pytest.fixture(name="likelihood")
def make_likelihood(norm):
    return Likelihood(norm)


@pytest.fixture(name="model")
def make_model(likelihood_factor, linear_factor, prior_a, prior_b):
    return likelihood_factor * linear_factor * prior_a * prior_b


@pytest.fixture(name="approx0")
def make_approx0(a_, b_, z_, x_, y_):
    return {
        a_: NormalMessage.from_mode(np.zeros((n_features, n_dims)), 100),
        b_: NormalMessage.from_mode(np.zeros(n_dims), 100),
        z_: NormalMessage.from_mode(np.zeros((n_obs, n_dims)), 100),
        x_: FixedMessage(x),
        y_: FixedMessage(y),
    }


@pytest.fixture(name="model_approx")
def make_model_approx(model, approx0):
    return EPMeanField.from_approx_dists(model, approx0)


def check_model_approx(mean_field, a_, b_, z_, x_, y_):
    X = np.c_[x, np.ones(len(x))]
    XTX = X.T.dot(X) + np.eye(3) * (error_std / prior_std) ** 2
    cov = np.linalg.inv(XTX) * error_std**2

    cov_a = cov[:2, :]
    cov_b = cov[2, :]

    # Analytic results
    mean_a = cov_a.dot(X.T.dot(y))
    mean_b = cov_b.dot(X.T.dot(y))
    a_std = cov_a.diagonal()[:, None] ** 0.5
    b_std = cov_b[[-1]] ** 0.5

    assert mean_field[a_].mean == pytest.approx(mean_a, rel=5e-2)
    assert mean_field[b_].mean == pytest.approx(mean_b, rel=5e-2)
    assert mean_field[a_].sigma == pytest.approx(a_std, rel=0.5)
    assert mean_field[b_].sigma == pytest.approx(b_std, rel=0.5)


@pytest.fixture(name="model_jac_approx")
def make_model_jac_approx(
    model, approx0, likelihood_factor, linear_factor_jac, prior_a, prior_b
):
    model = likelihood_factor * linear_factor_jac * prior_a * prior_b
    model_jac_approx = EPMeanField.from_approx_dists(model, approx0)
    return model_jac_approx


def test_jacobian(
    a_,
    b_,
    x_,
    z_,
    linear_factor,
    linear_factor_jac,
):
    n, m, d = 5, 4, 3
    x = np.random.rand(n, d)
    a = np.random.rand(d, m)
    b = np.random.rand(m)

    values = {x_: x, a_: a, b_: b}

    g0 = {z_: np.random.rand(n, m)}
    fval0, fjac0 = linear_factor.func_jacobian(values)
    fval1, fjac1 = linear_factor_jac.func_jacobian(values)

    fgrad0 = fjac0.grad(g0)
    fgrad1 = fjac1.grad(g0)

    assert fval0 == fval1
    assert (fgrad0 - fgrad1).norm() < 1e-6
    assert (fval0.deterministic_values - fval1.deterministic_values).norm() < 1e-6


def test_laplace(
    model_approx,
    a_,
    b_,
    x_,
    y_,
    z_,
):
    laplace = LaplaceOptimiser()
    opt = EPOptimiser(model_approx.factor_graph, default_optimiser=laplace)
    model_approx = opt.run(model_approx)
    mean_field = model_approx.mean_field
    check_model_approx(mean_field, a_, b_, z_, x_, y_)


def _test_parallel_laplace(
    model_approx,
    a_,
    b_,
    x_,
    y_,
    z_,
):
    laplace = LaplaceOptimiser()
    opt = ParallelEPOptimiser(
        model_approx.factor_graph,
        n_cores=len(model_approx.factors) + 1,
        default_optimiser=laplace,
    )
    model_approx = opt.run(model_approx)
    mean_field = model_approx.mean_field
    check_model_approx(mean_field, a_, b_, z_, x_, y_)


def _test_laplace_jac(
    model_jac_approx,
    a_,
    b_,
    x_,
    y_,
    z_,
):
    laplace = LaplaceOptimiser()
    opt = EPOptimiser(model_jac_approx.factor_graph, default_optimiser=laplace)
    model_approx = opt.run(model_jac_approx)

    mean_field = model_approx.mean_field
    check_model_approx(mean_field, a_, b_, z_, x_, y_)


@pytest.fixture(name="normal_model_approx")
def make_normal_model_approx(
    model_approx,
    approx0,
    linear_factor,
    a_,
    b_,
    y_,
    z_,
):
    y = model_approx.mean_field[y_].mean
    normal_factor = NormalMessage(y, np.full_like(y, error_std)).as_factor(z_)
    prior_a = NormalMessage(np.zeros_like(a), np.full_like(a, prior_std)).as_factor(
        a_, "prior_a"
    )
    prior_b = NormalMessage(np.zeros_like(b), np.full_like(b, prior_std)).as_factor(
        b_, "prior_b"
    )

    new_model = normal_factor * linear_factor * prior_a * prior_b
    return EPMeanField.from_approx_dists(new_model, approx0)


def test_exact_updates(
    normal_model_approx,
    a_,
    b_,
    x_,
    y_,
    z_,
):
    laplace = LaplaceOptimiser()
    opt = EPOptimiser.from_meanfield(normal_model_approx, default_optimiser=laplace)
    new_approx = opt.run(normal_model_approx)
    mean_field = new_approx.mean_field
    check_model_approx(mean_field, a_, b_, z_, x_, y_)
