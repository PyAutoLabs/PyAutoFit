import numpy as np
import pytest
from scipy import stats

from autofit import graphical as graph
from autofit.graphical import Variable, messages
from autofit.messages.normal import NormalMessage


def normal_loglike(x, centre, precision):
    diff = np.asanyarray(x) - centre
    se = np.square(diff)
    return np.sum(-precision * se - np.log(2 * np.pi) + np.log(precision)) / 2


def normal_loglike_jacobian(x, centre, precision):
    diff = np.asanyarray(x) - centre
    se = np.square(diff)
    loglike = np.sum(-precision * se - np.log(2 * np.pi) + np.log(precision)) / 2
    jac_x = -precision * diff
    jac_c = np.sum(precision * diff)
    jac_p = np.sum(1 / precision - se) / 2

    return loglike, (jac_x, jac_c, jac_p)


def normal_loglike_t(x, centre, precision):
    # Make log transform of precision so that optimisation is unbounded
    _precision = np.exp(precision)
    return normal_loglike(x, centre, _precision)


def normal_loglike_t_jacobian(x, centre, precision):
    # Make log transform of precision so that optimisation is unbounded
    _precision = np.exp(precision)
    val, jac = normal_loglike_jacobian(x, centre, _precision)
    jac_x, jac_c, jac_p = jac
    jac_p *= _precision
    return val, (jac_x, jac_c, jac_p)


def hierarchical_loglike(centre, precision, *xs):
    return normal_loglike(xs, centre, precision)


def hierarchical_loglike_jac(centre, precision, *xs):
    loglike, (jac_x, jac_c, jac_p) = normal_loglike_jacobian(xs, centre, precision)
    return loglike, (jac_c, jac_p) + tuple(jac_x)


def hierarchical_loglike_t(centre, logt, *xs):
    return normal_loglike_t(xs, centre, logt)


def hierarchical_loglike_t_jac(centre, precision, *xs):
    loglike, (jac_x, jac_c, jac_p) = normal_loglike_t_jacobian(xs, centre, precision)
    return loglike, (jac_c, jac_p) + tuple(jac_x)


n = 5


@pytest.fixture(name="centres")
def make_centres():
    np.random.seed(1)

    mu = 0.5
    sigma = 0.5
    centre_dist = stats.norm(loc=mu, scale=sigma)

    return centre_dist.rvs(n)


@pytest.fixture(name="widths")
def make_widths():
    np.random.seed(1)

    a = 10
    b = 4000
    precision_dist = stats.invgamma(a, scale=b)

    return precision_dist.rvs(n) ** -0.5


@pytest.fixture(name="model_approx")
def make_model_approx(centres, widths):
    centres_ = [Variable(f"x_{i}") for i in range(n)]
    mu_ = Variable("mu")
    logt_ = Variable("logt")

    centre_likelihoods = [
        NormalMessage(c, w).as_factor(x) for c, w, x in zip(centres, widths, centres_)
    ]
    normal_likelihoods = [
        graph.Factor(
            normal_loglike_t,
            centre,
            mu_,
            logt_,
            factor_jacobian=normal_loglike_t_jacobian,
        )
        for centre in centres_
    ]

    model = graph.utils.prod(centre_likelihoods + normal_likelihoods)

    model_approx = graph.EPMeanField.from_approx_dists(
        model,
        {
            mu_: NormalMessage(0, 10),
            logt_: NormalMessage(0, 10),
            **{x_: NormalMessage(0, 10) for x_ in centres_},
        },
    )

    return model_approx


def _test_simple(model_approx, centres):
    laplace = graph.LaplaceOptimiser()
    ep_opt = graph.EPOptimiser(model_approx, default_optimiser=laplace)
    new_approx = ep_opt.run(model_approx, max_steps=10)

    mu_ = new_approx.factor_graph.name_variable_dict["mu"]
    logt_ = new_approx.factor_graph.name_variable_dict["logt"]

    assert new_approx.mean_field[mu_].mean == pytest.approx(np.mean(centres), rel=1.0)
    assert new_approx.mean_field[logt_].mean == pytest.approx(
        np.log(np.std(centres) ** -2), rel=2.0, abs=1.0
    )


def test_hierarchical(centres, widths):
    centres_ = [Variable(f"x_{i}") for i in range(n)]
    mu_ = Variable("mu")
    logt_ = Variable("logt")

    centre_likelihoods = [
        messages.NormalMessage(c, w).as_factor(x)
        for c, w, x in zip(centres, widths, centres_)
    ]

    hierarchical_factor = graph.Factor(
        hierarchical_loglike_t,
        mu_,
        logt_,
        *centres_,
        factor_jacobian=hierarchical_loglike_t_jac,
    )

    model = graph.utils.prod(centre_likelihoods) * hierarchical_factor

    model_approx = graph.EPMeanField.from_approx_dists(
        model,
        {
            mu_: messages.NormalMessage(0.0, 10.0),
            logt_: messages.NormalMessage(0.0, 10.0),
            **{x_: messages.NormalMessage(0.0, 10.0) for x_ in centres_},
        },
    )

    laplace = graph.LaplaceOptimiser()
    ep_opt = graph.EPOptimiser(model_approx, default_optimiser=laplace)
    new_approx = ep_opt.run(model_approx, max_steps=5)
    print(new_approx)

    mu_ = new_approx.factor_graph.name_variable_dict["mu"]
    logt_ = new_approx.factor_graph.name_variable_dict["logt"]

    assert new_approx.mean_field[mu_].mean == pytest.approx(np.mean(centres), rel=0.2)
    assert new_approx.mean_field[logt_].mean == pytest.approx(
        np.log(np.std(centres) ** -2), rel=0.2
    )


@pytest.fixture(name="data")
def make_data():
    ## Generative model
    np.random.seed(1)

    mu = 2.0
    sigma = 0.5
    centre_dist = stats.norm(loc=mu, scale=sigma)
    mu_logt = 3.0
    sigma_logt = 0.5
    logt_dist = stats.norm(loc=mu_logt, scale=sigma_logt)

    n = 5
    n_samples_avg = 500

    centres = centre_dist.rvs(n)
    widths = np.exp(logt_dist.rvs(n)) ** -0.5
    n_samples = np.random.poisson(n_samples_avg, size=n)

    sample_dists = [stats.norm(*p) for p in zip(centres, widths)]
    data = [dist.rvs(n) for dist, n in zip(sample_dists, n_samples)]
    return data


def test_full(data):
    samples = {Variable(f"samples_{i}"): sample for i, sample in enumerate(data)}
    x_i_ = [Variable(f"x_{i}") for i in range(n)]
    logt_i_ = [Variable(f"logt_{i}") for i in range(n)]

    mu_x_ = Variable("mu_x")
    logt_x_ = Variable("logt_x")
    mu_logt_ = Variable("mu_logt")
    logt_logt_ = Variable("logt_logt")
    hierarchical_params = (mu_x_, logt_x_, mu_logt_, logt_logt_)

    # Setting up model
    data_loglikes = [
        graph.Factor(
            normal_loglike_t,
            s_,
            x_,
            logt_,
            factor_jacobian=normal_loglike_t_jacobian,
            name=f"normal_{i}",
        )
        for i, (s_, x_, logt_) in enumerate(zip(samples, x_i_, logt_i_))
    ]
    centre_loglikes = [
        graph.Factor(normal_loglike_t, x_, mu_x_, logt_x_) for x_ in x_i_
    ]
    precision_loglikes = [
        graph.Factor(normal_loglike_t, logt_, mu_logt_, logt_logt_) for logt_ in logt_i_
    ]
    priors = [
        messages.NormalMessage(0, 10).as_factor(v, name=f"prior_{v.name}")
        for v in hierarchical_params
    ]
    model = graph.utils.prod(
        data_loglikes + centre_loglikes + precision_loglikes + priors
    )


def test_full_hierachical(data):
    samples = {Variable(f"samples_{i}"): sample for i, sample in enumerate(data)}
    x_i_ = [Variable(f"x_{i}") for i in range(n)]
    logt_i_ = [Variable(f"logt_{i}") for i in range(n)]

    mu_x_ = Variable("mu_x")
    logt_x_ = Variable("logt_x")
    mu_logt_ = Variable("mu_logt")
    logt_logt_ = Variable("logt_logt")
    hierarchical_params = (mu_x_, logt_x_, mu_logt_, logt_logt_)

    # Setting up model
    data_loglikes = [
        graph.Factor(
            normal_loglike_t,
            s_,
            x_,
            logt_,
            factor_jacobian=normal_loglike_t_jacobian,
            name=f"normal_{i}",
        )
        for i, (s_, x_, logt_) in enumerate(zip(samples, x_i_, logt_i_))
    ]
    centre_loglike = graph.Factor(
        hierarchical_loglike_t,
        mu_x_,
        logt_x_,
        *x_i_,
        name="mean_loglike",
        factor_jacobian=hierarchical_loglike_t_jac,
    )
    logt_loglike = graph.Factor(
        hierarchical_loglike_t,
        mu_logt_,
        logt_logt_,
        *logt_i_,
        name="logt_loglike",
        factor_jacobian=hierarchical_loglike_t_jac,
    )
    priors = [
        messages.NormalMessage(0.0, 10.0).as_factor(v, name=f"prior_{v.name}")
        for v in hierarchical_params
    ]

    model = graph.utils.prod(data_loglikes + priors + [centre_loglike, logt_loglike])

    model_approx = graph.EPMeanField.from_approx_dists(
        model,
        {
            **{v: messages.NormalMessage(0.0, 10.0) for v in model.variables},
            **{v: messages.FixedMessage(sample) for v, sample in samples.items()},
        },
    )

    # Mean field approximation
    model_approx = graph.EPMeanField.from_approx_dists(
        model,
        {
            **{v: messages.NormalMessage(0.0, 10.0) for v in model.variables},
            **{v: messages.FixedMessage(sample) for v, sample in samples.items()},
        },
    )

    laplace = graph.LaplaceOptimiser()
    ep_opt = graph.EPOptimiser(model, default_optimiser=laplace)
    new_approx = ep_opt.run(model_approx, max_steps=10)
    new_approx.mean_field.subset(hierarchical_params)

    m = np.mean([np.mean(sample) for sample in data])
    logt = np.mean([np.log(np.std(sample) ** -2) for sample in data])

    assert new_approx.mean_field[mu_x_].mean == pytest.approx(m, rel=1.0)
    assert new_approx.mean_field[mu_logt_].mean == pytest.approx(logt, rel=1.0)
