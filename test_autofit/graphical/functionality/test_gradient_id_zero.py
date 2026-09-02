"""
Regression tests for PyAutoFit#1557: the first prior of a process (id 0) used to
compare (and hash) equal to the ``FactorValue`` sentinel, so its cavity gradient
overwrote the VJP seed and every multi-variable factor gradient was wrong.
"""
import itertools

import numpy as np
import pytest

import autofit as af
from autofit.graphical import LaplaceOptimiser
from autofit.graphical.utils import StatusFlag
from autofit.mapper.prior.abstract import Prior
from autofit.mapper.variable import FactorValue, Variable, VariableData


class Level:
    def __init__(self, x):
        self.x = x


class Analysis(af.Analysis):
    def log_likelihood_function(self, instance):
        return -0.5 * (51.0 - instance.x) ** 2 / 0.5**2


@pytest.fixture(autouse=True)
def fresh_prior_ids(monkeypatch):
    # Number priors as a fresh process would, so the first one takes id 0
    monkeypatch.setattr(Prior, "_ids", itertools.count())


@pytest.fixture(name="mu")
def make_mu():
    mu = af.GaussianPrior(50.0, 10.0)
    assert mu.id == 0
    return mu


@pytest.fixture(name="hierarchical_approx")
def make_hierarchical_approx(mu):
    model = af.Model(Level)
    model.x = af.GaussianPrior(50.0, 20.0)

    hierarchical = af.HierarchicalFactor(af.GaussianPrior, mean=mu, sigma=10.0)
    hierarchical.add_drawn_variable(model.x)

    graph = af.FactorGraphModel(af.AnalysisFactor(model, Analysis()), hierarchical)
    factor = next(f for f in graph.graph.factors if f.name.startswith("Hierarchical"))
    approx = graph.mean_field_approximation().factor_approximation(factor)
    return approx, mu, model.x


def test_prior_id_zero_is_not_the_factor_value_sentinel(mu):
    assert FactorValue.id < 0

    assert mu != FactorValue
    assert FactorValue != mu
    assert not (mu == FactorValue)
    assert not (FactorValue == mu)
    assert hash(mu) != hash(FactorValue)

    seed = VariableData({FactorValue: 1.0})
    seed.update({mu: 0.02})
    assert seed[FactorValue] == 1.0
    assert seed[mu] == 0.02
    assert len(seed) == 2


def test_prior_and_variable_with_the_same_id_are_distinct(mu):
    variable = Variable("x", id_=mu.id)

    assert variable != mu
    assert mu != variable
    assert not (variable == mu)
    assert not (mu == variable)

    # Equality by id among priors is unchanged
    assert mu == af.GaussianPrior(0.0, 1.0, id_=mu.id)
    assert af.GaussianPrior(0.0, 1.0, id_=mu.id) == mu


def test_gradient_matches_finite_differences(hierarchical_approx):
    approx, mu, x = hierarchical_approx
    values = {mu: np.float64(48.0), x: np.float64(55.0)}

    _, gradient = approx.func_gradient(values)

    eps = 1e-5
    for variable in values:
        plus = {**values, variable: values[variable] + eps}
        minus = {**values, variable: values[variable] - eps}
        finite_difference = float(approx(plus) - approx(minus)) / (2 * eps)

        assert abs(finite_difference) > 1e-2
        assert float(gradient[variable]) == pytest.approx(finite_difference, abs=1e-6)


class GroupAnalysis(af.Analysis):
    def __init__(self, y, noise):
        self.y = y
        self.noise = noise

    def log_likelihood_function(self, instance):
        return float(-0.5 * np.sum((self.y - instance.x) ** 2) / self.noise**2)


def test_hierarchical_ep_updates_the_parent_mean(tmp_path):
    """
    With the corrupted gradient the parent mean never left its starting mean
    field (50.0 +/- 15) and the hierarchical factor never recorded a SUCCESS.
    """
    # The Laplace optimiser's refinement step samples the global numpy RNG
    np.random.seed(0)
    rng = np.random.default_rng(0)
    mu = af.GaussianPrior(50.0, 10.0)
    assert mu.id == 0

    hierarchical = af.HierarchicalFactor(af.GaussianPrior, mean=mu, sigma=10.0)
    analysis_factors = []
    for truth in (45.0, 52.0, 58.0):
        model = af.Model(Level)
        model.x = af.GaussianPrior(50.0, 20.0)
        y = truth + 2.0 * rng.standard_normal(20)
        analysis_factors.append(af.AnalysisFactor(model, GroupAnalysis(y, 2.0)))
        hierarchical.add_drawn_variable(model.x)

    graph = af.FactorGraphModel(*analysis_factors, hierarchical)
    result = graph.optimise(
        af.LaplaceOptimiser(),
        paths=af.DirectoryPaths(name="ep", path_prefix=str(tmp_path)),
        max_steps=5,
    )
    mean_field = result.updated_ep_mean_field.mean_field

    # The x_i are pinned by their data, so the exact parent posterior is
    # mu | x ~ N(m, 1 / p) with p = 1 / 10^2 (prior) + n / 10^2 (hierarchical)
    x_means = [float(mean_field[f.prior_model.x].mean) for f in analysis_factors]
    precision = (1 + len(x_means)) / 10.0**2
    mu_mean = (50.0 + sum(x_means)) / 10.0**2 / precision

    assert float(mean_field[mu].mean) == pytest.approx(mu_mean, abs=0.1)
    assert float(mean_field[mu].sigma) == pytest.approx(precision**-0.5, rel=0.1)

    flags = [
        status.flag
        for factor in graph.graph.factors
        if factor.name.startswith("Hierarchical")
        for _, status in result.ep_history[factor].history
    ]
    assert StatusFlag.SUCCESS in flags, flags
