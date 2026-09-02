"""
PyAutoFit#1559 (D4) at the EP level: a ``TruncatedGaussianPrior`` scatter
hyper-prior must keep its ``(lower_limit, upper_limit)`` through the whole
message-passing loop -- the very first ``prior.message ** (1 / (count - 1))``
in ``message_dict`` used to reset it to ``(-inf, inf)``.
"""
import numpy as np
import pytest

import autofit as af
from autofit.graphical import LaplaceOptimiser
from autofit.graphical.utils import StatusFlag
from autofit.messages.truncated_normal import TruncatedNormalMessage


class Level:
    def __init__(self, x):
        self.x = x


class GroupAnalysis(af.Analysis):
    def __init__(self, y, noise):
        self.y = y
        self.noise = noise

    def log_likelihood_function(self, instance):
        return float(-0.5 * np.sum((self.y - instance.x) ** 2) / self.noise**2)


def test__truncated_scatter_keeps_its_support_through_ep_sweeps():
    # The Laplace optimiser's refinement step samples the global numpy RNG
    np.random.seed(0)
    rng = np.random.default_rng(0)

    hf = af.HierarchicalFactor(
        af.GaussianPrior,
        mean=af.GaussianPrior(mean=50.0, sigma=10.0),
        sigma=af.TruncatedGaussianPrior(
            mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=100.0
        ),
    )
    analysis_factors = []
    for truth in (45.0, 52.0, 58.0):
        model = af.Model(Level)
        model.x = af.GaussianPrior(50.0, 20.0)
        y = truth + 2.0 * rng.standard_normal(20)
        analysis_factors.append(af.AnalysisFactor(model, GroupAnalysis(y, 2.0)))
        hf.add_drawn_variable(model.x)

    graph = af.FactorGraphModel(*analysis_factors, hf)
    approx = graph.mean_field_approximation()

    # The starting mean field already carries the limits
    start = approx.mean_field[hf.sigma]
    assert isinstance(start, TruncatedNormalMessage)
    assert (start.lower_limit, start.upper_limit) == (0.0, 100.0)

    # Manual factor_approximation -> optimise -> project_mean_field loop, which
    # avoids EPOptimiser's DirectoryPaths output
    laplace = LaplaceOptimiser()
    flags = []
    for _ in range(3):
        for factor in graph.graph.factors:
            factor_approx = approx.factor_approximation(factor)
            new_dist, status = laplace.optimise(factor_approx)
            approx, status = approx.project_mean_field(
                new_dist, factor_approx, status=status
            )
            flags.append(status.flag)

    scatter = approx.mean_field[hf.sigma]
    assert isinstance(scatter, TruncatedNormalMessage)
    assert scatter.lower_limit == 0.0
    assert scatter.upper_limit == 100.0
    assert StatusFlag.EXCEPTION not in flags, flags
