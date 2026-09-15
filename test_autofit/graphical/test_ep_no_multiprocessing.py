"""
Expectation propagation never drives a factor search through a Python
multiprocessing pool (human ruling 2026-09-09, #1608).

RAL job 342351_0 hung 27 hours because the EP factor search was a Nautilus
built with `number_of_cores=10` over a non-JAX analysis: two forked likelihood
workers segfaulted, `multiprocessing.Pool` replaced them, and because the pool
never re-issues a dead worker's in-flight task the `Pool.map` inside the fit
blocked to the wall clock.

`AbstractSearch.optimise` therefore refuses `number_of_cores > 1` outright,
rather than downgrading it silently.
"""

import numpy as np
import pytest

import autofit as af
import autofit.graphical as g
from autofit.graphical.utils import Status
from test_autofit.graphical.gaussian.model import Analysis, Gaussian, make_data

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(name="factor_model")
def make_factor_model():
    x = np.arange(100)
    y = make_data(Gaussian(centre=50.0, normalization=25.0, sigma=10.0), x)

    prior_model = af.Model(
        Gaussian,
        centre=af.GaussianPrior(mean=50, sigma=20),
        normalization=af.GaussianPrior(mean=25, sigma=10),
        sigma=af.GaussianPrior(mean=10, sigma=10),
    )

    return g.AnalysisFactor(prior_model, analysis=Analysis(x=x, y=y))


@pytest.fixture(name="no_forking")
def make_no_forking(monkeypatch):
    """
    Make any attempt to fork a pool a hard failure, in every module that holds
    `fork_context` as a module-level name (plus the package namespace, which the
    dynesty search imports from inside its own method).
    """

    def no_fork_context(*args, **kwargs):
        raise AssertionError("must not fork")

    module_paths = (
        "autofit.non_linear.parallel",
        "autofit.non_linear.parallel.context",
        "autofit.non_linear.search.abstract_search",
        "autofit.non_linear.search.nest.dynesty.search.abstract",
        "autofit.non_linear.search.nest.nautilus.search",
        "autofit.non_linear.parallel.sneaky",
        "autofit.non_linear.parallel.process",
    )

    import importlib

    for module_path in module_paths:
        module = importlib.import_module(module_path)
        if hasattr(module, "fork_context"):
            monkeypatch.setattr(module, "fork_context", no_fork_context)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test__ep_refuses_multi_core_search(factor_model, no_forking):
    search = af.DynestyStatic(maxcall=5, number_of_cores=2)

    factor_approx = factor_model.mean_field_approximation().factor_approximation(
        factor_model
    )

    with pytest.raises(af.exc.SearchException, match="multiprocessing") as error:
        search.optimise(factor_approx)

    message = str(error.value)

    assert "number_of_cores=2" in message
    assert "number_of_cores=1" in message
    assert "use_jax" in message
    assert "DynestyStatic" in message


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test__ep_single_core_search_unchanged(factor_model, no_forking):
    search = af.DynestyStatic(maxcall=5, number_of_cores=1)

    result, status = search.optimise(
        factor_model.mean_field_approximation().factor_approximation(factor_model)
    )

    assert isinstance(result, g.MeanField)
    assert isinstance(status, Status)
