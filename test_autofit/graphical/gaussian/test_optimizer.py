import pytest

import autofit as af
import autofit.graphical as g
from autoconf.conf import output_path_for_test
from autofit.graphical.utils import Status
from test_autofit.graphical.gaussian.model import Analysis


@pytest.fixture(name="analysis")
def make_analysis(x, y):
    return Analysis(x=x, y=y)


@pytest.fixture(name="factor_model")
def make_factor_model(prior_model, analysis):
    return g.AnalysisFactor(prior_model, analysis=analysis)


@pytest.fixture(name="laplace")
def make_laplace():
    return g.LaplaceOptimiser()


@pytest.fixture(name="dynesty")
def make_dynesty():
    return af.DynestyStatic(name="", maxcall=5)


def test_default(factor_model, laplace):
    model = factor_model.optimise(laplace).model[0]

    assert model.centre.mean == pytest.approx(50, rel=0.1)
    assert model.normalization.mean == pytest.approx(25, rel=0.1)
    assert model.sigma.mean == pytest.approx(10, rel=0.1)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def _test_set_model_identifier(dynesty, prior_model, analysis):
    dynesty.fit(prior_model, analysis)

    identifier = dynesty.paths.identifier
    assert identifier is not None

    prior_model.centre = af.GaussianPrior(mean=20, sigma=20)
    dynesty.fit(prior_model, analysis)

    assert identifier != dynesty.paths.identifier


class TestDynesty:
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @output_path_for_test()
    def test_optimisation(self, factor_model, laplace, dynesty):
        factor_model.optimiser = dynesty
        factor_model.optimise(laplace)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_null_paths(self, factor_model):
        search = af.DynestyStatic(maxcall=5)
        result, status = search.optimise(
            factor_model.mean_field_approximation().factor_approximation(factor_model)
        )

        assert isinstance(result, g.MeanField)
        assert isinstance(status, Status)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @output_path_for_test()
    def test_optimise(self, factor_model, dynesty):
        result, status = dynesty.optimise(
            factor_model.mean_field_approximation().factor_approximation(factor_model)
        )

        assert isinstance(result, g.MeanField)
        assert isinstance(status, Status)
