import numpy as np
import pytest

import autofit.mapper.variable
from autofit import graphical as graph


@pytest.fixture(autouse=True)
def set_seed():
    np.random.seed(0)


@pytest.fixture(name="x")
def make_x():
    return autofit.mapper.variable.Variable("x")


@pytest.fixture(name="probit_factor")
def make_probit_factor(x):

    from scipy import stats

    return graph.Factor(stats.norm(loc=0.0, scale=1.0).logcdf, x)
