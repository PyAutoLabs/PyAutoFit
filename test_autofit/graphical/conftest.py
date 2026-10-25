import numpy as np
import pytest

import autofit.mapper.variable
from autofit import graphical as graph
from autonerves import conf


@pytest.fixture
def output_directory(tmp_path):
    """Each test owns its output, even when model identifiers coincide."""
    return tmp_path / "output"


@pytest.fixture(autouse=True)
def set_config_path(test_directory, output_directory):
    previous_configs = conf.instance.configs
    previous_output = conf.instance.output_path
    conf.instance.push(
        new_path=str(test_directory / "config"),
        output_path=str(output_directory),
    )
    yield
    conf.instance.configs = previous_configs
    conf.instance.output_path = previous_output


@pytest.fixture(autouse=True)
def do_remove_output():
    # tmp_path owns cleanup. The parent fixture removes the shared output tree,
    # which may belong to another worker running a non-graphical test.
    yield


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
