import os
from random import random

import pytest

import autofit as af


class MockFitness:
    def __init__(self, figure_of_merit=0.0, change_figure_of_merit=True):
        self.figure_of_merit = figure_of_merit
        self.change_figure_of_merit = change_figure_of_merit

    def __call__(self, parameters):
        if self.change_figure_of_merit:
            return -random() * 10
        return self.figure_of_merit


@pytest.fixture
def model_and_samples():
    model = af.Model(af.m.MockClassx4)
    model.one = af.UniformPrior(lower_limit=0.099, upper_limit=0.101)
    model.two = af.UniformPrior(lower_limit=0.199, upper_limit=0.201)
    model.three = af.UniformPrior(lower_limit=0.299, upper_limit=0.301)
    model.four = af.UniformPrior(lower_limit=0.399, upper_limit=0.401)

    initializer = af.InitializerPrior()

    unit_parameter_lists, parameter_lists, _ = initializer.samples_from_model(
        total_points=2,
        model=model,
        fitness=MockFitness(),
        paths=af.DirectoryPaths(),
    )

    return unit_parameter_lists, parameter_lists

@pytest.mark.parametrize("index, param_index, lower, upper", [
    (0, 0, 0.0, 1.0),
    (1, 0, 0.0, 1.0),
    (0, 1, 0.0, 1.0),
    (1, 1, 0.0, 1.0),
    (0, 2, 0.0, 1.0),
    (1, 2, 0.0, 1.0),
    (0, 3, 0.0, 1.0),
    (1, 3, 0.0, 1.0),
])
def test_unit_parameter_lists(model_and_samples, index, param_index, lower, upper):
    unit_parameter_lists, _ = model_and_samples
    assert lower < unit_parameter_lists[index][param_index] < upper

@pytest.mark.parametrize("index, param_index, lower, upper", [
    (0, 0, 0.099, 0.101),
    (1, 0, 0.099, 0.101),
    (0, 1, 0.199, 0.201),
    (1, 1, 0.199, 0.201),
    (0, 2, 0.299, 0.301),
    (1, 2, 0.299, 0.301),
    (0, 3, 0.399, 0.401),
    (1, 3, 0.399, 0.401),
])
def test_parameter_lists(model_and_samples, index, param_index, lower, upper):
    _, parameter_lists = model_and_samples
    assert lower < parameter_lists[index][param_index] < upper


def test__priors__samples_from_model__raise_exception_if_all_likelihoods_identical():
    model = af.Model(af.m.MockClassx4)

    initializer = af.InitializerPrior()

    with pytest.raises(af.exc.InitializerException):
        initializer.samples_from_model(
            total_points=2,
            model=model,
            fitness=MockFitness(change_figure_of_merit=False),
            paths=af.DirectoryPaths(),
        )


def test__priors__samples_in_test_mode():
    os.environ["PYAUTO_TEST_MODE"] = "1"

    model = af.Model(af.m.MockClassx4)
    model.one = af.UniformPrior(lower_limit=0.099, upper_limit=0.101)
    model.two = af.UniformPrior(lower_limit=0.199, upper_limit=0.201)
    model.three = af.UniformPrior(lower_limit=0.299, upper_limit=0.301)
    model.four = af.UniformPrior(lower_limit=0.399, upper_limit=0.401)

    initializer = af.InitializerPrior()

    (
        unit_parameter_lists,
        parameter_lists,
        figure_of_merit_list,
    ) = initializer.samples_from_model(
        total_points=2,
        model=model,
        fitness=None,
        paths=af.DirectoryPaths(),
    )

    assert 0.0 < unit_parameter_lists[0][0] < 1.0
    assert 0.0 < unit_parameter_lists[1][0] < 1.0
    assert 0.0 < unit_parameter_lists[0][1] < 1.0
    assert 0.0 < unit_parameter_lists[1][1] < 1.0
    assert 0.0 < unit_parameter_lists[0][2] < 1.0
    assert 0.0 < unit_parameter_lists[1][2] < 1.0
    assert 0.0 < unit_parameter_lists[0][3] < 1.0
    assert 0.0 < unit_parameter_lists[1][3] < 1.0

    assert 0.099 < parameter_lists[0][0] < 0.101
    assert 0.099 < parameter_lists[1][0] < 0.101
    assert 0.199 < parameter_lists[0][1] < 0.201
    assert 0.199 < parameter_lists[1][1] < 0.201
    assert 0.299 < parameter_lists[0][2] < 0.301
    assert 0.299 < parameter_lists[1][2] < 0.301
    assert 0.399 < parameter_lists[0][3] < 0.401
    assert 0.399 < parameter_lists[1][3] < 0.401

    assert figure_of_merit_list == [-1.0e99, -1.0e100]

    os.environ["PYAUTO_TEST_MODE"] = "0"


def test__ball__samples_sample_centre_of_priors():
    model = af.Model(af.m.MockClassx4)
    model.one = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    model.two = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)
    model.three = af.UniformPrior(lower_limit=0.0, upper_limit=3.0)
    model.four = af.UniformPrior(lower_limit=0.0, upper_limit=4.0)

    initializer = af.InitializerBall(lower_limit=0.4999, upper_limit=0.5001)

    (
        unit_parameter_lists,
        parameter_lists,
        figure_of_merit_list,
    ) = initializer.samples_from_model(
        total_points=2,
        model=model,
        fitness=MockFitness(),
        paths=af.DirectoryPaths(),
    )

    assert 0.4999 < unit_parameter_lists[0][0] < 0.5001
    assert 0.4999 < unit_parameter_lists[1][0] < 0.5001
    assert 0.4999 < unit_parameter_lists[0][1] < 0.5001
    assert 0.4999 < unit_parameter_lists[1][1] < 0.5001
    assert 0.4999 < unit_parameter_lists[0][2] < 0.5001
    assert 0.4999 < unit_parameter_lists[1][2] < 0.5001
    assert 0.4999 < unit_parameter_lists[0][3] < 0.5001
    assert 0.4999 < unit_parameter_lists[1][3] < 0.5001

    assert 0.499 < parameter_lists[0][0] < 0.501
    assert 0.499 < parameter_lists[1][0] < 0.501
    assert 0.999 < parameter_lists[0][1] < 1.001
    assert 0.999 < parameter_lists[1][1] < 1.001
    assert 1.499 < parameter_lists[0][2] < 1.501
    assert 1.499 < parameter_lists[1][2] < 1.501
    assert 1.999 < parameter_lists[0][3] < 2.001
    assert 1.999 < parameter_lists[1][3] < 2.001

    initializer = af.InitializerBall(lower_limit=0.7999, upper_limit=0.8001)

    (
        unit_parameter_lists,
        parameter_lists,
        figure_of_merit_list,
    ) = initializer.samples_from_model(
        total_points=2,
        model=model,
        fitness=MockFitness(),
        paths=af.DirectoryPaths(),
    )

    assert 0.799 < parameter_lists[0][0] < 0.801
    assert 0.799 < parameter_lists[1][0] < 0.801
    assert 1.599 < parameter_lists[0][1] < 1.601
    assert 1.599 < parameter_lists[1][1] < 1.601
    assert 2.399 < parameter_lists[0][2] < 2.401
    assert 2.399 < parameter_lists[1][2] < 2.401
    assert 3.199 < parameter_lists[0][3] < 3.201
    assert 3.199 < parameter_lists[1][3] < 3.201


@pytest.mark.parametrize(
    "unit_value, physical_value",
    [
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
    ],
)
def test_invert_physical(unit_value, physical_value):
    prior = af.UniformPrior(
        lower_limit=0.0,
        upper_limit=1.0,
    )
    assert prior.unit_value_for(unit_value) == pytest.approx(physical_value)


@pytest.mark.parametrize(
    "unit_value, physical_value",
    [
        (1.0, 0.0),
        (2.0, 0.5),
        (3.0, 1.0),
    ],
)
def test_invert_physical_offset(unit_value, physical_value):
    prior = af.UniformPrior(
        lower_limit=1.0,
        upper_limit=3.0,
    )
    assert prior.unit_value_for(unit_value) == pytest.approx(physical_value)


@pytest.mark.parametrize(
    "unit_value, physical_value",
    [
        (-float("inf"), 0.0),
        (0.0, 0.5),
        (float("inf"), 1.0),
    ],
)
def test_invert_gaussian(unit_value, physical_value):
    prior = af.GaussianPrior(
        mean=0.0,
        sigma=3.0,
    )
    assert prior.unit_value_for(unit_value) == physical_value


@pytest.fixture(name="model")
def make_model():
    return af.Model(
        af.ex.Gaussian,
        centre=af.UniformPrior(1.0, 2.0),
        normalization=af.UniformPrior(2.0, 3.0),
        sigma=af.UniformPrior(-2.0, -1.0),
    )


def test_initializer_bounds(model):
    initializer = af.InitializerParamBounds(
        {
            model.centre: (1.0, 2.0),
            model.normalization: (2.0, 3.0),
            model.sigma: (-2.0, -1.0),
        }
    )

    parameter_list = initializer._generate_unit_parameter_list(model)
    assert len(parameter_list) == 3
    for parameter in parameter_list:
        assert 0.0 <= parameter <= 1.0


def test__initializer_bounds__info_from_model(model):

    initializer =  af.InitializerParamBounds(
        {
            model.centre: (1.0, 2.0),
            model.normalization: (2.0, 3.0),
            model.sigma: (-2.0, -1.0),
        }
    )

    info = initializer.info_from_model(model)

    assert "Total Free Parameters = 3" in info
    assert "centre: Start[(1.0, 2.0)]" in info


def test__initializer_start_point(model):
    initializer = af.InitializerParamStartPoints(
        {
            model.centre: 1.5,
            model.normalization: 2.5,
            model.sigma: -1.5,
        }
    )

    parameter_list = initializer._generate_unit_parameter_list(model)
    assert len(parameter_list) == 3
    for parameter in parameter_list:
        assert 0.0 <= parameter <= 1.0


def test__initializer_start_point__info_from_model(model):
    initializer = af.InitializerParamStartPoints(
        {
            model.centre: 1.5,
            model.normalization: 2.5,
            model.sigma: -1.5,
        }
    )
    info = initializer.info_from_model(model)

    assert "Total Free Parameters = 3" in info
    assert "centre: Start[1.5]" in info


def test_offset(model):
    initializer = af.InitializerParamBounds(
        {
            model.centre: (1.5, 2.0),
            model.normalization: (2.5, 3.0),
            model.sigma: (-1.5, -1.0),
        }
    )

    parameter_list = initializer._generate_unit_parameter_list(model)
    assert len(parameter_list) == 3
    for parameter in parameter_list:
        assert 0.5 <= parameter <= 1.0


def test_missing_parameter(model):
    initializer = af.InitializerParamBounds(
        {
            model.centre: (1.5, 2.0),
            model.normalization: (2.5, 3.0),
        },
        lower_limit=0.5,
        upper_limit=0.5,
    )
    parameter_list = initializer._generate_unit_parameter_list(model)

    assert len(parameter_list) == 3
    for parameter in parameter_list:
        assert 0.5 <= parameter <= 1.0

    assert 0.5 in parameter_list
