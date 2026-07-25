import itertools

import pytest

import autofit as af
from autofit.mapper.prior.deferred import DeferredArgument


@pytest.fixture(autouse=True)
def reset_prior_count():
    af.Prior._ids = itertools.count()


@pytest.fixture(name="uniform_dict")
def make_uniform_dict():
    return {"type": "Uniform", "lower_limit": 2.0, "upper_limit": 3.0}


@pytest.fixture(name="uniform_prior")
def make_uniform_prior(uniform_dict):
    return af.Prior.from_dict(uniform_dict)


@pytest.fixture(name="log_uniform_dict")
def make_log_uniform_dict():
    return {"type": "LogUniform", "lower_limit": 0.2, "upper_limit": 0.3}


@pytest.fixture(name="log_uniform_prior")
def make_log_uniform_prior(log_uniform_dict):
    return af.Prior.from_dict(log_uniform_dict)


@pytest.fixture(name="gaussian_dict")
def make_gaussian_dict():
    return {
        "type": "Gaussian",
        "mean": 3,
        "sigma": 4,
        "id": 0,
    }

@pytest.fixture(name="truncated_gaussian_dict")
def make_truncated_gaussian_dict():
    return {
        "type": "TruncatedGaussian",
        "lower_limit": -10.0,
        "upper_limit": 10.0,
        "mean": 3,
        "sigma": 4,
        "id": 0,
    }

@pytest.fixture(name="gaussian_prior")
def make_gaussian_prior(gaussian_dict):
    return af.Prior.from_dict(gaussian_dict)

@pytest.fixture(name="truncated_gaussian_prior")
def make_truncated_gaussian_prior(truncated_gaussian_dict):
    return af.Prior.from_dict(truncated_gaussian_dict)

@pytest.fixture(name="relative_width_dict")
def make_relative_width_dict():
    return {"type": "Relative", "value": 1.0}


@pytest.fixture(name="absolute_width_dict")
def make_absolute_width_dict():
    return {"type": "Absolute", "value": 2.0}


@pytest.fixture(name="relative_width_modifier")
def make_relative_width_modifier(relative_width_dict):
    return af.WidthModifier.from_dict(relative_width_dict)


@pytest.fixture(name="absolute_width_modifier")
def make_absolute_width_modifier(absolute_width_dict):
    return af.WidthModifier.from_dict(absolute_width_dict)


class TestWidth:
    def test_relative(self, relative_width_modifier):
        assert isinstance(relative_width_modifier, af.RelativeWidthModifier)
        assert relative_width_modifier.value == 1.0

    def test_absolute(self, absolute_width_modifier):
        assert isinstance(absolute_width_modifier, af.AbsoluteWidthModifier)
        assert absolute_width_modifier.value == 2.0

    def test_default(self):
        modifier = af.WidthModifier.for_class_and_attribute_name(
            af.ex.Gaussian, "not_real"
        )
        assert modifier.value == 0.5
        assert isinstance(modifier, af.RelativeWidthModifier)


class TestDict:
    def test_uniform(self, uniform_prior, uniform_dict, remove_ids):

        print(uniform_dict)
        print(remove_ids(uniform_prior.dict()))

        assert remove_ids(uniform_prior.dict()) == uniform_dict

    def test_log_uniform(self, log_uniform_prior, log_uniform_dict, remove_ids):
        assert remove_ids(log_uniform_prior.dict()) == log_uniform_dict

    def test_gaussian(self, gaussian_prior, gaussian_dict):
        assert gaussian_prior.dict() == gaussian_dict


class TestFromDict:
    def test_uniform(self, uniform_prior):
        # assert isinstance(uniform_prior, af.UniformPrior)
        assert uniform_prior.lower_limit == 2
        assert uniform_prior.upper_limit == 3

    def test_log_uniform(self, log_uniform_prior, absolute_width_modifier):
        # assert isinstance(log_uniform_prior, af.LogUniformPrior)
        assert log_uniform_prior.lower_limit == 0.2
        assert log_uniform_prior.upper_limit == 0.3

    def test_gaussian(self, gaussian_prior):
        assert isinstance(gaussian_prior, af.GaussianPrior)
        assert gaussian_prior.mean == 3
        assert gaussian_prior.sigma == 4

    def test_truncated_gaussian(self, truncated_gaussian_prior):
        assert isinstance(truncated_gaussian_prior, af.TruncatedGaussianPrior)
        assert truncated_gaussian_prior.lower_limit == -10
        assert truncated_gaussian_prior.upper_limit == 10
        assert truncated_gaussian_prior.mean == 3
        assert truncated_gaussian_prior.sigma == 4

    def test_constant(self):
        result = af.Prior.from_dict({"type": "Constant", "value": 1.5})
        assert result == 1.5

    def test_deferred(self):
        result = af.Prior.from_dict({"type": "Deferred"})
        assert isinstance(result, DeferredArgument)
