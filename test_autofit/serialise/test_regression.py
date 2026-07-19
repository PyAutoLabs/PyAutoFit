import json

import pytest

from autonerves.dictable import to_dict, from_dict
from autofit.mapper.mock.mock_model import MockClassx2Tuple
import autofit as af


def test_tuple():
    model = af.Model(MockClassx2Tuple)
    json.dumps(model.dict())


class AttributeNotInInit:
    def __init__(self):
        self.attribute = 1


def test_embedded():
    model = af.Collection(instance=AttributeNotInInit())

    from_dict(to_dict(model))


@pytest.fixture(name="alpha")
def make_alpha():
    return af.GaussianPrior(mean=1, sigma=0.03)


@pytest.fixture(name="gamma")
def make_gamma():
    return 0.0


def test_serialise_compound(alpha, gamma):
    beta = 1.0 + gamma - 2 * alpha
    assert from_dict(to_dict(beta)).instance_from_prior_medians() == -1


def test_add_gamma(alpha, gamma):
    beta = 1.0 + -2 * alpha + gamma
    assert beta.instance_from_prior_medians() == -1


def test_negative_addition(alpha):
    assert (-alpha + 1).instance_from_prior_medians() == 0.0
