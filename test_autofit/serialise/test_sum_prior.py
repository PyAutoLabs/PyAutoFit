import pytest

import autofit as af
from autofit.mapper.prior.arithmetic.compound import SumPrior


@pytest.fixture(name="prior_dict")
def make_prior_dict():
    return {
        "type": "compound",
        "compound_type": "SumPrior",
        "left": {
            "type": "Gaussian",
            "id": 0,
            "mean": 1.0,
            "sigma": 2.0,
        },
        "right": {
            "type": "Gaussian",
            "id": 0,
            "mean": 1.0,
            "sigma": 2.0,
        },
    }


def test_sum_prior(prior_dict):
    prior = af.GaussianPrior(
        mean=1.0,
        sigma=2.0,
    )

    sum_prior = prior + prior

    d = sum_prior.dict()

    # The operand names are found by inspecting the caller's frames, so their
    # values depend on the call stack; check they are written, not what they are.
    assert isinstance(d.pop("left_name"), str)
    assert isinstance(d.pop("right_name"), str)

    assert d == prior_dict


def test_sum_prior_from_dict(prior_dict):
    sum_prior = af.Model.from_dict(prior_dict)

    assert isinstance(sum_prior, SumPrior)
    assert isinstance(sum_prior.left, af.GaussianPrior)
    assert isinstance(sum_prior.right, af.GaussianPrior)
    assert sum_prior.left.mean == 1.0
    assert sum_prior.left.sigma == 2.0
    assert sum_prior.right.mean == 1.0
    assert sum_prior.right.sigma == 2.0
