import json

import pytest

import autofit as af
from autofit.mapper.prior.arithmetic.compound import NegativePrior
from autonerves.dictable import to_dict, from_dict


@pytest.fixture(name="model")
def make_model():
    """
    A model where two priors are shared by two components through arithmetic,
    like a stellar mass ``(1 - f) * m`` and a dark mass ``f * m``.
    """
    m = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
    f = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)

    stars = af.Model(af.ex.Gaussian)
    halo = af.Model(af.ex.Gaussian)

    stars.normalization = (1.0 - f) * m
    halo.normalization = f * m

    for component in (stars, halo):
        component.centre = 50.0
        component.sigma = 10.0

    return af.Collection(stars=stars, halo=halo)


@pytest.fixture(name="loaded")
def make_loaded(model):
    return from_dict(json.loads(json.dumps(to_dict(model))))


def test_dict_is_json_serialisable(model):
    json.dumps(to_dict(model))


def test_round_trip_keeps_shared_priors(model, loaded):
    assert loaded.prior_count == model.prior_count == 2


def test_round_trip_keeps_identifier(model, loaded):
    assert loaded.identifier == model.identifier


def test_round_trip_gives_same_instance(model, loaded):
    values = {"m": 7.0, "f": 0.3}

    def instance(m):
        arguments = {
            prior: values[prior_name(m, prior)] for prior in m.priors_ordered_by_id
        }
        return m.instance_for_arguments(arguments)

    def prior_name(m, prior):
        (name,) = {path[-1] for path, p in m.path_priors_tuples if p is prior}
        return name

    original = instance(model)
    reloaded = instance(loaded)

    assert reloaded.stars.normalization == pytest.approx(original.stars.normalization) == 4.9
    assert reloaded.halo.normalization == pytest.approx(original.halo.normalization) == 2.1


def test_modified_prior_of_a_prior_serialises():
    f = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    negative = -f

    assert isinstance(negative, NegativePrior)
    assert negative.dict()["prior"]["type"] == "Uniform"
    assert negative.cls is float

    loaded = af.ModelObject.from_dict(json.loads(json.dumps(negative.dict())))
    assert isinstance(loaded, NegativePrior)
    assert loaded.instance_from_prior_medians() == -0.5


def test_info(model):
    assert "MultiplePrior" in model.info


def test_dict_without_operand_names_loads():
    d = {
        "type": "compound",
        "compound_type": "SumPrior",
        "left": 1.0,
        "right": {"type": "Uniform", "id": 0, "lower_limit": 0.0, "upper_limit": 1.0},
    }
    loaded = af.ModelObject.from_dict(d)
    assert loaded.instance_from_prior_medians() == 1.5
