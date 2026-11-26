"""
A model reloaded from its dictionary (the route ``paths.save_all`` uses for
``model.json``) keeps its free parameters in the order of the model it was
saved from.

Parameters are ordered by prior id, which is creation order. A prior replaced
after the model is built is created out of the order the dictionary is walked
in, so before the fix the reloaded model listed it first and mapped a
parameter vector onto the wrong priors.
"""
import json

import pytest

import autofit as af
from autofit.tools.util import NumpyEncoder
from autonerves.dictable import to_dict, from_dict


def reload(model):
    return from_dict(json.loads(json.dumps(to_dict(model), cls=NumpyEncoder)))


def make_replaced_prior_model():
    model = af.Collection(g=af.Model(af.ex.Gaussian))
    model.g.centre = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)
    return model


def make_compound_model():
    """
    Two components share ``m`` and ``f`` through ``(1 - f) * m`` and ``f * m``,
    and other priors are replaced or fixed after the model is built.
    """
    m = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
    f = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)

    stars = af.Model(af.ex.Gaussian)
    halo = af.Model(af.ex.Gaussian)
    model = af.Collection(galaxies=af.Collection(lens=af.Collection(stars=stars, halo=halo)))

    stars.normalization = (1.0 - f) * m
    halo.normalization = f * m
    halo.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=3.0)
    stars.sigma = af.GaussianPrior(mean=1.0, sigma=0.5)
    stars.centre = 0.5
    return model


@pytest.fixture(
    name="model",
    params=[make_replaced_prior_model, make_compound_model],
    ids=["replaced_prior", "compound"],
)
def make_model(request):
    return request.param()


def test_paths_keep_their_order(model):
    loaded = reload(model)

    assert loaded.paths == model.paths
    assert loaded.model_component_and_parameter_names == (
        model.model_component_and_parameter_names
    )


def test_identifier_is_kept(model):
    assert reload(model).identifier == model.identifier


def test_vector_gives_the_same_instance(model):
    loaded = reload(model)
    unit_vector = [
        0.1 + 0.8 * index / (model.prior_count - 1)
        for index in range(model.prior_count)
    ]

    assert loaded.prior_count == model.prior_count
    assert loaded.vector_from_unit_vector(unit_vector) == pytest.approx(
        model.vector_from_unit_vector(unit_vector)
    )

    original = model.instance_from_unit_vector(unit_vector)
    reloaded = loaded.instance_from_unit_vector(unit_vector)
    assert [
        value for _, value in reloaded.path_instance_tuples_for_class(float)
    ] == pytest.approx(
        [value for _, value in original.path_instance_tuples_for_class(float)]
    )


def test_replaced_prior_maps_vector_to_its_own_parameter():
    model = make_replaced_prior_model()
    loaded = reload(model)

    instance = loaded.instance_from_vector([1.0, 2.0, 3.0])

    assert [path for path, _ in loaded.prior_tuples_ordered_by_id] == [
        "normalization",
        "sigma",
        "centre",
    ]
    assert (instance.g.normalization, instance.g.sigma, instance.g.centre) == (
        1.0,
        2.0,
        3.0,
    )


def test_reloaded_prior_ids_stay_unique():
    model = make_compound_model()
    first = reload(model)
    second = reload(model)

    ids = [
        prior.id
        for prior in first.priors_ordered_by_id + second.priors_ordered_by_id
    ]
    assert len(set(ids)) == len(ids)
    for prior in first.priors_ordered_by_id:
        assert prior.message.id_ == prior.id
