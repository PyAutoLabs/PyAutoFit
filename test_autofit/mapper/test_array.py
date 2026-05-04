import pytest
import numpy as np

import autofit as af
from autoconf.dictable import to_dict


@pytest.fixture
def array():
    return af.Array(
        shape=(2, 2),
        prior=af.GaussianPrior(mean=0.0, sigma=1.0),
    )


@pytest.fixture
def array_3d():
    return af.Array(
        shape=(2, 2, 2),
        prior=af.GaussianPrior(mean=0.0, sigma=1.0),
    )


def test_prior_count(array):
    assert array.prior_count == 4


def test_prior_count_3d(array_3d):
    assert array_3d.prior_count == 8


def test_instance(array):
    instance = array.instance_from_prior_medians()
    print(array.info)
    assert (instance == np.array([[0.0, 0.0], [0.0, 0.0]])).all()


def test_instance_3d(array_3d):
    instance = array_3d.instance_from_prior_medians()
    assert (
        instance
        == np.array([
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ])
    ).all()


def test_modify_prior(array):
    array[0, 0] = 1.0
    assert array.prior_count == 3
    print(array.instance_from_prior_medians())
    assert (
        array.instance_from_prior_medians()
        == np.array([
            [1.0, 0.0],
            [0.0, 0.0],
        ])
    ).all()


def test_correlation(array):
    array[0, 0] = array[1, 1]
    array[0, 1] = array[1, 0]

    instance = array.random_instance()

    assert instance[0, 0] == instance[1, 1]
    assert instance[0, 1] == instance[1, 0]


@pytest.fixture
def array_dict():
    return {
        "arguments": {
            "indices": {
                "type": "list",
                "values": [
                    {"type": "tuple", "values": [0, 0]},
                    {"type": "tuple", "values": [0, 1]},
                    {"type": "tuple", "values": [1, 0]},
                    {"type": "tuple", "values": [1, 1]},
                ],
            },
            "prior_0_0": {
                "mean": 0.0,
                "sigma": 1.0,
                "type": "Gaussian",
            },
            "prior_0_1": {
                "mean": 0.0,
                "sigma": 1.0,
                "type": "Gaussian",
            },
            "prior_1_0": {
                "mean": 0.0,
                "sigma": 1.0,
                "type": "Gaussian",
            },
            "prior_1_1": {
                "mean": 0.0,
                "sigma": 1.0,
                "type": "Gaussian",
            },
            "shape": {"type": "tuple", "values": [2, 2]},
        },
        "type": "array",
    }


def test_to_dict(array, array_dict, remove_ids):
    assert remove_ids(to_dict(array)) == array_dict


def test_from_dict(array_dict):
    array = af.AbstractPriorModel.from_dict(array_dict)
    assert array.prior_count == 4
    assert (
        array.instance_from_prior_medians()
        == np.array([
            [0.0, 0.0],
            [0.0, 0.0],
        ])
    ).all()


@pytest.fixture
def array_1d():
    return af.Array(
        shape=(2,),
        prior=af.GaussianPrior(mean=0.0, sigma=1.0),
    )


def test_1d_array(array_1d):
    assert array_1d.prior_count == 2
    assert (array_1d.instance_from_prior_medians() == np.array([0.0, 0.0])).all()


def test_1d_array_modify_prior(array_1d):
    array_1d[0] = 1.0
    assert array_1d.prior_count == 1
    assert (array_1d.instance_from_prior_medians() == np.array([1.0, 0.0])).all()


def test_gaussian_prior_model_for_arguments_with_fixed_element(array):
    """
    ``Array.gaussian_prior_model_for_arguments`` is invoked by
    ``AbstractSearch.optimise`` to build a posterior ``GaussianPrior``
    model from search arguments. Fixed scalar elements (set via
    ``arr[i, j] = float``) must pass through unchanged — they have no
    prior to update from posterior samples. Mirrors the try/except
    already in ``_instance_for_arguments``.
    """
    array[0, 0] = 1.5
    array[1, 1] = 2.5

    arguments = {
        prior: af.GaussianPrior(mean=10.0, sigma=0.1)
        for prior in array.priors
    }
    new_array = array.gaussian_prior_model_for_arguments(arguments)

    assert new_array.prior_count == 2
    instance = new_array.instance_from_prior_medians()
    assert instance[0, 0] == 1.5
    assert instance[1, 1] == 2.5
    assert instance[0, 1] == 10.0
    assert instance[1, 0] == 10.0


def test_tree_flatten(array):
    children, aux = array.tree_flatten()
    assert len(children) == 4
    assert aux == ((2, 2),)

    new_array = af.Array.tree_unflatten(aux, children)
    assert new_array.prior_count == 4
    assert (
        new_array.instance_from_prior_medians()
        == np.array([
            [0.0, 0.0],
            [0.0, 0.0],
        ])
    ).all()


