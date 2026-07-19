import pytest

import autofit as af
from autonerves.dictable import to_dict, from_dict


def test_trivial():
    instance = af.ModelInstance(dict(t=1))
    linear_interpolator = af.LinearInterpolator([instance])

    result = linear_interpolator[linear_interpolator.t == 1]

    assert result is instance


@pytest.fixture(name="linear_interpolator")
def make_linear_interpolator(instances):
    return af.LinearInterpolator(instances)


@pytest.fixture(name="spline_interpolator")
def make_spline_interpolator(instances):
    return af.SplineInterpolator(instances)


def test_spline_interpolator(spline_interpolator):
    result = spline_interpolator[spline_interpolator.t == 1.5]

    assert result.t == 1.5
    assert result.gaussian.centre == 0.5


def test_spline_relationships(spline_interpolator):
    relationships = spline_interpolator.relationships(spline_interpolator.t)
    assert relationships.gaussian.centre(1.5) == 0.5


def test_smooth_spline_interpolator(instances):
    interpolator = af.SplineInterpolator(
        instances
        + [
            af.ModelInstance(
                dict(
                    t=3.0,
                    gaussian=af.ex.Gaussian(centre=4.0, normalization=3.0, sigma=-3.0),
                )
            ),
        ]
    )

    result = interpolator[interpolator.t == 1.5]

    assert result.t == 1.5
    assert result.gaussian.centre < 0.5


@pytest.mark.parametrize(
    "t, centre", [(0.0, -1.0), (1.0, 0.0), (1.5, 0.5), (2.0, 1.0), (3.0, 2.0)]
)
def test_linear(t, centre, linear_interpolator):
    result = linear_interpolator[linear_interpolator.t == t]

    assert result.t == t
    assert result.gaussian.centre == centre
    assert result.gaussian.normalization == t
    assert result.gaussian.sigma == -t


@pytest.mark.parametrize("sigma", [-0.5, 0.0, 0.5, 1.0])
def test_alternate_attribute(linear_interpolator, sigma):
    result = linear_interpolator[linear_interpolator.gaussian.sigma == sigma]

    assert result.gaussian.sigma == sigma
    assert result.t == -sigma
    assert result.gaussian.normalization == -sigma


def test_deeper_attributes():
    collection = af.Collection(
        model=af.Model(
            af.ex.Gaussian,
            centre=0.0,
            normalization=1.0,
            sigma=-1.0,
        )
    )

    instance_1 = af.Collection(
        t=1.0,
        collection=collection,
    ).instance_from_prior_medians()
    instance_2 = af.Collection(
        t=2.0,
        collection=collection,
    ).instance_from_prior_medians()

    linear_interpolator = af.LinearInterpolator([instance_1, instance_2])

    result = linear_interpolator[linear_interpolator.t == 1.5]

    assert result.collection.model.centre == 0.0
    assert result.collection.model.normalization == 1.0
    assert result.collection.model.sigma == -1.0


def test_to_dict(linear_interpolator, linear_interpolator_dict):
    assert to_dict(linear_interpolator) == linear_interpolator_dict


def test_from_dict(linear_interpolator_dict):
    interpolator = from_dict(linear_interpolator_dict)
    assert interpolator[interpolator.t == 1.5].t == 1.5


@pytest.fixture(name="instance_dict")
def make_instance_dict():
    return {
        "type": "instance",
        "class_path": "autofit.mapper.model.ModelInstance",
        "arguments": {
            "child_items": {
                "type": "dict",
                "arguments": {
                    "t": 1.0,
                    "gaussian": {
                        "type": "instance",
                        "class_path": "autofit.example.model.Gaussian",
                        "arguments": {
                            "centre": 0.0,
                            "sigma": -1.0,
                            "normalization": 1.0,
                        },
                    },
                },
            }
        },
    }


@pytest.fixture(name="linear_interpolator_dict")
def make_linear_interpolator_dict(instance_dict):
    return {
        "type": "instance",
        "class_path": "autofit.interpolator.linear.LinearInterpolator",
        "arguments": {
            "instances": {
                "type": "list",
                "values": [
                    {
                        "type": "instance",
                        "class_path": "autofit.mapper.model.ModelInstance",
                        "arguments": {
                            "child_items": {
                                "type": "dict",
                                "arguments": {
                                    "t": 1.0,
                                    "gaussian": {
                                        "type": "instance",
                                        "class_path": "autofit.example.model.Gaussian",
                                        "arguments": {
                                            "normalization": 1.0,
                                            "centre": 0.0,
                                            "sigma": -1.0,
                                        },
                                    },
                                },
                            }
                        },
                    },
                    {
                        "type": "instance",
                        "class_path": "autofit.mapper.model.ModelInstance",
                        "arguments": {
                            "child_items": {
                                "type": "dict",
                                "arguments": {
                                    "t": 2.0,
                                    "gaussian": {
                                        "type": "instance",
                                        "class_path": "autofit.example.model.Gaussian",
                                        "arguments": {
                                            "normalization": 2.0,
                                            "centre": 1.0,
                                            "sigma": -2.0,
                                        },
                                    },
                                },
                            }
                        },
                    },
                ],
            }
        },
    }


def test_instance_as_dict(model_instance, instance_dict):
    assert to_dict(model_instance) == instance_dict


def test_instance_from_dict(model_instance, instance_dict):
    instance = from_dict(instance_dict)
    assert instance.t == 1.0

    gaussian = instance.gaussian
    assert isinstance(gaussian, af.ex.Gaussian)
    assert gaussian.centre == 0.0
    assert gaussian.normalization == 1.0
    assert gaussian.sigma == -1.0


def test_relationships(linear_interpolator):
    relationships = linear_interpolator.relationships(linear_interpolator.t)
    assert relationships.gaussian.centre(1.0) == 0.0
