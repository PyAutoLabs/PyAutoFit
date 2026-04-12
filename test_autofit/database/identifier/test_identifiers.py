import numpy as np
import pytest

import autofit as af
from autofit import conf
from autofit.mapper.model_object import Identifier


def test_unique_tag_is_used():
    identifier = af.DynestyStatic("name", unique_tag="tag").paths._identifier

    assert "tag" in identifier.hash_list


def test_class_path():
    identifier = Identifier(
        Class,
    )
    (string,) = identifier.hash_list
    assert "test_autofit.database.identifier.test_identifiers.Class" in string


class Class:
    __doc__ = "hello"

    def __init__(self, one=1, two=2, three=3):
        self.one = one
        self.two = two
        self.three = three
        self.four = None

    __identifier_fields__ = ("one", "two")

    def __eq__(self, other):
        return self.one == other.one


class ExcludeClass:
    def __init__(self, one=1, two=2, three=3):
        self.one = one
        self.two = two
        self.three = three

    __exclude_identifier_fields__ = ("three",)


class AttributeClass:
    def __init__(self):
        self.attribute = None


def test_exclude_identifier_fields():
    other = ExcludeClass(three=4)
    assert Identifier(other) == Identifier(ExcludeClass())

    other.__exclude_identifier_fields__ = tuple()

    assert Identifier(other) != Identifier(ExcludeClass())


def test_numpy_array():
    identifier = Identifier(np.array([0]))
    assert identifier.hash_list == []


def test_hash_list():
    identifier = Identifier(Class())
    assert identifier.hash_list == ["Class", "one", "1", "two", "2"]


def test_constructor_only():
    attribute = AttributeClass()
    attribute.attribute = 1

    assert Identifier(AttributeClass()) == Identifier(attribute)


def test_exclude_does_no_effect_constructor():
    attribute = AttributeClass()
    attribute.__exclude_identifier_fields__ = tuple()
    attribute.attribute = 1

    assert Identifier(AttributeClass()) == Identifier(attribute)


class PrivateClass:
    def __init__(self, argument):
        self._argument = argument


def test_private_not_included():
    instance = PrivateClass(argument="one")
    identifier = str(Identifier(instance))

    instance._argument = "two"
    assert Identifier(instance) == identifier


def test_missing_field():
    instance = Class()
    instance.__identifier_fields__ = ("five",)

    with pytest.raises(AssertionError):
        Identifier(instance)


def test_change_class():
    gaussian_0 = af.Model(
        af.ex.Gaussian, normalization=af.UniformPrior(lower_limit=1e-6, upper_limit=1e6)
    )
    gaussian_1 = af.Model(
        af.ex.Gaussian, normalization=af.LogUniformPrior(lower_limit=1e-6, upper_limit=1e6)
    )

    assert Identifier(gaussian_0) != Identifier(gaussian_1)


def test_tiny_change():
    # noinspection PyTypeChecker
    instance = Class(one=1.0)
    identifier = str(Identifier(instance))

    instance.one += 1e-9
    print(instance.one)

    assert identifier == Identifier(instance)


def test_infinity():
    # noinspection PyTypeChecker
    instance = Class(one=float("inf"))
    str(Identifier(instance))


def test_identifier_fields():
    other = Class(three=4)
    assert Identifier(Class()) == Identifier(other)

    other.__identifier_fields__ = ("one", "two", "three")
    assert Identifier(Class()) != Identifier(other)


def test_unique_tag():
    search = af.m.MockSearch()

    search.fit(model=af.Model(af.ex.Gaussian), analysis=af.m.MockAnalysis())

    identifier = search.paths.identifier

    search = af.m.MockSearch(unique_tag="dataset")

    search.fit(
        model=af.Model(af.ex.Gaussian),
        analysis=af.m.MockAnalysis(),
    )

    assert search.paths.identifier != identifier


def test_prior():
    identifier = af.UniformPrior().identifier
    assert identifier == af.UniformPrior().identifier
    assert identifier != af.UniformPrior(lower_limit=0.5).identifier
    assert identifier != af.UniformPrior(upper_limit=0.5).identifier


def test_model():
    identifier = af.Model(af.ex.Gaussian, centre=af.UniformPrior()).identifier
    assert identifier == af.Model(af.ex.Gaussian, centre=af.UniformPrior()).identifier
    assert (
        identifier
        != af.Model(af.ex.Gaussian, centre=af.UniformPrior(upper_limit=0.5)).identifier
    )


def test_collection():
    identifier = af.Collection(
        gaussian=af.Model(af.ex.Gaussian, centre=af.UniformPrior())
    ).identifier
    assert (
        identifier
        == af.Collection(
            gaussian=af.Model(af.ex.Gaussian, centre=af.UniformPrior())
        ).identifier
    )
    assert (
        identifier
        != af.Collection(
            gaussian=af.Model(af.ex.Gaussian, centre=af.UniformPrior(upper_limit=0.5))
        ).identifier
    )


def test_instance():
    identifier = af.Collection(gaussian=af.ex.Gaussian()).identifier
    assert identifier == af.Collection(gaussian=af.ex.Gaussian()).identifier
    assert identifier != af.Collection(gaussian=af.ex.Gaussian(centre=0.5)).identifier


def test__identifier_description():
    model = af.Collection(
        gaussian=af.Model(
            af.ex.Gaussian,
            centre=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
            normalization=af.LogUniformPrior(lower_limit=0.001, upper_limit=0.01),
            sigma=af.GaussianPrior(
                mean=0.5, sigma=2.0,
            ),
        )
    )

    identifier = Identifier([model])

    description = identifier.description.splitlines()


    i = 0
    assert description[i] == "Collection"
    i += 1
    assert description[i] == "item_number"
    i += 1
    assert description[i] == "0"
    i += 1
    assert description[i] == "gaussian"
    i += 1
    assert description[i] == "Model"
    i += 1
    assert description[i] == "cls"
    i += 1
    assert description[i] == "autofit.example.model.Gaussian"
    i += 1
    assert description[i] == "centre"
    i += 1
    assert description[i] == "UniformPrior"
    i += 1
    assert description[i] == "lower_limit"
    i += 1
    assert description[i] == "0.0"
    i += 1
    assert description[i] == "upper_limit"
    i += 1
    assert description[i] == "1.0"
    i += 1
    assert description[i] == "normalization"
    i += 1
    assert description[i] == "LogUniformPrior"
    i += 1
    assert description[i] == "lower_limit"
    i += 1
    assert description[i] == "0.001"
    i += 1
    assert description[i] == "upper_limit"
    i += 1
    assert description[i] == "0.01"
    i += 1
    assert description[i] == "sigma"
    i += 1
    assert description[i] == "GaussianPrior"
    i += 1
    assert description[i] == "mean"
    i += 1
    assert description[i] == "0.5"
    i += 1
    assert description[i] == "sigma"
    i += 1
    assert description[i] == "2.0"
    i += 1


def test__identifier_description__after_model_and_instance():
    model = af.Collection(
        gaussian=af.Model(
            af.ex.Gaussian,
            centre=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
            normalization=af.LogUniformPrior(lower_limit=0.001, upper_limit=0.01),
            sigma=af.GaussianPrior(
                mean=0.5, sigma=2.0,
            ),
        )
    )

    max_log_likelihood_instance = model.instance_from_prior_medians()

    samples_summary = af.m.MockSamplesSummary(
        model=model,
        max_log_likelihood_instance=max_log_likelihood_instance,
        prior_means=[1.0, 3.0, 5.0],
    )

    result = af.mock.MockResult(
        samples_summary=samples_summary,
    )

    model.gaussian.centre = result.model_centred.gaussian.centre
    model.gaussian.normalization = result.instance.gaussian.normalization

    identifier = Identifier([model])

    description = identifier.description

    print(description)

    assert "Collection" in description
    assert "item_number" in description
    assert "0" in description
    assert "gaussian" in description
    assert "centre" in description
    assert "TruncatedGaussianPrior" in description
    assert "mean" in description
    assert "1.0" in description
    assert "sigma" in description
    assert "1.0" in description


def test__identifier_description__after_take_attributes():
    model = af.Collection(
        gaussian=af.Model(
            af.ex.Gaussian,
            centre=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
            normalization=af.LogUniformPrior(lower_limit=0.001, upper_limit=0.01),
            sigma=af.GaussianPrior(
                mean=0.5, sigma=2.0,
            ),
        )
    )

    model.take_attributes(source=model)

    identifier = Identifier([model])

    description = identifier.description.splitlines()

    # THIS TEST FAILS DUE TO THE BUG DESCRIBED IN A GITHUB ISSUE.

    i = 0

    assert description[i] == "Collection"
    i += 1
    assert description[i] == "item_number"
    i += 1
    assert description[i] == "0"
    i += 1
    assert description[i] == "gaussian"
    i += 1
    assert description[i] == "Model"
    i += 1
    assert description[i] == "cls"
    i += 1
    assert description[i] == "autofit.example.model.Gaussian"
    i += 1
    assert description[i] == "centre"
    i += 1
    assert description[i] == "UniformPrior"
    i += 1
    assert description[i] == "lower_limit"
    i += 1
    assert description[i] == "0.0"
    i += 1
    assert description[i] == "upper_limit"
    i += 1
    assert description[i] == "1.0"
    i += 1
    assert description[i] == "normalization"
    i += 1
    assert description[i] == "LogUniformPrior"
    i += 1
    assert description[i] == "lower_limit"
    i += 1
    assert description[i] == "0.001"
    i += 1
    assert description[i] == "upper_limit"
    i += 1
    assert description[i] == "0.01"
    i += 1
    assert description[i] == "sigma"
    i += 1
    assert description[i] == "GaussianPrior"
    i += 1
    assert description[i] == "mean"
    i += 1
    assert description[i] == "0.5"
    i += 1
    assert description[i] == "sigma"
    i += 1
    assert description[i] == "2.0"
    i += 1


def test_dynesty_static():
    assert Identifier(af.DynestyStatic()).hash_list == [
        "DynestyStatic",
        "nlive",
        "50",
        "bound",
        "multi",
        "sample",
        "auto",
        "bootstrap",
        "enlarge",
        "walks",
        "5",
        "facc",
        "0.2",
        "slices",
        "5",
        "fmove",
        "0.9",
        "max_move",
        "100",
    ]


def test_integer_keys():
    assert str(Identifier({1: 1}))
