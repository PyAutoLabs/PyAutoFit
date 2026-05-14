from numbers import Number

import pytest

import autofit as af
from autofit.mapper.mock.mock_model import (
    WithConstants,
    Parameter,
    ModelWithTupleConstant,
)
from autofit.mapper.prior.constant import Constant


@pytest.fixture
def model():
    return af.Model(WithConstants)


def test_model(model):
    assert model.constant_1 != model.constant_2

    assert model.constant_1 == 1.0
    assert model.constant_2 == 1.0


def test_instance(model):
    instance = model.instance_from_unit_vector([])

    assert instance.constant_1 == 1.0
    assert instance.constant_2 == 1.0

    assert instance.constant_1 == instance.constant_2


def test_collection(model):
    collection = af.Collection(model, Constant(1.0))

    instance = collection.instance_from_unit_vector([])
    assert instance[0].constant_1 == 1.0
    assert instance[0].constant_2 == 1.0

    assert instance[1] == 1.0

    assert isinstance(instance[0].constant_1, float)
    assert isinstance(instance[1], float)


def test_kwarg():
    model = af.Model(Parameter, value=1.0)
    assert model.value != af.Model(Parameter, value=1.0).value


def test_tuple_constant():
    model = af.Model(ModelWithTupleConstant)

    assert model.constant.constant_0 != model.constant.constant_1

    instance = model.instance_from_unit_vector([])
    assert instance.constant[0] == instance.constant[1]


def test_model_info(model):
    assert (
        model.info
        == """Total Free Parameters = 0

model                                                                           WithConstants (N=0)

constant_1                                                                      1.0
constant_2                                                                      1.0"""
    )


def test_set_constant(model):
    model.constant_1 = 2.0
    assert model.constant_1 == 2.0

    model.constant_2 = 2.0
    assert model.constant_2 == 2.0
    assert model.constant_1 != model.constant_2


def test_set_value_in_collection():
    collection = af.Collection(1.0, 1.0)

    assert collection[0] == 1.0
    assert collection[1] == 1.0
    assert collection[0] != collection[1]

    collection[0] = 2.0
    assert collection[0] == 2.0

    collection[1] = 2.0
    assert collection[1] == 2.0
    assert collection[0] != collection[1]


def test_equality_comparison():
    assert af.Constant(1.0) == 1.0
    assert 1.0 == af.Constant(1.0)


def test_inequality():
    assert af.Constant(1.0) != 2.0
    assert 2.0 != af.Constant(1.0)
    assert af.Constant(1.0) != 2.0
    assert 2.0 != af.Constant(1)


def test_lt():
    assert af.Constant(1.0) < 2.0
    assert 1.0 < af.Constant(2.0)
    assert not af.Constant(2.0) < 1.0
    assert not 2.0 < af.Constant(1.0)
    assert af.Constant(1.0) < Constant(2.0)


def test_gt():
    assert af.Constant(2.0) > 1.0
    assert 2.0 > af.Constant(1.0)
    assert not af.Constant(1.0) > 2.0
    assert not 1.0 > af.Constant(2.0)
    assert af.Constant(2.0) > Constant(1.0)


def test_le():
    assert af.Constant(1.0) <= 2.0
    assert 1.0 <= af.Constant(2.0)
    assert not af.Constant(2.0) <= 1.0
    assert not 2.0 <= af.Constant(1.0)
    assert af.Constant(1.0) <= Constant(2.0)


def test_ge():
    assert af.Constant(2.0) >= 1.0
    assert 2.0 >= af.Constant(1.0)
    assert not af.Constant(1.0) >= 2.0
    assert not 1.0 >= af.Constant(2.0)
    assert af.Constant(2.0) >= Constant(1.0)


def test_is_number():
    assert isinstance(Constant(1.0), Number)


def test_hash():
    assert hash(af.Constant(1.0)) != hash(af.Constant(1.0))

    constant = af.Constant(1.0)
    deserialized = af.Constant.from_dict(constant.dict())
    assert hash(deserialized)


def test_float_id():
    constant = af.Constant(1.0)
    constant.id = 1.0

    assert hash(constant)


def test_model_with_constants():
    instance = af.Model(
        af.ex.Gaussian,
        centre=af.Constant(1.0),
    ).instance_from_prior_medians()

    assert instance.centre == 1.0
    assert isinstance(instance.centre, float)
    assert not isinstance(instance.centre, Constant)


def test_as_float():
    constant = af.Constant(1.0)
    assert float(constant) == 1.0

    constant_2 = af.Constant(2.0)
    assert float(constant_2) == 2.0

    assert constant.id != constant_2.id

    assert isinstance(float(constant), float)


def test_is_instance():
    constant = af.Constant(1.0)

    assert isinstance(constant, float)


class _ClassWithExcludedField:
    # Mirrors the LightProfileLinear pattern: an internal attribute set in
    # __init__ that is declared not part of the model identity. The info
    # property must respect that contract and suppress `token`.
    __exclude_identifier_fields__ = ("token",)

    def __init__(self, value: float = 0.5):
        self.value = value
        self.token = 7


def test_info_honors_exclude_identifier_fields():
    model = af.Collection(thing=_ClassWithExcludedField())

    info = model.info
    assert "token" not in info
    assert "value" in info
