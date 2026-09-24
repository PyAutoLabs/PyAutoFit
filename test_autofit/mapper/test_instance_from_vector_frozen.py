"""
A frozen model builds instances through cached structure (``_vector_priors`` and
``Model._instance_plan``); an unfrozen model rediscovers that structure on every call.
Both must produce the same instance, attribute by attribute and type by type, and raise
the same exception when an assertion fails.
"""
import copy

import numpy as np
import pytest

import autofit as af
from autofit import exc
from autofit.example.model import PhysicalNFW


def _gaussian():
    return af.Model(af.ex.Gaussian)


def _nested_collection():
    return af.Collection(
        a=af.Model(af.ex.Gaussian),
        inner=af.Collection(
            b=af.Model(af.ex.Exponential),
            c=af.Model(af.ex.Gaussian),
        ),
    )


def _shared_priors():
    g0 = af.Model(af.ex.Gaussian)
    g1 = af.Model(af.ex.Gaussian)
    g1.centre = g0.centre
    g1.sigma = g0.sigma
    return af.Collection(g0=g0, g1=g1)


def _tuple_priors():
    return af.Model(PhysicalNFW)


def _deferred():
    model = af.Model(af.ex.Gaussian)
    model.sigma = af.DeferredArgument()
    return af.Collection(g=model, other=af.Model(af.ex.Exponential))


def _constants():
    model = af.Model(af.ex.Gaussian)
    model.normalization = af.Constant(2.0)
    model.sigma = 3.0
    return af.Collection(a=1.0, b=af.Constant(2.0), g=model)


def _child_model_attribute():
    # Attributes that are not constructor arguments are set on the instance after
    # construction by the post-construction loop.
    model = af.Model(af.ex.Gaussian)
    model.extra_constant = af.Constant(4.0)
    model.extra_value = "a string"
    return model


def _assertions():
    model = af.Collection(
        g0=af.Model(af.ex.Gaussian), g1=af.Model(af.ex.Gaussian)
    )
    model.add_assertion(model.g0.centre < model.g1.centre)
    model.g1.add_assertion(model.g1.sigma > model.g1.normalization)
    return model


MODELS = {
    "gaussian": _gaussian,
    "nested_collection": _nested_collection,
    "shared_priors": _shared_priors,
    "tuple_priors": _tuple_priors,
    "deferred": _deferred,
    "constants": _constants,
    "non_constructor_attributes": _child_model_attribute,
    "assertions": _assertions,
}


def assert_equal_instances(a, b, path="instance"):
    assert type(a) is type(b), f"{path}: {type(a)} != {type(b)}"
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_equal_instances(x, y, f"{path}[{i}]")
    elif isinstance(a, dict):
        assert list(a) == list(b), f"{path}: keys {list(a)} != {list(b)}"
        for key in a:
            assert_equal_instances(a[key], b[key], f"{path}[{key!r}]")
    elif isinstance(a, (int, float, str, bool, type(None), np.ndarray, np.generic)):
        assert np.array_equal(a, b), f"{path}: {a} != {b}"
    elif isinstance(a, type):
        assert a is b, path
    else:
        assert list(vars(a)) == list(vars(b)), (
            f"{path}: attributes {list(vars(a))} != {list(vars(b))}"
        )
        for key in vars(a):
            assert_equal_instances(
                getattr(a, key), getattr(b, key), f"{path}.{key}"
            )


def _build(model, vector):
    try:
        return model.instance_from_vector(vector), None
    except exc.FitException as e:
        return None, str(e)


@pytest.mark.parametrize("name", list(MODELS))
def test_frozen_matches_unfrozen(name):
    unfrozen = MODELS[name]()
    frozen = copy.deepcopy(unfrozen)
    frozen.freeze()

    np.random.seed(1)
    raised = 0
    for _ in range(50):
        vector = unfrozen.random_vector_from_priors
        expected, expected_error = _build(unfrozen, vector)
        # Twice, so the second frozen call runs off the populated cache.
        for _ in range(2):
            result, error = _build(frozen, vector)
            assert error == expected_error
            if expected_error is None:
                assert_equal_instances(result, expected)
        raised += expected_error is not None

    if name == "assertions":
        assert 0 < raised < 50


def test_frozen_cache_populated_and_reset_on_unfreeze():
    model = _nested_collection()
    model.freeze()
    model.instance_from_vector(model.random_vector_from_priors)

    assert any(key[0] == "_vector_priors" for key in model._frozen_cache)
    child = model.a
    assert any(key[0] == "_instance_plan" for key in child._frozen_cache)

    model.unfreeze()
    assert model._frozen_cache == {}
    assert child._frozen_cache == {}

    # An edit after unfreezing is seen by the next frozen build.
    child.sigma = af.Constant(7.0)
    model.freeze()
    instance = model.instance_from_vector(model.random_vector_from_priors)
    assert instance.a.sigma == 7.0
    assert model.prior_count == len(model._vector_priors())


def test_wrong_vector_length_raises():
    model = _gaussian()
    model.freeze()
    with pytest.raises(AssertionError):
        model.instance_from_vector([1.0, 2.0])
