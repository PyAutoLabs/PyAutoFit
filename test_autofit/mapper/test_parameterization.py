import functools
import itertools

import pytest

import autofit as af

from autofit.text import formatter as frm


@pytest.fixture(autouse=True)
def reset_ids():
    af.Prior._ids = itertools.count()


def test_parameterization():
    model = af.Collection(collection=af.Collection(gaussian=af.Model(af.ex.Gaussian)))

    parameterization = model.parameterization
    assert parameterization == (
        """model                                                                           Collection (N=3)
    collection                                                                  Collection (N=3)
        gaussian                                                                Gaussian (N=3)"""
    )


def test_root():
    model = af.Model(af.ex.Gaussian)
    parameterization = model.parameterization
    assert parameterization == (
        "model                                                                           Gaussian (N=3)"
    )


def test_instance():
    model = af.Collection(collection=af.Collection(gaussian=af.ex.Gaussian()))

    parameterization = model.parameterization
    assert parameterization == (
        """model                                                                           Collection (N=0)
    collection                                                                  Collection (N=0)
        gaussian                                                                Gaussian (N=0)"""
    )


def test_tuple_prior():
    centre = af.TuplePrior()
    centre.centre_0 = af.UniformPrior()
    centre.centre_1 = af.UniformPrior()

    model = af.Model(af.ex.Gaussian, centre=centre)
    parameterization = model.parameterization
    assert parameterization == (
        "model                                                                           Gaussian (N=4)"
    )


@pytest.fixture(name="formatter")
def make_info_dict():
    formatter = frm.TextFormatter(line_length=20, indent=4)
    formatter.add(("one", "one"), 1)
    formatter.add(("one", "two"), 2)
    formatter.add(("one", "three", "four"), 4)
    formatter.add(("three", "four"), 4)

    return formatter


class TestGenerateModelInfo:
    def test_info_string(self, formatter):
        ls = formatter.list

        assert ls[0] == "one"
        assert len(ls[1]) == 21
        assert ls[1] == "    one             1"
        assert ls[2] == "    two             2"
        assert ls[3] == "    three"
        assert ls[4] == "        four        4"
        assert ls[5] == "three"
        assert ls[6] == "    four            4"

    def test_basic(self):
        mm = af.ModelMapper()
        mm.mock_class = af.m.MockClassx2
        model_info = mm.info

        assert (
            model_info
            == """Total Free Parameters = 2

model                                                                           ModelMapper (N=2)
    mock_class                                                                  MockClassx2 (N=2)

mock_class
    one                                                                         UniformPrior [0], lower_limit = 0.0, upper_limit = 1.0
    two                                                                         UniformPrior [1], lower_limit = 0.0, upper_limit = 2.0"""
        )

    def test_with_instance(self):
        mm = af.ModelMapper()
        mm.mock_class = af.m.MockClassx2

        mm.mock_class.two = 1.0

        model_info = mm.info

        assert (
            model_info
            == """Total Free Parameters = 1

model                                                                           ModelMapper (N=1)
    mock_class                                                                  MockClassx2 (N=1)

mock_class
    one                                                                         UniformPrior [0], lower_limit = 0.0, upper_limit = 1.0
    two                                                                         1.0"""
        )

    def test_with_tuple(self):
        mm = af.ModelMapper()
        mm.tuple = (0, 1)

        assert (
            mm.info
            == """Total Free Parameters = 0

model                                                                           ModelMapper (N=0)

tuple                                                                           (0, 1)"""
        )

    # noinspection PyUnresolvedReferences
    def test_tuple_instance_model_info(self, mapper):
        mapper.mock_cls = af.m.MockChildTuplex2
        info = mapper.info

        mapper.mock_cls.tup_0 = 1.0

        assert len(mapper.mock_cls.tup.instance_tuples) == 1
        assert len(mapper.mock_cls.instance_tuples) == 1

        assert len(info.split("\n")) == len(mapper.info.split("\n"))


def test_parameterization_cache_does_not_leak_into_instance():
    """Regression: ``parameterization`` is cached in
    ``self.__dict__["_parameterization_cache"]`` so that
    ``Collection._instance_for_arguments`` and ``ModelInstance.dict``
    (which skip underscore-prefixed keys) do not propagate the cached
    string onto the constructed instance. A plain
    ``functools.cached_property`` would write to ``__dict__["parameterization"]``
    without an underscore, leaking the string into ``ModelInstance.dict``
    and downstream JAX pytree flattening — see commit 4564ae9a1."""

    model = af.Collection(gaussian=af.Model(af.ex.Gaussian))

    # Touch model.info → exercises the same propagation path that every
    # workspace script hits at construction time.
    _ = model.info
    _ = model.parameterization  # second access uses the cache

    # The cache must live behind an underscore key on the model.
    assert "_parameterization_cache" in model.__dict__
    assert "parameterization" not in model.__dict__

    instance = model.instance_from_prior_medians()

    # Neither the cached key nor the public name may appear on the
    # constructed instance.
    assert "parameterization" not in instance.__dict__
    assert "_parameterization_cache" not in instance.__dict__
    assert "parameterization" not in instance.dict
    assert "_parameterization_cache" not in instance.dict

    # The instance must yield only model components when iterated.
    for child in instance:
        assert not isinstance(child, str)


def test_cached_property_names_classmethod_walks_mro():
    """The ``_cached_property_names`` classmethod on AbstractModel exposes the
    autoconf ``cached_property_names`` MRO walker. It must pick up
    descriptors declared on any ancestor and memoise the result on the class."""

    import functools

    import autofit as af

    # Build a synthetic subclass with a cached_property to verify the walker
    # finds it. We use af.Collection because both AbstractPriorModel and
    # ModelInstance inherit from AbstractModel.
    class SyntheticCollection(af.Collection):
        @functools.cached_property
        def synthetic_value(self):
            return "a synthetic cached string"

    names = SyntheticCollection._cached_property_names()
    assert "synthetic_value" in names

    # Result is memoised on the synthetic class.
    assert "__cached_property_names_cache__" in SyntheticCollection.__dict__

    # Plain af.Collection (no synthetic_value) has its own cache.
    base_names = af.Collection._cached_property_names()
    assert "synthetic_value" not in base_names


class _GuardedCollection(af.Collection):
    """Module-level subclass used by
    ``test_cached_property_excluded_from_all_dict_walks`` — must live at
    module scope so ``pickle.dumps`` can locate the class on round-trip."""

    @functools.cached_property
    def derived(self):
        return "leaky-string"


def test_cached_property_excluded_from_all_dict_walks():
    """Regression: a future ``@functools.cached_property`` declared anywhere
    in the model class hierarchy must not surface through any of:
    ``Collection._instance_for_arguments`` (via ``instance.__dict__``),
    ``ModelInstance.dict``, ``ModelInstance.tree_flatten()``,
    ``AbstractModel.items()``, ``ModelObject._dict``, or pickling via
    ``__getstate__``.

    Covers the class of bug PyAutoFit#1300 fixed for ``parameterization``;
    this test will fail if a maintainer reintroduces an un-prefixed
    cached_property on the model hierarchy without the
    ``_cached_property_names`` defense applied at every site."""

    import pickle

    model = _GuardedCollection(gaussian=af.Model(af.ex.Gaussian))

    # Trigger the cache. After this, model.__dict__["derived"] = "leaky-string".
    _ = model.derived
    assert model.__dict__.get("derived") == "leaky-string"

    instance = model.instance_from_prior_medians()

    # Site 1+4: Collection._instance_for_arguments + ModelInstance.dict
    assert "derived" not in instance.__dict__
    assert "derived" not in instance.dict

    # Site 4 also feeds tree_flatten — no string leaves.
    leaves = instance.dict.values()
    for leaf in leaves:
        assert not isinstance(leaf, str)

    # Site 3: AbstractModel.items() on the model itself.
    assert all(key != "derived" for key, _ in model.items())

    # Site 5: __getstate__ drops the cached value from pickles.
    state = model.__getstate__()
    assert "derived" not in state

    # Round-trip via pickle: the unpickled model re-computes the cached value,
    # rather than carrying the pickled string on the wire.
    blob = pickle.dumps(model)
    revived = pickle.loads(blob)
    assert "derived" not in revived.__dict__
    # Touching it recomputes.
    assert revived.derived == "leaky-string"


def test_integer_attributes():
    model = af.Model(af.ex.Gaussian)

    model.centre = 2

    print(model.info)

    assert (
        model.info
        == """Total Free Parameters = 2

model                                                                           Gaussian (N=2)

centre                                                                          2
normalization                                                                   UniformPrior [1], lower_limit = 0.0, upper_limit = 1.0
sigma                                                                           UniformPrior [2], lower_limit = 0.0, upper_limit = 1.0"""
    )
