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
    autonerves ``cached_property_names`` MRO walker. It must pick up
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


# ---------------------------------------------------------------------------
# Equivalence witness for the memoised ``parameterization`` implementation.
#
# ``_reference_parameterization`` below is the pre-optimisation algorithm,
# copied verbatim from ``AbstractPriorModel.parameterization``: every prefix of
# every leaf path is resolved with ``object_for_path`` and each model node is
# asked for its own ``prior_count``. It is the oracle the current
# implementation (which memoises the per-subtree prior set bottom-up) must
# reproduce byte-for-byte.
# ---------------------------------------------------------------------------


def _reference_parameterization(model):
    """The pre-optimisation ``parameterization`` algorithm, verbatim."""
    from autofit.mapper.prior.abstract import Prior
    from autofit.mapper.prior.constant import Constant
    from autofit.mapper.prior.tuple_prior import TuplePrior
    from autofit.mapper.prior_model.abstract import AbstractPriorModel
    from autofit.mapper.prior_model.prior_model import Model
    from autofit.mapper.prior_model.representative import find_groups
    from autofit.text.formatter import TextFormatter
    from autofit.tools.util import info_whitespace

    formatter = TextFormatter(line_length=info_whitespace())

    paths = []

    for t in model.path_instance_tuples_for_class(
        (
            Prior,
            float,
            Constant,
            tuple,
        ),
        ignore_children=True,
    ):
        for i in range(len(t[0])):
            path = t[0][:i]
            obj = model.object_for_path(path)
            if isinstance(obj, TuplePrior):
                continue
            if isinstance(obj, AbstractPriorModel):
                n = obj.prior_count
            else:
                n = 0
            if isinstance(obj, Model):
                name = obj.cls.__name__
            else:
                name = type(obj).__name__

            paths.append((("model",) + path, f"{name} (N={n})"))

    for group in find_groups(paths, limit=0):
        formatter.add(*group)

    return formatter.text


def _build_benchmark_model(n_groups=2, n_gal=2, n_profiles=2):
    """A nested model with priors shared across groups.

    A shrunken copy of the micro-benchmark model that motivated memoising the
    per-node prior count: deep nesting plus one prior shared by every group,
    so a node's count is not simply the sum of its children's counts.
    """
    shared = af.UniformPrior(0.0, 1.0)
    groups = []
    for _ in range(n_groups):
        gals = []
        for _ in range(n_gal):
            profiles = {f"p{k}": af.Model(af.ex.Gaussian) for k in range(n_profiles)}
            profiles["p0"].centre = shared
            gals.append(af.Collection(**profiles))
        groups.append(af.Collection(galaxies=gals))
    return af.Collection(groups=groups)


def _nested_collections():
    return af.Collection(
        outer=af.Collection(
            inner=af.Collection(
                gaussian=af.Model(af.ex.Gaussian),
                other=af.Model(af.ex.Gaussian),
            ),
            gaussian=af.Model(af.ex.Gaussian),
        )
    )


def _shared_prior():
    shared = af.UniformPrior(0.0, 1.0)
    one = af.Model(af.ex.Gaussian)
    two = af.Model(af.ex.Gaussian)
    one.centre = shared
    two.centre = shared
    return af.Collection(one=one, two=two)


def _list_children():
    return af.Collection([af.Model(af.ex.Gaussian), af.Model(af.ex.Gaussian)])


def _tuple_prior_model():
    centre = af.TuplePrior()
    centre.centre_0 = af.UniformPrior()
    centre.centre_1 = af.UniformPrior()
    return af.Model(af.ex.Gaussian, centre=centre)


def _constant_and_float():
    return af.Collection(a=1.0, b=af.Constant(2.0), g=af.Model(af.ex.Gaussian))


def _zero_dimension_child():
    return af.Collection(instance=af.ex.Gaussian(), model=af.Model(af.ex.Gaussian))


def _integer_attribute():
    model = af.Model(af.ex.Gaussian)
    model.centre = 2
    return model


@pytest.mark.parametrize(
    "factory",
    [
        _nested_collections,
        _shared_prior,
        _list_children,
        _tuple_prior_model,
        _constant_and_float,
        _zero_dimension_child,
        _integer_attribute,
        _build_benchmark_model,
    ],
)
def test_parameterization_matches_reference(factory):
    model = factory()

    # The reference is computed on the same object, before ``parameterization``
    # is ever touched, so both see identical prior ids and the
    # ``_parameterization_cache`` cannot short-circuit the comparison.
    reference = _reference_parameterization(model)
    assert "_parameterization_cache" not in model.__dict__

    assert model.parameterization == reference


def test_shared_prior_counted_once():
    """A prior shared by two children is counted once by their parent."""
    model = _shared_prior()

    lines = model.parameterization.split("\n")

    # Two Gaussians (3 free parameters each) sharing one prior => 5, not 6.
    assert lines[0].split()[-2:] == ["Collection", "(N=5)"]
    assert model.prior_count == 5


def test_parameterization_walks_each_node_once():
    """``parameterization`` must not re-walk the tree once per leaf path.

    The witness: the number of recursive ``path_instances_of_class`` calls a
    whole ``parameterization`` makes is no more than the number a single
    leaf-path listing makes -- i.e. at most one visit per node.
    """
    import autofit.mapper.model as model_module

    from autofit.mapper.prior.abstract import Prior
    from autofit.mapper.prior.constant import Constant

    def count_calls(fn):
        calls = {"n": 0}
        original = model_module.path_instances_of_class

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        model_module.path_instances_of_class = counting
        try:
            fn()
        finally:
            model_module.path_instances_of_class = original
        return calls["n"]

    model = _build_benchmark_model()
    parameterization_calls = count_calls(lambda: model.parameterization)

    other = _build_benchmark_model()
    listing_calls = count_calls(
        lambda: other.path_instance_tuples_for_class(
            (
                Prior,
                float,
                Constant,
                tuple,
            ),
            ignore_children=True,
        )
    )

    assert parameterization_calls == listing_calls
