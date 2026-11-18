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


def test_nested_sampler_identifiers_unchanged_by_clipper():
    """
    The clipper entering the MLE search identifiers (PyAutoFit#1493) must not
    re-key any nested sampler: these hash lists were captured on main before
    the change and pin the archived nested-sampling results' directories.
    """
    assert Identifier(af.Nautilus()).hash_list == [
        "Nautilus",
        "n_live",
        "3000",
        "n_update",
        "enlarge_per_dim",
        "1.1",
        "n_points_min",
        "split_threshold",
        "100",
        "n_networks",
        "4",
        "n_like_new_bound",
        "seed",
        "n_shell",
        "1",
        "n_eff",
        "500",
    ]
    assert Identifier(af.DynestyDynamic()).hash_list == [
        "DynestyDynamic",
        "bound",
        "multi",
        "sample",
        "auto",
        "enlarge",
        "bootstrap",
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


def test_mcmc_identifiers_unchanged_by_clipper():
    assert Identifier(af.Emcee()).hash_list == [
        "Emcee",
        "nwalkers",
        "50",
    ]
    assert Identifier(af.Zeus()).hash_list == [
        "Zeus",
        "nwalkers",
        "50",
        "tune",
        "True",
        "tolerance",
        "0.05",
        "patience",
        "5",
        "mu",
        "1.0",
        "light_mode",
        "False",
    ]


def test_nested_samplers_have_no_clipper():
    """
    Tripwire for the hard constraint of PyAutoFit#1493: the clipper is resolved
    by the point-estimate searches (``posterior_kind.POINT``, formerly
    ``AbstractMLE``) and must stay there. Hoisting it to NonLinearSearch would put
    it within reach of the nested samplers' identifier machinery and silently
    re-key the nested-sampling archive; this fails loudly instead.
    """
    for search in [af.Nautilus(), af.DynestyStatic(), af.DynestyDynamic()]:
        assert not hasattr(search, "clipper")


@pytest.mark.parametrize("cls", [af.MultiStartAdam, af.LBFGS])
def test_clipper_forks_mle_identifier(cls):
    default = Identifier(cls())
    assert Identifier(cls(clipper=af.ClipperNone())) == default
    box = Identifier(cls(clipper=af.ClipperPriorBox()))
    assert box != default
    assert Identifier(cls(clipper=af.ClipperPriorBox(margin=1.0e-3))) != box


def test_drawer_identifier_ignores_clipper():
    """
    Drawer takes the point-estimate clipper (formerly from AbstractMLE) but never consumes
    it, so a setting that cannot affect its result must not re-key it.
    """
    assert Identifier(af.Drawer()).hash_list == [
        "Drawer",
        "total_draws",
        "50",
    ]
    assert Identifier(af.Drawer(clipper=af.ClipperPriorBox())) == Identifier(
        af.Drawer()
    )


def _gaussian_with_explicit_priors():
    """
    A Gaussian whose priors are all given explicitly, so that its identifier does not
    move when the test configuration's default priors are edited.
    """
    return af.Model(
        af.ex.Gaussian,
        centre=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
        normalization=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
        sigma=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
    )


def test_assertion_free_model_identifier_unchanged():
    """
    Tripwire: hashing assertions must not re-key the models which carry none. These are
    the identifiers captured from `main` before assertions entered the identifier, and a
    change here orphans every result on disk.
    """
    assert (
        str(Identifier(_gaussian_with_explicit_priors()))
        == "9929b2be4248f0d116f5c1c034bda870"
    )
    assert (
        str(Identifier(af.Collection(g=_gaussian_with_explicit_priors())))
        == "5b45f48f8ab1205471ce531904ace327"
    )


def _asserted_collection(reverse=False):
    a = af.Model(af.ex.Gaussian)
    b = af.Model(af.ex.Gaussian)
    collection = af.Collection(a=a, b=b)
    if reverse:
        collection.add_assertion(b.sigma < a.sigma)
    else:
        collection.add_assertion(a.sigma < b.sigma)
    return collection


def test_assertion_changes_identifier():
    """
    Without this a model ordered by an assertion targets the output directory of the
    unordered model and loads its completed result instead of running.
    """
    unordered = af.Collection(a=af.Model(af.ex.Gaussian), b=af.Model(af.ex.Gaussian))
    assert Identifier(_asserted_collection()) != Identifier(unordered)


def test_assertion_direction_changes_identifier():
    assert Identifier(_asserted_collection()) != Identifier(
        _asserted_collection(reverse=True)
    )


def test_assertion_identifier_is_deterministic():
    assert Identifier(_asserted_collection()) == Identifier(_asserted_collection())


def test_number_of_assertions_changes_identifier():
    a = af.Model(af.ex.Gaussian)
    b = af.Model(af.ex.Gaussian)
    collection = af.Collection(a=a, b=b)
    collection.add_assertion(a.sigma < b.sigma)
    one = str(Identifier(collection))

    collection.add_assertion(a.centre < b.centre)
    assert str(Identifier(collection)) != one


def test_assertion_on_child_changes_collection_identifier():
    gaussian = af.Model(af.ex.Gaussian)
    gaussian.add_assertion(gaussian.centre < gaussian.sigma)

    assert Identifier(af.Collection(gaussian=gaussian)) != Identifier(
        af.Collection(gaussian=af.Model(af.ex.Gaussian))
    )


def test_compound_assertion_changes_identifier():
    """
    A chained assertion is a `CompoundAssertion`, whose operands are themselves
    assertions held in public attributes, and an arithmetic operand is a `CompoundPrior`
    holding its sides in underscore prefixed attributes. Both must reach the hash.
    """
    a = af.Model(af.ex.Gaussian)
    b = af.Model(af.ex.Gaussian)
    chained = af.Collection(a=a, b=b)
    chained.add_assertion((a.sigma < b.sigma) < b.centre)

    c = af.Model(af.ex.Gaussian)
    d = af.Model(af.ex.Gaussian)
    arithmetic = af.Collection(a=c, b=d)
    arithmetic.add_assertion(c.sigma * 2 < d.sigma)

    identifiers = {
        str(Identifier(chained)),
        str(Identifier(arithmetic)),
        str(Identifier(_asserted_collection())),
    }
    assert len(identifiers) == 3


def test_assertion_name_does_not_change_identifier():
    """
    The name is logged when an assertion is violated and cannot change the fit.
    """
    a = af.Model(af.ex.Gaussian)
    b = af.Model(af.ex.Gaussian)
    collection = af.Collection(a=a, b=b)
    collection.add_assertion(a.sigma < b.sigma, name="ordered")

    assert Identifier(collection) == Identifier(_asserted_collection())


class ZeroDimProfile:
    def __init__(self, centre=0.0, slope=1.0):
        self.centre = centre
        self.slope = slope


def _zero_dim_model(centre, slope):
    return af.Collection(
        lens=ZeroDimProfile(centre=centre, slope=slope),
        gaussian=af.Model(af.ex.Gaussian),
    )


def test_instance_with_0d_numpy_array_attribute():
    model = _zero_dim_model(np.array(0.5), np.array(2.0))

    first = str(Identifier(model))
    second = str(Identifier(_zero_dim_model(np.array(0.5), np.array(2.0))))

    assert first == second
    assert first == str(Identifier(_zero_dim_model(0.5, 2.0)))
    assert first != str(Identifier(_zero_dim_model(np.array(0.5), np.array(3.0))))
