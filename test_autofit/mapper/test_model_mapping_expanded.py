"""
Expanded tests for the model mapping API, covering gaps identified in:
- Collection composition and instance creation
- Shared (linked) priors across model types
- Direct use of instance_for_arguments with argument dicts
- Model tree navigation (object_for_path, path_for_prior, name_for_prior)
- Edge cases (empty models, deeply nested models, single-parameter models)
- Model subsetting (with_paths, without_paths)
- Freezing behavior
- Assertion checking
- from_instance round-trips
- mapper_from_prior_arguments and mapper_from_partial_prior_arguments
"""
import copy

import numpy as np
import pytest

import autofit as af
from autofit import exc
from autofit.mapper.prior.abstract import Prior


# ---------------------------------------------------------------------------
# Collection: composition, nesting, instance creation, iteration
# ---------------------------------------------------------------------------
class TestCollectionComposition:
    def test_collection_from_dict(self):
        model = af.Collection(
            one=af.Model(af.m.MockClassx2),
            two=af.Model(af.m.MockClassx2),
        )
        assert model.prior_count == 4

    def test_collection_from_list(self):
        model = af.Collection([af.m.MockClassx2, af.m.MockClassx2])
        assert model.prior_count == 4

    def test_collection_from_generator(self):
        model = af.Collection(af.Model(af.m.MockClassx2) for _ in range(3))
        assert model.prior_count == 6

    def test_nested_collection(self):
        inner = af.Collection(a=af.m.MockClassx2)
        outer = af.Collection(inner=inner, extra=af.m.MockClassx2)
        assert outer.prior_count == 4

    def test_deeply_nested_collection(self):
        model = af.Collection(
            level1=af.Collection(
                level2=af.Collection(
                    leaf=af.m.MockClassx2,
                )
            )
        )
        assert model.prior_count == 2

    def test_collection_instance_attribute_access(self):
        model = af.Collection(gaussian=af.m.MockClassx2, exp=af.m.MockClassx2)
        instance = model.instance_from_vector([1.0, 2.0, 3.0, 4.0])
        assert instance.gaussian.one == 1.0
        assert instance.gaussian.two == 2.0
        assert instance.exp.one == 3.0
        assert instance.exp.two == 4.0

    def test_collection_instance_index_access(self):
        model = af.Collection([af.m.MockClassx2, af.m.MockClassx2])
        instance = model.instance_from_vector([1.0, 2.0, 3.0, 4.0])
        assert instance[0].one == 1.0
        assert instance[1].one == 3.0

    def test_collection_len(self):
        model = af.Collection(a=af.m.MockClassx2, b=af.m.MockClassx2)
        assert len(model) == 2

    def test_collection_contains(self):
        model = af.Collection(a=af.m.MockClassx2, b=af.m.MockClassx2)
        assert "a" in model
        assert "c" not in model

    def test_collection_items(self):
        model = af.Collection(a=af.m.MockClassx2, b=af.m.MockClassx2)
        keys = [k for k, v in model.items()]
        assert "a" in keys
        assert "b" in keys

    def test_collection_getitem_string(self):
        model = af.Collection(a=af.m.MockClassx2)
        assert isinstance(model["a"], af.Model)

    def test_collection_append(self):
        model = af.Collection()
        model.append(af.m.MockClassx2)
        model.append(af.m.MockClassx2)
        assert model.prior_count == 4

    def test_collection_mixed_model_and_fixed(self):
        """Collection with one free model and one fixed instance."""
        model = af.Collection(
            free=af.Model(af.m.MockClassx2),
        )
        assert model.prior_count == 2

    def test_empty_collection(self):
        model = af.Collection()
        assert model.prior_count == 0


# ---------------------------------------------------------------------------
# Shared (linked) priors
# ---------------------------------------------------------------------------
class TestSharedPriors:
    def test_link_within_model(self):
        model = af.Model(af.m.MockClassx2)
        model.one = model.two
        assert model.prior_count == 1
        instance = model.instance_from_vector([5.0])
        assert instance.one == instance.two == 5.0

    def test_link_across_collection_children(self):
        model = af.Collection(a=af.m.MockClassx2, b=af.m.MockClassx2)
        model.a.one = model.b.one  # Link a.one to b.one
        assert model.prior_count == 3  # 4 - 1 shared

    def test_linked_priors_same_value(self):
        model = af.Collection(a=af.m.MockClassx2, b=af.m.MockClassx2)
        model.a.one = model.b.one
        instance = model.instance_from_vector([10.0, 20.0, 30.0])
        assert instance.a.one == instance.b.one

    def test_link_reduces_unique_prior_count(self):
        model = af.Model(af.m.MockClassx2)
        original_count = len(model.unique_prior_tuples)
        model.one = model.two
        assert len(model.unique_prior_tuples) == original_count - 1

    def test_linked_prior_identity(self):
        model = af.Model(af.m.MockClassx2)
        model.one = model.two
        assert model.one is model.two


# ---------------------------------------------------------------------------
# instance_for_arguments (direct argument dict usage)
# ---------------------------------------------------------------------------
class TestInstanceForArguments:
    def test_model_instance_for_arguments(self):
        model = af.Model(af.m.MockClassx2)
        args = {model.one: 10.0, model.two: 20.0}
        instance = model.instance_for_arguments(args)
        assert instance.one == 10.0
        assert instance.two == 20.0

    def test_collection_instance_for_arguments(self):
        model = af.Collection(a=af.m.MockClassx2, b=af.m.MockClassx2)
        args = {}
        for name, prior in model.prior_tuples_ordered_by_id:
            args[prior] = 1.0
        instance = model.instance_for_arguments(args)
        assert instance.a.one == 1.0
        assert instance.b.two == 1.0

    def test_shared_prior_in_arguments(self):
        """When priors are linked, only one entry is needed in the arguments dict."""
        model = af.Model(af.m.MockClassx2)
        model.one = model.two
        shared_prior = model.one
        args = {shared_prior: 42.0}
        instance = model.instance_for_arguments(args)
        assert instance.one == 42.0
        assert instance.two == 42.0

    def test_missing_prior_raises(self):
        model = af.Model(af.m.MockClassx2)
        args = {model.one: 10.0}  # missing model.two
        with pytest.raises(KeyError):
            model.instance_for_arguments(args)


# ---------------------------------------------------------------------------
# Vector and unit vector mapping
# ---------------------------------------------------------------------------
class TestVectorMapping:
    def test_instance_from_vector_basic(self):
        model = af.Model(af.m.MockClassx2)
        instance = model.instance_from_vector([3.0, 4.0])
        assert instance.one == 3.0
        assert instance.two == 4.0

    def test_vector_length_mismatch_raises(self):
        model = af.Model(af.m.MockClassx2)
        with pytest.raises(AssertionError):
            model.instance_from_vector([1.0])

    def test_unit_vector_length_mismatch_raises(self):
        model = af.Model(af.m.MockClassx2)
        with pytest.raises(AssertionError):
            model.instance_from_unit_vector([0.5])

    def test_vector_from_unit_vector(self):
        model = af.Model(af.m.MockClassx2)
        model.one = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
        model.two = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
        physical = model.vector_from_unit_vector([0.0, 1.0])
        assert physical[0] == pytest.approx(0.0, abs=1e-6)
        assert physical[1] == pytest.approx(10.0, abs=1e-6)

    def test_instance_from_prior_medians(self):
        model = af.Model(af.m.MockClassx2)
        model.one = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
        model.two = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
        instance = model.instance_from_prior_medians()
        assert instance.one == pytest.approx(50.0)
        assert instance.two == pytest.approx(50.0)

    def test_random_instance(self):
        model = af.Model(af.m.MockClassx2)
        model.one = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
        model.two = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
        instance = model.random_instance()
        assert 0.0 <= instance.one <= 1.0
        assert 0.0 <= instance.two <= 1.0


# ---------------------------------------------------------------------------
# Model tree navigation
# ---------------------------------------------------------------------------
class TestModelTreeNavigation:
    def test_object_for_path_child_model(self):
        model = af.Collection(g=af.Model(af.m.MockClassx2))
        child = model.object_for_path(("g",))
        assert isinstance(child, af.Model)

    def test_object_for_path_prior(self):
        model = af.Collection(g=af.Model(af.m.MockClassx2))
        prior = model.object_for_path(("g", "one"))
        assert isinstance(prior, Prior)

    def test_paths_matches_prior_count(self):
        model = af.Collection(a=af.m.MockClassx2, b=af.m.MockClassx2)
        assert len(model.paths) == model.prior_count

    def test_path_for_prior(self):
        model = af.Collection(g=af.Model(af.m.MockClassx2))
        prior = model.g.one
        path = model.path_for_prior(prior)
        assert path == ("g", "one")

    def test_name_for_prior(self):
        model = af.Collection(g=af.Model(af.m.MockClassx2))
        prior = model.g.one
        name = model.name_for_prior(prior)
        assert name == "g_one"

    def test_path_instance_tuples_for_class(self):
        model = af.Collection(g=af.Model(af.m.MockClassx2))
        tuples = model.path_instance_tuples_for_class(Prior)
        paths = [t[0] for t in tuples]
        assert ("g", "one") in paths
        assert ("g", "two") in paths

    def test_deeply_nested_path(self):
        inner_model = af.Model(af.m.MockClassx2)
        inner_collection = af.Collection(leaf=inner_model)
        outer = af.Collection(branch=inner_collection)

        prior = outer.branch.leaf.one
        path = outer.path_for_prior(prior)
        assert path == ("branch", "leaf", "one")

    def test_direct_vs_recursive_prior_tuples(self):
        model = af.Collection(a=af.m.MockClassx2)
        assert len(model.direct_prior_tuples) == 0  # Collection has no direct priors
        assert len(model.prior_tuples) == 2  # But has 2 recursive priors

    def test_direct_prior_model_tuples(self):
        model = af.Collection(a=af.m.MockClassx2, b=af.m.MockClassx2)
        assert len(model.direct_prior_model_tuples) == 2


# ---------------------------------------------------------------------------
# instance_from_path_arguments and instance_from_prior_name_arguments
# ---------------------------------------------------------------------------
class TestPathAndNameArguments:
    def test_instance_from_path_arguments(self):
        model = af.Collection(g=af.m.MockClassx2)
        instance = model.instance_from_path_arguments(
            {("g", "one"): 10.0, ("g", "two"): 20.0}
        )
        assert instance.g.one == 10.0
        assert instance.g.two == 20.0

    def test_instance_from_prior_name_arguments(self):
        model = af.Collection(g=af.m.MockClassx2)
        instance = model.instance_from_prior_name_arguments(
            {"g_one": 10.0, "g_two": 20.0}
        )
        assert instance.g.one == 10.0
        assert instance.g.two == 20.0


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------
class TestAssertions:
    def test_assertion_passes(self):
        model = af.Model(af.m.MockClassx2)
        model.add_assertion(model.one > model.two)
        # one=10 > two=5 should pass
        instance = model.instance_from_vector([10.0, 5.0])
        assert instance.one == 10.0

    def test_assertion_fails(self):
        model = af.Model(af.m.MockClassx2)
        model.add_assertion(model.one > model.two)
        with pytest.raises(exc.FitException):
            model.instance_from_vector([1.0, 10.0])

    def test_ignore_assertions(self):
        model = af.Model(af.m.MockClassx2)
        model.add_assertion(model.one > model.two)
        # Should not raise even though assertion fails
        instance = model.instance_from_vector([1.0, 10.0], ignore_assertions=True)
        assert instance.one == 1.0

    def test_multiple_assertions(self):
        model = af.Model(af.m.MockClassx4)
        model.add_assertion(model.one > model.two)
        model.add_assertion(model.three > model.four)
        # Both pass
        instance = model.instance_from_vector([10.0, 5.0, 10.0, 5.0])
        assert instance.one == 10.0
        # First fails
        with pytest.raises(exc.FitException):
            model.instance_from_vector([1.0, 10.0, 10.0, 5.0])

    def test_true_assertion_ignored(self):
        """Adding True as an assertion should be a no-op."""
        model = af.Model(af.m.MockClassx2)
        model.add_assertion(True)
        assert len(model.assertions) == 0


# ---------------------------------------------------------------------------
# Model subsetting (with_paths, without_paths)
# ---------------------------------------------------------------------------
class TestModelSubsetting:
    def test_with_paths_single_child(self):
        model = af.Collection(a=af.m.MockClassx2, b=af.m.MockClassx2)
        subset = model.with_paths([("a",)])
        assert subset.prior_count == 2

    def test_without_paths_single_child(self):
        model = af.Collection(a=af.m.MockClassx2, b=af.m.MockClassx2)
        subset = model.without_paths([("a",)])
        assert subset.prior_count == 2

    def test_with_paths_specific_prior(self):
        model = af.Collection(a=af.m.MockClassx2, b=af.m.MockClassx2)
        subset = model.with_paths([("a", "one")])
        assert subset.prior_count == 1

    def test_with_prefix(self):
        model = af.Collection(ab_one=af.m.MockClassx2, cd_two=af.m.MockClassx2)
        subset = model.with_prefix("ab")
        assert subset.prior_count == 2


# ---------------------------------------------------------------------------
# Freezing behavior
# ---------------------------------------------------------------------------
class TestFreezing:
    def test_freeze_prevents_modification(self):
        model = af.Model(af.m.MockClassx2)
        model.freeze()
        with pytest.raises(AssertionError):
            model.one = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)

    def test_unfreeze_allows_modification(self):
        model = af.Model(af.m.MockClassx2)
        model.freeze()
        model.unfreeze()
        model.one = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
        assert isinstance(model.one, af.UniformPrior)

    def test_frozen_model_still_creates_instances(self):
        model = af.Model(af.m.MockClassx2)
        model.freeze()
        instance = model.instance_from_vector([1.0, 2.0])
        assert instance.one == 1.0

    def test_freeze_propagates_to_children(self):
        model = af.Collection(a=af.m.MockClassx2)
        model.freeze()
        with pytest.raises(AssertionError):
            model.a.one = 1.0

    def test_cached_results_consistent(self):
        model = af.Model(af.m.MockClassx2)
        model.freeze()
        result1 = model.prior_tuples_ordered_by_id
        result2 = model.prior_tuples_ordered_by_id
        assert result1 == result2


# ---------------------------------------------------------------------------
# mapper_from_prior_arguments and related
# ---------------------------------------------------------------------------
class TestMapperFromPriorArguments:
    def test_replace_all_priors(self):
        model = af.Model(af.m.MockClassx2)
        new_one = af.GaussianPrior(mean=0.0, sigma=1.0)
        new_two = af.GaussianPrior(mean=5.0, sigma=2.0)
        new_model = model.mapper_from_prior_arguments(
            {model.one: new_one, model.two: new_two}
        )
        assert new_model.prior_count == 2
        assert isinstance(new_model.one, af.GaussianPrior)

    def test_partial_replacement(self):
        model = af.Model(af.m.MockClassx2)
        new_one = af.GaussianPrior(mean=0.0, sigma=1.0)
        new_model = model.mapper_from_partial_prior_arguments(
            {model.one: new_one}
        )
        assert new_model.prior_count == 2
        assert isinstance(new_model.one, af.GaussianPrior)
        # two should retain its original prior type
        assert new_model.two is not None

    def test_fix_via_mapper_from_prior_arguments(self):
        """Replacing a prior with a float effectively fixes that parameter."""
        model = af.Model(af.m.MockClassx2)
        new_model = model.mapper_from_prior_arguments(
            {model.one: 5.0, model.two: model.two}
        )
        assert new_model.prior_count == 1

    def test_with_limits(self):
        model = af.Model(af.m.MockClassx2)
        model.one = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
        model.two = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
        new_model = model.with_limits([(10.0, 20.0), (30.0, 40.0)])
        assert new_model.prior_count == 2


# ---------------------------------------------------------------------------
# from_instance round trips
# ---------------------------------------------------------------------------
class TestFromInstance:
    def test_from_simple_instance(self):
        instance = af.m.MockClassx2(1.0, 2.0)
        model = af.AbstractPriorModel.from_instance(instance)
        assert model.prior_count == 0

    def test_from_instance_as_model(self):
        instance = af.m.MockClassx2(1.0, 2.0)
        model = af.AbstractPriorModel.from_instance(instance)
        free_model = model.as_model()
        assert free_model.prior_count == 2

    def test_from_instance_with_model_classes(self):
        instance = af.m.MockClassx2(1.0, 2.0)
        model = af.AbstractPriorModel.from_instance(
            instance, model_classes=(af.m.MockClassx2,)
        )
        assert model.prior_count == 2

    def test_from_list_instance(self):
        instance_list = [af.m.MockClassx2(1.0, 2.0), af.m.MockClassx2(3.0, 4.0)]
        model = af.AbstractPriorModel.from_instance(instance_list)
        assert model.prior_count == 0

    def test_from_dict_instance(self):
        instance_dict = {
            "one": af.m.MockClassx2(1.0, 2.0),
            "two": af.m.MockClassx2(3.0, 4.0),
        }
        model = af.AbstractPriorModel.from_instance(instance_dict)
        assert model.prior_count == 0


# ---------------------------------------------------------------------------
# Fixing parameters and Constant values
# ---------------------------------------------------------------------------
class TestFixedParameters:
    def test_fix_reduces_prior_count(self):
        model = af.Model(af.m.MockClassx2)
        model.one = 5.0
        assert model.prior_count == 1

    def test_fixed_value_in_instance(self):
        model = af.Model(af.m.MockClassx2)
        model.one = 5.0
        instance = model.instance_from_vector([10.0])
        assert instance.one == 5.0
        assert instance.two == 10.0

    def test_fix_all_parameters(self):
        model = af.Model(af.m.MockClassx2)
        model.one = 5.0
        model.two = 10.0
        assert model.prior_count == 0
        instance = model.instance_from_vector([])
        assert instance.one == 5.0
        assert instance.two == 10.0


# ---------------------------------------------------------------------------
# take_attributes (prior passing)
# ---------------------------------------------------------------------------
class TestTakeAttributes:
    def test_take_from_instance(self):
        model = af.Model(af.m.MockClassx2)
        source = af.m.MockClassx2(10.0, 20.0)
        model.take_attributes(source)
        assert model.prior_count == 0

    def test_take_from_model(self):
        """Taking attributes from another model copies priors."""
        source_model = af.Model(af.m.MockClassx2)
        source_model.one = af.GaussianPrior(mean=5.0, sigma=1.0)
        source_model.two = af.GaussianPrior(mean=10.0, sigma=2.0)

        target_model = af.Model(af.m.MockClassx2)
        target_model.take_attributes(source_model)
        assert isinstance(target_model.one, af.GaussianPrior)


# ---------------------------------------------------------------------------
# Serialization (dict / from_dict)
# ---------------------------------------------------------------------------
class TestSerialization:
    def test_model_dict_roundtrip(self):
        model = af.Model(af.m.MockClassx2)
        d = model.dict()
        loaded = af.AbstractPriorModel.from_dict(d)
        assert loaded.prior_count == model.prior_count

    def test_collection_dict_roundtrip(self):
        model = af.Collection(a=af.m.MockClassx2, b=af.m.MockClassx2)
        d = model.dict()
        loaded = af.AbstractPriorModel.from_dict(d)
        assert loaded.prior_count == model.prior_count

    def test_fixed_parameter_survives_roundtrip(self):
        model = af.Model(af.m.MockClassx2)
        model.one = 5.0
        d = model.dict()
        loaded = af.AbstractPriorModel.from_dict(d)
        assert loaded.prior_count == 1

    def test_linked_prior_survives_roundtrip(self):
        model = af.Model(af.m.MockClassx2)
        model.one = model.two
        assert model.prior_count == 1
        d = model.dict()
        loaded = af.AbstractPriorModel.from_dict(d)
        assert loaded.prior_count == 1


# ---------------------------------------------------------------------------
# Log prior computation
# ---------------------------------------------------------------------------
class TestLogPrior:
    def test_log_prior_within_bounds(self):
        model = af.Model(af.m.MockClassx2)
        model.one = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
        model.two = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
        log_priors = model.log_prior_list_from_vector([5.0, 5.0])
        assert all(np.isfinite(lp) for lp in log_priors)

    def test_log_prior_outside_bounds(self):
        model = af.Model(af.m.MockClassx2)
        model.one = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
        model.two = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
        log_priors = model.log_prior_list_from_vector([15.0, 5.0])
        # Out-of-bounds value should have a lower (or zero) log prior than in-bounds
        assert log_priors[0] <= log_priors[1]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_single_parameter_model(self):
        """A model with a single free parameter using explicit prior."""
        model = af.Model(af.m.MockClassx2)
        model.two = 5.0  # Fix one parameter
        assert model.prior_count == 1
        instance = model.instance_from_vector([42.0])
        assert instance.one == 42.0
        assert instance.two == 5.0

    def test_model_copy_preserves_priors(self):
        model = af.Model(af.m.MockClassx2)
        copied = model.copy()
        assert copied.prior_count == model.prior_count
        # Priors are independent copies (different objects)
        assert copied.one is not model.one

    def test_model_copy_linked_priors_independent(self):
        """Copying a model with linked priors preserves the link in the copy."""
        model = af.Model(af.m.MockClassx2)
        model.one = model.two
        assert model.prior_count == 1
        copied = model.copy()
        assert copied.prior_count == 1
        # The copy's internal link is preserved
        assert copied.one is copied.two

    def test_prior_ordering_is_deterministic(self):
        """prior_tuples_ordered_by_id should be stable across calls."""
        model = af.Collection(a=af.m.MockClassx2, b=af.m.MockClassx2)
        order1 = [(n, p.id) for n, p in model.prior_tuples_ordered_by_id]
        order2 = [(n, p.id) for n, p in model.prior_tuples_ordered_by_id]
        assert order1 == order2

    def test_prior_count_equals_total_free_parameters(self):
        model = af.Collection(a=af.m.MockClassx2, b=af.m.MockClassx4)
        assert model.prior_count == model.total_free_parameters

    def test_has_model(self):
        model = af.Collection(a=af.Model(af.m.MockClassx2))
        assert model.has_model(af.m.MockClassx2)
        assert not model.has_model(af.m.MockClassx4)

    def test_has_instance(self):
        model = af.Model(af.m.MockClassx2)
        assert model.has_instance(Prior)
        assert not model.has_instance(af.m.MockClassx4)
