import random

import numpy as np
import pytest

import autofit as af


@pytest.fixture(name="initial_model")
def make_initial_model():
    return af.Model(af.m.MockClassx2)


class TestParamNames:
    def test_has_prior(self):
        prior_model = af.Model(af.m.MockClassx2)
        assert "one" == prior_model.name_for_prior(prior_model.one)


class ExtendedMockClass(af.m.MockClassx2):
    def __init__(self, one, two, three):
        super().__init__(one, two)
        self.three = three


# noinspection PyUnresolvedReferences
class TestRegression:
    def test_set_tuple_instance(self):
        mm = af.ModelMapper()
        mm.mock_cls = af.m.MockChildTuplex2

        assert mm.prior_count == 4

        mm.mock_cls.tup_0 = 0.0
        mm.mock_cls.tup_1 = 0.0

        assert mm.prior_count == 2

    def test_get_tuple_instances(self):
        mm = af.ModelMapper()
        mm.mock_cls = af.m.MockChildTuplex2

        assert isinstance(mm.mock_cls.tup_0, af.Prior)
        assert isinstance(mm.mock_cls.tup_1, af.Prior)

    def test_tuple_parameter(self, mapper):
        mapper.with_float = af.m.MockWithFloat
        mapper.with_tuple = af.m.MockWithTuple

        assert mapper.prior_count == 3

        mapper.with_tuple.tup_0 = mapper.with_float.value

        assert mapper.prior_count == 2

    def test_parameter_name_ordering(self):
        mm = af.ModelMapper()
        mm.one = af.m.MockClassRelativeWidth
        mm.two = af.m.MockClassRelativeWidth

        mm.one.one.id = mm.two.three.id + 1

        assert mm.model_component_and_parameter_names == [
            "one.two",
            "one.three",
            "two.one",
            "two.two",
            "two.three",
            "one.one",
        ]

    def test_parameter_name_list(self):
        mm = af.ModelMapper()
        mm.one = af.m.MockClassRelativeWidth
        mm.two = af.m.MockClassRelativeWidth

        assert mm.parameter_names == ["one", "two", "three", "one", "two", "three"]

    def test_parameter_name_distinction(self):
        mm = af.ModelMapper()
        mm.ls = af.Collection(
            [
                af.Model(af.m.MockClassRelativeWidth),
                af.Model(af.m.MockClassRelativeWidth),
            ]
        )
        assert mm.model_component_and_parameter_names == [
            "ls.0.one",
            "ls.0.two",
            "ls.0.three",
            "ls.1.one",
            "ls.1.two",
            "ls.1.three",
        ]

    def test__parameter_labels(self):
        mm = af.ModelMapper()
        mm.one = af.m.MockClassRelativeWidth
        mm.two = af.m.MockClassx2

        assert mm.parameter_labels == [
            "one_label",
            "two_label",
            "three_label",
            "one_label",
            "two_label",
        ]

    def test__superscripts(self):
        mm = af.ModelMapper()
        mm.one = af.m.MockClassRelativeWidth
        mm.two = af.m.MockClassx2NoSuperScript

        assert mm.superscripts == ["r", "r", "r", "two", "two"]

        model = af.Collection(group=mm)

        assert model.superscripts == ["r", "r", "r", "two", "two"]

    def test__superscript_overwrite_via_config(self):
        mm = af.ModelMapper()
        mm.one = af.m.MockClassRelativeWidth
        mm.two = af.m.MockClassx2NoSuperScript
        mm.three = af.m.MockClassx3

        assert mm.superscripts_overwrite_via_config == [
            "r",
            "r",
            "r",
            "",
            "",
            "",
            "",
            "",
        ]

    def test__parameter_labels_with_superscripts_latex(self):
        mm = af.ModelMapper()
        mm.one = af.m.MockClassRelativeWidth
        mm.two = af.m.MockClassx2NoSuperScript

        assert mm.parameter_labels_with_superscripts == [
            r"one_label^{\rm r}",
            r"two_label^{\rm r}",
            r"three_label^{\rm r}",
            r"one_label^{\rm two}",
            r"two_label^{\rm two}",
        ]

        assert mm.parameter_labels_with_superscripts_latex == [
            r"$one_label^{\rm r}$",
            r"$two_label^{\rm r}$",
            r"$three_label^{\rm r}$",
            r"$one_label^{\rm two}$",
            r"$two_label^{\rm two}$",
        ]

    def test_name_for_prior(self):
        ls = af.Collection(
            [
                af.m.MockClassRelativeWidth(1, 2, 3),
                af.Model(af.m.MockClassRelativeWidth),
            ]
        )
        assert ls.name_for_prior(ls[1].one) == "1_one"

    def test_tuple_parameter_float(self, mapper):
        mapper.with_float = af.m.MockWithFloat
        mapper.with_tuple = af.m.MockWithTuple

        mapper.with_float.value = 1.0

        assert mapper.prior_count == 2

        mapper.with_tuple.tup_0 = mapper.with_float.value

        assert mapper.prior_count == 1

        instance = mapper.instance_from_unit_vector([0.0])

        assert instance.with_float.value == 1
        assert instance.with_tuple.tup == (1.0, 0.0)


class TestModelingMapper:
    def test__argument_extraction(self):
        mapper = af.ModelMapper()
        mapper.mock_class = af.m.MockClassx2
        assert 1 == len(mapper.prior_model_tuples)

        assert len(mapper.prior_tuples_ordered_by_id) == 2

    def test_attribution(self):
        mapper = af.ModelMapper()

        mapper.mock_class = af.m.MockClassx2

        assert hasattr(mapper, "mock_class")
        assert hasattr(mapper.mock_class, "one")

    def test_tuple_arg(self):
        mapper = af.ModelMapper()

        mapper.mock_profile = af.m.MockClassx3TupleFloat

        assert 3 == len(mapper.prior_tuples_ordered_by_id)


class TestInstances:
    def test_attribute(self):
        mm = af.ModelMapper()
        mm.cls_1 = af.m.MockClassx2

        assert 1 == len(mm.prior_model_tuples)
        assert isinstance(mm.cls_1, af.Model)

    def test__instance_from_unit_vector(self):
        mapper = af.ModelMapper(mock_cls=af.m.MockClassx2Tuple)

        model_map = mapper.instance_from_unit_vector([1.0, 1.0])

        assert model_map.mock_cls.one_tuple == (1.0, 2.0)

    def test__instance_from_vector(self):
        mapper = af.ModelMapper(mock_cls=af.m.MockClassx2Tuple)

        model_map = mapper.instance_from_vector([1.0, 0.5])

        assert model_map.mock_cls.one_tuple == (1.0, 0.5)

    def test_inheritance(self):
        mapper = af.ModelMapper(mock_cls=af.m.MockChildTuplex2)

        model_map = mapper.instance_from_unit_vector([1.0, 1.0, 1.0, 1.0])

        assert model_map.mock_cls.tup == (1.0, 1.0)

    def test__multiple_classes(self):
        mapper = af.ModelMapper(
            mock_child_cls_0=af.m.MockChildTuplex3,
            mock_child_cls_1=af.m.MockChildTuplex2,
            mock_child_cls_2=af.m.MockChildTuplex2,
            mock_child_tuple=af.m.MockChildTuple,
            mock_child_cls_3=af.m.MockChildTuplex3,
        )

        model_map = mapper.instance_from_unit_vector(
            [0.5 for _ in range(len(mapper.prior_tuples_ordered_by_id))]
        )

        assert isinstance(model_map.mock_child_cls_1, af.m.MockChildTuplex2)
        assert isinstance(model_map.mock_child_cls_2, af.m.MockChildTuplex2)
        assert isinstance(model_map.mock_child_tuple, af.m.MockChildTuple)

        assert isinstance(model_map.mock_child_cls_0, af.m.MockChildTuplex3)
        assert isinstance(model_map.mock_child_cls_3, af.m.MockChildTuplex3)

    def test__in_order_of_class_constructor(self):
        mapper = af.ModelMapper(mock_cls_0=af.m.MockChildTuplex2)

        model_map = mapper.instance_from_unit_vector([0.25, 0.5, 0.75, 1.0])

        assert model_map.mock_cls_0.tup == (0.25, 0.5)
        assert model_map.mock_cls_0.one == 1.5
        assert model_map.mock_cls_0.two == 2.0

        mapper = af.ModelMapper(
            mock_cls_0=af.m.MockChildTuplex2,
            mock_cls_1=af.m.MockChildTuple,
            mock_cls_2=af.m.MockChildTuplex2,
        )

        model_map = mapper.instance_from_unit_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )

        assert model_map.mock_cls_0.tup == pytest.approx((0.1, 0.2))
        assert model_map.mock_cls_0.one == pytest.approx(0.6)
        assert model_map.mock_cls_0.two == pytest.approx(0.8)

        assert model_map.mock_cls_1.tup == pytest.approx((0.5, 0.6))

        assert model_map.mock_cls_2.tup == pytest.approx((0.7, 0.8))
        assert model_map.mock_cls_2.one == pytest.approx(1.8)
        assert model_map.mock_cls_2.two == pytest.approx(2.0)

    def test__check_order_for_different_unit_values(self):
        mapper = af.ModelMapper(
            mock_cls_0=af.m.MockChildTuplex2,
            mock_cls_1=af.m.MockChildTuple,
            mock_cls_2=af.m.MockChildTuplex2,
        )

        mapper.mock_cls_0.tup.tup_0 = af.UniformPrior(0.0, 1.0)
        mapper.mock_cls_0.tup.tup_1 = af.UniformPrior(0.0, 1.0)
        mapper.mock_cls_0.one = af.UniformPrior(0.0, 1.0)
        mapper.mock_cls_0.two = af.UniformPrior(0.0, 1.0)

        mapper.mock_cls_1.tup.tup_0 = af.UniformPrior(0.0, 1.0)
        mapper.mock_cls_1.tup.tup_1 = af.UniformPrior(0.0, 1.0)

        mapper.mock_cls_2.tup.tup_0 = af.UniformPrior(0.0, 1.0)
        mapper.mock_cls_2.tup.tup_1 = af.UniformPrior(0.0, 1.0)
        mapper.mock_cls_2.one = af.UniformPrior(0.0, 1.0)
        mapper.mock_cls_2.two = af.UniformPrior(0.0, 1.0)

        model_map = mapper.instance_from_unit_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )

        assert model_map.mock_cls_0.tup == pytest.approx((0.1, 0.2))
        assert model_map.mock_cls_0.one == pytest.approx(0.3)
        assert model_map.mock_cls_0.two == pytest.approx(0.4)

        assert model_map.mock_cls_1.tup == pytest.approx((0.5, 0.6))

        assert model_map.mock_cls_2.tup == pytest.approx((0.7, 0.8))
        assert model_map.mock_cls_2.one == pytest.approx(0.9)
        assert model_map.mock_cls_2.two == pytest.approx(1.0)

    def test__check_order_for_different_unit_values_and_set_priors_equal_to_one_another(
        self,
    ):
        mapper = af.ModelMapper(
            mock_cls_0=af.m.MockChildTuplex2,
            mock_cls_1=af.m.MockChildTuple,
            mock_cls_2=af.m.MockChildTuplex2,
        )

        mapper.mock_cls_0.tup.tup_0 = af.UniformPrior(0.0, 1.0)
        mapper.mock_cls_0.tup.tup_1 = af.UniformPrior(0.0, 1.0)
        mapper.mock_cls_0.one = af.UniformPrior(0.0, 1.0)
        mapper.mock_cls_0.two = af.UniformPrior(0.0, 1.0)

        mapper.mock_cls_1.tup.tup_0 = af.UniformPrior(0.0, 1.0)
        mapper.mock_cls_1.tup.tup_1 = af.UniformPrior(0.0, 1.0)

        mapper.mock_cls_2.tup.tup_0 = af.UniformPrior(0.0, 1.0)
        mapper.mock_cls_2.tup.tup_1 = af.UniformPrior(0.0, 1.0)
        mapper.mock_cls_2.one = af.UniformPrior(0.0, 1.0)
        mapper.mock_cls_2.two = af.UniformPrior(0.0, 1.0)

        mapper.mock_cls_0.one = mapper.mock_cls_0.two
        mapper.mock_cls_2.tup.tup_1 = mapper.mock_cls_1.tup.tup_1

        model_map = mapper.instance_from_unit_vector(
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )

        assert model_map.mock_cls_0.tup == pytest.approx((0.2, 0.3))
        assert model_map.mock_cls_0.one == pytest.approx(0.4)
        assert model_map.mock_cls_0.two == pytest.approx(0.4)

        assert model_map.mock_cls_1.tup == pytest.approx((0.5, 0.6))

        assert model_map.mock_cls_2.tup == pytest.approx((0.7, 0.6))
        assert model_map.mock_cls_2.one == pytest.approx(0.8)
        assert model_map.mock_cls_2.two == pytest.approx(0.9)

    def test__instance_from_vector__check_order(self):
        mapper = af.ModelMapper(
            mock_cls_0=af.m.MockChildTuplex2,
            mock_cls_1=af.m.MockChildTuple,
            mock_cls_2=af.m.MockChildTuplex2,
        )

        model_map = mapper.instance_from_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )

        assert model_map.mock_cls_0.tup == (0.1, 0.2)
        assert model_map.mock_cls_0.one == 0.3
        assert model_map.mock_cls_0.two == 0.4

        assert model_map.mock_cls_1.tup == (0.5, 0.6)

        assert model_map.mock_cls_2.tup == (0.7, 0.8)
        assert model_map.mock_cls_2.one == 0.9
        assert model_map.mock_cls_2.two == 1.0

    def test__instance_from_prior_medians(self):
        mapper = af.ModelMapper(mock_cls_0=af.m.MockChildTuplex2)

        model_map = mapper.instance_from_prior_medians()

        model_2 = mapper.instance_from_unit_vector([0.5, 0.5, 0.5, 0.5])

        assert model_map.mock_cls_0.tup == model_2.mock_cls_0.tup == (0.5, 0.5)
        assert model_map.mock_cls_0.one == model_2.mock_cls_0.one == 1.0
        assert model_map.mock_cls_0.two == model_2.mock_cls_0.two == 1.0

        mapper = af.ModelMapper(
            mock_cls_0=af.m.MockChildTuplex2,
            mock_cls_1=af.m.MockChildTuple,
            mock_cls_2=af.m.MockChildTuplex2,
        )

        model_map = mapper.instance_from_prior_medians()

        model_2 = mapper.instance_from_unit_vector(
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        )

        assert model_map.mock_cls_0.tup == model_2.mock_cls_0.tup == (0.5, 0.5)
        assert model_map.mock_cls_0.one == model_2.mock_cls_0.one == 1.0
        assert model_map.mock_cls_0.two == model_2.mock_cls_0.two == 1.0

        assert model_map.mock_cls_1.tup == model_2.mock_cls_1.tup == (0.5, 0.5)

        assert model_map.mock_cls_2.tup == model_2.mock_cls_2.tup == (0.5, 0.5)
        assert model_map.mock_cls_2.one == model_2.mock_cls_2.one == 1.0
        assert model_map.mock_cls_2.two == model_2.mock_cls_2.two == 1.0

    def test__from_prior_medians__one_model__set_one_parameter_to_another(self):
        mapper = af.ModelMapper(mock_cls_0=af.m.MockChildTuplex2)

        mapper.mock_cls_0.one = mapper.mock_cls_0.two

        model_map = mapper.instance_from_prior_medians()

        model_2 = mapper.instance_from_unit_vector([0.5, 0.5, 0.5])

        assert model_map.mock_cls_0.tup == model_2.mock_cls_0.tup == (0.5, 0.5)
        assert model_map.mock_cls_0.one == model_2.mock_cls_0.one == 1.0
        assert model_map.mock_cls_0.two == model_2.mock_cls_0.two == 1.0

    def test_log_prior_list_from_vector(self):
        mapper = af.ModelMapper()
        mapper.mock_class = af.Model(af.m.MockClassx2)
        mapper.mock_class.one = af.GaussianPrior(mean=1.0, sigma=2.0)
        mapper.mock_class.two = af.LogUniformPrior(lower_limit=1e-8, upper_limit=10.0)

        log_prior_list = mapper.log_prior_list_from_vector(vector=[0.0, 5.0])

        # Density-form log-priors (PyAutoLabs/PyAutoFit#1266):
        #   Gaussian(mean=1, sigma=2) at value=0:   -(0-1)**2 / (2*4) = -0.125
        #   LogUniform(...)            at value=5:  -log(5)
        assert log_prior_list[0] == -0.125
        assert log_prior_list[1] == pytest.approx(-np.log(5.0), 1.0e-12)

    def test_random_unit_vector_within_limits(self):
        mapper = af.ModelMapper()
        mapper.mock_class = af.Model(af.m.MockClassx2)

        random.seed(1)

        assert mapper.random_unit_vector_within_limits(
            lower_limit=0.0, upper_limit=1.0
        ) == pytest.approx([0.13436, 0.84743], 1.0e-4)

        assert mapper.random_unit_vector_within_limits(
            lower_limit=0.2, upper_limit=0.8
        ) == pytest.approx([0.65826, 0.35304], 1.0e-4)

    def test_random_vector_from_prior_within_limits(self):
        random.seed(1)

        mapper = af.ModelMapper()
        mapper.mock_class = af.Model(af.m.MockClassx2)

        vector = mapper.random_vector_from_priors_within_limits(
            lower_limit=0.499999, upper_limit=0.500001
        )

        assert vector == pytest.approx([0.5, 1.0], 1.0e-3)

        vector = mapper.random_vector_from_priors_within_limits(
            lower_limit=0.899999, upper_limit=0.900001
        )

        assert vector == pytest.approx([0.9, 1.8], 1.0e-3)

        vector = mapper.random_vector_from_priors_within_limits(
            lower_limit=0.2, upper_limit=0.3
        )

        assert vector == pytest.approx([0.268, 0.402], abs=0.1)

    def test_random_vector_from_prior(self):
        mapper = af.Collection(mock_class=af.Model(af.m.MockClassx2))

        random.seed(1)

        assert mapper.random_vector_from_priors == pytest.approx(
            [0.1343, 1.6948], 1.0e-2
        )
        assert mapper.random_vector_from_priors == pytest.approx(
            [0.7637, 0.5101], 1.0e-2
        )

        # By default, this seeded random will draw a value < -0.15, which is below the lower limit below. This
        # test ensures that this value is resampled to the next draw, which is above 0.15

        mapper = af.Collection(
            mock_class=af.Model(
                af.m.MockClassx2, one=af.UniformPrior(lower_limit=0.15, upper_limit=1.0)
            )
        )

        mapper.mock_class.one.lower_limit = 0.15

        assert mapper.random_vector_from_priors == pytest.approx(
            [0.5711, 0.8989], 1.0e-2
        )

    def test_vector_from_prior_medians(self):
        mapper = af.ModelMapper()
        mapper.mock_class = af.Model(af.m.MockClassx2)

        assert mapper.physical_values_from_prior_medians == [0.5, 1.0]


class TestUtility:
    def test_prior_prior_model_dict(self):
        mapper = af.ModelMapper(mock_class=af.m.MockClassx2)

        assert len(mapper.prior_prior_model_dict) == 2
        assert (
            mapper.prior_prior_model_dict[mapper.prior_tuples_ordered_by_id[0][1]].cls
            == af.m.MockClassx2
        )
        assert (
            mapper.prior_prior_model_dict[mapper.prior_tuples_ordered_by_id[1][1]].cls
            == af.m.MockClassx2
        )

    def test_name_for_prior(self):
        mapper = af.ModelMapper(mock_class=af.m.MockClassx2)

        assert mapper.name_for_prior(mapper.priors[0]) == "mock_class_one"
        assert mapper.name_for_prior(mapper.priors[1]) == "mock_class_two"


class TestPriorReplacement:
    def test_prior_replacement(self):
        mapper = af.ModelMapper(mock_class=af.m.MockClassx2)
        result = mapper.mapper_from_prior_means([10, 5])

        assert isinstance(result.mock_class.one, af.TruncatedGaussianPrior)
        assert {prior.id for prior in mapper.priors} == {
            prior.id for prior in result.priors
        }

    def test_replace_priors_with_gaussians_from_tuples(self):
        mapper = af.ModelMapper(mock_class=af.m.MockClassx2)
        result = mapper.mapper_from_prior_means([10, 5])

        assert isinstance(result.mock_class.one, af.TruncatedGaussianPrior)

    def test_replacing_priors_for_profile(self):
        mapper = af.ModelMapper(mock_class=af.m.MockClassx3TupleFloat)
        result = mapper.mapper_from_prior_means([10, 5, 5])

        assert isinstance(
            result.mock_class.one_tuple.unique_prior_tuples[0][1], af.TruncatedGaussianPrior
        )
        assert isinstance(
            result.mock_class.one_tuple.unique_prior_tuples[1][1], af.TruncatedGaussianPrior
        )
        assert isinstance(result.mock_class.two, af.TruncatedGaussianPrior)

    def test_replace_priors_for_two_classes(self):
        mapper = af.ModelMapper(one=af.m.MockClassx2, two=af.m.MockClassx2)

        result = mapper.mapper_from_prior_means([1, 2, 3, 4])

        assert result.one.one.mean == 1
        assert result.one.two.mean == 2
        assert result.two.one.mean == 3
        assert result.two.two.mean == 4

    def test__mapper_from_uniform_floats(self):
        mapper = af.ModelMapper(mock_class=af.m.MockClassx2)
        result = mapper.mapper_from_uniform_floats([10, 5], b=1.0)

        assert isinstance(result.mock_class.one, af.UniformPrior)
        assert {prior.id for prior in mapper.priors} == {
            prior.id for prior in result.priors
        }


class TestArguments:
    def test_same_argument_name(self):
        mapper = af.ModelMapper()

        mapper.one = af.Model(af.m.MockClassx2)
        mapper.two = af.Model(af.m.MockClassx2)

        instance = mapper.instance_from_vector([0.1, 0.2, 0.3, 0.4])

        assert instance.one.one == 0.1
        assert instance.one.two == 0.2
        assert instance.two.one == 0.3
        assert instance.two.two == 0.4


class TestIndependentPriorModel:
    def test_associate_prior_model(self):
        prior_model = af.Model(af.m.MockClassx2)

        mapper = af.ModelMapper()

        mapper.prior_model = prior_model

        assert len(mapper.prior_model_tuples) == 1

        instance = mapper.instance_from_vector([0.1, 0.2])

        assert instance.prior_model.one == 0.1
        assert instance.prior_model.two == 0.2


@pytest.fixture(name="list_prior_model")
def make_list_prior_model():
    return af.Collection([af.Model(af.m.MockClassx2), af.Model(af.m.MockClassx2)])


class TestListPriorModel:
    def test_instance_from_vector(self, list_prior_model):
        mapper = af.ModelMapper()
        mapper.list = list_prior_model

        instance = mapper.instance_from_vector([0.1, 0.2, 0.3, 0.4])

        assert isinstance(instance.list, af.ModelInstance)
        print(instance.list.items)
        assert len(instance.list) == 2
        assert instance.list[0].one == 0.1
        assert instance.list[0].two == 0.2
        assert instance.list[1].one == 0.3
        assert instance.list[1].two == 0.4

    def test_prior_results_for_gaussian_tuples(self, list_prior_model):
        mapper = af.ModelMapper()
        mapper.list = list_prior_model

        gaussian_mapper = mapper.mapper_from_prior_means(
            [1, 2, 3, 4]
        )

        assert len(gaussian_mapper.list) == 2
        assert gaussian_mapper.list[0].one.mean == 1
        assert gaussian_mapper.list[0].two.mean == 2
        assert gaussian_mapper.list[1].one.mean == 3
        assert gaussian_mapper.list[1].two.mean == 4
        assert gaussian_mapper.list[0].one.sigma == 1
        assert gaussian_mapper.list[0].two.sigma == 2
        assert gaussian_mapper.list[1].one.sigma == 1
        assert gaussian_mapper.list[1].two.sigma == 2

    def test_prior_results_for_gaussian_tuples__include_override_from_width_file(
        self, list_prior_model
    ):
        mapper = af.ModelMapper()
        mapper.list = list_prior_model

        gaussian_mapper = mapper.mapper_from_prior_means(
            [1, 2, 3, 4]
        )

        assert len(gaussian_mapper.list) == 2
        assert gaussian_mapper.list[0].one.mean == 1
        assert gaussian_mapper.list[0].two.mean == 2
        assert gaussian_mapper.list[1].one.mean == 3
        assert gaussian_mapper.list[1].two.mean == 4
        assert gaussian_mapper.list[0].one.sigma == 1
        assert gaussian_mapper.list[0].two.sigma == 2
        assert gaussian_mapper.list[1].one.sigma == 1
        assert gaussian_mapper.list[1].two.sigma == 2

    def test_automatic_boxing(self):
        mapper = af.ModelMapper()
        mapper.list = [af.Model(af.m.MockClassx2), af.Model(af.m.MockClassx2)]

        assert isinstance(mapper.list, af.Collection)


@pytest.fixture(name="mock_with_instance")
def make_mock_with_instance():
    mock_with_instance = af.Model(af.m.MockClassx2)
    mock_with_instance.one = 3.0
    return mock_with_instance


class Testinstance:
    def test__instance_prior_count(self, mock_with_instance):
        mapper = af.ModelMapper()
        mapper.mock_class = mock_with_instance

        assert len(mapper.unique_prior_tuples) == 1

    def test__retrieve_instances(self, mock_with_instance):
        assert len(mock_with_instance.instance_tuples) == 1

    def test_instance_prior_reconstruction(self, mock_with_instance):
        mapper = af.ModelMapper()
        mapper.mock_class = mock_with_instance

        instance = mapper.instance_for_arguments({mock_with_instance.two: 0.5})

        assert instance.mock_class.one == 3
        assert instance.mock_class.two == 0.5

    def test__instance_in_config(self):
        mapper = af.ModelMapper()

        mock_with_instance = af.Model(af.m.MockClassx2Instance, one=3)

        mapper.mock_class = mock_with_instance

        instance = mapper.instance_for_arguments({mock_with_instance.two: 0.5})

        assert instance.mock_class.one == 3
        assert instance.mock_class.two == 0.5

    def test__set_float(self):
        prior_model = af.Model(af.m.MockClassx2)
        prior_model.one = 3
        prior_model.two = 4.0
        assert prior_model.one == 3
        assert prior_model.two == 4.0

    def test__list_prior_model_instances(self, mapper):
        prior_model = af.Model(af.m.MockClassx2)
        prior_model.one = 3.0
        prior_model.two = 4.0

        mapper.mock_list = [prior_model]
        assert isinstance(mapper.mock_list, af.Collection)
        assert len(mapper.instance_tuples) == 2

    def test__set_for_tuple_prior(self):
        prior_model = af.Model(af.m.MockChildTuplex3)
        prior_model.tup_0 = 1.0
        prior_model.tup_1 = 2.0
        prior_model.one = 1.0
        prior_model.two = 1.0
        prior_model.three = 1.0
        instance = prior_model.instance_for_arguments({})
        assert instance.tup == (1.0, 2.0)


@pytest.fixture(name="mock_config")
def make_mock_config():
    return


@pytest.fixture(name="mapper_with_one")
def make_mapper_with_one():
    mapper = af.ModelMapper()
    mapper.one = af.Model(af.m.MockClassx2)
    return mapper


@pytest.fixture(name="mapper_with_list")
def make_mapper_with_list():
    mapper = af.ModelMapper()
    mapper.list = [af.Model(af.m.MockClassx2), af.Model(af.m.MockClassx2)]
    return mapper


class TestGaussianWidthConfig:
    def test_relative_widths(self, mapper):
        mapper.relative_width = af.m.MockClassRelativeWidth
        new_mapper = mapper.mapper_from_prior_means([1, 1, 1])

        assert new_mapper.relative_width.one.mean == 1.0
        assert new_mapper.relative_width.one.sigma == 0.1

        assert new_mapper.relative_width.two.mean == 1.0
        assert new_mapper.relative_width.two.sigma == 0.5

        assert new_mapper.relative_width.three.mean == 1.0
        assert new_mapper.relative_width.three.sigma == 1.0

    def test_prior_classes(self, mapper_with_one):
        assert mapper_with_one.prior_class_dict == {
            mapper_with_one.one.one: af.m.MockClassx2,
            mapper_with_one.one.two: af.m.MockClassx2,
        }

    def test_prior_classes_list(self, mapper_with_list):
        assert mapper_with_list.prior_class_dict == {
            mapper_with_list.list[0].one: af.m.MockClassx2,
            mapper_with_list.list[0].two: af.m.MockClassx2,
            mapper_with_list.list[1].one: af.m.MockClassx2,
            mapper_with_list.list[1].two: af.m.MockClassx2,
        }

    def test_no_override(self):
        mapper = af.ModelMapper()

        mapper.one = af.Model(af.m.MockClassx2)

        af.ModelMapper()

        assert mapper.one is not None
