import copy

import pytest

import autofit as af


@pytest.fixture(name="instance_prior_model")
def make_instance_prior_model():
    instance = af.m.MockClassx2(1.0, 2.0)
    return af.AbstractPriorModel.from_instance(instance)


@pytest.fixture(name="list_prior_model")
def make_list_prior_model():
    instance = [af.m.MockClassx2(1.0, 2.0)]
    return af.AbstractPriorModel.from_instance(instance)


@pytest.fixture(name="complex_prior_model")
def make_complex_prior_model():
    instance = af.m.MockComplexClass(af.m.MockClassx2(1.0, 2.0))
    return af.AbstractPriorModel.from_instance(instance)


def test_class_assertion():
    with pytest.raises(AssertionError):
        af.Model("Hello")


class TestAsModel:
    def test_instance(self, instance_prior_model):
        model = instance_prior_model.as_model()
        assert model.prior_count == 2

    def test_complex(self, complex_prior_model):
        assert complex_prior_model.prior_count == 0
        model = complex_prior_model.as_model()
        assert model.prior_count == 2
        assert model.simple.prior_count == 2

    def test_galaxies(self):
        galaxies = af.ModelInstance()
        galaxies.one = af.m.MockComponents()
        instance = af.ModelInstance()
        instance.galaxies = galaxies
        model = instance.as_model(model_classes=(af.m.MockComponents,))
        assert model.prior_count == 1


class TestFromInstance:
    def test_model_mapper(self):
        instance = af.ModelInstance()
        instance.simple = af.m.MockClassx2(1.0, 2.0)

        result = af.AbstractPriorModel.from_instance(instance)

        assert isinstance(result, af.ModelMapper)

    def test_with_model_classes(self):
        instance = af.m.MockComplexClass(af.m.MockClassx2(1.0, 2.0))
        model = af.AbstractPriorModel.from_instance(
            instance, model_classes=(af.m.MockClassx2,)
        )
        assert model.prior_count == 2

    def test_list_with_model_classes(self):
        instance = [
            af.m.MockClassx2(1.0, 2.0),
            af.m.MockComplexClass(af.m.MockClassx2(1.0, 2.0)),
        ]
        model = af.AbstractPriorModel.from_instance(
            instance, model_classes=(af.m.MockComplexClass,)
        )

        assert model.prior_count == 2
        assert model[0].prior_count == 0
        assert model[1].prior_count == 2

    def test_dict_with_model_classes(self):
        instance = {
            "one": af.m.MockClassx2(1.0, 2.0),
            "two": af.m.MockComplexClass(af.m.MockClassx2(1.0, 2.0)),
        }
        model = af.AbstractPriorModel.from_instance(
            instance, model_classes=(af.m.MockComplexClass,)
        )

        assert model.prior_count == 2
        assert model[0].prior_count == 0
        assert model[1].prior_count == 2

        assert model.one.one == 1.0
        assert model.one.two == 2.0

        assert isinstance(model.two.simple.one, af.Prior)
        assert isinstance(model.two.simple.two, af.Prior)

    def test_instance(self, instance_prior_model):
        assert instance_prior_model.cls == af.m.MockClassx2
        assert instance_prior_model.prior_count == 0
        assert instance_prior_model.one == 1.0
        assert instance_prior_model.two == 2.0

        new_instance = instance_prior_model.instance_for_arguments({})
        assert isinstance(new_instance, af.m.MockClassx2)
        assert new_instance.one == 1.0
        assert new_instance.two == 2.0

    def test_complex(self, complex_prior_model):
        assert complex_prior_model.cls == af.m.MockComplexClass
        assert complex_prior_model.prior_count == 0
        assert isinstance(complex_prior_model.simple, af.Model)
        assert complex_prior_model.simple.cls == af.m.MockClassx2
        assert complex_prior_model.simple.one == 1.0

        new_instance = complex_prior_model.instance_for_arguments({})
        assert isinstance(new_instance, af.m.MockComplexClass)
        assert isinstance(new_instance.simple, af.m.MockClassx2)
        assert new_instance.simple.one == 1.0

    def test_list(self, list_prior_model):
        assert isinstance(list_prior_model, af.Collection)
        assert isinstance(list_prior_model[0], af.Model)
        assert list_prior_model[0].one == 1.0

    def test_dict(self):
        instance = {"simple": af.m.MockClassx2(1.0, 2.0)}
        prior_model = af.AbstractPriorModel.from_instance(instance)
        assert isinstance(prior_model, af.Collection)
        assert isinstance(prior_model.simple, af.Model)
        assert prior_model.simple.one == 1.0

        new_instance = prior_model.instance_for_arguments({})
        assert isinstance(new_instance.simple, af.m.MockClassx2)

        prior_model = af.AbstractPriorModel.from_instance(new_instance)
        assert isinstance(prior_model, af.Collection)
        assert isinstance(prior_model.simple, af.Model)
        assert prior_model.simple.one == 1.0


class TestSum:
    def test_add_prior_models(self):
        mock_cls_0 = af.Model(af.m.MockChildTuplex2)
        mock_cls_1 = af.Model(af.m.MockChildTuplex2)

        mock_cls_0.one = 1.0
        mock_cls_1.two = 0.0

        result = mock_cls_0 + mock_cls_1

        assert isinstance(result, af.Model)
        assert result.cls == af.m.MockChildTuplex2
        assert isinstance(result.one, af.Prior)
        assert isinstance(result.two, af.Prior)

    def test_fail_for_mismatch(self):
        mock_cls_0 = af.Model(af.m.MockChildTuplex2)
        mock_cls_1 = af.Model(af.m.MockChildTuplex3)

        with pytest.raises(TypeError):
            mock_cls_0 + mock_cls_1

    def test_add_children(self):
        mock_components_1 = af.Model(
            af.m.MockComponents,
            components_0=af.Collection(mock_cls_0=af.m.MockChildTuplex2),
            components_1=af.Collection(mock_cls_2=af.m.MockChildTuplex3),
        )
        mock_components_2 = af.Model(
            af.m.MockComponents,
            components_0=af.Collection(mock_cls_1=af.m.MockChildTuplex2),
            components_1=af.Collection(mock_cls_3=af.m.MockChildTuplex3),
        )

        result = mock_components_1 + mock_components_2

        assert (
            result.components_0.mock_cls_0 == mock_components_1.components_0.mock_cls_0
        )
        assert (
            result.components_0.mock_cls_1 == mock_components_2.components_0.mock_cls_1
        )

        assert (
            result.components_1.mock_cls_2 == mock_components_1.components_1.mock_cls_2
        )
        assert (
            result.components_1.mock_cls_3 == mock_components_2.components_1.mock_cls_3
        )

    def test_prior_model_override(self):
        mock_components_1 = af.Model(
            af.m.MockComponents,
            components_0=af.Collection(light=af.m.MockChildTuplex2()),
            components_1=af.Collection(mass=af.m.MockChildTuplex3),
        )
        mock_components_2 = af.Model(
            af.m.MockComponents,
            components_0=af.Collection(light=af.m.MockChildTuplex2),
            components_1=af.Collection(mass=af.m.MockChildTuplex3()),
        )

        result = mock_components_1 + mock_components_2

        assert result.components_1.mass == mock_components_1.components_1.mass
        assert result.components_0.light == mock_components_2.components_0.light


class TestFloatAnnotation:
    # noinspection PyUnresolvedReferences
    def test_prior_linking(self):
        mapper = af.ModelMapper()
        mapper.a = af.m.MockClassx2
        mapper.b = af.m.MockClassx2

        assert mapper.prior_count == 4

        mapper.a.one = mapper.b.one

        assert mapper.prior_count == 3

        mapper.a.two = mapper.b.two

        assert mapper.prior_count == 2

        mapper.a.one = mapper.a.two
        mapper.b.one = mapper.b.two

        assert mapper.prior_count == 1


class TestHashing:
    def test_is_hashable(self):
        assert hash(af.AbstractPriorModel()) is not None
        assert hash(af.Model(af.m.MockClassx2)) is not None
        assert (
            hash(af.AnnotationPriorModel(af.m.MockClassx2, af.m.MockClassx2, "one"))
            is not None
        )


class StringDefault:
    def __init__(self, value="a string"):
        self.value = value


class TestStringArguments:
    def test_string_default(self):
        prior_model = af.Model(StringDefault)
        assert prior_model.prior_count == 0

        assert prior_model.instance_for_arguments({}).value == "a string"


class TestPriorModelArguments:
    def test_list_arguments(self):
        prior_model = af.Model(af.m.MockListClass)

        assert prior_model.prior_count == 0

        prior_model = af.Model(af.m.MockListClass, ls=[af.m.MockClassx2])

        assert prior_model.prior_count == 2

        prior_model = af.Model(
            af.m.MockListClass, ls=[af.m.MockClassx2, af.m.MockClassx2]
        )

        assert prior_model.prior_count == 4

    def test_float_argument(self):
        prior = af.UniformPrior(0.5, 2.0)
        prior_model = af.Model(af.m.MockComponents, parameter=prior)

        assert prior_model.prior_count == 1
        assert prior_model.priors[0] is prior

        prior_model = af.Model(af.m.MockComponents, parameter=4.0)
        assert prior_model.prior_count == 0
        assert prior_model.parameter == 4.0

        instance = prior_model.instance_for_arguments({})
        assert instance.parameter == 4.0

    def test_arbitrary_keyword_arguments(self):
        prior_model = af.Model(
            af.m.MockComponents,
            mock_cls_0=af.m.MockChildTuplex2,
            mock_cls_1=af.m.MockChildTuplex3,
        )
        assert prior_model.prior_count == 10
        instance = prior_model.instance_from_unit_vector(
            [0.5] * prior_model.prior_count
        )
        assert isinstance(instance.mock_cls_0, af.m.MockChildTuplex2)
        assert isinstance(instance.mock_cls_1, af.m.MockChildTuplex3)


class TestCase:
    def test_complex_class(self):
        prior_model = af.Model(af.m.MockComplexClass)

        assert hasattr(prior_model, "simple")
        assert prior_model.simple.prior_count == 2
        assert prior_model.prior_count == 2

    def test_create_instance(self):
        mapper = af.ModelMapper()
        mapper.complex = af.m.MockComplexClass

        instance = mapper.instance_from_unit_vector([1.0, 0.0])

        assert instance.complex.simple.one == 1.0
        assert instance.complex.simple.two == 0.0

    def test_instantiate_with_list_arguments(self):
        mapper = af.ModelMapper()
        mapper.list_object = af.Model(
            af.m.MockListClass, ls=[af.m.MockClassx2, af.m.MockClassx2]
        )

        assert len(mapper.list_object.ls) == 2

        assert mapper.list_object.prior_count == 4
        assert mapper.prior_count == 4

        instance = mapper.instance_from_unit_vector([0.1, 0.2, 0.3, 0.4])

        assert len(instance.list_object.ls) == 2
        assert instance.list_object.ls[0].one == pytest.approx(0.1)
        assert instance.list_object.ls[0].two == pytest.approx(0.4)
        assert instance.list_object.ls[1].one == pytest.approx(0.3)
        assert instance.list_object.ls[1].two == pytest.approx(0.8)

    def test_mix_instances_and_models(self):
        mapper = af.ModelMapper()
        mapper.list_object = af.Model(
            af.m.MockListClass, ls=[af.m.MockClassx2, af.m.MockClassx2(1, 2)]
        )

        assert mapper.prior_count == 2

        instance = mapper.instance_from_unit_vector([0.1, 0.2])

        assert len(instance.list_object.ls) == 2
        assert instance.list_object.ls[0].one == pytest.approx(0.1)
        assert instance.list_object.ls[0].two == pytest.approx(0.4)
        assert instance.list_object.ls[1].one == pytest.approx(1)
        assert instance.list_object.ls[1].two == pytest.approx(2)


class TestCollectionPriorModel:
    def test_keyword_arguments(self):
        prior_model = af.Collection(one=af.m.MockClassx2, two=af.m.MockClassx2(1, 2))

        assert len(prior_model.direct_prior_model_tuples) == 1
        assert len(prior_model) == 2

        instance = prior_model.instance_for_arguments(
            {prior_model.one.one: 0.1, prior_model.one.two: 0.2}
        )

        assert instance.one.one == 0.1
        assert instance.one.two == 0.2

        assert instance.two.one == 1
        assert instance.two.two == 2

    def test_mix_instances_in_grouped_list_prior_model(self):
        prior_model = af.Collection([af.m.MockClassx2, af.m.MockClassx2(1, 2)])

        assert len(prior_model.direct_prior_model_tuples) == 1
        assert prior_model.prior_count == 2

        mapper = af.ModelMapper()
        mapper.ls = prior_model

        instance = mapper.instance_from_unit_vector([0.1, 0.2])

        assert len(instance.ls) == 2

        assert instance.ls[0].one == pytest.approx(0.1)
        assert instance.ls[0].two == pytest.approx(0.4)
        assert instance.ls[1].one == pytest.approx(1)
        assert instance.ls[1].two == pytest.approx(2)

        assert len(prior_model.prior_class_dict) == 2

    def test_list_in_grouped_list_prior_model(self):
        prior_model = af.Collection([[af.m.MockClassx2]])

        assert len(prior_model.direct_prior_model_tuples) == 1
        assert prior_model.prior_count == 2

    def test_list_prior_model_with_dictionary(self, simple_model):
        assert isinstance(simple_model.simple, af.Model)

    def test_override_with_instance(self, simple_model):
        simple_instance = af.m.MockClassx2(1, 2)

        simple_model.simple = simple_instance

        assert len(simple_model) == 1
        assert simple_model.simple == simple_instance

    def test_names_of_priors(self):
        collection = af.Collection([af.UniformPrior(), af.UniformPrior()])
        assert collection.name_for_prior(collection[0]) == "0"


@pytest.fixture(name="simple_model")
def make_simple_model():
    return af.Collection({"simple": af.m.MockClassx2})


class TestCopy:
    def test_simple(self, simple_model):
        assert simple_model.prior_count > 0
        assert copy.deepcopy(simple_model).prior_count == simple_model.prior_count

    def test_embedded(self, simple_model):
        model = af.Collection(simple=simple_model)
        assert copy.deepcopy(model).prior_count == model.prior_count

    def test_circular(self):
        one = af.Model(af.m.MockClassx2)

        one.one = af.Model(af.m.MockClassx2)
        one.one.one = one

        # noinspection PyUnresolvedReferences
        assert one.prior_count == one.one.prior_count
        assert copy.deepcopy(one).prior_count == one.prior_count


def test_composition(simple_model):
    assert simple_model.composition == ["simple.one", "simple.two"]
