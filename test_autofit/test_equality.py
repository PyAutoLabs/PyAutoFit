from copy import deepcopy

import pytest

import autofit as af


@pytest.fixture(name="prior_model")
def make_prior_model():
    return af.Model(af.m.MockClassx2Tuple)


class TestCase:
    def test_prior_model(self, prior_model):
        prior_model_copy = deepcopy(prior_model)
        assert prior_model == prior_model_copy

        prior_model_copy.centre_0 = af.UniformPrior()

        assert prior_model != prior_model_copy

    def test_list_prior_model(self, prior_model):
        list_prior_model = af.Collection([prior_model])
        list_prior_model_copy = deepcopy(list_prior_model)
        assert list_prior_model == list_prior_model_copy

        list_prior_model[0].centre_0 = af.UniformPrior()

        assert list_prior_model != list_prior_model_copy

    def test_model_mapper(self, prior_model):
        model_mapper = af.ModelMapper()
        model_mapper.prior_model = prior_model
        model_mapper_copy = deepcopy(model_mapper)

        assert model_mapper == model_mapper_copy

        model_mapper.prior_model.centre_0 = af.UniformPrior()

        assert model_mapper != model_mapper_copy

    def test_non_trivial_equality(self):
        mock_components = af.Model(
            af.m.MockComponents,
            components_0=af.Collection(mock_cls_0=af.m.MockChildTuplex2),
            components_1=af.Collection(
                mock_cls_2=af.m.MockChildTuplex3
            ),
        )

        model_mapper = af.ModelMapper()
        model_mapper.mock_components = mock_components
        model_mapper_copy = deepcopy(model_mapper)

        assert model_mapper == model_mapper_copy

        model_mapper.mock_components.components_0.tup_0 = af.UniformPrior()

        assert model_mapper != model_mapper_copy

    def test_model_instance_equality(self):
        model_instance = af.ModelInstance()
        model_instance.profile = af.m.MockClassx2Tuple()
        model_instance_copy = deepcopy(model_instance)

        assert model_instance == model_instance_copy

        model_instance.profile.centre = (1.0, 2.0)

        assert model_instance != model_instance_copy
