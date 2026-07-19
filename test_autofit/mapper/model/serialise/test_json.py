import itertools
import os
from pathlib import Path
import pytest
import json

import autofit as af
from autonerves.dictable import from_dict


@pytest.fixture(name="collection_dict")
def make_collection_dict(model_dict):
    return {
        "arguments": {"gaussian": model_dict},
        "type": "collection",
    }


@pytest.fixture(name="model")
def make_model():
    return af.Model(af.ex.Gaussian, centre=af.UniformPrior(upper_limit=2.0))


@pytest.fixture(autouse=True)
def reset_prior_id():
    af.Prior._ids = itertools.count()


class TestTuple:
    def test_tuple_prior(self):
        tuple_prior = af.TuplePrior()
        tuple_prior.tup_0 = 0
        tuple_prior.tup_1 = 1

        result = af.Model.from_dict(tuple_prior.dict())
        assert isinstance(result, af.TuplePrior)

    def test_model_with_tuple(self):
        tuple_model = af.Model(af.m.MockWithTuple)
        tuple_model.instance_from_prior_medians()
        model_dict = tuple_model.dict()

        model = af.Model.from_dict(model_dict)
        instance = model.instance_from_prior_medians()
        assert instance.tup == (0.5, 0.5)


class TestFromDict:
    def test_model_from_dict(self, model_dict):
        model = af.Model.from_dict(model_dict)
        assert model.cls == af.ex.Gaussian
        assert model.prior_count == 3
        assert model.centre.upper_limit == 2.0

    def test_instance_from_dict(self, instance_dict):
        instance = from_dict(instance_dict)
        assert isinstance(instance, af.ex.Gaussian)
        assert instance.centre == 0.0
        assert instance.normalization == 0.1
        assert instance.sigma == 0.01

    def test_collection_from_dict(self, collection_dict):
        collection = af.Model.from_dict(collection_dict)
        assert isinstance(collection, af.Collection)
        assert len(collection) == 1


class TestToDict:
    def test_model_priors(self, model, model_dict, remove_ids):
        assert remove_ids(model.dict()) == model_dict

    def test_model_floats(self, instance_dict):
        model = af.Model(af.ex.Gaussian, centre=0.0, normalization=0.1, sigma=0.01)

        assert model.dict() == instance_dict

    def test_collection(self, model, collection_dict, remove_ids):
        collection = af.Collection(gaussian=model)
        assert remove_ids(collection.dict()) == collection_dict

    def test_collection_instance(self, instance_dict):
        collection = af.Collection(gaussian=af.ex.Gaussian())
        assert collection.dict() == {
            "type": "collection",
            "arguments": {
                "gaussian": {
                    "type": "instance",
                    "class_path": "autofit.example.model.Gaussian",
                    "arguments": {"sigma": 0.01, "normalization": 0.1, "centre": 0.0},
                }
            },
        }

    @pytest.mark.parametrize(
        "path",
        [
            "autofit.example.different.Gaussian",
            "autofit.example.model.Gossian",
        ],
    )
    def test_bad_class_path(self, model_dict, path):
        model_dict["class_path"] = path

        model = af.Model.from_dict(model_dict)
        assert isinstance(model, af.Collection)
        assert model.centre.upper_limit == 2.0


class TestFromJson:
    def test__from_json(self, model_dict):
        model = af.Model.from_dict(model_dict)

        model_file = Path(__file__).parent / "model.json"

        try:
            os.remove(model_file)
        except OSError:
            pass

        with open(model_file, "w+") as f:
            json.dump(model.dict(), f, indent=4)

        model = af.Model.from_json(file=model_file)

        assert model.cls == af.ex.Gaussian
        assert model.prior_count == 3
        assert model.centre.upper_limit == 2.0

        os.remove(model_file)
