import importlib.util

import pytest

from autofit import SearchOutput
from autofit.database.aggregator.scrape import _add_files


# `astropy` ships via the `[optional]` extras. Envs installed without them
# must skip rather than fail with `No module named 'astropy'`.
requires_astropy = pytest.mark.skipif(
    importlib.util.find_spec("astropy") is None,
    reason="requires astropy (installed via the [optional] extras)",
)


class MockFit:
    def __init__(self):
        self.jsons = {}

    id = "id"

    def set_json(self, name, json):
        self.jsons[name] = json

    def __setitem__(self, key, value):
        pass

    def set_pickle(self, key, value):
        pass

    def set_array(self, key, value):
        pass

    def set_fits(self, key, value):
        pass


@pytest.fixture(name="fit")
def make_fit(directory):
    fit = MockFit()
    _add_files(
        fit=fit,
        item=SearchOutput(directory / "search_output"),
    )
    return fit


@requires_astropy
def test_add_files(fit):
    assert fit.jsons["model"] == {
        "class_path": "autofit.example.model.Gaussian",
        "type": "model",
        "arguments": {
            "centre": {
                "type": "Gaussian",
                "id": 0,
                "mean": 1.0,
                "sigma": 1.0,
            },
            "normalization": {
                "type": "Gaussian",
                "id": 1,
                "mean": 1.0,
                "sigma": 1.0,
            },
            "sigma": {
                "type": "Gaussian",
                "id": 2,
                "mean": 1.0,
                "sigma": 1.0,
            },
        },
    }


@requires_astropy
def test_add_recursive(fit):
    assert fit.jsons["directory.example"] == {
        "hello": "world",
    }
