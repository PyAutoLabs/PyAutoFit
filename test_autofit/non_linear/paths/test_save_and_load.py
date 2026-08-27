import numpy as np
import pytest

import autofit as af

# `astropy` ships via the `[optional]` extras; skip the module there
# rather than fail collection with `No module named 'astropy'`.
pytest.importorskip(
    "astropy",
    reason="requires astropy (installed via the [optional] extras)",
)

from astropy.io import fits


@pytest.fixture(name="dictionary")
def make_dictionary():
    return {"hello": "world"}


@pytest.fixture(name="directory_paths")
def make_directory_paths():
    return af.DirectoryPaths()


@pytest.fixture(name="database_paths")
def make_database_paths(session):
    return af.DatabasePaths(session)


def test_directory_json(dictionary, directory_paths):
    directory_paths.save_json("test", dictionary)
    assert directory_paths.load_json("test") == dictionary


def test_database_json(dictionary, database_paths):
    database_paths.save_json("test", dictionary)
    assert database_paths.load_json("test") == dictionary


@pytest.fixture(name="array")
def make_array():
    return np.array([1, 2, 3])


def test_directory_array(array, directory_paths):
    directory_paths.save_array("test", array)
    assert (directory_paths.load_array("test") == array).all()


def test_database_array(array, database_paths):
    database_paths.save_array("test", array)
    assert (database_paths.load_array("test") == array).all()
