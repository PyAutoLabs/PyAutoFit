from pathlib import Path

import pytest

import autofit as af
from autofit import SearchOutput


@pytest.fixture(name="aggregator")
def make_path_aggregator(session):
    fits = [
        af.db.Fit(id="complete", is_complete=True),
        af.db.Fit(id="incomplete", is_complete=False),
    ]
    for i, fit in enumerate(fits):
        fit["dataset"] = {"name": "dataset"}
        fit["pipeline"] = f"pipeline{i}"
    session.add_all(fits)
    session.flush()
    return af.Aggregator(session)


@pytest.fixture(name="directory")
def make_directory():
    return Path(__file__).parent


@pytest.fixture(name="aggregator_directory")
def make_aggregator_directory(directory):
    directory = Path(__file__).resolve().parent

    return directory.parent / "tools" / "files" / "aggregator"


@pytest.fixture(name="search_output")
def make_search_output(directory):
    return SearchOutput(directory / "search_output")
