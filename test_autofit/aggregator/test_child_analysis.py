import importlib.util
from pathlib import Path

import pytest

from autofit import SearchOutput
from autofit.aggregator import Aggregator
import autofit as af


# `astropy` ships via the `[optional]` extras. Envs installed without them
# must skip rather than fail with `No module named 'astropy'`.
requires_astropy = pytest.mark.skipif(
    importlib.util.find_spec("astropy") is None,
    reason="requires astropy (installed via the [optional] extras)",
)


@pytest.fixture(name="directory")
def make_directory():
    return Path(__file__).parent


@pytest.fixture(name="search_output")
def make_search_output(directory):
    return SearchOutput(directory / "search_output")


@pytest.fixture(name="child_analyses")
def make_child_analyses(search_output):
    return search_output.children


def test_child_analysis(child_analyses):
    assert len(child_analyses) == 2


def test_child_analysis_pickles(child_analyses):
    assert child_analyses[0].example == "hello world"


def test_child_analysis_values(directory):
    aggregator = Aggregator.from_directory(
        directory,
        completed_only=True,
    )

    assert list(aggregator.child_values("example")) == [["hello world", "hello world"]]
    assert list(aggregator)[0].child_values("example") == ["hello world", "hello world"]


@pytest.fixture(name="aggregator")
def make_aggregator(session, directory):
    aggregator = af.Aggregator(session)
    aggregator.add_directory(
        directory,
        completed_only=True,
    )
    return aggregator


@requires_astropy
def test_database_aggregator(aggregator):
    assert list(aggregator.child_values("example")) == [
        ["hello world", "hello world"],
    ]


@requires_astropy
def test_child_values(aggregator):
    fit, *_ = list(aggregator)
    assert fit.child_values("example") == ["hello world", "hello world"]
