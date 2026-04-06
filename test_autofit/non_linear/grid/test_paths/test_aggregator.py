import pytest

from autofit.aggregator.aggregator import Aggregator


@pytest.fixture(name="aggregator")
def make_aggregator(sample_name_paths, grid_search_10_result):
    return Aggregator.from_directory(sample_name_paths.output_path)


def test_aggregate(aggregator):
    assert len(aggregator) == 25
    assert len(aggregator.grid_search_outputs) == 1


@pytest.fixture(name="grid_search")
def make_grid_search(aggregator):
    (grid_search,) = aggregator.grid_searches()
    return grid_search


def test_correspondence(grid_search, aggregator):
    assert len(grid_search.children) == 25
    assert grid_search.children[0] is list(aggregator)[0]


def test_attributes(grid_search):
    assert grid_search.is_complete
    assert isinstance(grid_search.unique_tag, str)


def test_files(grid_search):
    assert grid_search.result


def test_jsons(grid_search):
    assert len(grid_search.jsons) == 1
