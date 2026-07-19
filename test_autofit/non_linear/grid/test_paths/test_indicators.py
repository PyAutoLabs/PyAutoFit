from pathlib import Path

import pytest

import autofit as af
from autonerves.conf import output_path_for_test
from autofit.database.aggregator.scrape import Scraper

output_directory = Path(__file__).parent / "output"


@pytest.fixture(name="parent_search")
@output_path_for_test(output_directory)
def make_parent_search(model_gaussian_x1):
    search = af.m.MockSearch(name="parent")
    search.paths.model = model_gaussian_x1
    return search


@pytest.fixture(name="database_parent_search")
def make_database_parent_search(session):
    return af.DynestyStatic(session=session)


def _make_grid_search(mapper, session=None):
    search = af.SearchGridSearch(
        search=af.m.MockMLE(session=session), number_of_steps=2
    )
    search.fit(
        model=mapper,
        analysis=af.m.MockAnalysis(),
        grid_priors=[
            mapper.component.one_tuple.one_tuple_0,
            mapper.component.one_tuple.one_tuple_1,
        ],
    )
    return search


@pytest.fixture(name="grid_search")
def make_grid_search(mapper):
    search = _make_grid_search(mapper)
    search.paths.save_all()
    return search


@pytest.fixture(name="database_grid_search")
def make_database_grid_search(mapper, session):
    return _make_grid_search(mapper, session=session)


class TestDirectory:
    def test_is_grid_search(self, grid_search):
        assert grid_search.paths.is_grid_search


@output_path_for_test(output_directory)
def test_incomplete(grid_search, session):
    grid_search.save_metadata()

    Scraper(directory=output_directory, session=session).scrape()

    session.commit()

    aggregator = af.Aggregator(session)
    aggregator = aggregator(aggregator.search.is_complete)
    assert len(aggregator) == 0


class TestDatabase:
    def test_is_grid_search(self, database_grid_search):
        assert database_grid_search.paths.is_grid_search
