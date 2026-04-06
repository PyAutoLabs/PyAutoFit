import pytest

import autofit as af
from autofit import Sample
from autofit.non_linear.mock.mock_samples_summary import MockSamplesSummary


@pytest.fixture(name="mapper")
def make_mapper():
    return af.Collection(component=af.Model(af.m.MockClassx2Tuple))


@pytest.fixture(name="grid_search")
def make_grid_search(mapper):
    mock_search = af.m.MockSearch()
    mock_search.paths = af.DirectoryPaths(name="")
    search = af.SearchGridSearch(number_of_steps=10, search=mock_search)
    return search


@pytest.fixture(name="sample_name_paths")
def make_sample_name_paths():
    return af.DirectoryPaths(name="sample_name")


@pytest.fixture(name="grid_search_10_result")
def make_grid_search_10_result(mapper, sample_name_paths):
    grid_search = af.SearchGridSearch(
        search=af.m.MockMLE(
            samples_summary=MockSamplesSummary(
                model=mapper,
                median_pdf_sample=Sample(
                    log_likelihood=1.0,
                    log_prior=0.0,
                    weight=0.0,
                    kwargs={
                        "component.one_tuple.one_tuple_0": 0,
                        "component.one_tuple.one_tuple_1": 1,
                    },
                ),
            )
        ),
        number_of_steps=5,
    )
    grid_search.search.paths = sample_name_paths
    return grid_search.fit(
        model=mapper,
        analysis=af.m.MockAnalysis(),
        grid_priors=[
            mapper.component.one_tuple.one_tuple_0,
            mapper.component.one_tuple.one_tuple_1,
        ],
    )
