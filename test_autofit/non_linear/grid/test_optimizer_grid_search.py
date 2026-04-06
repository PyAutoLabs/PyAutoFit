import csv
import pickle

import numpy as np
import pytest

import autofit as af
from autofit import exc


def test_unpickle_result():
    # noinspection PyTypeChecker
    result = af.GridSearchResult(
        samples=[af.Samples(model=af.Model(af.ex.Gaussian), sample_list=[])],
        lower_limits_lists=[[1]],
        grid_priors=[],
    )
    result = pickle.loads(pickle.dumps(result))
    assert result is not None


class TestGridSearchablePriors:
    def test_generated_models(self, grid_search, mapper):
        mappers = list(
            grid_search.model_mappers(
                mapper,
                grid_priors=[
                    mapper.component.one_tuple.one_tuple_0,
                    mapper.component.one_tuple.one_tuple_1,
                ],
            )
        )

        assert len(mappers) == 100

        assert mappers[0].component.one_tuple.one_tuple_0.lower_limit == 0.0
        assert mappers[0].component.one_tuple.one_tuple_0.upper_limit == 0.1
        assert mappers[0].component.one_tuple.one_tuple_1.lower_limit == 0.0
        assert mappers[0].component.one_tuple.one_tuple_1.upper_limit == 0.2

        assert mappers[-1].component.one_tuple.one_tuple_0.lower_limit == 0.9
        assert mappers[-1].component.one_tuple.one_tuple_0.upper_limit == 1.0
        assert mappers[-1].component.one_tuple.one_tuple_1.lower_limit == 1.8
        assert mappers[-1].component.one_tuple.one_tuple_1.upper_limit == 2.0

    def test_non_grid_searched_dimensions(self, mapper):
        search = af.m.MockSearch()
        search.paths = af.DirectoryPaths(name="")
        grid_search = af.SearchGridSearch(number_of_steps=10, search=search)

        mappers = list(
            grid_search.model_mappers(
                mapper, grid_priors=[mapper.component.one_tuple.one_tuple_0]
            )
        )

        assert len(mappers) == 10

        assert mappers[0].component.one_tuple.one_tuple_0.lower_limit == 0.0
        assert mappers[0].component.one_tuple.one_tuple_0.upper_limit == 0.1
        assert mappers[0].component.one_tuple.one_tuple_1.lower_limit == 0.0
        assert mappers[0].component.one_tuple.one_tuple_1.upper_limit == 2.0

        assert mappers[-1].component.one_tuple.one_tuple_0.lower_limit == 0.9
        assert mappers[-1].component.one_tuple.one_tuple_0.upper_limit == 1.0
        assert mappers[-1].component.one_tuple.one_tuple_1.lower_limit == 0.0
        assert mappers[-1].component.one_tuple.one_tuple_1.upper_limit == 2.0

    def test_tied_priors(self, grid_search, mapper):
        mapper.component.one_tuple.one_tuple_0 = mapper.component.one_tuple.one_tuple_1

        mappers = list(
            grid_search.model_mappers(
                grid_priors=[
                    mapper.component.one_tuple.one_tuple_0,
                    mapper.component.one_tuple.one_tuple_1,
                ],
                model=mapper,
            )
        )

        assert len(mappers) == 10

        assert mappers[0].component.one_tuple.one_tuple_0.lower_limit == 0.0
        assert mappers[0].component.one_tuple.one_tuple_0.upper_limit == 0.2
        assert mappers[0].component.one_tuple.one_tuple_1.lower_limit == 0.0
        assert mappers[0].component.one_tuple.one_tuple_1.upper_limit == 0.2

        assert mappers[-1].component.one_tuple.one_tuple_0.lower_limit == 1.8
        assert mappers[-1].component.one_tuple.one_tuple_0.upper_limit == 2.0
        assert mappers[-1].component.one_tuple.one_tuple_1.lower_limit == 1.8
        assert mappers[-1].component.one_tuple.one_tuple_1.upper_limit == 2.0

        for mapper in mappers:
            assert (
                mapper.component.one_tuple.one_tuple_0
                == mapper.component.one_tuple.one_tuple_1
            )

    def test_different_prior_width(self, grid_search, mapper):
        mapper.component.one_tuple.one_tuple_0 = af.UniformPrior(0.0, 2.0)
        mappers = list(
            grid_search.model_mappers(
                grid_priors=[mapper.component.one_tuple.one_tuple_0], model=mapper
            )
        )

        assert len(mappers) == 10

        assert mappers[0].component.one_tuple.one_tuple_0.lower_limit == 0.0
        assert mappers[0].component.one_tuple.one_tuple_0.upper_limit == 0.2

        assert mappers[-1].component.one_tuple.one_tuple_0.lower_limit == 1.8
        assert mappers[-1].component.one_tuple.one_tuple_0.upper_limit == 2.0

        mapper.component.one_tuple.one_tuple_0 = af.UniformPrior(1.0, 1.5)
        mappers = list(
            grid_search.model_mappers(
                mapper, grid_priors=[mapper.component.one_tuple.one_tuple_0]
            )
        )

        assert len(mappers) == 10

        assert mappers[0].component.one_tuple.one_tuple_0.lower_limit == 1.0
        assert mappers[0].component.one_tuple.one_tuple_0.upper_limit == 1.05

        assert mappers[-1].component.one_tuple.one_tuple_0.lower_limit == 1.45
        assert mappers[-1].component.one_tuple.one_tuple_0.upper_limit == 1.5

    def test_raises_exception_for_bad_limits(self, grid_search, mapper):
        mapper.component.one_tuple.one_tuple_0 = af.GaussianPrior(
            0.0, 2.0,
        )
        with pytest.raises(exc.PriorException):
            list(
                grid_search.make_arguments(
                    [[0, 1]], grid_priors=[mapper.component.one_tuple.one_tuple_0]
                )
            )
        mapper.component.one_tuple.one_tuple_0 = af.UniformPrior(
            lower_limit=float("-inf"), upper_limit=float("inf")
        )
        with pytest.raises(exc.PriorException):
            list(
                grid_search.make_arguments(
                    [[0, 1]], grid_priors=[mapper.component.one_tuple.one_tuple_0]
                )
            )


@pytest.fixture(name="grid_search_05")
def make_grid_search_05():
    search = af.SearchGridSearch(search=af.m.MockMLE(), number_of_steps=2)
    search.search.paths = af.DirectoryPaths(name="sample_name")
    return search


@pytest.fixture(autouse=True)
def empty_args():
    af.m.MockMLE.init_args = list()


def test_csv_headers(grid_search_10_result, sample_name_paths):
    with open(sample_name_paths.output_path / "results.csv") as f:
        reader = csv.reader(f)
        headers = next(reader)

    assert headers == [
        "index",
        "component_one_tuple_0",
        "component_one_tuple_1",
        "log_likelihood_increase",
    ]


def test_output_result_json(grid_search_10_result, sample_name_paths):
    assert isinstance(sample_name_paths.load_json("result"), dict)


class TestGridNLOBehaviour:
    def test_results(self, grid_search_05, mapper):
        result = grid_search_05.fit(
            model=mapper,
            analysis=af.m.MockAnalysis(),
            grid_priors=[
                mapper.component.one_tuple.one_tuple_0,
                mapper.component.one_tuple.one_tuple_1,
            ],
        )

        assert len(result.samples) == 4
        assert result.no_dimensions == 2

    def test_results_10(self, grid_search_10_result):
        assert len(grid_search_10_result.samples) == 25
        assert grid_search_10_result.no_dimensions == 2
        assert grid_search_10_result.log_likelihoods().native.shape == (5, 5)

    def test_passes_attributes(self):
        search = af.DynestyStatic()
        search.paths = af.DirectoryPaths(name="")
        grid_search = af.SearchGridSearch(number_of_steps=10, search=search)

        grid_search.nlive = 20
        grid_search.facc = 0.3

        search = grid_search.search_instance("name_path")

        assert search.nlive is grid_search.nlive
        assert grid_search.paths.output_path != search.paths.output_path


@pytest.fixture(name="grid_search_result")
def make_grid_search_result():
    one = af.m.MockResultGrid(1)
    two = af.m.MockResultGrid(2)

    # noinspection PyTypeChecker
    return af.GridSearchResult(
        samples=[one, two], lower_limits_lists=[[1], [2]], grid_priors=[[1], [2]]
    )


class TestGridSearchResult:
    def test_best_result(self, grid_search_result):
        assert grid_search_result.best_samples.log_likelihood == 2

    def test_attributes(self, grid_search_result):
        assert grid_search_result.model == 2

    def test_best_model(self, grid_search_result):
        assert grid_search_result.best_model == 2

    def test_all_models(self, grid_search_result):
        assert grid_search_result.all_models == [1, 2]

    def test__result_derived_properties(self):
        lower_limit_lists = [[0.0, 0.0], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]]

        # noinspection PyTypeChecker
        grid_search_result = af.GridSearchResult(
            samples=None,
            grid_priors=[
                af.UniformPrior(lower_limit=-2.0, upper_limit=2.0),
                af.UniformPrior(lower_limit=-3.0, upper_limit=3.0),
            ],
            lower_limits_lists=lower_limit_lists,
        )

        assert grid_search_result.shape == (2, 2)
        assert grid_search_result.physical_step_sizes == (2.0, 3.0)
        assert grid_search_result.physical_centres_lists == [
            [-1.0, -1.5],
            [-1.0, 1.5],
            [1.0, -1.5],
            [1.0, 1.5],
        ]
        assert grid_search_result.physical_upper_limits_lists == [
            [0.0, 0.0],
            [0.0, 3.0],
            [2.0, 0.0],
            [2.0, 3.0],
        ]

    def test__results_on_native_grid(self, grid_search_result):
        assert (
            grid_search_result.samples.native
            == np.array(
                [
                    [grid_search_result.samples[0], grid_search_result.samples[1]],
                ]
            )
        ).all()

        assert (
            grid_search_result.log_likelihoods().native
            == np.array(
                [
                    [1, 2],
                ]
            )
        ).all()


@pytest.mark.parametrize(
    "n_dimensions, n_steps",
    [
        (2, 2),
        (3, 3),
        (2, 3),
        (3, 2),
        (4, 4),
    ],
)
def test_higher_dimensions(n_dimensions, n_steps):
    shape = n_dimensions * (n_steps,)
    total = n_steps**n_dimensions
    model = af.Model(af.ex.Gaussian)
    result = af.GridSearchResult(
        samples=total
        * [
            af.SamplesPDF(
                model,
                [
                    af.Sample(
                        1.0,
                        1.0,
                        1.0,
                        {"centre": 1.0, "sigma": 1.0, "normalization": 1.0},
                    )
                ],
            ),
        ],
        grid_priors=[],
        lower_limits_lists=total * [n_dimensions * [0.0]],
    )

    assert result.shape == shape
    assert result.samples.native.shape == shape
    assert result.log_likelihoods().native.shape == shape
