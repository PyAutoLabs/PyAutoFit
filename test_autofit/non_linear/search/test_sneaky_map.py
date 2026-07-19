import multiprocessing as mp
import pytest
import warnings

# noinspection PyProtectedMember
from dynesty.dynesty import _function_wrapper
from emcee.ensemble import _FunctionWrapper

import autofit as af
from autonerves.conf import output_path_for_test
from autofit.non_linear import fitness as f
from autofit.non_linear.mock.mock_analysis import MockAnalysis
from autofit.non_linear.parallel import SneakyPool, SneakyJob

# noinspection PyProtectedMember
from autofit.non_linear.parallel.sneaky import _is_likelihood_function


@pytest.fixture(name="search")
def make_search():
    return af.DynestyStatic(name="", number_of_cores=2, maxcall=2)


@pytest.fixture(name="model")
def make_model():
    return af.Model(af.ex.Gaussian)


@pytest.fixture(name="analysis")
def make_analysis():
    return Analysis()


@pytest.fixture(name="paths")
def make_paths(model, search):
    paths = search.paths
    paths.model = model
    return model


@pytest.fixture(name="fitness")
def make_fitness(
    paths,
    model,
    analysis,
):
    return Fitness(
        model=model,
        analysis=analysis,
    )


class Analysis(af.Analysis):
    def log_likelihood_function(self, instance: af.ex.Gaussian):
        return -((10 - instance.centre) ** 2)


class Fitness(f.Fitness):
    def __call__(self, parameter_list):
        return -1


@pytest.fixture(name="sneaky_job")
def make_sneaky_job(fitness):
    return SneakyJob(identity, 1, fitness)


def test_perform_fitness(pool, fitness):
    results = list(pool.map(fitness, range(2)))
    assert results == [-1, -1]


def test_sneaky_removal(sneaky_job):
    assert sneaky_job.args == [1]
    assert sneaky_job.fitness_index == 1


def test_sneaky_call(sneaky_job, fitness):
    assert sneaky_job.perform(fitness) == ([1, fitness],)


def identity(*args):
    return args


def test_sneaky_pool(fitness, pool):
    results = list(pool.map(identity, [(0, fitness), (1, fitness)]))

    assert len(results) == 2

    for i, t in enumerate(results):
        assert t[0][0] == i
        assert isinstance(t[0][1], Fitness)


def test_arbitrary_map(pool):
    args_list = [(1,), (1, 2), (1, 2, 3)]
    results = list(pool.map(identity, args_list))

    assert results == [([1],), ([1, 2],), ([1, 2, 3],)]


def get_pid(*_):
    return mp.current_process().pid


@pytest.fixture(name="pool")
def make_pool(fitness):
    return SneakyPool(
        processes=1, paths=af.DirectoryPaths(name="test"), fitness=fitness
    )


def test_process_ids(pool):
    results = set(pool.map(get_pid, range(10)))
    assert results == set(pool.map(get_pid, range(10)))


class MyException(Exception):
    pass


def raise_exception(*_):
    raise MyException()


def test_raising_error(pool):
    with pytest.raises(MyException):
        list(pool.map(raise_exception, range(10)))


@output_path_for_test()
def test_sneaky_map(search, model, analysis):

    result = search.fit(model, analysis)

    assert isinstance(result.instance, af.ex.Gaussian)


@pytest.mark.parametrize(
    "function",
    [
        _FunctionWrapper(lambda: 1, None, None),
        Fitness(af.Model(af.ex.Gaussian), MockAnalysis()),
        _function_wrapper(lambda: 1, [], {}, "loglikelihood"),
    ],
)
def test_is_likelihood_function(function):
    assert _is_likelihood_function(function)
