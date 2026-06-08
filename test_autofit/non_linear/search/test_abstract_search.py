import os
import pytest

import autofit as af
from autoconf import conf

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(name="mapper")
def make_mapper():
    return af.ModelMapper()


@pytest.fixture(name="mock_list")
def make_mock_list():
    return [af.Model(af.m.MockClassx4), af.Model(af.m.MockClassx4)]


@pytest.fixture(name="result")
def make_result():
    mapper = af.ModelMapper()
    mapper.component = af.m.MockClassx2Tuple
    # noinspection PyTypeChecker
    return af.mock.MockResult(
        samples_summary=af.m.MockSamplesSummary(
            model=mapper,
            prior_means=[0, 1],
        ),
    )


def test__environment_variable_override():
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"

    conf.instance["general"]["parallel"]["warn_environment_variables"] = True

    with pytest.warns(af.exc.SearchWarning):
        af.mock.MockSearch(number_of_cores=2)

    conf.instance["general"]["parallel"]["warn_environment_variables"] = False

class TestResult:
    def test_model(self, result):

        component = result.model.component
        assert component.one_tuple.one_tuple_0.mean == 0.5
        assert component.one_tuple.one_tuple_1.mean == 1

    def test_model_centred(self, result):

        component = result.model_centred.component
        assert component.one_tuple.one_tuple_0.mean == 0
        assert component.one_tuple.one_tuple_1.mean == 1
        assert component.one_tuple.one_tuple_0.sigma == 0.2
        assert component.one_tuple.one_tuple_1.sigma == 0.2

    def test_model_centred_absolute(self, result):
        component = result.model_centred_absolute(a=2.0).component
        assert component.one_tuple.one_tuple_0.mean == 0
        assert component.one_tuple.one_tuple_1.mean == 1
        assert component.one_tuple.one_tuple_0.sigma == 2.0
        assert component.one_tuple.one_tuple_1.sigma == 2.0

    def test_model_centred_relative(self, result):
        component = result.model_centred_relative(r=1.0).component
        assert component.one_tuple.one_tuple_0.mean == 0
        assert component.one_tuple.one_tuple_1.mean == 1
        assert component.one_tuple.one_tuple_0.sigma == 0.0
        assert component.one_tuple.one_tuple_1.sigma == 1.0

    def test_raises(self, result):
        with pytest.raises(af.exc.PriorException):
            result.model.mapper_from_prior_means(
                result.samples_summary.prior_means, a=2.0, r=1.0
            )


class TestSearchConfig:
    def test__explicit_params_accessible(self):
        search = af.DynestyStatic(nlive=100)
        assert search.nlive == 100

    def test__run_params_accessible(self):
        search = af.DynestyStatic(dlogz=0.5)
        assert search.dlogz == 0.5

    def test__unique_tag(self):
        search = af.DynestyStatic(unique_tag="my_tag")
        assert search.unique_tag == "my_tag"

    def test__path_prefix_and_name(self):
        from pathlib import Path

        search = af.DynestyStatic(
            path_prefix="prefix",
            name="my_search",
        )
        assert search.path_prefix == Path("prefix")
        assert search.name == "my_search"

    def test__identifier_fields_differ_across_searches(self):
        emcee = af.Emcee()
        dynesty = af.DynestyStatic()

        assert emcee.__identifier_fields__ != dynesty.__identifier_fields__
        assert "nwalkers" in emcee.__identifier_fields__
        assert "nlive" in dynesty.__identifier_fields__

    def test__bypass_fake_samples_support_multi_batch_checks(self):
        model = af.Model(af.m.MockClassx2)
        parameter_vector = [1.0, 2.0]

        sample_list = af.DynestyStatic._build_fake_samples(
            model=model,
            parameter_vector=parameter_vector,
            log_likelihood=-10.0,
        )

        assert len(sample_list) == 4
        assert sample_list[0].parameter_lists_for_model(model) == parameter_vector
        assert sample_list[0].log_likelihood == -10.0
        assert [sample.log_likelihood for sample in sample_list] == [
            -10.0,
            -11.0,
            -12.0,
            -13.0,
        ]
        assert all(sample.weight > 0.0 for sample in sample_list)


class TestUpdaterPathsRefresh:
    """
    The cached ``SearchUpdater`` must be invalidated when ``self.paths``
    is reassigned to a new object — otherwise output (samples, viz,
    profiling) keeps writing under the FIRST paths the search ever saw.
    The EP loop does this routinely:
    ``AbstractSearch.optimise(factor_approx)`` reassigns ``self.paths``
    to a fresh ``SubDirectoryPaths`` per factor and per EP iteration.
    """

    def test__updater_refreshes_when_paths_reassigned(self):
        search = af.DynestyStatic(name="updater_paths_test_a")
        first_updater = search._updater
        first_paths = search.paths
        # Sanity: cache hit on second access with no path change.
        assert search._updater is first_updater

        search.paths = af.DirectoryPaths(name="updater_paths_test_b")
        second_updater = search._updater

        assert second_updater is not first_updater, (
            "Expected a fresh SearchUpdater after self.paths was reassigned"
        )
        assert second_updater._paths is search.paths
        assert second_updater._paths is not first_paths


class TestLabels:
    def test_param_names(self):
        model = af.Model(af.m.MockClassx4)
        assert [
            "one",
            "two",
            "three",
            "four",
        ] == model.model_component_and_parameter_names

    def test_label_config(self):
        assert conf.instance["notation"]["label"]["label"]["one"] == "one_label"
        assert conf.instance["notation"]["label"]["label"]["two"] == "two_label"
        assert conf.instance["notation"]["label"]["label"]["three"] == "three_label"
        assert conf.instance["notation"]["label"]["label"]["four"] == "four_label"
