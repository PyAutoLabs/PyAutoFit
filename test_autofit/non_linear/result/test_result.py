import pytest

import autofit as af
from autofit import Sample
from autofit.non_linear.mock.mock_samples_summary import MockSamplesSummary


@pytest.fixture(name="result")
def make_result():
    mapper = af.ModelMapper()
    mapper.component = af.m.MockClassx2Tuple
    sample = Sample(
        log_likelihood=1.0,
        log_prior=0.0,
        weight=0.0,
        kwargs={
            "component.one_tuple.one_tuple_0": 0,
            "component.one_tuple.one_tuple_1": 1,
        },
    )
    # noinspection PyTypeChecker
    return af.mock.MockResult(
        samples=af.m.MockSamples(
            sample_list=[sample],
            # max_log_likelihood_instance=[0, 1],
            prior_means=[0, 1],
            model=mapper,
        ),
        samples_summary=MockSamplesSummary(
            model=mapper,
            max_log_likelihood_instance=[0, 1],
            median_pdf_sample=sample,
        ),
    )


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

    def test_model_absolute(self, result):
        component = result.model_centred_absolute(a=2.0).component
        assert component.one_tuple.one_tuple_0.mean == 0
        assert component.one_tuple.one_tuple_1.mean == 1
        assert component.one_tuple.one_tuple_0.sigma == 2.0
        assert component.one_tuple.one_tuple_1.sigma == 2.0

    def test_model_relative(self, result):
        component = result.model_centred_relative(r=1.0).component
        assert component.one_tuple.one_tuple_0.mean == 0
        assert component.one_tuple.one_tuple_1.mean == 1
        assert component.one_tuple.one_tuple_0.sigma == 0.0
        assert component.one_tuple.one_tuple_1.sigma == 1.0

    def test_model_bounded(self, result):
        component = result.model_centred_max_lh_bounded(b=1.0).component

        assert component.one_tuple.one_tuple_0.lower_limit == -1.0
        assert component.one_tuple.one_tuple_1.lower_limit == 0.0
        assert component.one_tuple.one_tuple_0.upper_limit == 1.0
        assert component.one_tuple.one_tuple_1.upper_limit == 2.0

    def test_raises(self, result):
        with pytest.raises(af.exc.PriorException):
            result.model.mapper_from_prior_means(
                result.samples.prior_means, a=2.0, r=1.0
            )

    def test_start_point(self, result):
        start_point = result.start_point

        assert isinstance(start_point, af.InitializerParamStartPoints)

        expected = result.samples.max_log_likelihood(as_instance=False)
        parameter_dict = {
            ".".join(result.model.path_for_prior(prior)): value
            for prior, value in start_point.parameter_dict.items()
        }
        for path, value in zip(
            [".".join(path) for path in result.model.paths], expected
        ):
            assert parameter_dict[path] == pytest.approx(
                (value - 1e-8, value + 1e-8)
            )


@pytest.fixture(name="results")
def make_results_collection():
    results = af.ResultsCollection()

    results.add("first", "one")
    results.add("second", "two")

    return results


class TestResultsCollection:
    def test_with_name(self, results):
        assert results.from_name("first") == "one"
        assert results.from_name("second") == "two"

    def test_with_index(self, results):
        assert results[0] == "one"
        assert results[1] == "two"
        assert results.first == "one"
        assert results.last == "two"
        assert len(results) == 2

    def test_missing_result(self, results):
        with pytest.raises(af.exc.PipelineException):
            results.from_name("third")
