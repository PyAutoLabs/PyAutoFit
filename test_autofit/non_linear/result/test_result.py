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


class _RejectsLowValue:
    """
    A model component whose constructor rejects part of its own prior range, mimicking a
    validated parameterization (e.g. `ell_comps` outside the ellipticity unit disk) which
    only fires when a stored vector is materialized on the host.
    """

    def __init__(self, value):
        if value < 0.75:
            raise af.exc.FitException("value must be at least 0.75")
        self.value = value


class _NoSamplesPaths:
    """Paths double standing in for a resumed run whose `samples.csv` was never written."""

    @property
    def samples(self):
        raise FileNotFoundError()


def _samples_with_rejected_best():
    """
    Samples whose maximum likelihood point is rejected by the model, but whose
    second-best point is not.
    """
    model = af.Model(_RejectsLowValue)
    model.value = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)

    return af.SamplesPDF(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=[[0.1], [0.9]],
            log_likelihood_list=[2.0, 1.0],
            log_prior_list=[0.0, 0.0],
            weight_list=[1.0, 1.0],
        ),
        samples_info={"log_evidence": 1.0},
    )


class TestResultInstanceRejectedBestPoint:
    def test__summary_still_raises_on_a_rejected_best_point(self):
        """
        The summary holds a single sample, so it has nothing to substitute and must keep
        raising -- the recovery belongs on the full samples, not here.
        """
        samples = _samples_with_rejected_best()

        with pytest.raises(af.exc.SamplesException):
            samples.summary().instance

    def test__result_instance__falls_back_to_the_best_valid_sample(self):
        samples = _samples_with_rejected_best()

        result = af.Result(
            samples_summary=samples.summary(),
            samples=samples,
        )

        assert result.instance.value == pytest.approx(0.9)
        assert result.max_log_likelihood_instance.value == pytest.approx(0.9)

    def test__result_instance__is_none_when_there_are_no_samples_to_recover_from(self):
        samples = _samples_with_rejected_best()

        result = af.Result(
            samples_summary=samples.summary(),
            paths=_NoSamplesPaths(),
        )

        assert result.instance is None

    def test__result_instance__recovered_instance_is_computed_once(self):
        samples = _samples_with_rejected_best()

        result = af.Result(
            samples_summary=samples.summary(),
            samples=samples,
        )

        assert result.instance is result.instance
