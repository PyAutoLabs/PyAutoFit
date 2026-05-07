from pathlib import Path
import pytest

import autofit as af

from autofit.text import text_util

text_path = Path(__file__).resolve().parent / "files" / "samples"


@pytest.fixture(name="model")
def make_model():
    return af.ModelMapper(mock_class=af.m.MockClassx2)


@pytest.fixture(name="samples")
def make_samples(model):
    parameters = [[1.0, 2.0], [1.2, 2.2]]

    log_likelihood_list = [1.0, 0.0]

    return af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=[0.0, 0.0],
            weight_list=log_likelihood_list,
            model=model
        )
    )


def test__results_to_file(samples):

    result_info =text_util.result_info_from(
        samples=samples,
    )

    assert  "Maximum Log Likelihood                                                          1.00000000\n" in result_info

def test__search_summary_to_file(model):
    file_search_summary = text_path / "search.summary"

    parameters = [[1.0, 2.0], [1.2, 2.2]]

    log_likelihood_list = [1.0, 0.0]

    samples = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=[0.0, 0.0],
            weight_list=log_likelihood_list,
            model=model
        ),
        samples_info={
            "time": "1",
            "total_accepted_samples" : 2
        }
    )

    text_util.search_summary_to_file(
        samples=samples,
        log_likelihood_function_time=1.0,
        filename=file_search_summary
    )

    results = open(file_search_summary)
    lines = results.readlines()
    assert lines[0] == "Total Samples = 2\n"
    results.close()

    samples = af.m.MockSamplesNest(
        model=model,
        sample_list=af.Sample.from_lists(
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list + [2.0],
            log_prior_list=[1.0, 1.0],
            weight_list=log_likelihood_list,
            model=model
        ),
        samples_info={
            "total_samples": 10,
            "total_accepted_samples": 2,
            "time": "1",
            "number_live_points": 1,
            "log_evidence": 1.0
        }
    )

    text_util.search_summary_to_file(samples=samples, log_likelihood_function_time=1.0, filename=file_search_summary)

    results = open(file_search_summary)
    lines = results.readlines()
    assert lines[0] == "Total Samples = 10\n"
    assert lines[1] == "Total Accepted Samples = 2\n"
    assert lines[2] == "Acceptance Ratio = 0.2\n"
    assert lines[3] == "Time To Run = 0:00:01\n"
    assert lines[4] == "Time Per Sample (seconds) = 0.1\n"
    assert lines[5] == "Log Likelihood Function Evaluation Time (seconds) = 1.0\n"
    results.close()
