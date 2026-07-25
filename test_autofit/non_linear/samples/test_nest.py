import pytest

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

def test__samples_within_parameter_range(samples_x5):
    model = af.ModelMapper(mock_class_1=af.m.MockClassx4)

    parameters = [
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [21.0, 22.0, 23.0, 24.0],
        [0.0, 1.0, 2.0, 3.0],
    ]

    samples_x5 = af.m.MockSamplesNest(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=[1.0, 2.0, 3.0, 10.0, 5.0],
            log_prior_list=[0.0, 0.0, 0.0, 0.0, 0.0],
            weight_list=[1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        samples_info={
            "total_samples" : 10,
            "log_evidence" : 0.0,
            "number_live_points" : 5,
            }
        )

    samples_range = samples_x5.samples_within_parameter_range(
        parameter_index=0, parameter_range=[-1.0, 100.0]
    )

    assert len(samples_range.parameter_lists) == 5
    assert samples_x5.parameter_lists[0] == samples_range.parameter_lists[0]

    samples_range = samples_x5.samples_within_parameter_range(
        parameter_index=0, parameter_range=[1.0, 100.0]
    )

    assert len(samples_range.parameter_lists) == 1
    assert samples_range.parameter_lists[0] == [21.0, 22.0, 23.0, 24.0]

    samples_range = samples_x5.samples_within_parameter_range(
        parameter_index=2, parameter_range=[1.5, 2.5]
    )

    assert len(samples_range.parameter_lists) == 4
    assert samples_range.parameter_lists[0] == [0.0, 1.0, 2.0, 3.0]
    assert samples_range.parameter_lists[1] == [0.0, 1.0, 2.0, 3.0]
    assert samples_range.parameter_lists[2] == [0.0, 1.0, 2.0, 3.0]
    assert samples_range.parameter_lists[3] == [0.0, 1.0, 2.0, 3.0]


def test__acceptance_ratio_is_correct():
    model = af.ModelMapper(mock_class_1=af.m.MockClassx4)

    samples_x5 = af.m.MockSamplesNest(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=5 * [[]],
            log_likelihood_list=[1.0, 2.0, 3.0, 4.0, 5.0],
            log_prior_list=5 * [0.0],
            weight_list=5 * [0.0],
        ),
        samples_info={
            "total_samples" : 10,
            "log_evidence" : 0.0,
            "number_live_points" : 5,
            "total_accepted_samples": 5
            }
    )

    assert samples_x5.acceptance_ratio == 0.5
