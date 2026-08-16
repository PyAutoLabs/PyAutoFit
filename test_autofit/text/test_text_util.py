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


def _gradient_samples(model, samples_info):
    parameters = [[1.0, 2.0], [1.2, 2.2]]
    log_likelihood_list = [1.0, 0.0]

    return af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=[0.0, 0.0],
            weight_list=log_likelihood_list,
            model=model,
        ),
        samples_info=samples_info,
    )


def test__search_summary__nan_lane_step_counters(model):
    """
    The gradient searches report both non-finite failure modes, plus rates
    normalised by lane-steps (``n_starts * total_steps``) -- raw counts are not
    comparable between runs of different size.
    """
    samples = _gradient_samples(
        model=model,
        samples_info={
            "total_samples": 10,
            "time": None,
            "n_starts": 8,
            "total_steps": 100,
            "n_resurrections": 40,
            "n_value_nan_lane_steps": 40,
            "n_grad_nan_lane_steps": 8,
        },
    )

    summary = "".join(text_util.search_summary_from_samples(samples=samples))

    assert "Resurrections = 40\n" in summary
    assert "Value-NaN Lane-Steps = 40\n" in summary
    assert "Gradient-NaN Lane-Steps = 8\n" in summary

    # 40 / (8 * 100) and 8 / (8 * 100).
    assert "Value-NaN Lane-Step Rate = 0.05\n" in summary
    assert "Gradient-NaN Lane-Step Rate = 0.01\n" in summary


def test__search_summary__nan_counters_absent_for_other_searches(model):
    """
    ``search_summary_from_samples`` is on EVERY search's summary path, so a
    search without the counters -- Nautilus, which is jit-only and has no
    gradient to be non-finite -- must be completely unaffected.
    """
    parameters = [[1.0, 2.0], [1.2, 2.2]]
    log_likelihood_list = [1.0, 0.0]

    # MockSamplesNest, not MockSamples: Nautilus produces a SamplesNest, and
    # `total_accepted_samples` (the neighbouring duck-typed block) only exists
    # there. The samples_info below is Nautilus's real key set -- see
    # `NautilusSearch.samples_info_from`.
    samples = af.m.MockSamplesNest(
        model=model,
        sample_list=af.Sample.from_lists(
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=[0.0, 0.0],
            weight_list=log_likelihood_list,
            model=model,
        ),
        samples_info={
            "total_samples": 10,
            "total_accepted_samples": 2,
            "time": None,
            "number_live_points": 1,
            "log_evidence": 1.0,
        },
    )

    summary = "".join(text_util.search_summary_from_samples(samples=samples))

    assert "NaN" not in summary
    assert "Resurrections" not in summary
    assert "Lane-Step" not in summary

    # The MCMC block it sits beside is untouched.
    assert "Total Accepted Samples = 2\n" in summary


def test__search_summary__nan_rates_omitted_when_no_lane_steps_taken(model):
    """
    A search that died before its first step has a zero denominator. The counts
    still report; the rates are omitted rather than raising ZeroDivisionError.
    """
    samples = _gradient_samples(
        model=model,
        samples_info={
            "total_samples": 10,
            "time": None,
            "n_starts": 8,
            "total_steps": 0,
            "n_resurrections": 0,
            "n_value_nan_lane_steps": 0,
            "n_grad_nan_lane_steps": 0,
        },
    )

    summary = "".join(text_util.search_summary_from_samples(samples=samples))

    assert "Value-NaN Lane-Steps = 0\n" in summary
    assert "Rate" not in summary


def test__search_summary__clipper_none_emits_nothing(model):
    """
    The default path must be untouched. A ``ClipperNone`` run reports no
    clipping lines at all -- this file is read by tooling and sits in every
    archived run's output, so the unclipped summary stays byte-identical.
    """
    samples = _gradient_samples(
        model=model,
        samples_info={
            "total_samples": 10,
            "time": None,
            "n_starts": 8,
            "total_steps": 100,
            "n_resurrections": 0,
            "n_value_nan_lane_steps": 0,
            "n_grad_nan_lane_steps": 0,
            "clipper": "ClipperNone",
            "n_clipped_lane_steps": 0,
        },
    )

    summary = "".join(text_util.search_summary_from_samples(samples=samples))

    assert "Clipper" not in summary
    assert "Clipped" not in summary


def test__search_summary__clipper_counted_reports_count_and_rate(model):
    """
    ``MultiStartGradient`` enforces the constraint itself every step, so it
    knows how often the clipper fired and reports the count and the rate.
    """
    samples = _gradient_samples(
        model=model,
        samples_info={
            "total_samples": 10,
            "time": None,
            "n_starts": 8,
            "total_steps": 100,
            "n_resurrections": 0,
            "n_value_nan_lane_steps": 0,
            "n_grad_nan_lane_steps": 0,
            "clipper": "ClipperPriorBox",
            "n_clipped_lane_steps": 40,
        },
    )

    summary = "".join(text_util.search_summary_from_samples(samples=samples))

    assert "Clipper = ClipperPriorBox\n" in summary
    assert "Clipped Lane-Steps = 40\n" in summary

    # 40 / (8 * 100), the same lane-step denominator as the NaN rates.
    assert "Clipped Lane-Step Rate = 0.05\n" in summary


def test__search_summary__clipper_without_count_says_not_measured(model):
    """
    ``LBFGS`` is declarative -- it hands bounds to scipy and never calls
    ``project``, so there is no mask and nothing to count. A ``0`` here would
    read as "never fired" when it means "cannot know", so it must not appear.
    """
    samples = _gradient_samples(
        model=model,
        samples_info={
            "total_samples": 10,
            "time": None,
            "clipper": "ClipperPriorBox",
        },
    )

    summary = "".join(text_util.search_summary_from_samples(samples=samples))

    assert "Clipper = ClipperPriorBox\n" in summary
    assert "Clipped Lane-Steps = not measured (bounds enforced by scipy)\n" in summary
    assert "Clipped Lane-Steps = 0" not in summary
    assert "Clipped Lane-Step Rate" not in summary


def test__search_summary__clipper_absent_for_other_searches(model):
    """
    A search predating the ``Clipper`` writes no ``clipper`` key, and must be
    completely unaffected -- the same invariant the NaN counters hold.
    """
    samples = _gradient_samples(
        model=model,
        samples_info={
            "total_samples": 10,
            "time": None,
            "n_starts": 8,
            "total_steps": 100,
            "n_resurrections": 0,
            "n_value_nan_lane_steps": 0,
            "n_grad_nan_lane_steps": 0,
        },
    )

    summary = "".join(text_util.search_summary_from_samples(samples=samples))

    assert "Clipper" not in summary
    assert "Clipped" not in summary


def test__search_summary__constrained_lane_steps_reported(model):
    """
    The trapped-lane counter (PyAutoFit#1475) reached ``samples_info`` when it
    shipped but was never emitted, so the artefact a user reads to find out what
    the search did did not report it.
    """
    samples = _gradient_samples(
        model=model,
        samples_info={
            "total_samples": 10,
            "time": None,
            "n_starts": 8,
            "total_steps": 100,
            "n_resurrections": 0,
            "n_value_nan_lane_steps": 0,
            "n_grad_nan_lane_steps": 0,
            "n_constrained_lane_steps": 667,
        },
    )

    summary = "".join(text_util.search_summary_from_samples(samples=samples))

    assert "Constrained Lane-Steps = 667\n" in summary


def test__search_summary__constrained_absent_when_never_written(model):
    """
    A ``search_internal`` written before the trapped-lane counter existed has no
    such key. A zero it never wrote must not be reported as a measured zero --
    ``0`` and "not written" are different findings.
    """
    samples = _gradient_samples(
        model=model,
        samples_info={
            "total_samples": 10,
            "time": None,
            "n_starts": 8,
            "total_steps": 100,
            "n_resurrections": 0,
            "n_value_nan_lane_steps": 0,
            "n_grad_nan_lane_steps": 0,
        },
    )

    summary = "".join(text_util.search_summary_from_samples(samples=samples))

    assert "Constrained" not in summary
