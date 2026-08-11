import os
from pathlib import Path
import pytest

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


class _RejectsLowStoredValue:
    def __init__(self, value):
        if value < 0.5:
            raise af.exc.FitException("stored value is outside the current domain")
        self.value = value


def _guarded_samples(parameter_lists, log_likelihood_list, weight_list):
    model = af.Model(_RejectsLowStoredValue)
    model.value = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    return af.SamplesPDF(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameter_lists,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=[0.0] * len(parameter_lists),
            weight_list=weight_list,
        ),
    )


def _factor_graph_samples():
    shared_value = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    factor_models = [
        af.Collection(galaxies=af.Model(_RejectsLowStoredValue, value=shared_value))
        for _ in range(2)
    ]
    factor_graph = af.FactorGraphModel(
        *[
            af.AnalysisFactor(
                prior_model=factor_model,
                analysis=af.m.MockAnalysis(),
            )
            for factor_model in factor_models
        ]
    )
    model = factor_graph.global_prior_model
    samples = af.SamplesPDF(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=[[0.9]],
            log_likelihood_list=[1.0],
            log_prior_list=[0.0],
            weight_list=[1.0],
        ),
    )
    return samples, factor_models[0]


@pytest.mark.parametrize("use_summary", [False, True])
def test__subsamples__rebuilds_instance_after_model_rebind(use_summary):
    samples, factor_model = _factor_graph_samples()
    source = samples.summary() if use_summary else samples

    global_instance = source.instance
    child = source.subsamples(model=factor_model)

    assert child._instance is None
    assert child.instance is not global_instance
    assert child.instance.galaxies.value == pytest.approx(0.9)


@pytest.mark.parametrize("method_name", ["with_paths", "without_paths"])
def test__path_filter__clears_model_derived_caches(samples_x5, method_name):
    samples_x5.instance
    samples_x5.paths
    samples_x5.names

    paths = [("mock_class_1", "one")]
    filtered = getattr(samples_x5, method_name)(paths)

    assert filtered._instance is None
    assert filtered._paths is None
    assert filtered._names is None


def test__table__headers(samples_x5):
    assert samples_x5._headers == [
        "mock_class_1.one",
        "mock_class_1.two",
        "mock_class_1.three",
        "mock_class_1.four",
        "log_likelihood",
        "log_prior",
        "log_posterior",
        "weight",
    ]


def test__table__rows(samples_x5):
    rows = list(samples_x5._rows)
    assert rows == [
        [0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 2.0, 3.0, 2.0, 0.0, 2.0, 1.0],
        [0.0, 1.0, 2.0, 3.0, 3.0, 0.0, 3.0, 1.0],
        [21.0, 22.0, 23.0, 24.0, 10.0, 0.0, 10.0, 1.0],
        [0.0, 1.0, 2.0, 3.0, 5.0, 0.0, 5.0, 1.0],
    ]


def test__table__write_table():
    model = af.Collection(mock_class_1=af.m.MockClassx4)

    parameters = [
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [21.0, 22.0, 23.0, 24.0],
        [0.0, 1.0, 2.0, 3.0],
    ]

    samples_x5 = af.Samples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=[1.0, 2.0, 3.0, 10.0, 5.0],
            log_prior_list=[0.0, 0.0, 0.0, 0.0, 0.0],
            weight_list=[1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    )

    filename = "samples.csv"
    samples_x5.write_table(filename=filename)

    assert Path(filename).exists()
    os.remove(filename)


def test__max_log_likelihood(samples_x5):
    assert samples_x5.max_log_likelihood(as_instance=False) == [21.0, 22.0, 23.0, 24.0]

    instance = samples_x5.max_log_likelihood(as_instance=True)

    assert instance.mock_class_1.one == 21.0
    assert instance.mock_class_1.two == 22.0
    assert instance.mock_class_1.three == 23.0
    assert instance.mock_class_1.four == 24.0


def test__max_log_likelihood__historical_invalid_best_uses_next_valid_instance():
    samples = _guarded_samples(
        parameter_lists=[[0.1], [0.9]],
        log_likelihood_list=[2.0, 1.0],
        weight_list=[0.5, 0.5],
    )

    assert samples.max_log_likelihood(as_instance=False) == [0.1]
    assert samples.max_log_likelihood().value == 0.9


def test__draw_randomly_via_pdf__historical_invalid_draw_is_retried(monkeypatch):
    from autofit.non_linear.samples import pdf

    samples = _guarded_samples(
        parameter_lists=[[0.1], [0.9]],
        log_likelihood_list=[2.0, 1.0],
        weight_list=[0.5, 0.5],
    )
    choices = iter([0, 1])
    monkeypatch.setattr(pdf.np.random, "choice", lambda *args, **kwargs: next(choices))

    assert samples.draw_randomly_via_pdf().value == 0.9


def test__draw_randomly_via_pdf__all_invalid_fails_clearly(monkeypatch):
    from autofit.non_linear.samples import pdf

    samples = _guarded_samples(
        parameter_lists=[[0.1]],
        log_likelihood_list=[1.0],
        weight_list=[1.0],
    )
    monkeypatch.setattr(pdf, "VALID_INSTANCE_MAX_ATTEMPTS", 2)

    with pytest.raises(
        af.exc.SamplesException,
        match="Could not draw a valid model instance.*after 2 attempts",
    ):
        samples.draw_randomly_via_pdf()


def test__max_log_posterior():
    model = af.Collection(mock_class_1=af.m.MockClassx4)

    parameters = [
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [21.0, 22.0, 23.0, 24.0],
    ]

    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=[1.0, 2.0, 3.0, 0.0, 5.0],
            log_prior_list=[1.0, 2.0, 3.0, 10.0, 6.0],
            weight_list=[1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    )

    assert samples_x5.log_posterior_list == [2.0, 4.0, 6.0, 10.0, 11.0]

    assert samples_x5.max_log_posterior(as_instance=False) == [21.0, 22.0, 23.0, 24.0]

    instance = samples_x5.max_log_posterior(as_instance=True)

    assert instance.mock_class_1.one == 21.0
    assert instance.mock_class_1.two == 22.0
    assert instance.mock_class_1.three == 23.0
    assert instance.mock_class_1.four == 24.0


def test__instance_from_sample_index():
    model = af.Collection(mock_class=af.m.MockClassx4)

    parameters = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.1, 2.1, 3.1, 4.1],
    ]

    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=[0.0, 0.0, 0.0, 0.0, 0.0],
            log_prior_list=[0.0, 0.0, 0.0, 0.0, 0.0],
            weight_list=[1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    )

    instance = samples_x5.from_sample_index(sample_index=0, as_instance=True)

    assert instance.mock_class.one == 1.0
    assert instance.mock_class.two == 2.0
    assert instance.mock_class.three == 3.0
    assert instance.mock_class.four == 4.0


def test__samples_above_weight_threshold_from():

    model = af.Collection(mock_class=af.m.MockClassx4)

    parameters = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.1, 2.1, 3.1, 4.1],
    ]

    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=[0.0, 0.0, 0.0, 0.0, 0.0],
            log_prior_list=[0.0, 0.0, 0.0, 0.0, 0.0],
            weight_list=[0.2, 0.2, 1.0, 1.0, 1.0],
        ),
    )

    samples_above_weight_threshold = samples_x5.samples_above_weight_threshold_from(
        weight_threshold=0.5
    )

    assert len(samples_above_weight_threshold) == 3
    assert samples_above_weight_threshold.sample_list[0].weight == 1.0


def test__samples_drawn_randomly_via_pdf_from():

    model = af.Collection(mock_class=af.m.MockClassx4)

    parameters = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.1, 2.1, 3.1, 4.1],
    ]

    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=[0.0, 0.0, 0.0, 0.0, 0.0],
            log_prior_list=[0.0, 0.0, 0.0, 0.0, 0.0],
            weight_list=[0.2, 0.2, 0.2, 0.2, 0.2],
        ),
    )

    samples_drawn_randomly_via_pdf = samples_x5.samples_drawn_randomly_via_pdf_from(
        total_draws=3
    )

    assert len(samples_drawn_randomly_via_pdf) == 3
    assert samples_drawn_randomly_via_pdf.sample_list[0].weight == 0.2


def test__addition_of_samples(samples_x5):
    samples = samples_x5 + samples_x5

    assert len(samples.sample_list) == 10
    assert samples.sample_list[0].log_likelihood == 1.0
    assert samples.sample_list[4].log_likelihood == 5.0
    assert samples.sample_list[5].log_likelihood == 1.0
    assert samples.sample_list[9].log_likelihood == 5.0


def test__sum_of_samples(samples_x5):
    samples = sum([samples_x5, samples_x5, samples_x5])

    assert len(samples.sample_list) == 15
    assert samples.sample_list[0].log_likelihood == 1.0
    assert samples.sample_list[4].log_likelihood == 5.0
    assert samples.sample_list[5].log_likelihood == 1.0
    assert samples.sample_list[9].log_likelihood == 5.0
    assert samples.sample_list[10].log_likelihood == 1.0
    assert samples.sample_list[14].log_likelihood == 5.0


def test__addition_of_samples__raises_error_if_model_mismatch(samples_x5):
    model = af.Collection(mock_class_1=af.m.MockClassx2)

    parameters = [
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [21.0, 22.0],
        [0.0, 1.0],
    ]

    samples_different_model = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=[1.0, 2.0],
            log_prior_list=[0.0, 0.0],
            weight_list=[1.0, 1.0],
        ),
    )

    with pytest.raises(af.exc.SamplesException):
        samples_x5 + samples_different_model


def test__sample_kwargs__mixed_dotted_and_dotless_string_keys():
    sample = af.Sample(
        log_likelihood=1.0,
        log_prior=0.0,
        weight=1.0,
        kwargs={
            "mock_class_1.one": 10.0,
            "dummy_0": 99.0,
        },
    )

    assert all(isinstance(key, tuple) for key in sample.kwargs)
    assert sample.is_path_kwargs is True
    assert sample.kwargs[("dummy_0",)] == 99.0
    assert sample.kwargs[("mock_class_1", "one")] == 10.0

    paths = [
        [("mock_class_1", "one")],
        [("dummy_0",)],
    ]
    assert sample.parameter_lists_for_paths(paths) == [10.0, 99.0]
