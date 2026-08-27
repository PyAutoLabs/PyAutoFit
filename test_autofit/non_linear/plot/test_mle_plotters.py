import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from autofit.non_linear.plot import mle_plotters
from autofit.non_linear.plot.plot_util import PlotKwargsError


class MockModel:
    def __init__(self, labels):
        self.parameter_labels_with_superscripts_latex = labels
        self.total_free_parameters = len(labels)


class MockSamples:
    def __init__(self):
        rng = np.random.default_rng(0)
        self.model = MockModel(["x", "y"])
        self.parameters_extract = rng.normal(size=(2, 30)).tolist()
        self.log_likelihood_list = rng.normal(size=30).tolist()
        self.samples_info = {"fom_history": rng.random(30).tolist()}


@pytest.fixture
def samples():
    return MockSamples()


@pytest.fixture(autouse=True)
def no_output(monkeypatch):
    monkeypatch.setattr(mle_plotters, "output_figure", lambda *args, **kwargs: None)


@pytest.mark.parametrize(
    "plot",
    [
        mle_plotters.subplot_parameters,
        mle_plotters.log_likelihood_vs_iteration,
        mle_plotters.figure_of_merit_vs_iteration,
    ],
)
def test__line_kwarg_is_forwarded_to_the_trace(plot, samples):
    plot(samples=samples, linewidth=3.0)


@pytest.mark.parametrize(
    "plot",
    [
        mle_plotters.subplot_parameters,
        mle_plotters.log_likelihood_vs_iteration,
        mle_plotters.figure_of_merit_vs_iteration,
    ],
)
def test__unknown_line_kwarg_is_rejected_and_named(plot, samples):
    with pytest.raises(PlotKwargsError) as error:
        plot(samples=samples, yticksize=16)

    assert "yticksize" in str(error.value)


def test__caller_colour_overrides_the_default_black_trace(samples, monkeypatch):
    import matplotlib.pyplot as plt

    captured = {}

    def fake_plot(*args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(plt, "plot", fake_plot)

    mle_plotters.log_likelihood_vs_iteration(samples=samples, c="red")

    assert captured["c"] == "red"


def test__default_trace_colour_is_black_when_no_kwarg_given(samples, monkeypatch):
    import matplotlib.pyplot as plt

    captured = {}

    def fake_plot(*args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(plt, "plot", fake_plot)

    mle_plotters.log_likelihood_vs_iteration(samples=samples)

    assert captured["c"] == "k"
