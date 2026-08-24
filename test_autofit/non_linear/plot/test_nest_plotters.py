import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from autofit.non_linear.plot import nest_plotters
from autofit.non_linear.plot.plot_util import PlotKwargsError


class MockModel:
    def __init__(self, labels):
        self.parameter_labels_with_superscripts_latex = labels
        self.total_free_parameters = len(labels)


class MockSamples:
    def __init__(self):
        rng = np.random.default_rng(0)
        self.model = MockModel(["x", "y"])
        self.parameter_lists = rng.normal(size=(60, 2)).tolist()
        self.weight_list = rng.random(60).tolist()


@pytest.fixture
def samples():
    return MockSamples()


@pytest.fixture(autouse=True)
def no_output(monkeypatch):
    monkeypatch.setattr(nest_plotters, "output_figure", lambda *args, **kwargs: None)


@pytest.fixture
def routed(monkeypatch):
    """Capture which of anesthetic's two entry points each kwarg reached."""
    import anesthetic
    from anesthetic.samples import NestedSamples

    captured = {"axes": {}, "plot": {}}

    real_make_2d_axes = anesthetic.make_2d_axes

    def fake_make_2d_axes(params, **kwargs):
        captured["axes"].update(kwargs)
        return real_make_2d_axes(params)

    def fake_plot_2d(self, axes=None, **kwargs):
        captured["plot"].update(kwargs)

    monkeypatch.setattr(anesthetic, "make_2d_axes", fake_make_2d_axes)
    monkeypatch.setattr(NestedSamples, "plot_2d", fake_plot_2d)
    return captured


def test__corner_anesthetic__axes_kwarg_is_routed_to_the_figure(samples, routed):
    # `figsize` is named by `make_2d_axes`, so it shapes the figure rather than
    # being handed to `plot_2d`.
    nest_plotters.corner_anesthetic(samples=samples, figsize=(4, 4))

    assert routed["axes"]["figsize"] == (4, 4)
    assert "figsize" not in routed["plot"]


def test__corner_anesthetic__style_kwarg_is_routed_to_the_plot(samples, routed):
    nest_plotters.corner_anesthetic(samples=samples, alpha=0.25)

    assert routed["plot"]["alpha"] == 0.25
    assert "alpha" not in routed["axes"]


def test__corner_anesthetic__reserved_kwarg_is_not_swallowed_by_the_decorator(samples):
    # `corner_anesthetic` is wrapped in `@log_plot_exception`, which catches
    # `TypeError`. Without the `PlotKwargsError` re-raise this would be reported
    # as an unconverged posterior and the caller would never see their mistake.
    with pytest.raises(PlotKwargsError) as error:
        nest_plotters.corner_anesthetic(samples=samples, weights=[1.0])

    assert "weights" in str(error.value)
