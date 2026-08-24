import numpy as np
import pytest

from autofit.non_linear.plot import samples_plotters
from autofit.non_linear.plot.plot_util import PlotKwargsError
from autofit.non_linear.plot.samples_plotters import _corner_range_from


def test__corner_range__real_spread_columns_preserved():
    data = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ]
    )

    plot_range = _corner_range_from(data)

    assert plot_range == [(1.0, 3.0), (10.0, 30.0)]


def test__corner_range__degenerate_columns_widened_to_nonzero_span():
    # Columns 0, 1, 2 have no dynamic range (all samples equal) — the input
    # that made `corner` raise "no dynamic range". Column 3 has real spread.
    data = np.array(
        [
            [0.0, 5.0, -2.0, 1.0],
            [0.0, 5.0, -2.0, 2.0],
            [0.0, 5.0, -2.0, 3.0],
        ]
    )

    plot_range = _corner_range_from(data)

    # Every column ends up with a strictly positive width, so `corner` gets a
    # valid range for all of them and does not raise.
    for lower, upper in plot_range:
        assert upper > lower

    # Degenerate windows stay centred on the constant value.
    for (lower, upper), value in zip(plot_range[:3], [0.0, 5.0, -2.0]):
        assert lower < value < upper

    # The real-spread column is left untouched.
    assert plot_range[3] == (1.0, 3.0)


class MockModel:
    def __init__(self, labels):
        self.parameter_labels_with_superscripts_latex = labels
        self.total_free_parameters = len(labels)


class MockSamples:
    def __init__(self, parameter_lists, weight_list=None):
        self.parameter_lists = parameter_lists
        self.weight_list = (
            weight_list if weight_list is not None else [1.0] * len(parameter_lists)
        )
        self.model = MockModel(["x", "y"])


@pytest.fixture
def samples():
    rng = np.random.default_rng(0)
    return MockSamples(
        parameter_lists=rng.normal(size=(50, 2)).tolist(),
        weight_list=rng.random(50).tolist(),
    )


@pytest.fixture
def corner_call(monkeypatch):
    """Capture the arguments `corner_cornerpy` hands to `corner.corner`."""
    import corner
    import functools

    captured = {}

    # `functools.wraps` keeps the real signature reachable via `__wrapped__`, so
    # the guard still validates against corner's genuine parameter list rather
    # than against this stub's bare `**kwargs`.
    @functools.wraps(corner.corner)
    def fake_corner(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(corner, "corner", fake_corner)
    monkeypatch.setattr(
        samples_plotters, "output_figure", lambda *args, **kwargs: None
    )
    return captured


def test__corner_cornerpy__weights_reach_corner_under_its_own_name(
    samples, corner_call
):
    # Regression: the call passed `weight_list=`, which `corner.corner` does not
    # name — it landed in the hist2d sink and every weighted posterior was
    # plotted unweighted, with no error.
    samples_plotters.corner_cornerpy(samples=samples)

    assert "weight_list" not in corner_call
    assert corner_call["weights"] == samples.weight_list


def test__corner_cornerpy__caller_kwarg_is_forwarded(samples, corner_call):
    samples_plotters.corner_cornerpy(samples=samples, bins=5, show_titles=True)

    assert corner_call["bins"] == 5
    assert corner_call["show_titles"] is True


def test__corner_cornerpy__hist2d_pass_through_kwarg_is_accepted(samples, corner_call):
    # `plot_datapoints` is named by `corner.core.hist2d`, not `corner.corner` —
    # a documented pass-through, so it must survive the guard.
    samples_plotters.corner_cornerpy(samples=samples, plot_datapoints=False)

    assert corner_call["plot_datapoints"] is False


def test__corner_cornerpy__unknown_kwarg_raises_naming_it(samples, corner_call):
    with pytest.raises(PlotKwargsError) as error:
        samples_plotters.corner_cornerpy(samples=samples, panelsize=3.5)

    assert "panelsize" in str(error.value)


def test__corner_cornerpy__wrong_library_kwarg_hints_the_right_name(
    samples, corner_call
):
    # What `autofit_workspace/scripts/plot/zeus_plotter.py` passed: zeus's name
    # for the weights. It must not be silently dropped, and the error should
    # point at corner's spelling.
    with pytest.raises(PlotKwargsError) as error:
        samples_plotters.corner_cornerpy(samples=samples, weight_list=None)

    assert "weight_list" in str(error.value)
    assert "weights" in str(error.value)


def test__corner_cornerpy__reserved_kwarg_cannot_be_overridden(samples, corner_call):
    with pytest.raises(PlotKwargsError) as error:
        samples_plotters.corner_cornerpy(samples=samples, data=[[1.0, 2.0]])

    assert "data" in str(error.value)


def test__corner_cornerpy__range_none_keeps_the_degenerate_column_guard(
    samples, corner_call
):
    # `emcee_plotter.py` passes `range=None`. Honouring that literally would hand
    # `corner` back the degenerate columns `_corner_range_from` exists to widen.
    samples_plotters.corner_cornerpy(samples=samples, range=None)

    assert corner_call["range"] == _corner_range_from(
        np.asarray(samples.parameter_lists)
    )


def test__corner_cornerpy__explicit_range_wins(samples, corner_call):
    samples_plotters.corner_cornerpy(samples=samples, range=[(0.0, 1.0), (0.0, 1.0)])

    assert corner_call["range"] == [(0.0, 1.0), (0.0, 1.0)]


def test__corner_cornerpy__degenerate_columns_still_render(monkeypatch):
    # The real `corner`, not the fake: constant columns must not reintroduce the
    # "no dynamic range" crash now that `range` is caller-overridable.
    import matplotlib

    matplotlib.use("Agg")
    monkeypatch.setattr(
        samples_plotters, "output_figure", lambda *args, **kwargs: None
    )

    degenerate = MockSamples(parameter_lists=[[1.0, 5.0]] * 20)

    samples_plotters.corner_cornerpy(samples=degenerate, bins=5)
