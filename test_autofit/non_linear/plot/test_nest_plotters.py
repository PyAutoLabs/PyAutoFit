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

    def latent_parameters_from(self, parameter_array, weights=None, recentre=False):
        # The latent set of this mock: the first two parameters become their sum.
        from autofit.mapper.prior_model.latent_parameters import LatentParameters

        return LatentParameters(
            names=["sum"] + ["p"] * (parameter_array.shape[1] - 2),
            superscripts=[None] * (parameter_array.shape[1] - 1),
            component_paths=[()] * (parameter_array.shape[1] - 1),
            parameter_array=__import__("numpy").column_stack(
                [parameter_array[:, 0] + parameter_array[:, 1], parameter_array[:, 2:]]
            ),
            latent_mask=[True] + [False] * (parameter_array.shape[1] - 2),
        )


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


def test__corner_anesthetic__repeated_labels_still_plot(monkeypatch):
    # A model with several components of one type repeats its LaTeX labels
    # (every Gaussian of a basis is "$\\epsilon_{1}^{g}$"). Keyed by label,
    # anesthetic raised KeyError, @log_plot_exception logged it as an
    # unconverged posterior and no plot was ever written.
    rng = np.random.default_rng(1)
    samples = MockSamples()
    samples.model = MockModel(["$x$", "$e$", "$e$"])
    samples.parameter_lists = rng.normal(size=(200, 3)).tolist()
    samples.weight_list = rng.random(200).tolist()

    written = []
    monkeypatch.setattr(
        nest_plotters, "output_figure", lambda *args, **kwargs: written.append(kwargs)
    )

    import matplotlib.pyplot as plt

    nest_plotters.corner_anesthetic(samples=samples)

    assert written, "corner_anesthetic gave up without writing the figure"
    ylabels = [ax.get_ylabel() for ax in plt.gcf().axes if ax.get_ylabel()]
    assert ylabels.count("$e$") >= 2  # the repeated label is still what is displayed
    plt.close("all")


def test__corner_anesthetic__labels_kwarg_is_reserved(samples):
    with pytest.raises(PlotKwargsError) as error:
        nest_plotters.corner_anesthetic(samples=samples, labels=["a", "b"])

    assert "labels" in str(error.value)


def test__corner_anesthetic__parameter_indices_plot_that_subset(monkeypatch, routed):
    # One corner plot per model component is made by passing the indices of its
    # parameters: only those columns are plotted, with their own labels, and the
    # figure is sized to the subset rather than the whole model.
    from anesthetic.samples import NestedSamples

    rng = np.random.default_rng(2)
    samples = MockSamples()
    samples.model = MockModel(["$a$", "$b$", "$c$"])
    samples.parameter_lists = rng.normal(size=(60, 3)).tolist()
    samples.weight_list = rng.random(60).tolist()

    seen = {}

    def fake_plot_2d(self, axes=None, **kwargs):
        seen["columns"] = self.shape[1]

    monkeypatch.setattr(NestedSamples, "plot_2d", fake_plot_2d)

    nest_plotters.corner_anesthetic(samples=samples, parameter_indices=[0, 2])

    assert seen["columns"] == 2
    assert list(routed["axes"]["labels"].values()) == ["$a$", "$c$"]
    assert routed["axes"]["figsize"] == (8, 8)


def test__corner_anesthetic__latent_plots_the_latent_set(monkeypatch, routed):
    from anesthetic.samples import NestedSamples

    rng = np.random.default_rng(3)
    samples = MockSamples()
    samples.model = MockModel(["$a$", "$b$", "$c$"])
    samples.parameter_lists = rng.normal(size=(60, 3)).tolist()
    samples.weight_list = rng.random(60).tolist()

    seen = {}

    def fake_plot_2d(self, axes=None, **kwargs):
        seen["first"] = self.to_numpy()[0, 0]

    monkeypatch.setattr(NestedSamples, "plot_2d", fake_plot_2d)

    nest_plotters.corner_anesthetic(samples=samples, latent=True)

    # Two parameters became one, labelled from the latent set's own names,
    # and the figure is sized to the latent set.
    assert list(routed["axes"]["labels"].values()) == ["$s$", "$p$"]
    assert seen["first"] == pytest.approx(sum(samples.parameter_lists[0][:2]))
    assert routed["axes"]["figsize"] == (8, 8)


def _samples_with(n_parameters, n_samples=60, seed=4):
    rng = np.random.default_rng(seed)
    samples = MockSamples()
    samples.model = MockModel([f"$x_{i}$" for i in range(n_parameters)])
    samples.parameter_lists = rng.normal(size=(n_samples, n_parameters)).tolist()
    samples.weight_list = rng.random(n_samples).tolist()
    return samples


def test__corner_anesthetic__few_parameters_keep_anesthetic_default_style(routed):
    nest_plotters.corner_anesthetic(samples=_samples_with(3))

    assert "kinds" not in routed["plot"]
    assert "upper" not in routed["axes"]


def test__corner_anesthetic__many_parameters_draw_the_compact_style(routed):
    # A KDE contour panel costs about a second and there are n(n-1)/2 of them,
    # and the upper triangle doubles the axes to build and save, so past the
    # threshold the lower triangle is histograms and the upper one is dropped.
    n = nest_plotters.DEFAULT_MAX_DETAILED_PARAMETERS + 1

    nest_plotters.corner_anesthetic(samples=_samples_with(n))

    assert routed["plot"]["kinds"] == {"diagonal": "kde_1d", "lower": "hist_2d"}
    assert routed["axes"]["upper"] is False


@pytest.mark.parametrize(
    "name, value", [("kind", "kde"), ("kinds", "kde"), ("upper", True)]
)
def test__corner_anesthetic__caller_style_wins_over_the_compact_default(
    routed, name, value
):
    n = nest_plotters.DEFAULT_MAX_DETAILED_PARAMETERS + 1

    nest_plotters.corner_anesthetic(samples=_samples_with(n), **{name: value})

    received = {**routed["axes"], **routed["plot"]}
    assert received[name] == value
    # anesthetic reads ``kinds`` over ``kind``, so no default may sit beside it.
    assert "kinds" not in routed["plot"] or name == "kinds"
    assert routed["axes"].get("upper", True) is True


def _label_state(fig):
    return [
        (
            ax.xaxis.label.get_visible(),
            ax.yaxis.label.get_visible(),
            [label.get_visible() for label in ax.get_xticklabels()],
            [label.get_visible() for label in ax.get_yticklabels()],
        )
        for ax in fig.axes
    ]


def _draw(samples, deferred):
    import matplotlib.pyplot as plt
    from anesthetic import make_2d_axes
    from anesthetic.samples import NestedSamples

    n = len(samples.model.parameter_labels_with_superscripts_latex)
    columns = [f"p{i}" for i in range(n)]
    nested = NestedSamples(
        np.asarray(samples.parameter_lists),
        weights=samples.weight_list,
        columns=columns,
        labels=samples.model.parameter_labels_with_superscripts_latex,
    )
    fig, axes = make_2d_axes(columns, labels=nested.get_labels_map())
    if deferred:
        with nest_plotters._deferred_shared_axes():
            nested.plot_2d(axes, kinds="hist")
    else:
        nested.plot_2d(axes, kinds="hist")
    fig.canvas.draw()
    state = _label_state(fig)
    plt.close(fig)
    return state


def test__deferred_shared_axes__runs_the_label_pass_once_with_the_same_result(
    monkeypatch,
):
    # pandas runs its shared-axis label pass after every panel, over every axis
    # of the figure: n^4 work for an n-parameter corner plot. Deferred, it runs
    # once per distinct call and the inner panels' labels end up hidden just as
    # before.
    from pandas.plotting._matplotlib import core as pandas_core

    samples = _samples_with(4, n_samples=200)
    expected = _draw(samples, deferred=False)

    real = pandas_core.handle_shared_axes
    calls = []

    def counting(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(pandas_core, "handle_shared_axes", counting)

    assert _draw(samples, deferred=True) == expected
    assert len(calls) == 1
    assert pandas_core.handle_shared_axes is counting


def test__deferred_shared_axes__restores_pandas_when_the_plot_raises():
    from pandas.plotting._matplotlib import core as pandas_core

    real = pandas_core.handle_shared_axes

    with pytest.raises(RuntimeError):
        with nest_plotters._deferred_shared_axes():
            raise RuntimeError("panel failed")

    assert pandas_core.handle_shared_axes is real


def test__deferred_shared_axes__positional_call_runs_straight_through(monkeypatch):
    # A future pandas calling it positionally must not lose the label pass.
    from pandas.plotting._matplotlib import core as pandas_core

    calls = []
    monkeypatch.setattr(
        pandas_core, "handle_shared_axes", lambda *args, **kwargs: calls.append(args)
    )

    with nest_plotters._deferred_shared_axes():
        pandas_core.handle_shared_axes(1, 2)

    assert calls == [(1, 2)]


def test__corner_anesthetic__edge_panels_show_latex_labels_not_column_keys(
    monkeypatch,
):
    # The samples are keyed p0, p1, ... without labels (labelled columns are
    # copied for every panel, half the cost of a big plot), so the LaTeX labels
    # must be put back on the axes after plot_2d names them by column.
    import matplotlib.pyplot as plt

    written = []
    monkeypatch.setattr(
        nest_plotters, "output_figure", lambda *args, **kwargs: written.append(kwargs)
    )

    nest_plotters.corner_anesthetic(samples=_samples_with(3, n_samples=200), kinds="hist")

    assert written
    # Inner panels keep a hidden column-name label, as before; only what is
    # drawn matters.
    shown = {
        axis.label.get_text()
        for ax in plt.gcf().axes
        for axis in (ax.xaxis, ax.yaxis)
        if axis.get_visible() and axis.label.get_visible() and axis.label.get_text()
    }
    assert shown == {"$x_0$", "$x_1$", "$x_2$"}
    plt.close("all")
