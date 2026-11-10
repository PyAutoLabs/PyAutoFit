"""Exact same-runtime parity: no image tolerance is permitted by this adapter."""
import matplotlib
import numpy as np
import pytest
from copy import deepcopy

matplotlib.use("Agg")

from autofit.non_linear.plot import compact_corner, nest_plotters
from autofit.non_linear.plot.plot_util import PlotKwargsError
from test_autofit.non_linear.plot.test_nest_plotters import _samples_with, _label_state


@pytest.fixture(autouse=True)
def cleanup():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def require_supported():
    if any(compact_corner.version(p) != v for p, v in compact_corner.VERSIONS.items()):
        pytest.skip("Real-runtime equivalence requires the explicitly supported versions")


def render(samples, backend, monkeypatch, **kwargs):
    import matplotlib.pyplot as plt

    written = []
    monkeypatch.setattr(nest_plotters, "output_figure", lambda **kw: written.append(kw))
    np.random.seed(10)
    nest_plotters.corner_anesthetic(
        samples, compact_backend=backend, kinds=compact_corner.KINDS,
        upper=False, figsize=(5, 5), dpi=70, **deepcopy(kwargs)
    )
    assert written, "A swallowed plotting error is not successful fallback"
    fig = plt.gcf()
    fig.canvas.draw()
    return fig, np.asarray(fig.canvas.buffer_rgba()).copy()


@pytest.mark.parametrize("options", [
    {}, {"q": .8, "density": True, "color": "darkred", "alpha": .4},
    {"lower_kwargs": {"bins": 17, "q": (0.1, .9)}},
    {"lower_kwargs": {"bins": [7, 12], "range": [[-2, 2], [-3, 3]],
                      "levels": [.95, .68], "cmin": .01, "cmax": 100}},
    {"diagonal_kwargs": {"bw_scale": 1.5, "ngrid_kde": 64}},
    {"parameter_indices": [2, 0]},
])
def test_real_runtime_exact_parity(monkeypatch, options):
    require_supported()
    samples = _samples_with(3, n_samples=120)
    samples.model.parameter_labels_with_superscripts_latex = ["$x$", "$x$", "$z$"]
    samples.weight_list[0:10] = [0.] * 10
    baseline, original = render(samples, "anesthetic", monkeypatch, **options)
    calls = []
    draw = compact_corner.draw
    def witnessed(*args):
        calls.append(True)
        return draw(*args)
    monkeypatch.setattr(compact_corner, "draw", witnessed)
    candidate, actual = render(samples, "array", monkeypatch, **options)
    assert calls == [True]
    assert _label_state(candidate) == _label_state(baseline)
    for a, b in zip(candidate.axes, baseline.axes):
        assert len(a.lines) == len(b.lines)
        assert len(a.collections) == len(b.collections)
        for left, right in zip(a.lines, b.lines):
            np.testing.assert_array_equal(left.get_xydata(), right.get_xydata())
        for left, right in zip(a.collections, b.collections):
            np.testing.assert_array_equal(left.get_array(), right.get_array())
            np.testing.assert_array_equal(left.get_coordinates(), right.get_coordinates())
    np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("mutation", ["nan", "float32", "constant", "negative", "version", "style", "label_none"])
def test_unsupported_inputs_fall_back_before_drawing(monkeypatch, mutation):
    samples = _samples_with(3, n_samples=80)
    options = {}
    if mutation == "nan":
        samples.parameter_lists[0][0] = float("nan")
    elif mutation == "float32":
        samples.parameter_lists = np.asarray(samples.parameter_lists, dtype=np.float32)
    elif mutation == "constant":
        samples.parameter_lists = np.asarray(samples.parameter_lists)
        samples.parameter_lists[:, 0] = 1.
        assert compact_corner.prepare(samples.parameter_lists, samples.weight_list,
                                      {"upper": False}, {"kinds": compact_corner.KINDS}) is None
        # The baseline KDE itself fails for this degenerate fixture; the adapter
        # must delegate, not present an empty figure as a successful new path.
        from anesthetic.samples import NestedSamples
        seen = []
        monkeypatch.setattr(NestedSamples, "plot_2d", lambda *a, **kw: seen.append(True))
        monkeypatch.setattr(compact_corner, "draw", lambda *a: pytest.fail("partial draw"))
        render(samples, "array", monkeypatch)
        assert seen == [True]
        return
    elif mutation == "negative":
        # Probe preparation only: negative weights are not valid KDE data.
        weights = np.asarray(samples.weight_list)
        weights[0] = -1
        assert compact_corner.prepare(np.asarray(samples.parameter_lists), weights,
                                      {"upper": False}, {"kinds": compact_corner.KINDS}) is None
        return
    elif mutation == "version":
        monkeypatch.setattr(compact_corner, "version", lambda p: "unvalidated")
    elif mutation == "label_none":
        options["label"] = None
    else:
        options["diagonal_kwargs"] = {"linewidth": 2}
    monkeypatch.setattr(compact_corner, "draw", lambda *a: pytest.fail("fallback drew partially"))
    baseline, original = render(samples, "anesthetic", monkeypatch, **options)
    candidate, actual = render(samples, "array", monkeypatch, **options)
    np.testing.assert_array_equal(actual, original)


def test_invalid_backend_is_not_swallowed():
    with pytest.raises(PlotKwargsError, match="compact_backend"):
        nest_plotters.corner_anesthetic(_samples_with(2), compact_backend="unknown")


def test_quantile_cache_is_local_and_per_column(monkeypatch):
    require_supported()
    import anesthetic.utils

    calls = []
    original = anesthetic.utils.quantile
    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(anesthetic.utils, "quantile", count)
    samples = _samples_with(4)
    for _ in range(2):
        prepared = compact_corner.prepare(np.asarray(samples.parameter_lists), samples.weight_list,
                                         {"upper": False}, {"kinds": compact_corner.KINDS})
        assert prepared is not None
    assert len(calls) == 8


def test_automatic_compact_style_and_global_state(monkeypatch):
    require_supported()
    import matplotlib.pyplot as plt
    from pandas.plotting._matplotlib import core

    samples = _samples_with(11, n_samples=60)
    samples.weight_list = [1.] * 60
    handler = core.handle_shared_axes
    images = []
    monkeypatch.setattr(nest_plotters, "output_figure", lambda **kw: None)
    for backend in ("anesthetic", "array"):
        nest_plotters.corner_anesthetic(samples, compact_backend=backend, figsize=(6, 6), dpi=60)
        fig = plt.gcf()
        assert len(fig.axes) == 77
        fig.canvas.draw()
        images.append(np.asarray(fig.canvas.buffer_rgba()).copy())
        assert core.handle_shared_axes is handler
        plt.close(fig)
    np.testing.assert_array_equal(*images)
