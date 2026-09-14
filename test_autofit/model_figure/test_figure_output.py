"""
``ModelPlotter.figure`` output -- the ``show | png | svg | pdf`` branching of
``autofit/non_linear/plot/plot_util.py:output_figure``, with ``svg`` added.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

import autofit as af


@pytest.fixture
def model():
    return af.Model(af.ex.Gaussian)


@pytest.mark.parametrize("format", ["png", "svg", "pdf"])
def test_figure_writes_the_file(model, tmp_path, format):
    figure = af.ModelPlotter(model).figure(
        path=tmp_path, filename="model", format=format
    )

    written = tmp_path / f"model.{format}"
    assert written.exists()
    assert written.stat().st_size > 0
    assert figure is not None


def test_filename_is_honoured(model, tmp_path):
    af.ModelPlotter(model).figure(path=tmp_path, filename="my_model", format="png")

    assert (tmp_path / "my_model.png").exists()


def test_a_missing_directory_is_created(model, tmp_path):
    target = tmp_path / "deep" / "nested"
    af.ModelPlotter(model).figure(path=target, format="png")

    assert (target / "model.png").exists()


def test_show_returns_nothing_so_a_notebook_draws_it_once(model, tmp_path, monkeypatch):
    """
    The show path shows the figure and hands nothing back.

    A returned ``Figure`` is rendered a second time by IPython's ``image/png``
    formatter, as the ``execute_result`` of a cell ending on
    ``af.ModelPlotter(model).figure()`` -- so the tutorials drew every model
    twice.  ``output_figure``, the convention this mirrors, returns nothing and
    always closes; so does this.
    """
    shown = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shown.append(True))
    open_before = set(plt.get_fignums())

    figure = af.ModelPlotter(model).figure(path=tmp_path, format="show")

    assert shown == [True]
    assert figure is None
    assert set(plt.get_fignums()) == open_before
    assert list(tmp_path.iterdir()) == []


def test_an_unknown_format_raises_rather_than_silently_doing_nothing(model, tmp_path):
    with pytest.raises(ValueError):
        af.ModelPlotter(model).figure(path=tmp_path, format="jpeg")


def test_the_spec_is_built_once_and_cached(model):
    plotter = af.ModelPlotter(model)

    assert plotter.spec() is plotter.spec()
    assert plotter.spec(collapse=False) is not plotter.spec(collapse=True)


def test_detail_must_be_names_or_priors(model):
    with pytest.raises(ValueError):
        af.ModelPlotter(model).presentation(detail="everything")
