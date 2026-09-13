"""
The EP hook: `graph_model.png` and `graph_state.png` beside `graph.info`.

`Visualise` already writes the convergence curves (`graph.png`,
`graph_factors.png`) on every `visualise_interval` tick. Behind `output.yaml`'s
`model_figure` key it now also writes the factor graph itself -- once as
structure, and once per tick with the run painted on it.

Three things are asserted here that no unit test of the figure layers can see,
because they are properties of the *run*:

* the model view is written once and the state view every tick, so the state
  file always shows the sweep that just finished;
* a config which says nothing about `model_figure` writes neither file, which
  is what keeps the figure off for every existing user;
* a renderer that raises leaves the fit intact -- the run returns, the
  convergence curves are still written, and the failure goes to the log.

The graph is the two-factor, one-shared-variable double of
`test_factor_failure_recovery` fitted by exact factor fits, so a three-sweep
run is milliseconds.
"""

import matplotlib

matplotlib.use("Agg")

import pytest

from autofit import graphical as graph
from autofit.graphical.expectation_propagation.factor_optimiser import ExactFactorFit
from autofit.graphical.expectation_propagation.history import EPHistory
from autofit.non_linear.paths.directory import DirectoryPaths

from test_autofit.graphical.functionality.test_factor_failure_recovery import (
    make_shared_variable_approx,
)
from test_autofit.non_linear.paths.test_model_figure_output import model_figure_key

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

MAX_STEPS = 3


def run_ep(tmp_path, name="ep_figures", max_steps=MAX_STEPS):
    """
    A real `EPOptimiser.run` with `visualise_interval=1`, writing into
    `tmp_path`. Returns the optimiser's output directory.
    """
    model_approx, factor_graph, prior, likelihood = make_shared_variable_approx()

    optimiser = graph.EPOptimiser(
        factor_graph,
        factor_optimisers={prior: ExactFactorFit(), likelihood: ExactFactorFit()},
        # `kl_tol=None` disables the convergence check: this graph is exact and
        # would otherwise terminate after one sweep, before a second tick.
        ep_history=EPHistory(kl_tol=None),
        paths=DirectoryPaths(name=name, path_prefix=str(tmp_path)),
    )
    optimiser.run(
        model_approx,
        max_steps=max_steps,
        visualise_interval=1,
        max_consecutive_failures=100,
    )
    return optimiser.output_path


def written(output_path, filename):
    target = output_path / filename

    assert target.exists(), f"{target} was not written"
    contents = target.read_bytes()
    assert contents[:8] == PNG_MAGIC
    assert len(contents) > 1024, f"{target} is {len(contents)} bytes -- empty canvas"
    return target


@pytest.fixture
def figure_kinds(monkeypatch):
    """
    Every `EPPlotter.figure` call's `kind`, in order, with the real figure
    still drawn.

    Call counts are what prove the state file is refreshed per tick; comparing
    bytes cannot, because two sweeps of an exactly converging graph render
    identically.
    """
    from autofit.model_figure.ep.plotter import EPPlotter

    original = EPPlotter.figure
    kinds = []

    def spy(self, *args, **kwargs):
        kinds.append(kwargs.get("kind", "model"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(EPPlotter, "figure", spy)
    return kinds


def test_both_figures_are_written_beside_the_convergence_curves(tmp_path):
    with model_figure_key(True):
        output_path = run_ep(tmp_path)

    written(output_path, "graph_model.png")
    written(output_path, "graph_state.png")
    # the curves the hook must not have displaced
    assert (output_path / "graph.png").exists()
    assert (output_path / "graph_factors.png").exists()


def test_the_model_view_is_drawn_once_and_the_state_view_every_tick(
    tmp_path, figure_kinds
):
    """
    Structure cannot change mid-run, so `graph_model.png` is written on the
    first tick and never again; the state view is redrawn on every one.
    """
    with model_figure_key(True):
        run_ep(tmp_path)

    assert figure_kinds.count("model") == 1
    assert figure_kinds[0] == "model"

    # one tick per sweep plus the final call at the end of `run`
    assert figure_kinds.count("state") == MAX_STEPS + 1
    assert figure_kinds.count("state") == len(figure_kinds) - 1


def test_the_state_file_is_rewritten_rather_than_left_at_the_first_sweep(tmp_path):
    """
    The file the last tick wrote is the one on disk: its modification time is
    at or after the model view's, which only the first tick writes.
    """
    with model_figure_key(True):
        output_path = run_ep(tmp_path)

    model = written(output_path, "graph_model.png")
    state = written(output_path, "graph_state.png")

    assert state.stat().st_mtime_ns >= model.stat().st_mtime_ns


@pytest.mark.parametrize("value", [False, None])
def test_nothing_is_written_when_the_key_is_off_or_absent(tmp_path, value):
    """
    `None` is the case that matters most: a config which has never heard of
    `model_figure` must write no figure, even though `output.yaml`'s `default:`
    is `true`.
    """
    with model_figure_key(value):
        output_path = run_ep(tmp_path)

    assert not (output_path / "graph_model.png").exists()
    assert not (output_path / "graph_state.png").exists()
    # ... and the curves are unaffected either way
    assert (output_path / "graph.png").exists()


def test_a_raising_plotter_leaves_the_run_intact(tmp_path, monkeypatch, caplog):
    """
    A figure is a diagnostic, never a dependency: if it cannot be drawn the
    sweep still finishes, the run still returns and the reason goes to the log.
    """
    import logging

    from autofit.model_figure.ep.plotter import EPPlotter

    def explode(self, *args, **kwargs):
        raise RuntimeError("no pixels today")

    monkeypatch.setattr(EPPlotter, "figure", explode)

    with caplog.at_level(logging.INFO):
        with model_figure_key(True):
            output_path = run_ep(tmp_path)

    assert not (output_path / "graph_model.png").exists()
    assert (output_path / "graph.png").exists()
    assert any("no pixels today" in record.message for record in caplog.records)


def test_no_figure_is_drawn_without_a_factor_graph(tmp_path):
    """
    Every other caller of `Visualise` -- `stochastic.py` -- constructs it
    without a graph and calls it with no arguments. That must go on working,
    and draw no factor-graph figure.
    """
    from autofit.graphical.expectation_propagation.visualise import Visualise

    visualiser = Visualise(EPHistory(kl_tol=None), tmp_path)

    with model_figure_key(True):
        visualiser()

    assert visualiser.factor_graph is None
    assert not (tmp_path / "graph_model.png").exists()
    assert (tmp_path / "graph.png").exists()
