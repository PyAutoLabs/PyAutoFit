import csv

import numpy as np
import pytest

from autofit import graphical as graph
from autofit.graphical.expectation_propagation.diagnostics import EPDiagnostics
from autofit.mapper.variable import Variable
from autofit.messages.normal import NormalMessage
from autofit.non_linear.paths.directory import DirectoryPaths


def make_model_approx():
    """
    A tiny exact-conjugate normal/normal EP graph (prior_x * like_x on a
    single scalar variable x). Cheap enough to run to convergence in a
    handful of factor updates, which is all these diagnostics tests need.
    """
    x = Variable("x")
    prior = NormalMessage(1.0, 2.0).as_factor(x, name="prior_x")
    likelihood = NormalMessage(3.0, 0.5).as_factor(x, name="like_x")
    fg = graph.FactorGraph([prior, likelihood])
    model_approx = graph.EPMeanField.from_approx_dists(fg, {x: NormalMessage(0.0, 10.0)})
    return model_approx, x


def test_snapshot_records_rows():
    model_approx, x = make_model_approx()

    opt = graph.EPOptimiser.from_meanfield(model_approx, paths=False)
    opt.run(model_approx, max_steps=4)

    factor_rows = opt.diagnostics.factor_rows
    assert factor_rows

    expected_columns = {
        "step",
        "factor",
        "success",
        "updated",
        "flag",
        "log_evidence",
        "kl_divergence",
    }
    for row in factor_rows:
        assert set(row.keys()) == expected_columns

    variable_rows = opt.diagnostics.variable_rows
    assert variable_rows

    # one variable_row per (step, variable) -- here there is a single
    # variable "x", so the count matches the number of factor updates
    steps = [row["step"] for row in variable_rows]
    assert len(steps) == len(set(steps))
    assert len(variable_rows) == len(factor_rows)

    # kl_divergence lives on factor_rows, not variable_rows -- first step
    # has no previous mean field to diverge from, so it is NaN.
    assert np.isnan(factor_rows[0]["kl_divergence"])
    assert all(np.isfinite(row["kl_divergence"]) for row in factor_rows[1:])


def test_csv_outputs_written(tmp_path):
    model_approx, x = make_model_approx()

    paths = DirectoryPaths(name="ep_diag_test", path_prefix=str(tmp_path))
    opt = graph.EPOptimiser.from_meanfield(model_approx, paths=paths)
    opt.run(model_approx, max_steps=4)

    output_path = opt.output_path
    assert output_path is not None

    ep_history_path = output_path / "ep_history.csv"
    mean_field_history_path = output_path / "mean_field_history.csv"
    assert ep_history_path.exists()
    assert mean_field_history_path.exists()

    with open(ep_history_path, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "step",
            "factor",
            "success",
            "updated",
            "flag",
            "log_evidence",
            "kl_divergence",
        ]
        ep_rows = list(reader)
    assert len(ep_rows) == len(opt.diagnostics.factor_rows)

    with open(mean_field_history_path, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["step", "factor", "variable", "mean", "std"]
        mf_rows = list(reader)
    assert len(mf_rows) == len(opt.diagnostics.variable_rows)

    assert (output_path / "mean_field_evolution.png").exists()
    assert (output_path / "graph_factors.png").exists()

    results_path = output_path / "ep_diagnostics.results"
    assert results_path.exists()
    results_text = results_path.read_text()
    assert "x" in results_text
    assert "WARNINGS" not in results_text


def test_mean_field_summary():
    model_approx, x = make_model_approx()

    opt = graph.EPOptimiser.from_meanfield(model_approx, paths=False)
    result = opt.run(model_approx, max_steps=4)

    summary = graph.mean_field_summary(result.mean_field)
    assert "variable" in summary
    assert "x" in summary

    message = result.mean_field[x]
    assert f"{message.mean:.6g}" in summary
    assert f"{message.std:.6g}" in summary


def test_sigma_collapse_floor():
    diagnostics = EPDiagnostics()
    diagnostics.variable_rows = [
        {"step": 0, "factor": "f", "variable": "collapsed", "mean": 0.0, "std": 1e-12},
    ]

    warnings_list = graph.check_sigma_collapse(diagnostics)

    assert len(warnings_list) == 1
    assert "collapsed" in warnings_list[0]
    assert "floor" in warnings_list[0]


def test_sigma_collapse_monotone():
    diagnostics = EPDiagnostics()

    shrinking_stds = np.geomspace(1.0, 1e-5, num=8)
    healthy_stds = [1.0, 1.05, 0.95, 1.02, 0.98, 1.01, 0.99, 1.0]

    rows = []
    for step, std in enumerate(shrinking_stds):
        rows.append(
            {"step": step, "factor": "f", "variable": "shrinking", "mean": 0.0, "std": float(std)}
        )
    for step, std in enumerate(healthy_stds):
        rows.append(
            {"step": step, "factor": "f", "variable": "healthy", "mean": 0.0, "std": float(std)}
        )
    diagnostics.variable_rows = rows

    warnings_list = graph.check_sigma_collapse(diagnostics)

    shrinking_warnings = [w for w in warnings_list if "shrinking" in w]
    healthy_warnings = [w for w in warnings_list if "healthy" in w]

    assert len(shrinking_warnings) == 1
    assert "monotonically" in shrinking_warnings[0]
    assert healthy_warnings == []


def test_no_collapse_on_healthy_run():
    model_approx, x = make_model_approx()

    opt = graph.EPOptimiser.from_meanfield(model_approx, paths=False)
    opt.run(model_approx, max_steps=4)

    assert graph.check_sigma_collapse(opt.diagnostics) == []


def test_parallel_end_of_run_guards():
    """
    F3: a real ParallelEPOptimiser run needs n_cores >= 3 and spins up a
    multiprocessing.Pool. The only existing ParallelEPOptimiser test in
    this repo, `_test_parallel_laplace` in
    test_autofit/graphical/regression/test_linear_regression.py, is
    prefixed with an underscore so pytest never collects it -- i.e. a
    real parallel run is already established as impractical to exercise
    routinely under this test suite (pool startup cost / environment
    flakiness), so we don't add a second one here.

    Instead we verify directly the end-of-run diagnostics guards that
    ParallelEPOptimiser.run shares with EPOptimiser.run (both call
    self._output_diagnostics(...) and self._warn_sigma_collapse() after
    the main loop): with paths=False, self.visualiser stays None and
    self.output_path stays None, and both calls must be no-ops rather
    than raising.
    """
    model_approx, x = make_model_approx()

    opt = graph.EPOptimiser.from_meanfield(model_approx, paths=False)

    assert opt.visualiser is None
    assert opt.output_path is None

    # no snapshots taken yet -- factor_rows/variable_rows are empty, which
    # is the same state ParallelEPOptimiser.run's guards must tolerate
    # before touching self.output_path / self.visualiser.
    opt._output_diagnostics()
    opt._output_diagnostics(final=True, model_approx=model_approx)
    opt._warn_sigma_collapse()
