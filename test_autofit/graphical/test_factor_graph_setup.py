"""
Factor graphs forward `modify_before_fit` to each factor's analysis and check that the
factors resolved the same compute setup (likelihood-lifecycle phase 5D).

Dummy analyses carry a duck-typed `setup` with a `setup_id`; nothing imports autoarray.
"""
import logging

import numpy as np
import pytest

import autofit as af
import autofit.graphical as g
from autofit import exc
from autofit.graphical.declarative.factor.analysis import FactorCallable
from autofit.non_linear.parallel import pool as pool_module
from autofit.non_linear.parallel import setup_id_of

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


class _Setup:
    def __init__(self, setup_id):
        self.setup_id = setup_id


class _Analysis(af.Analysis):
    """
    Records each `modify_before_fit` call and returns a *new* analysis carrying the
    resolved setup, so the test can see both references are rebound.
    """

    def __init__(self, setup_id=None, calls=None):
        super().__init__()
        self.setup_id = setup_id
        self.calls = calls if calls is not None else []
        self.modified = False

    def log_likelihood_function(self, instance):
        return 1.0 if self.modified else 0.0

    def modify_before_fit(self, paths, model):
        self.calls.append((paths, model))
        new = _Analysis(setup_id=self.setup_id, calls=self.calls)
        new.modified = True
        if self.setup_id is not None:
            new.setup = _Setup(self.setup_id)
        return new


def _factor(analysis, name):
    return g.AnalysisFactor(
        af.Model(af.ex.Gaussian), analysis=analysis, name=name
    )


def test_forwarding_calls_each_factor_once_with_its_model_and_sub_paths():
    calls_a, calls_b = [], []
    factor_a = _factor(_Analysis("numba_cpu_sparse_fp64", calls_a), "a")
    factor_b = _factor(_Analysis("numba_cpu_sparse_fp64", calls_b), "b")
    graph = g.FactorGraphModel(factor_a, factor_b)

    paths = af.DirectoryPaths(name="graph_setup")
    model = graph.global_prior_model

    assert graph.modify_before_fit(paths=paths, model=model) is graph

    assert len(calls_a) == 1 and len(calls_b) == 1
    child_paths_a, model_a = calls_a[0]
    child_paths_b, model_b = calls_b[0]
    assert "analyses/analysis_0" in str(child_paths_a.output_path)
    assert "analyses/analysis_1" in str(child_paths_b.output_path)
    assert model_a.prior_count == factor_a.prior_model.prior_count
    assert model_a is model[0]
    assert model_b is model[1]

    for factor in (factor_a, factor_b):
        assert factor.analysis.modified
        assert factor._factor.analysis is factor.analysis
        assert isinstance(factor._factor, FactorCallable)

    assert graph.setup_ids() == {
        "a": "numba_cpu_sparse_fp64",
        "b": "numba_cpu_sparse_fp64",
    }
    assert setup_id_of(graph) == "numba_cpu_sparse_fp64"


def test_rebound_analysis_is_what_the_graph_evaluates():
    factor = _factor(_Analysis(), "a")
    graph = g.FactorGraphModel(factor)
    instance = graph.global_prior_model.instance_from_prior_medians()

    assert graph.log_likelihood_function(instance) == 0.0
    graph.modify_before_fit(paths=af.DirectoryPaths(name="graph_eval"), model=None)
    assert graph.log_likelihood_function(instance) == 1.0


def test_disagreeing_setups_raise():
    graph = g.FactorGraphModel(
        _factor(_Analysis("numba_cpu_sparse_fp64"), "a"),
        _factor(_Analysis("numpy_cpu_fp64"), "b"),
    )

    with pytest.raises(exc.SearchException, match="different compute setups"):
        graph.modify_before_fit(paths=af.DirectoryPaths(name="graph_bad"), model=None)


def test_factors_without_a_setup_are_ignored():
    graph = g.FactorGraphModel(
        _factor(_Analysis("numba_cpu_sparse_fp64"), "a"),
        _factor(_Analysis(None), "b"),
    )
    graph.modify_before_fit(paths=af.DirectoryPaths(name="graph_mixed"), model=None)

    assert graph.setup_ids() == {"a": "numba_cpu_sparse_fp64"}
    graph.check_setup_agreement()


def test_no_setups_agree_trivially():
    graph = g.FactorGraphModel(_factor(_Analysis(), "a"), _factor(_Analysis(), "b"))
    graph.check_setup_agreement()
    assert setup_id_of(graph) is None


def test_hierarchical_factor_is_skipped():
    calls = []
    hierarchical = g.HierarchicalFactor(
        af.GaussianPrior,
        mean=af.GaussianPrior(mean=50, sigma=10),
        sigma=af.GaussianPrior(mean=10, sigma=5),
    )
    model = af.Model(af.ex.Gaussian)
    hierarchical.add_drawn_variable(model.centre)
    factor = g.AnalysisFactor(model, analysis=_Analysis("x", calls), name="a")

    graph = g.FactorGraphModel(factor, hierarchical)
    graph.modify_before_fit(paths=af.DirectoryPaths(name="graph_hier"), model=None)

    assert len(calls) == 1
    assert graph.setup_ids() == {"a": "x"}


def test_runtime_gates_check_setup_agreement():
    from autofit.non_linear.search import preflight

    graph = g.FactorGraphModel(
        _factor(_Analysis(), "a"),
        _factor(_Analysis(), "b"),
    )
    graph.model_factors[0].analysis.setup = _Setup("one")
    graph.model_factors[1].analysis.setup = _Setup("two")

    with pytest.raises(exc.SearchException, match="different compute setups"):
        preflight.check_runtime_gates(af.Emcee(), graph, graph.global_prior_model)


def test_whole_graph_fit_banner_reports_the_agreed_setup(monkeypatch, caplog, tmp_path):
    monkeypatch.setenv("PYAUTO_TEST_MODE", "2")
    monkeypatch.setattr(pool_module, "detected_cores", lambda: 8)

    class _FitAnalysis(_Analysis):
        def log_likelihood_function(self, instance):
            return -1.0

        def modify_before_fit(self, paths, model):
            self.setup = _Setup("numba_cpu_sparse_fp64")
            return self

    graph = g.FactorGraphModel(
        _factor(_FitAnalysis(), "a"), _factor(_FitAnalysis(), "b")
    )
    search = af.Emcee(
        name="graph_fit", path_prefix=str(tmp_path), number_of_cores="auto"
    )

    with caplog.at_level(logging.INFO):
        search.fit(model=graph.global_prior_model, analysis=graph)

    banners = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Parallel:")]
    assert banners == [
        "Parallel: 8 workers (auto: detected cores)"
    ]
