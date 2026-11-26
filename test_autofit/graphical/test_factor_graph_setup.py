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


class _DefaultAnalysis(af.Analysis):
    """
    Resolves a default setup (`setup_is_default`) in `modify_before_fit`, whose backend
    follows the setup_id prefix, and records `adopt_backend` calls.
    """

    def __init__(self, setup_id, default=True):
        super().__init__(use_jax=False)
        self._setup_id = setup_id
        self._default = default
        self.setup = None
        self.adopted = []

    @property
    def setup_is_default(self):
        return self._default

    def adopt_backend(self, use_jax):
        self.adopted.append(use_jax)
        self.setup = None
        self._use_jax = use_jax

    def log_likelihood_function(self, instance):
        return 0.0

    def modify_before_fit(self, paths, model):
        if not self.adopted:
            self.setup = _Setup(self._setup_id)
            self._use_jax = self._setup_id.startswith("jax")
        return self


def _modify(graph, name):
    return graph.modify_before_fit(paths=af.DirectoryPaths(name=name), model=None)


def test_base_analysis_has_no_default_setup_and_adopt_is_a_no_op():
    analysis = _Analysis()
    assert analysis.setup_is_default is False
    analysis.adopt_backend(True)
    assert analysis.is_jax is False


def test_disagreeing_default_setups_adopt_one_backend(monkeypatch):
    monkeypatch.delenv("PYAUTO_DISABLE_JAX", raising=False)
    a = _DefaultAnalysis("numba_cpu_sparse_fp64")
    b = _DefaultAnalysis("jax_cpu_dense_fp64_nufft")
    graph = g.FactorGraphModel(_factor(a, "a"), _factor(b, "b"))

    _modify(graph, "graph_default_mixed")

    import importlib.util

    expected = importlib.util.find_spec("jax") is not None
    assert a.adopted == [expected] and b.adopted == [expected]
    assert graph.setup_ids() == {}
    assert graph.factors_disagreeing_on_backend() == []
    graph.check_setup_agreement()


def test_agreeing_default_setups_are_kept():
    a = _DefaultAnalysis("numba_cpu_sparse_fp64")
    b = _DefaultAnalysis("numba_cpu_sparse_fp64")
    graph = g.FactorGraphModel(_factor(a, "a"), _factor(b, "b"))

    _modify(graph, "graph_default_agree")

    assert a.adopted == [] and b.adopted == []
    assert setup_id_of(graph) == "numba_cpu_sparse_fp64"


def test_graph_use_jax_false_pushes_numpy_down_to_default_factors():
    a = _DefaultAnalysis("numba_cpu_sparse_fp64")
    b = _DefaultAnalysis("jax_cpu_dense_fp64_nufft")
    graph = g.FactorGraphModel(_factor(a, "a"), _factor(b, "b"), use_jax=False)

    _modify(graph, "graph_default_numpy")

    assert a.adopted == [False] and b.adopted == [False]
    assert graph.is_jax is False
    graph.check_backend_agreement()


def test_graph_use_jax_pushes_down_even_when_default_setups_agree():
    a = _DefaultAnalysis("jax_cpu_dense_fp64")
    b = _DefaultAnalysis("jax_cpu_dense_fp64")
    graph = g.FactorGraphModel(_factor(a, "a"), _factor(b, "b"), use_jax=False)

    _modify(graph, "graph_default_agree_numpy")

    assert a.adopted == [False] and b.adopted == [False]
    graph.check_backend_agreement()


def test_default_factors_follow_an_explicit_setup():
    explicit = _DefaultAnalysis("numba_cpu_sparse_fp64", default=False)
    default = _DefaultAnalysis("jax_cpu_dense_fp64_nufft")
    graph = g.FactorGraphModel(_factor(explicit, "a"), _factor(default, "b"))

    _modify(graph, "graph_explicit_lead")

    assert explicit.adopted == []
    assert default.adopted == [False]
    assert graph.setup_ids() == {"a": "numba_cpu_sparse_fp64"}


def test_disagreeing_explicit_setups_still_raise():
    a = _DefaultAnalysis("numba_cpu_sparse_fp64", default=False)
    b = _DefaultAnalysis("jax_cpu_dense_fp64", default=False)
    c = _DefaultAnalysis("jax_cpu_dense_fp64_nufft")
    graph = g.FactorGraphModel(_factor(a, "a"), _factor(b, "b"), _factor(c, "c"))

    with pytest.raises(exc.SearchException, match="setups passed explicitly"):
        _modify(graph, "graph_explicit_bad")

    assert a.adopted == [] and b.adopted == [] and c.adopted == []


def test_model_analysis_forwards_default_setup_and_adopt_backend():
    from autofit.non_linear.analysis.model_analysis import ModelAnalysis

    inner = _DefaultAnalysis("jax_cpu_dense_fp64")
    inner._use_jax = True
    wrapper = ModelAnalysis(analysis=inner, model=af.Model(af.ex.Gaussian))

    assert wrapper.setup_is_default is True
    wrapper.adopt_backend(False)

    assert inner.adopted == [False]
    assert wrapper.is_jax is False
