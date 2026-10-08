"""
``Analysis.is_jax``, the single "is this analysis JAX?" probe, and how the wrappers
forward it (search-extensibility decision D12):

- ``ModelAnalysis`` and ``AnalysisFactor`` take their wrapped analysis's ``is_jax`` and
  ``gradient_mode``;
- ``FactorGraphModel`` derives ``is_jax`` from its factors, and whole-graph fitting
  requires every factor to agree; per-factor expectation propagation allows mixed
  factors;
- hierarchical factors carry the flag and honour ``PYAUTO_DISABLE_JAX``;
- the flags survive pytree flattening and pickling.
"""

import importlib.util
import pickle

import pytest

import autofit as af
import autofit.graphical as g
from autofit import exc
from autofit.non_linear.analysis.model_analysis import ModelAnalysis

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

requires_jax = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None, reason="jax is not installed"
)


class NumpyAnalysis(af.Analysis):
    def __init__(self, use_jax=False):
        super().__init__(use_jax=use_jax)

    def log_likelihood_function(self, instance):
        # Traceable on both backends: a `use_jax=True` analysis is jitted by every search
        # (search-extensibility phase A2), so a host `float(...)` would not trace.
        return -0.5 * (instance.centre - 50.0) ** 2


class ForwardAnalysis(NumpyAnalysis):
    gradient_mode = "forward"


class NoInitAnalysis(af.Analysis):
    """An analysis that never calls ``Analysis.__init__``."""

    def __init__(self):
        pass

    def log_likelihood_function(self, instance):
        return 0.0


def _factor(analysis, name):
    return g.AnalysisFactor(af.Model(af.ex.Gaussian), analysis=analysis, name=name)


def test_is_jax_false_by_default():
    assert NumpyAnalysis().is_jax is False


def test_is_jax_false_without_analysis_init():
    assert NoInitAnalysis().is_jax is False


@requires_jax
def test_is_jax_true_when_use_jax():
    assert NumpyAnalysis(use_jax=True).is_jax is True


@requires_jax
def test_disable_jax_switch(monkeypatch):
    monkeypatch.setenv("PYAUTO_DISABLE_JAX", "1")

    assert NumpyAnalysis(use_jax=True).is_jax is False


@requires_jax
def test_model_analysis_forwards_is_jax_and_gradient_mode():
    analysis = ForwardAnalysis(use_jax=True)

    wrapped = analysis.with_model(af.Model(af.ex.Gaussian))

    assert isinstance(wrapped, ModelAnalysis)
    assert wrapped.is_jax is True
    assert wrapped.gradient_mode == "forward"
    assert pickle.loads(pickle.dumps(wrapped)).is_jax is True


def test_model_analysis_explicit_use_jax_is_honoured():
    wrapped = ModelAnalysis(
        analysis=NumpyAnalysis(), model=af.Model(af.ex.Gaussian), use_jax=False
    )

    assert wrapped.is_jax is False
    assert wrapped.gradient_mode == "reverse"


@requires_jax
def test_analysis_factor_forwards_is_jax_and_gradient_mode():
    factor = _factor(ForwardAnalysis(use_jax=True), "forward")

    assert factor.is_jax is True
    assert factor.gradient_mode == "forward"
    assert _factor(NumpyAnalysis(), "numpy").is_jax is False


@requires_jax
def test_factor_graph_model_inherits_use_jax_from_its_factors():
    graph = g.FactorGraphModel(
        _factor(NumpyAnalysis(use_jax=True), "a"),
        _factor(NumpyAnalysis(use_jax=True), "b"),
    )

    assert graph.is_jax is True
    assert graph.factors_disagreeing_on_backend() == []
    graph.check_backend_agreement()


def test_numpy_factor_graph_model_is_numpy():
    graph = g.FactorGraphModel(
        _factor(NumpyAnalysis(), "a"), _factor(NumpyAnalysis(), "b")
    )

    assert graph.is_jax is False
    graph.check_backend_agreement()


@requires_jax
def test_mixed_factor_graph_whole_graph_check_names_the_disagreeing_factors():
    graph = g.FactorGraphModel(
        _factor(NumpyAnalysis(), "numpy_factor"),
        _factor(NumpyAnalysis(use_jax=True), "jax_factor"),
    )

    assert graph.is_jax is False
    assert graph.factors_disagreeing_on_backend() == ["jax_factor"]

    with pytest.raises(exc.SearchException, match="jax_factor"):
        graph.check_backend_agreement()


@requires_jax
def test_explicit_graph_use_jax_must_agree_with_the_factors():
    graph = g.FactorGraphModel(_factor(NumpyAnalysis(), "numpy_factor"), use_jax=True)

    assert graph.is_jax is True

    with pytest.raises(exc.SearchException, match="numpy_factor"):
        graph.check_backend_agreement()


@requires_jax
def test_whole_graph_fit_of_a_mixed_graph_raises(monkeypatch):
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)
    graph = g.FactorGraphModel(
        _factor(NumpyAnalysis(), "numpy_factor"),
        _factor(NumpyAnalysis(use_jax=True), "jax_factor"),
    )

    with pytest.raises(exc.SearchException, match="jax_factor"):
        af.Drawer(total_draws=2).fit(model=graph.global_prior_model, analysis=graph)


class _BackendReached(Exception):
    pass


@requires_jax
def test_whole_graph_check_survives_a_model_analysis_wrapper(monkeypatch):
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)
    graph = g.FactorGraphModel(
        _factor(NumpyAnalysis(), "numpy_factor"),
        _factor(NumpyAnalysis(use_jax=True), "jax_factor"),
    )
    search = af.Drawer(total_draws=2)

    def reached(*args, **kwargs):
        raise _BackendReached()

    monkeypatch.setattr(search, "_fit", reached)

    with pytest.raises(exc.SearchException, match="jax_factor"):
        search.fit(
            model=graph.global_prior_model,
            analysis=ModelAnalysis(graph, graph.global_prior_model),
        )


@requires_jax
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_mixed_factor_graph_still_runs_per_factor_ep(monkeypatch):
    """
    Per-factor EP fits each factor with its own analysis, so a numpy factor and a JAX
    factor in one graph each run on their own backend.
    """
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)
    graph = g.FactorGraphModel(
        g.AnalysisFactor(
            af.Model(af.ex.Gaussian),
            analysis=NumpyAnalysis(),
            optimiser=af.Drawer(total_draws=3),
            name="numpy_factor",
        ),
        g.AnalysisFactor(
            af.Model(af.ex.Gaussian),
            analysis=NumpyAnalysis(use_jax=True),
            optimiser=af.Drawer(total_draws=3),
            name="jax_factor",
        ),
    )

    result = graph.optimise(
        af.LaplaceOptimiser(),
        paths=False,
        ep_history=af.EPHistory(kl_tol=None),
        max_steps=1,
    )

    assert result is not None


def test_factor_graph_gradient_mode_is_the_common_mode_else_reverse():
    forward = g.FactorGraphModel(
        _factor(ForwardAnalysis(), "a"), _factor(ForwardAnalysis(), "b")
    )
    mixed = g.FactorGraphModel(
        _factor(ForwardAnalysis(), "a"), _factor(NumpyAnalysis(), "b")
    )

    assert forward.gradient_mode == "forward"
    assert mixed.gradient_mode == "reverse"


@requires_jax
def test_factor_graph_flags_survive_pytree_flattening_and_pickling():
    graph = g.FactorGraphModel(_factor(NumpyAnalysis(), "numpy_factor"), use_jax=True)

    children, aux_data = graph.tree_flatten()
    rebuilt = g.FactorGraphModel.tree_unflatten(aux_data, children)

    assert rebuilt.is_jax is True
    assert pickle.loads(pickle.dumps(graph)).is_jax is True


@requires_jax
def test_legacy_factor_graph_pickle_keeps_its_explicit_backend():
    """
    Graphs pickled before the backend was derived from the factors stored an explicit
    ``_use_jax`` attribute (``False`` by default). Unpickling migrates it, so the graph
    keeps that choice rather than re-deriving JAX from its children, and
    ``tree_flatten`` still works.
    """
    graph = g.FactorGraphModel(
        _factor(NumpyAnalysis(use_jax=True), "a"),
        _factor(NumpyAnalysis(use_jax=True), "b"),
        use_jax=False,
    )

    legacy_state = dict(graph.__dict__)
    del legacy_state["_explicit_use_jax"]
    legacy_state["_use_jax"] = False
    legacy = g.FactorGraphModel.__new__(g.FactorGraphModel)
    legacy.__dict__.update(legacy_state)

    restored = pickle.loads(pickle.dumps(legacy))

    assert restored.is_jax is False
    children, aux_data = restored.tree_flatten()
    assert aux_data[2] is False
    assert g.FactorGraphModel.tree_unflatten(aux_data, children).is_jax is False


@requires_jax
def test_hierarchical_factor_carries_the_flag(monkeypatch):
    def hierarchical(use_jax):
        factor = g.HierarchicalFactor(
            af.GaussianPrior,
            mean=af.GaussianPrior(mean=0.0, sigma=1.0),
            sigma=1.0,
            use_jax=use_jax,
        )
        factor.add_drawn_variable(af.GaussianPrior(mean=0.0, sigma=1.0))
        return factor

    assert all(factor.is_jax for factor in hierarchical(True).factors)
    assert not any(factor.is_jax for factor in hierarchical(False).factors)

    monkeypatch.setenv("PYAUTO_DISABLE_JAX", "1")

    assert not any(factor.is_jax for factor in hierarchical(True).factors)


@requires_jax
def test_factor_graph_derives_jax_through_a_hierarchical_factor():
    h = af.HierarchicalFactor(
        af.GaussianPrior,
        mean=af.GaussianPrior(mean=0, sigma=1),
        sigma=1.0,
        use_jax=True,
    )
    h.add_drawn_variable(af.GaussianPrior(mean=0, sigma=1))
    graph = af.FactorGraphModel(h)

    assert all(factor.is_jax for factor in h.factors)
    assert graph.is_jax is True
    graph.check_backend_agreement()


def test_supports_jax_visualization_reads_is_jax():
    assert NumpyAnalysis().supports_jax_visualization is False
