"""
The JAX preflight ``NonLinearSearch.fit`` runs before a backend builds any state
(search-extensibility A3b; survey 02 §5.3, ``autofit.non_linear.search.preflight``).

- the trace preflight: one ``jax.eval_shape`` of the declared objective kind (and of its
  gradient kind for ``gradient == "uses"``) at the prior medians, raising a
  ``SearchException`` that names the search with the tracer error chained;
- the optional numerical probe (``PYAUTO_JAX_PREFLIGHT_PROBE=1``), off by default;
- the x64 check: warn once when a JAX analysis runs with ``jax_enable_x64`` off, raise
  for a search declaring ``requires_fp64``;
- placement: after the test-mode bypass (``PYAUTO_TEST_MODE>=2`` never runs it) and after
  the ``jax_use='required'`` gate.

No backend executes: every fit here either raises before ``_fit`` or bypasses it.
"""

import logging

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import autofit as af
from autofit import exc
from autofit.non_linear.search import capabilities as cap
from autofit.non_linear.search import preflight

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


class NumpyCastAnalysis(af.Analysis):
    """
    A ``use_jax=True`` analysis whose likelihood casts a parameter with
    ``np.asarray``: it cannot be traced.
    """

    def __init__(self):
        super().__init__(use_jax=True)

    def log_likelihood_function(self, instance):
        centre = np.asarray(instance.centre)
        return -0.5 * float((centre - 50.0) ** 2)


class TraceableAnalysis(af.Analysis):
    def __init__(self, value=None):
        super().__init__(use_jax=True)
        self.value = value

    def log_likelihood_function(self, instance):
        import jax.numpy as jnp

        if self.value is not None:
            return jnp.asarray(self.value)
        return -0.5 * (instance.centre - 50.0) ** 2


def _model():
    return af.Model(af.ex.Gaussian)


def _backend_must_not_run(*args, **kwargs):
    raise AssertionError("the backend ran before the preflight")


def test_preflight_raises_on_an_np_asarray_likelihood_with_the_error_chained(
    monkeypatch,
):
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)
    search = af.DynestyStatic()
    monkeypatch.setattr(search, "_fit", _backend_must_not_run)

    with pytest.raises(exc.SearchException) as error:
        search.fit(model=_model(), analysis=NumpyCastAnalysis())

    message = str(error.value)
    assert "DynestyStatic" in message
    assert "'scalar'" in message
    assert "TracerArrayConversionError" in message
    assert preflight.PREFLIGHT_ENV in message

    cause = error.value.__cause__
    assert cause is not None
    assert type(cause).__name__ == "TracerArrayConversionError"


def test_preflight_traces_the_batched_kind_of_a_batched_search():
    with pytest.raises(exc.SearchException, match="'batched'") as error:
        preflight.trace_preflight(af.Nautilus(), NumpyCastAnalysis(), _model())

    assert "Nautilus" in str(error.value)
    assert error.value.__cause__ is not None


def test_preflight_kinds_follow_the_declared_capabilities():
    assert preflight.preflight_kinds(af.DynestyStatic()) == ("scalar",)
    assert preflight.preflight_kinds(af.Nautilus()) == ("batched",)

    class GradientSearch(af.DynestyStatic):
        gradient = cap.Gradient.USES

    class BatchedGradientSearch(af.DynestyStatic):
        gradient = cap.Gradient.USES
        batched = True

    assert preflight.preflight_kinds(GradientSearch()) == ("scalar", "value_and_grad")
    assert preflight.preflight_kinds(BatchedGradientSearch()) == (
        "batched",
        "batched_value_and_grad",
    )


def test_preflight_traces_the_gradient_of_a_gradient_search(monkeypatch):
    """
    A gradient search's gradient kind is traced too: the trace sees the gradient's own
    failure (here a function JAX can evaluate but not differentiate).
    """
    import jax.numpy as jnp

    @jax.custom_jvp
    def no_derivative(x):
        return x

    @no_derivative.defjvp
    def _no_derivative_jvp(primals, tangents):
        raise NotImplementedError("this likelihood has no derivative")

    class NonDifferentiableAnalysis(af.Analysis):
        def __init__(self):
            super().__init__(use_jax=True)

        def log_likelihood_function(self, instance):
            return -0.5 * no_derivative(instance.centre - 50.0) ** 2 * jnp.ones(())

    class GradientSearch(af.DynestyStatic):
        gradient = cap.Gradient.USES

    preflight.trace_preflight(af.DynestyStatic(), NonDifferentiableAnalysis(), _model())

    with pytest.raises(exc.SearchException, match="'value_and_grad'") as error:
        preflight.trace_preflight(GradientSearch(), NonDifferentiableAnalysis(), _model())

    assert isinstance(error.value.__cause__, NotImplementedError)


def test_preflight_passes_a_traceable_likelihood_and_evaluates_nothing(monkeypatch):
    calls = []

    class CountingAnalysis(TraceableAnalysis):
        def log_likelihood_function(self, instance):
            calls.append(type(instance.centre).__name__)
            return super().log_likelihood_function(instance)

    preflight.trace_preflight(af.DynestyStatic(), CountingAnalysis(), _model())

    # One trace: the likelihood saw tracers, never concrete values.
    assert len(calls) == 1
    assert "Tracer" in calls[0] or "ShapedArray" in calls[0]


def test_preflight_is_skipped_by_its_environment_flag(monkeypatch):
    monkeypatch.setenv(preflight.PREFLIGHT_ENV, "0")
    preflight.trace_preflight(af.DynestyStatic(), NumpyCastAnalysis(), _model())


def test_preflight_is_a_no_op_for_a_numpy_analysis():
    class NumpyAnalysis(af.Analysis):
        def log_likelihood_function(self, instance):
            return float(np.asarray(instance.centre))

    preflight.trace_preflight(af.DynestyStatic(), NumpyAnalysis(), _model())


def test_preflight_is_bypassed_by_test_mode_2(monkeypatch):
    """
    ``PYAUTO_TEST_MODE=2`` returns before the preflight, so even a non-traceable JAX
    likelihood completes a smoke run (the bypass evaluates it eagerly).
    """
    monkeypatch.setenv("PYAUTO_TEST_MODE", "2")

    result = af.DynestyStatic().fit(model=_model(), analysis=NumpyCastAnalysis())

    assert isinstance(result, af.Result)


def test_required_gate_runs_before_the_preflight(monkeypatch):
    """
    A ``jax_use='required'`` search given a numpy analysis gets the shared REQUIRED
    message, not a preflight error.
    """
    pytest.importorskip("optax")
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)

    def preflight_must_not_run(*args, **kwargs):
        raise AssertionError("the preflight ran before the REQUIRED gate")

    monkeypatch.setattr(preflight, "trace_preflight", preflight_must_not_run)

    class NumpyAnalysis(af.Analysis):
        def log_likelihood_function(self, instance):
            return 0.0

    with pytest.raises(exc.SearchException) as error:
        af.MultiStartAdam().fit(model=_model(), analysis=NumpyAnalysis())

    assert str(error.value) == cap.JAX_REQUIRED_MESSAGE.format(search="MultiStartAdam")


def test_numerical_probe_is_off_by_default(monkeypatch, caplog):
    monkeypatch.delenv(preflight.PROBE_ENV, raising=False)

    with caplog.at_level(logging.WARNING, logger=preflight.__name__):
        preflight.trace_preflight(
            af.DynestyStatic(), TraceableAnalysis(value=np.nan), _model()
        )

    assert "numerical probe" not in caplog.text


def test_numerical_probe_warns_on_an_invalid_value_at_the_prior_medians(
    monkeypatch, caplog
):
    monkeypatch.setenv(preflight.PROBE_ENV, "1")

    with caplog.at_level(logging.WARNING, logger=preflight.__name__):
        preflight.trace_preflight(
            af.DynestyStatic(), TraceableAnalysis(value=np.nan), _model()
        )

    assert "DynestyStatic numerical probe" in caplog.text


def test_numerical_probe_is_quiet_on_a_valid_value(monkeypatch, caplog):
    monkeypatch.setenv(preflight.PROBE_ENV, "1")

    with caplog.at_level(logging.WARNING, logger=preflight.__name__):
        preflight.trace_preflight(af.DynestyStatic(), TraceableAnalysis(), _model())

    assert "numerical probe" not in caplog.text


# --- x64 ------------------------------------------------------------------------------


@pytest.fixture
def float32(monkeypatch):
    """
    Run the test with ``jax_enable_x64`` off, restoring it (and the once-per-process
    warning flag) afterwards.
    """
    previous = bool(jax.config.jax_enable_x64)
    monkeypatch.setattr(preflight, "_x64_warning_emitted", False)
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_fp32_warns_once(float32, caplog):
    analysis = TraceableAnalysis()

    with pytest.warns(UserWarning, match="32-bit precision") as record:
        with caplog.at_level(logging.WARNING, logger=preflight.__name__):
            preflight.check_x64(af.DynestyStatic(), analysis)
            preflight.check_x64(af.DynestyStatic(), analysis)

    assert len([w for w in record if "32-bit" in str(w.message)]) == 1
    assert caplog.text.count("32-bit precision") == 1


def test_fp32_raises_for_a_search_declaring_requires_fp64(float32):
    class Fp64Search(af.DynestyStatic):
        requires_fp64 = True

    with pytest.raises(exc.SearchException, match="Fp64Search requires float64"):
        preflight.check_x64(Fp64Search(), TraceableAnalysis())


def test_fp32_check_is_silent_for_numpy_and_for_x64(monkeypatch, recwarn):
    monkeypatch.setattr(preflight, "_x64_warning_emitted", False)

    class NumpyAnalysis(af.Analysis):
        def log_likelihood_function(self, instance):
            return 0.0

    preflight.check_x64(af.DynestyStatic(), NumpyAnalysis())

    if jax.config.jax_enable_x64:
        preflight.check_x64(af.DynestyStatic(), TraceableAnalysis())

    assert not [w for w in recwarn if "32-bit" in str(w.message)]


def test_no_public_search_requires_fp64():
    from autofit.non_linear.search.registry import entries

    assert not any(entry.capabilities["requires_fp64"] for entry in entries())
