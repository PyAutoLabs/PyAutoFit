"""Tests for the ``use_jax_for_visualization`` flag on ``Analysis``."""

import pytest

import autofit as af


class _FittableAnalysis(af.Analysis):
    """Minimal Analysis subclass with a trivial ``fit_from`` for dispatch tests."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fit_from_calls = 0

    def log_likelihood_function(self, instance):
        return 0.0

    def fit_from(self, instance):
        self.fit_from_calls += 1
        return ("fit", instance)


class _JitFittableAnalysis(af.Analysis):
    """Analysis with a ``fit_from`` returning a JIT-traceable array.

    ``_FittableAnalysis.fit_from`` returns a Python tuple with a string literal,
    which is not tracer-compatible. For the JIT-enabled dispatch path we need a
    ``fit_from`` whose output is entirely JAX-compatible.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fit_from_calls = 0

    def log_likelihood_function(self, instance):
        return 0.0

    def fit_from(self, instance):
        import jax.numpy as jnp

        self.fit_from_calls += 1
        return jnp.asarray(instance) * 2.0


def test_default_flag_is_false():
    analysis = af.Analysis()
    assert analysis._use_jax is False
    assert analysis._use_jax_for_visualization is False
    assert analysis.supports_jax_visualization is False


def test_flag_requires_use_jax(caplog):
    with caplog.at_level("WARNING"):
        analysis = af.Analysis(use_jax=False, use_jax_for_visualization=True)
    assert analysis._use_jax_for_visualization is False
    assert any("requires use_jax=True" in r.message for r in caplog.records)


def test_flag_accepted_when_use_jax_true():
    analysis = af.Analysis(use_jax=True, use_jax_for_visualization=True)
    assert analysis._use_jax is True
    assert analysis._use_jax_for_visualization is True
    assert analysis.supports_jax_visualization is True


def test_pyauto_disable_jax_env_var_clears_both_flags(monkeypatch):
    monkeypatch.setenv("PYAUTO_DISABLE_JAX", "1")
    analysis = af.Analysis(use_jax=True, use_jax_for_visualization=True)
    assert analysis._use_jax is False
    assert analysis._use_jax_for_visualization is False


def test_fit_for_visualization_dispatches_through_jit_when_flag_set():
    import jax.numpy as jnp

    analysis = _JitFittableAnalysis(use_jax=True, use_jax_for_visualization=True)

    assert getattr(analysis, "_jitted_fit_from", None) is None

    result_1 = analysis.fit_for_visualization(instance=1.0)
    assert analysis._jitted_fit_from is not None
    assert jnp.allclose(result_1, jnp.asarray(2.0))

    jitted_after_first = analysis._jitted_fit_from
    result_2 = analysis.fit_for_visualization(instance=3.0)
    assert analysis._jitted_fit_from is jitted_after_first
    assert jnp.allclose(result_2, jnp.asarray(6.0))


def test_fit_for_visualization_works_without_flag():
    analysis = _FittableAnalysis()
    result = analysis.fit_for_visualization(instance="sentinel")
    assert result == ("fit", "sentinel")
    assert analysis.fit_from_calls == 1
    assert getattr(analysis, "_jitted_fit_from", None) is None


def test_subclass_can_override_supports_jax_visualization():
    class ForcedAnalysis(af.Analysis):
        @property
        def supports_jax_visualization(self):
            return True

    analysis = ForcedAnalysis()
    assert analysis._use_jax_for_visualization is False
    assert analysis.supports_jax_visualization is True
