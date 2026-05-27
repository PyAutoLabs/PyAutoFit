"""Tests for the visualization path on ``Analysis``.

The ``use_jax_for_visualization`` flag has been removed — visualization
now always follows ``use_jax``. These tests verify the simplified
``fit_for_visualization`` dispatch and the ``supports_jax_visualization``
property.
"""

import importlib.util

import pytest

import autofit as af


def _jax_installed() -> bool:
    return importlib.util.find_spec("jax") is not None


class _FittableAnalysis(af.Analysis):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fit_from_calls = 0

    def log_likelihood_function(self, instance):
        return 0.0

    def fit_from(self, instance):
        self.fit_from_calls += 1
        return ("fit", instance)


def test_default_flags():
    analysis = af.Analysis()
    assert analysis._use_jax is False
    assert analysis.supports_jax_visualization is False


@pytest.mark.skipif(not _jax_installed(), reason="jax not installed")
def test_use_jax_enables_jax_visualization():
    analysis = af.Analysis(use_jax=True)
    assert analysis._use_jax is True
    assert analysis.supports_jax_visualization is True


@pytest.mark.skipif(_jax_installed(), reason="jax installed")
def test_use_jax_true_falls_back_when_jax_missing(recwarn):
    analysis = af.Analysis(use_jax=True)
    assert analysis._use_jax is False
    assert any("JAX is not installed" in str(w.message) for w in recwarn)


def test_pyauto_disable_jax_env_var(monkeypatch):
    monkeypatch.setenv("PYAUTO_DISABLE_JAX", "1")
    analysis = af.Analysis(use_jax=True)
    assert analysis._use_jax is False


def test_fit_for_visualization_delegates_to_fit_from():
    analysis = _FittableAnalysis()
    result = analysis.fit_for_visualization(instance="sentinel")
    assert result == ("fit", "sentinel")
    assert analysis.fit_from_calls == 1


def test_subclass_can_override_supports_jax_visualization():
    class ForcedAnalysis(af.Analysis):
        @property
        def supports_jax_visualization(self):
            return True

    analysis = ForcedAnalysis()
    assert analysis.supports_jax_visualization is True
