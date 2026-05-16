"""Tests for the ``use_jax_for_visualization`` flag on ``Analysis``."""

import importlib.util

import pytest

import autofit as af


def _jax_installed() -> bool:
    """Check jax availability without importing it (per numpy-only rule)."""
    return importlib.util.find_spec("jax") is not None


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


@pytest.mark.skipif(not _jax_installed(), reason="jax not installed; fallback path tested below")
def test_flag_accepted_when_use_jax_true():
    analysis = af.Analysis(use_jax=True, use_jax_for_visualization=True)
    assert analysis._use_jax is True
    assert analysis._use_jax_for_visualization is True
    assert analysis.supports_jax_visualization is True


@pytest.mark.skipif(_jax_installed(), reason="jax installed; happy path tested above")
def test_use_jax_true_falls_back_to_numpy_when_jax_missing(recwarn):
    """When jax isn't installed, use_jax=True should silently downgrade
    to use_jax=False after emitting a UserWarning. Affects 3.9/3.10
    where the [jax] extra is gated out."""
    analysis = af.Analysis(use_jax=True, use_jax_for_visualization=True)
    assert analysis._use_jax is False
    assert analysis._use_jax_for_visualization is False
    assert any("JAX is not installed" in str(w.message) for w in recwarn)


def test_pyauto_disable_jax_env_var_clears_both_flags(monkeypatch):
    monkeypatch.setenv("PYAUTO_DISABLE_JAX", "1")
    analysis = af.Analysis(use_jax=True, use_jax_for_visualization=True)
    assert analysis._use_jax is False
    assert analysis._use_jax_for_visualization is False


def test_pyauto_disable_jax_overrides_sentinel_default(monkeypatch):
    """PYAUTO_DISABLE_JAX=1 must still force both off even when the user
    constructs Analysis(use_jax=True) and lets the sentinel resolve. This is
    a numpy-only check — JAX-conditional sentinel-resolution assertions live
    in autofit_workspace_test/scripts/jax_assertions/fitness_dispatch.py."""
    monkeypatch.setenv("PYAUTO_DISABLE_JAX", "1")
    analysis = af.Analysis(use_jax=True)
    assert analysis._use_jax is False
    assert analysis._use_jax_for_visualization is False


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
