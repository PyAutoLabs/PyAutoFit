import logging
from unittest.mock import MagicMock

from autonerves import conf

from autofit.non_linear.search.updater import SearchUpdater


def _make_updater() -> SearchUpdater:
    return SearchUpdater(
        paths=MagicMock(),
        timer=MagicMock(),
        search_logger=logging.getLogger("test_updater"),
        plot_results_func=MagicMock(),
        samples_from_func=MagicMock(),
        disable_output=False,
        iterations_per_full_update=1.0,
    )


def test__compute_latent_samples__skipped_in_test_mode(monkeypatch):
    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")
    monkeypatch.delenv("PYAUTO_SKIP_LATENTS", raising=False)

    analysis = MagicMock()
    result = _make_updater()._compute_latent_samples(
        samples=MagicMock(),
        samples_save=MagicMock(),
        analysis=analysis,
        fitness=MagicMock(),
        during_analysis=False,
    )

    assert result is None
    analysis.compute_latent_samples.assert_not_called()


def test__compute_latent_samples__skipped_by_explicit_env_var(monkeypatch):
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)
    monkeypatch.setenv("PYAUTO_SKIP_LATENTS", "1")

    analysis = MagicMock()
    result = _make_updater()._compute_latent_samples(
        samples=MagicMock(),
        samples_save=MagicMock(),
        analysis=analysis,
        fitness=MagicMock(),
        during_analysis=False,
    )

    assert result is None
    analysis.compute_latent_samples.assert_not_called()


def test__compute_latent_samples__config_gate_still_works(monkeypatch):
    """
    With neither env var set, the existing ``latent_after_fit`` /
    ``latent_during_fit`` config gate must still short-circuit
    when both are disabled. Regression for the new skip_latents()
    check not bypassing the original logic.
    """
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)
    monkeypatch.delenv("PYAUTO_SKIP_LATENTS", raising=False)

    original_after = conf.instance["output"]["latent_after_fit"]
    original_during = conf.instance["output"]["latent_during_fit"]
    conf.instance["output"]["latent_after_fit"] = False
    conf.instance["output"]["latent_during_fit"] = False
    try:
        analysis = MagicMock()
        result = _make_updater()._compute_latent_samples(
            samples=MagicMock(),
            samples_save=MagicMock(),
            analysis=analysis,
            fitness=MagicMock(),
            during_analysis=False,
        )
        assert result is None
        analysis.compute_latent_samples.assert_not_called()
    finally:
        conf.instance["output"]["latent_after_fit"] = original_after
        conf.instance["output"]["latent_during_fit"] = original_during
