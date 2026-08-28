import logging
from unittest.mock import MagicMock

from autonerves import conf
from autonerves.conf import with_config

import autofit as af
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


class _RejectsLowValue:
    """
    A model component whose constructor rejects part of its own prior range, mimicking a
    validated parameterization (e.g. `ell_comps` outside the ellipticity unit disk) which
    only fires when a stored vector is materialized on the host.
    """

    def __init__(self, value):
        if value < 0.75:
            raise af.exc.FitException("value must be at least 0.75")
        self.value = value


def _samples_with_rejected_best():
    """
    Samples whose maximum likelihood point is rejected by the model, with a second
    zero-weight row so the weight-threshold prune has something to remove.
    """
    model = af.Model(_RejectsLowValue)
    model.value = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)

    return af.SamplesPDF(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=[[0.1], [0.9]],
            log_likelihood_list=[2.0, 1.0],
            log_prior_list=[0.0, 0.0],
            weight_list=[1.0, 0.0],
        ),
        samples_info={"log_evidence": 1.0},
    )


def _updater_for(samples, paths) -> SearchUpdater:
    return SearchUpdater(
        paths=paths,
        timer=MagicMock(),
        search_logger=logging.getLogger("test_updater"),
        plot_results_func=MagicMock(),
        samples_from_func=lambda model, search_internal: samples,
        disable_output=False,
        iterations_per_full_update=1.0,
    )


def test__save_samples__persists_samples_when_the_best_point_is_rejected():
    """
    A stored best point the model rejects must not cost the run its samples: both
    `save_samples` and `save_samples_summary` are called, and only the instance is
    withheld (PyAutoFit #1535).
    """
    samples = _samples_with_rejected_best()
    paths = MagicMock()

    (
        samples_out,
        samples_summary,
        instance,
        samples_save,
    ) = _updater_for(samples, paths)._save_samples(
        model=samples.model,
        search_internal=None,
        during_analysis=False,
    )

    assert instance is None
    assert samples_out is samples
    assert samples_summary is not None

    paths.save_samples.assert_called_once()
    paths.save_samples_summary.assert_called_once()

    assert paths.save_samples.call_args.kwargs["samples"] is samples_save


def test__save_samples__weight_threshold_prune_runs_when_the_best_point_is_rejected():
    """
    The early return this replaces also skipped the weight-threshold prune, so every
    zero-weight row survived into `samples.csv` (PyAutoFit #1487).
    """
    samples = _samples_with_rejected_best()
    paths = MagicMock()

    _, _, instance, samples_save = _updater_for(samples, paths)._save_samples(
        model=samples.model,
        search_internal=None,
        during_analysis=False,
    )

    assert instance is None
    assert len(samples.sample_list) == 2
    assert [sample.weight for sample in samples_save.sample_list] == [1.0]


@with_config(
    "general",
    "output",
    "samples_to_csv",
    value=True,
)
def test__save_samples__writes_samples_csv_when_the_best_point_is_rejected(
    output_directory,
):
    samples = _samples_with_rejected_best()
    paths = af.DirectoryPaths(
        "rejected_best_point",
        path_prefix=str(output_directory),
    )

    _, _, instance, _ = _updater_for(samples, paths)._save_samples(
        model=samples.model,
        search_internal=None,
        during_analysis=False,
    )

    assert instance is None
    assert (paths._files_path / "samples.csv").exists()
    assert (paths._files_path / "samples_summary.json").exists()
