"""
The family bases out of the MRO (search-extensibility A4c, decision D14).

``AbstractMCMC`` / ``AbstractNest`` / ``AbstractMLE`` are deprecated thin subclasses of
``NonLinearSearch``: subclassing one warns, and a subclass keeps the former defaults.
The public searches apply their ``posterior_kind`` defaults in their own ``__init__``
and plot through the per-kind plotter.
"""

import json
import warnings
from pathlib import Path

import pytest

import autofit as af
from autofit.non_linear.initializer import InitializerBall, InitializerPrior
from autofit.non_linear.samples.mcmc import SamplesMCMC
from autofit.non_linear.samples.nest import SamplesNest
from autofit.non_linear.search import posterior_kind
from autofit.non_linear.search.mcmc.abstract_mcmc import AbstractMCMC
from autofit.non_linear.search.mcmc.auto_correlations import AutoCorrelationsSettings
from autofit.non_linear.search.mle.abstract_mle import AbstractMLE
from autofit.non_linear.search.nest.abstract_nest import AbstractNest
from autonerves.dictable import get_class_path

from test_autofit.non_linear.search.conformance_roster import searches_under_test


def _subclass(base):
    """Define a minimal external search on a deprecated base, as a user would."""

    class External(base):
        __identifier_fields__ = ()

        def _fit(self, model, analysis):
            raise NotImplementedError

    return External


@pytest.mark.parametrize("base", [AbstractMCMC, AbstractNest, AbstractMLE])
def test_subclassing_a_deprecated_family_base_warns(base):
    with pytest.warns(FutureWarning, match=f"subclasses {base.__name__}"):
        _subclass(base)


@pytest.mark.parametrize("base", [AbstractMCMC, AbstractNest, AbstractMLE])
def test_deprecated_bases_are_distinct_classes(base):
    assert base is not af.NonLinearSearch
    assert issubclass(base, af.NonLinearSearch)


def test_deprecated_mcmc_keeps_its_defaults():
    with pytest.warns(FutureWarning):
        External = _subclass(AbstractMCMC)

    search = External()

    assert isinstance(search.initializer, InitializerBall)
    assert (search.initializer.lower_limit, search.initializer.upper_limit) == (
        0.49,
        0.51,
    )
    assert isinstance(search.auto_correlation_settings, AutoCorrelationsSettings)
    assert External.samples_cls is SamplesMCMC
    assert not hasattr(search, "clipper")


def test_deprecated_nest_keeps_its_defaults():
    with pytest.warns(FutureWarning):
        External = _subclass(AbstractNest)

    assert isinstance(External().initializer, InitializerPrior)
    assert External.samples_cls is SamplesNest
    with pytest.raises(ValueError, match="nested sampling"):
        External(initializer=af.InitializerParamBounds({}))


def test_deprecated_mle_keeps_its_defaults():
    with pytest.warns(FutureWarning):
        External = _subclass(AbstractMLE)

    search = External()

    assert isinstance(search.initializer, InitializerBall)
    assert isinstance(search.clipper, af.ClipperNone)


def test_public_searches_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        for entry in searches_under_test():
            if not entry.missing_requirements():
                entry.resolve()()


@pytest.mark.parametrize(
    "cls, initializer",
    [
        (af.Emcee, InitializerBall),
        (af.Zeus, InitializerBall),
        (af.BlackJAXNUTS, InitializerBall),
        (af.SMC, InitializerPrior),
        (af.DynestyStatic, InitializerPrior),
        (af.Nautilus, InitializerPrior),
        (af.Drawer, InitializerBall),
        (af.BFGS, InitializerBall),
        (af.MultiStartAdam, InitializerBall),
    ],
)
def test_posterior_kind_defaults_applied(cls, initializer):
    assert type(cls().initializer) is initializer


@pytest.mark.parametrize("cls", [af.DynestyStatic, af.DynestyDynamic, af.Nautilus])
def test_nested_samplers_still_reject_param_bounds(cls):
    with pytest.raises(ValueError, match="nested sampling"):
        cls(initializer=af.InitializerParamBounds({}))


def test_smc_is_weighted_and_takes_no_auto_correlation_settings():
    search = af.SMC()

    assert str(af.SMC.posterior_kind) == "weighted"
    assert af.SMC.produces_evidence is True
    assert not hasattr(search, "auto_correlation_settings")


def test_old_smc_search_json_with_auto_correlation_settings_loads(caplog):
    golden = json.loads((Path(__file__).parent / "seed_golden.json").read_text())
    old = golden["SMC"]["main_search_json"]
    assert "auto_correlation_settings" in old["arguments"]

    loaded = af.from_dict(old)

    assert isinstance(loaded, af.SMC)
    assert "unknown keyword argument" not in caplog.text


# `samples_info["class_path"]` is the search's `samples_cls`; main 8437a3b76's values.
SAMPLES_CLASS_PATHS = {
    "Emcee": "autofit.non_linear.samples.mcmc.SamplesMCMC",
    "Zeus": "autofit.non_linear.samples.mcmc.SamplesMCMC",
    "BlackJAXNUTS": "autofit.non_linear.samples.mcmc.SamplesMCMC",
    "SMC": "autofit.non_linear.search.mcmc.blackjax.smc.samples.SamplesSMC",
    "DynestyStatic": "autofit.non_linear.samples.nest.SamplesNest",
    "DynestyDynamic": "autofit.non_linear.samples.nest.SamplesNest",
    "Nautilus": "autofit.non_linear.samples.nest.SamplesNest",
    "NSS": "autofit.non_linear.search.nest.nss.samples.NSSamples",
    "Drawer": "autofit.non_linear.samples.samples.Samples",
    "BFGS": "autofit.non_linear.samples.samples.Samples",
    "LBFGS": "autofit.non_linear.samples.samples.Samples",
    "MultiStartAdam": "autofit.non_linear.samples.samples.Samples",
    "MultiStartADABelief": "autofit.non_linear.samples.samples.Samples",
    "MultiStartLion": "autofit.non_linear.samples.samples.Samples",
    "MultiStartProdigy": "autofit.non_linear.samples.samples.Samples",
}


@pytest.mark.parametrize("entry", searches_under_test(), ids=lambda e: e.name)
def test_samples_class_path_unchanged(entry):
    assert (
        get_class_path(entry.resolve().samples_cls) == SAMPLES_CLASS_PATHS[entry.name]
    )


@pytest.mark.parametrize(
    "cls, kind",
    [
        (af.Emcee, posterior_kind.CHAIN),
        (af.SMC, posterior_kind.WEIGHTED),
        (af.Nautilus, posterior_kind.WEIGHTED),
        (af.BFGS, posterior_kind.POINT),
    ],
)
def test_plot_results_dispatches_on_posterior_kind(cls, kind, monkeypatch):
    calls = []
    monkeypatch.setattr(
        type(kind), "plot_results", lambda self, search, samples: calls.append(self)
    )

    cls().plot_results(samples=None)

    assert calls == [kind]


def test_search_without_a_posterior_kind_plots_nothing_of_its_own():
    with pytest.raises(NotImplementedError):
        af.m.MockSearch().plot_results(samples=None)


# Adversary review 07, finding 4: positional constructor compatibility.


def test_drawer_positional_slots_are_unchanged():
    """`Drawer`'s explicit `clipper` is appended, so main's positional calls bind as before."""
    from autofit.non_linear.clipper import ClipperNone

    drawer = af.Drawer(None, None, None, 50, None, 12, 34, True)

    assert drawer.total_draws == 50
    assert drawer.iterations_per_full_update == 12
    assert drawer.iterations_per_quick_update == 34
    assert drawer.silence is True
    assert isinstance(drawer.clipper, ClipperNone)


def test_drawer_clipper_still_accepted_by_keyword_and_serialized():
    from autofit.non_linear.clipper import ClipperNone
    from autonerves.dictable import get_arguments

    assert "clipper" in get_arguments(af.Drawer)
    assert isinstance(af.Drawer(clipper=None).clipper, ClipperNone)
    assert "clipper" in af.to_dict(af.Drawer())["arguments"]


def _smc_positional_prefix():
    # name, path_prefix, unique_tag, num_particles, kernel, num_mcmc_steps,
    # num_integration_steps, target_ess, step_size, whiten_inflate, max_smc_steps,
    # batch_size, seed, initializer, inverse_mass_matrix
    return (
        None,
        None,
        None,
        256,
        "mala",
        5,
        8,
        0.5,
        None,
        2.0,
        200,
        0,
        None,
        None,
        None,
    )


def test_smc_old_positional_auto_correlation_slot_binds_as_before():
    pytest.importorskip("blackjax")
    settings = AutoCorrelationsSettings(check_for_convergence=False)

    with pytest.warns(FutureWarning, match="auto_correlation_settings"):
        search = af.SMC(*_smc_positional_prefix(), settings, 3, 7, 1, True)

    assert search.iterations_per_quick_update == 3
    assert search.iterations_per_full_update == 7
    assert search.number_of_cores == 1
    assert search.silence is True
    assert "auto_correlation_settings" not in af.to_dict(search)["arguments"]


def test_smc_too_many_positional_arguments_raise():
    pytest.importorskip("blackjax")

    with pytest.raises(TypeError, match="deprecated positional"):
        af.SMC(*_smc_positional_prefix(), None, 3, 7, 1, True, None, "extra")


def test_smc_keyword_construction_does_not_warn():
    pytest.importorskip("blackjax")

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        search = af.SMC(iterations_per_quick_update=3, silence=True)

    assert search.iterations_per_quick_update == 3
    assert search.silence is True


class _MockGalaxy:
    def __init__(self, light=None):
        self.light = light


class _MockShear:
    latent_parameters = {
        ("gamma_1", "gamma_2"): lambda gamma_1, gamma_2, **kwargs: {
            "magnitude": (gamma_1**2 + gamma_2**2) ** 0.5
        }
    }

    def __init__(self, gamma_1=0.0, gamma_2=0.0):
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2


def _lens_like_model():
    return af.Collection(
        galaxies=af.Collection(
            lens=af.Model(_MockGalaxy, light=af.Model(af.m.MockClassx2NoSuperScript)),
            source=af.Model(_MockGalaxy, light=af.Model(af.m.MockClassx2NoSuperScript)),
        )
    )


def _weighted_plot_calls(model, monkeypatch, tmp_path, latent=False):
    from types import SimpleNamespace
    import autofit.non_linear.plot as plot_module
    from autofit.non_linear.search.posterior_kind import WeightedDefaults

    calls = []
    monkeypatch.setattr(
        plot_module, "corner_anesthetic", lambda **kwargs: calls.append(kwargs)
    )
    search = SimpleNamespace(paths=SimpleNamespace(image_path=tmp_path))
    samples = SimpleNamespace(
        model=model,
        parameter_lists=[[0.1] * model.prior_count] * 3,
        weight_list=[1.0] * 3,
    )
    defaults = WeightedDefaults()
    defaults.plot_results(search=search, samples=samples)
    if latent:
        defaults.plot_latent_results(search=search, samples=samples)
    return calls


def test_weighted_plot_results_one_corner_per_component(monkeypatch, tmp_path):
    # The full corner plot of a many-parameter model is too dense to read, so
    # each top-level component (each galaxy here) also gets its own.
    calls = _weighted_plot_calls(_lens_like_model(), monkeypatch, tmp_path)

    assert [call.get("filename", "corner_anesthetic") for call in calls] == [
        "corner_anesthetic",
        "corner_anesthetic_lens",
        "corner_anesthetic_source",
    ]
    assert "parameter_indices" not in calls[0]
    assert calls[1]["parameter_indices"] == [0, 1]
    assert calls[2]["parameter_indices"] == [2, 3]


def test_weighted_plot_results_single_component_has_no_duplicate_corner(
    monkeypatch, tmp_path
):
    # One component's corner plot would repeat the full one.
    calls = _weighted_plot_calls(
        af.Model(af.m.MockClassx2NoSuperScript), monkeypatch, tmp_path
    )

    assert [call.get("filename", "corner_anesthetic") for call in calls] == [
        "corner_anesthetic"
    ]


def test_weighted_plot_results_components_key_missing_from_config_defaults_on(
    monkeypatch, tmp_path
):
    # A workspace config written before the per-component plots existed does
    # not mention them; they default on rather than raising KeyError.
    from autofit.non_linear.search.posterior_kind import WeightedDefaults

    real_should_plot = WeightedDefaults.should_plot

    def should_plot(self, name, default=None):
        if name == "corner_anesthetic_components":
            if default is None:
                raise KeyError(name)
            return default
        return real_should_plot(self, name, default)

    monkeypatch.setattr(WeightedDefaults, "should_plot", should_plot)

    calls = _weighted_plot_calls(_lens_like_model(), monkeypatch, tmp_path)

    assert len(calls) == 3


def test_weighted_plot_results_latent_corner_only_when_a_component_declares_one(
    monkeypatch, tmp_path
):
    from autofit.non_linear.search.posterior_kind import WeightedDefaults

    # The latent plots are off by default; every switch on for this test.
    monkeypatch.setattr(
        WeightedDefaults, "should_plot", lambda self, name, default=None: True
    )

    # Nothing latent: the latent plot would repeat the plain one, so none.
    calls = _weighted_plot_calls(_lens_like_model(), monkeypatch, tmp_path, latent=True)
    assert not any("latent" in call.get("filename", "") for call in calls)

    # A shear declares its magnitude: a latent plot, and one for the lens
    # since the per-component switch is on too; none for the source, which
    # has no latent parameter and whose plot would repeat its plain one.
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                _MockGalaxy,
                light=af.Model(
                    _MockShear,
                    gamma_1=af.UniformPrior(-1.0, 1.0),
                    gamma_2=af.UniformPrior(-1.0, 1.0),
                ),
            ),
            source=af.Model(_MockGalaxy, light=af.Model(af.m.MockClassx2NoSuperScript)),
        )
    )
    calls = _weighted_plot_calls(model, monkeypatch, tmp_path, latent=True)

    latent_calls = [call for call in calls if call.get("latent")]
    assert [call["filename"] for call in latent_calls] == [
        "corner_anesthetic_latent",
        "corner_anesthetic_latent_lens",
    ]
    assert latent_calls[1]["parameter_indices"] == [0]  # magnitude alone


def test_weighted_plot_results_latent_switch_is_independent(monkeypatch, tmp_path):
    # Latent on, the other two off: the latent plot is the only corner plot.
    from autofit.non_linear.search.posterior_kind import WeightedDefaults

    monkeypatch.setattr(
        WeightedDefaults,
        "should_plot",
        lambda self, name, default=None: name == "corner_anesthetic_latent",
    )
    model = af.Collection(
        lens=af.Model(
            _MockGalaxy,
            light=af.Model(
                _MockShear,
                gamma_1=af.UniformPrior(-1.0, 1.0),
                gamma_2=af.UniformPrior(-1.0, 1.0),
            ),
        )
    )

    calls = _weighted_plot_calls(model, monkeypatch, tmp_path, latent=True)

    assert [call.get("filename", "corner_anesthetic") for call in calls] == [
        "corner_anesthetic_latent"
    ]


def test_weighted_plot_results_never_makes_the_latent_plots(monkeypatch, tmp_path):
    # The latent plots are latent output: ``plot_results`` (every
    # ``iterations_per_full_update``) leaves them to ``plot_latent_results``,
    # which the updater calls once the fit is complete.
    from autofit.non_linear.search.posterior_kind import WeightedDefaults

    monkeypatch.setattr(
        WeightedDefaults, "should_plot", lambda self, name, default=None: True
    )
    model = af.Collection(
        lens=af.Model(
            _MockGalaxy,
            light=af.Model(
                _MockShear,
                gamma_1=af.UniformPrior(-1.0, 1.0),
                gamma_2=af.UniformPrior(-1.0, 1.0),
            ),
        )
    )

    calls = _weighted_plot_calls(model, monkeypatch, tmp_path, latent=False)

    assert not any(call.get("latent") for call in calls)


def test_search_plot_latent_results_dispatches_on_posterior_kind(monkeypatch):
    from autofit.non_linear.search import posterior_kind

    calls = []
    monkeypatch.setattr(
        posterior_kind.WeightedDefaults,
        "plot_latent_results",
        lambda self, search, samples: calls.append(self),
    )

    af.Nautilus().plot_latent_results(samples=None)
    assert calls == [posterior_kind.WEIGHTED]

    # Other kinds, and a search without one, plot nothing and do not raise.
    af.Emcee().plot_latent_results(samples=None)
    af.m.MockSearch().plot_latent_results(samples=None)


def test_weighted_plot_latent_results_single_component_has_no_duplicate(
    monkeypatch, tmp_path
):
    # One component with a latent parameter and no other: its per-component
    # latent plot would repeat the full latent plot, so none is made.
    from autofit.non_linear.search.posterior_kind import WeightedDefaults

    monkeypatch.setattr(
        WeightedDefaults, "should_plot", lambda self, name, default=None: True
    )
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                _MockGalaxy,
                light=af.Model(
                    _MockShear,
                    gamma_1=af.UniformPrior(-1.0, 1.0),
                    gamma_2=af.UniformPrior(-1.0, 1.0),
                ),
            ),
        )
    )

    calls = _weighted_plot_calls(model, monkeypatch, tmp_path, latent=True)

    assert [call.get("filename", "corner_anesthetic") for call in calls] == [
        "corner_anesthetic",
        "corner_anesthetic_latent",
    ]


def _without_key(monkeypatch, missing, values):
    # A config written before ``missing`` existed (should_plot raises KeyError
    # without a default), with the given value for every other key.
    from autofit.non_linear.search.posterior_kind import WeightedDefaults

    def should_plot(self, name, default=None):
        if name == missing:
            if default is None:
                raise KeyError(name)
            return default
        return values[name]

    monkeypatch.setattr(WeightedDefaults, "should_plot", should_plot)


def test_weighted_plot_results_components_follow_corner_anesthetic_when_key_absent(
    monkeypatch, tmp_path
):
    # Switching the corner plot off in an older config switches off the
    # per-component plots it does not mention, rather than leaving them on.
    _without_key(
        monkeypatch, "corner_anesthetic_components", {"corner_anesthetic": False}
    )

    calls = _weighted_plot_calls(_lens_like_model(), monkeypatch, tmp_path)

    assert calls == []


def _shear_model(shear_cls):
    return af.Collection(
        lens=af.Model(
            _MockGalaxy,
            light=af.Model(
                shear_cls,
                gamma_1=af.UniformPrior(-1.0, 1.0),
                gamma_2=af.UniformPrior(-1.0, 1.0),
            ),
        ),
        source=af.Model(_MockGalaxy, light=af.Model(af.m.MockClassx2NoSuperScript)),
    )


def test_weighted_plot_latent_results_components_have_their_own_switch(
    monkeypatch, tmp_path
):
    # The latent plot alone: its per-component plots are switched separately.
    from autofit.non_linear.search.posterior_kind import WeightedDefaults

    monkeypatch.setattr(
        WeightedDefaults,
        "should_plot",
        lambda self, name, default=None: name == "corner_anesthetic_latent",
    )

    calls = _weighted_plot_calls(
        _shear_model(_MockShear), monkeypatch, tmp_path, latent=True
    )

    assert [call["filename"] for call in calls] == ["corner_anesthetic_latent"]


class _MockRaisingShear(_MockShear):
    latent_parameters = {
        ("gamma_1", "gamma_2"): lambda gamma_1, gamma_2, **kwargs: 1 / 0
    }


def test_weighted_plot_latent_results_logs_a_raising_conversion(
    monkeypatch, tmp_path, caplog
):
    # A component's conversion is its own code: one that raises is logged and
    # no latent plot is made, rather than failing the fit's final output.
    from autofit.non_linear.search.posterior_kind import WeightedDefaults

    monkeypatch.setattr(
        WeightedDefaults, "should_plot", lambda self, name, default=None: True
    )

    with caplog.at_level("WARNING"):
        calls = _weighted_plot_calls(
            _shear_model(_MockRaisingShear), monkeypatch, tmp_path, latent=True
        )

    assert not any(call.get("latent") for call in calls)
    assert "latent parameters could not be computed" in caplog.text


def test_weighted_plot_latent_results_skipped_in_test_mode(monkeypatch, tmp_path):
    from autofit.non_linear.search.posterior_kind import WeightedDefaults

    monkeypatch.setattr(
        WeightedDefaults, "should_plot", lambda self, name, default=None: True
    )
    monkeypatch.setattr("autofit.non_linear.test_mode.skip_visualization", lambda: True)

    calls = _weighted_plot_calls(
        _shear_model(_MockShear), monkeypatch, tmp_path, latent=True
    )

    assert not any(call.get("latent") for call in calls)


def test_weighted_plot_results_skips_full_corner_of_a_very_big_model(
    monkeypatch, tmp_path
):
    # Past corner_anesthetic.max_parameters_search the full plot is large and
    # slow; the per-component plots are still made.
    from autofit.non_linear.search.posterior_kind import WeightedDefaults

    monkeypatch.setattr(
        WeightedDefaults,
        "_too_many_parameters_for_full_corner",
        staticmethod(lambda samples: True),
    )
    calls = _weighted_plot_calls(_lens_like_model(), monkeypatch, tmp_path)

    assert [call.get("filename", "corner_anesthetic") for call in calls] == [
        "corner_anesthetic_lens",
        "corner_anesthetic_source",
    ]


@pytest.mark.parametrize("limit, expected", [(3, True), (4, False)])
def test_too_many_parameters_for_full_corner_reads_config(monkeypatch, limit, expected):
    from types import SimpleNamespace
    from autofit.non_linear.search import posterior_kind

    monkeypatch.setattr(
        posterior_kind,
        "conf",
        SimpleNamespace(
            instance={
                "visualize": {
                    "plots_settings": {
                        "corner_anesthetic": {"max_parameters_search": limit}
                    }
                }
            }
        ),
    )
    samples = SimpleNamespace(model=_lens_like_model())

    assert (
        posterior_kind.WeightedDefaults._too_many_parameters_for_full_corner(samples)
        is expected
    )


def test_too_many_parameters_for_full_corner_older_config_defaults_to_30(monkeypatch):
    from types import SimpleNamespace
    from autofit.non_linear.search import posterior_kind

    monkeypatch.setattr(
        posterior_kind,
        "conf",
        SimpleNamespace(instance={"visualize": {"plots_settings": {"corner_anesthetic": {}}}}),
    )
    small = SimpleNamespace(model=SimpleNamespace(total_free_parameters=30))
    big = SimpleNamespace(model=SimpleNamespace(total_free_parameters=31))

    assert not posterior_kind.WeightedDefaults._too_many_parameters_for_full_corner(small)
    assert posterior_kind.WeightedDefaults._too_many_parameters_for_full_corner(big)


@pytest.mark.parametrize(
    "many_components, expected",
    [
        (True, "per-component corner plots are still made"),
        (False, "No corner plot is made"),
    ],
)
def test_skipped_full_corner_logs_what_is_actually_made(
    monkeypatch, tmp_path, caplog, many_components, expected
):
    # A single-component model has no per-component plots to fall back on, so
    # the log must not promise them.
    import logging
    from autofit.non_linear.search.posterior_kind import WeightedDefaults

    monkeypatch.setattr(
        WeightedDefaults,
        "_too_many_parameters_for_full_corner",
        staticmethod(lambda samples: True),
    )
    # Built here, not at collection: component grouping reads the test config.
    model = (
        _lens_like_model()
        if many_components
        else af.Model(af.m.MockClassx2NoSuperScript)
    )
    with caplog.at_level(logging.INFO):
        _weighted_plot_calls(model, monkeypatch, tmp_path)

    assert expected in caplog.text
