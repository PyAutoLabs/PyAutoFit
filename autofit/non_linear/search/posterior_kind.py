"""
Per-``posterior_kind`` defaults and plotters (search-extensibility A4c, decision D14).

The ``AbstractMCMC`` / ``AbstractNest`` / ``AbstractMLE`` family bases carried three
defaults (initializer, clipper, auto-correlation settings) and one plot routine. They are
no longer in any public search's MRO; what they carried lives here, keyed by the
search's declared ``posterior_kind`` (``autofit.non_linear.search.capabilities``):

- ``CHAIN`` (Emcee, Zeus, BlackJAXNUTS): ``InitializerBall(0.49, 0.51)``, default
  ``AutoCorrelationsSettings``, the cornerpy corner plot once the chain is converged;
- ``WEIGHTED`` (Dynesty, Nautilus, NSS, SMC): ``InitializerPrior`` (nested samplers
  also reject ``InitializerParamBounds``), the anesthetic corner plot;
- ``POINT`` (BFGS, LBFGS, MultiStart, Drawer): ``InitializerBall(0.49, 0.51)``,
  ``ClipperNone``, the parameter / log-likelihood / figure-of-merit traces.

Each concrete search applies its kind's defaults in its own ``__init__``, never on
``NonLinearSearch``: the #1493 clipper tripwire requires that the clipper never reaches
the nested samplers. ``NonLinearSearch.plot_results`` plots through ``plotter_for``.
"""

import logging
import warnings
from typing import Optional

import numpy as np

from autonerves import conf

from autofit.non_linear.search.capabilities import PosteriorKind

logger = logging.getLogger(__name__)


class PosteriorDefaults:
    kind: PosteriorKind = None
    plots_section: str = None
    # Whether ``plot_results`` runs at every ``iterations_per_full_update`` as
    # well as once the fit is complete. Progress plots (a point search's
    # likelihood against iteration) are worth watching live; a posterior's
    # corner plot is not -- it needs converged samples and is the slowest plot
    # a search makes -- so the chain and weighted kinds draw theirs at the end.
    plots_during_fit: bool = True

    def initializer_for(self, initializer=None):
        """The search's initializer: ``initializer``, or this kind's default."""
        return initializer if initializer is not None else self.default_initializer()

    def default_initializer(self):
        raise NotImplementedError

    def should_plot(self, name: str, default: Optional[bool] = None) -> bool:
        """
        Whether the ``plots_search`` config switches on the named plot. A key a
        user's (older) config does not mention falls back to ``default`` when one
        is given, so a new plot does not break an existing workspace.
        """
        try:
            return conf.instance["visualize"]["plots_search"][self.plots_section][name]
        except KeyError:
            if default is None:
                raise
            return default

    def plot_results(self, search, samples):
        raise NotImplementedError

    def plot_latent_results(self, search, samples):
        """
        The plots of the latent parameters, made once the fit is complete (see
        ``WeightedDefaults.plot_latent_results``); none for most kinds.
        """


class ChainDefaults(PosteriorDefaults):
    kind = PosteriorKind.CHAIN
    plots_section = "mcmc"
    plots_during_fit = False

    def default_initializer(self):
        from autofit.non_linear.initializer import InitializerBall

        return InitializerBall(lower_limit=0.49, upper_limit=0.51)

    @staticmethod
    def auto_correlation_settings_for(auto_correlation_settings=None, **defaults):
        """
        The chain's auto-correlation settings: ``auto_correlation_settings``, or
        ``AutoCorrelationsSettings(**defaults)`` built at construction, so a run-time
        ``PYAUTO_TEST_MODE`` reaches its reduced ``check_size`` (the former import-time
        default instance did not).
        """
        from autofit.non_linear.search.mcmc.auto_correlations import (
            AutoCorrelationsSettings,
        )

        if auto_correlation_settings is None:
            return AutoCorrelationsSettings(**defaults)
        return auto_correlation_settings

    def plot_results(self, search, samples):
        from autofit.non_linear.plot import corner_cornerpy

        if not samples.pdf_converged:
            return

        if self.should_plot("corner_cornerpy"):
            corner_cornerpy(
                samples=samples,
                path=search.paths.image_path / "search",
                format="png",
            )


class WeightedDefaults(PosteriorDefaults):
    kind = PosteriorKind.WEIGHTED
    plots_section = "nest"
    plots_during_fit = False

    def default_initializer(self):
        from autofit.non_linear.initializer import InitializerPrior

        return InitializerPrior()

    def nested_initializer_for(self, initializer=None):
        """
        A nested sampler's initializer: its initial live points are drawn from the prior,
        so ``InitializerParamBounds`` (a bounded start) is rejected.
        """
        from autofit.non_linear.initializer import InitializerParamBounds

        if isinstance(initializer, InitializerParamBounds):
            raise ValueError(
                "InitializerParamBounds cannot be used for nested sampling"
            )
        return self.initializer_for(initializer)

    def plot_results(self, search, samples):
        from autofit.non_linear.plot import corner_anesthetic

        path = search.paths.image_path / "search"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            corner = self.should_plot("corner_anesthetic")

            # An older config without the key: the per-component plots follow
            # the full one, so switching that off switches them off too.
            components = self.should_plot(
                "corner_anesthetic_components", default=corner
            )
            groups = samples.model.parameter_indices_by_component if components else {}

            if corner and self._too_many_parameters_for_full_corner(samples):
                logger.info(
                    "Skipping the full corner_anesthetic plot: the model has more "
                    "free parameters than corner_anesthetic.max_parameters_search "
                    "(visualize/plots_settings). "
                    + (
                        "The per-component corner plots are still made."
                        if len(groups) >= 2
                        else "No corner plot is made; plot a subset with "
                        "corner_anesthetic(samples, parameter_indices=...)."
                    )
                )
            elif corner:
                corner_anesthetic(samples=samples, path=path, format="png")

            if components:
                self._corner_per_component(
                    corner_anesthetic, samples, path, groups=groups
                )

    @staticmethod
    def _too_many_parameters_for_full_corner(samples) -> bool:
        """
        Whether the model is too big for one corner plot of every parameter,
        per ``corner_anesthetic.max_parameters_search`` (30 for an older config
        without the key). Past that the full plot is large and slow to draw,
        and the per-component plots are the readable ones anyway.
        """
        total = getattr(samples.model, "total_free_parameters", None)
        if not isinstance(total, int):
            return False

        from autofit.non_linear.plot.nest_plotters import _setting

        settings = conf.instance["visualize"]["plots_settings"]["corner_anesthetic"]
        return total > _setting(settings, "max_parameters_search", 30)

    def plot_latent_results(self, search, samples):
        """
        The corner plot(s) of the latent parameters: each component's sampled
        parameters re-expressed in the ones it is read in (a shear's magnitude
        and angle rather than gamma_1 / gamma_2), where a component declares the
        conversion (``AbstractPriorModel.latent_parameters_from``). Switched by
        ``corner_anesthetic_latent`` independently of the plots above, so it can
        be the only corner plot or sit beside them; skipped when no component
        declares anything, as it would repeat the plain plot, and per component
        (``corner_anesthetic_latent_components``) only for the components that
        have a latent parameter. Skipped in test mode like every other plot.

        Latent output is made after the fit, not at every ``iterations_per_full_update``
        (``SearchUpdater.visualize``), the same rule as ``latent.csv``.
        """
        from autofit.non_linear.plot import corner_anesthetic
        from autofit.non_linear.test_mode import skip_visualization

        if skip_visualization():
            return

        if not self.should_plot("corner_anesthetic_latent", default=False):
            return

        path = search.paths.image_path / "search"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # The conversions are a component's own code, run here outside the
            # plot's exception handling: one that raises is logged and the fit
            # carries on without the latent plots, as it does without the
            # latent columns of ``latent.csv``.
            try:
                latent = samples.model.latent_parameters_from(
                    np.asarray(samples.parameter_lists)
                )
            except Exception:
                logger.warning(
                    "The model's latent parameters could not be computed, so "
                    "their corner plots are not made.",
                    exc_info=True,
                )
                return

            if not latent.has_latent:
                return

            corner_anesthetic(
                samples=samples,
                path=path,
                filename="corner_anesthetic_latent",
                format="png",
                latent=True,
            )

            # Per component only where the model has several (one alone would
            # repeat the plot above), and only the components with a latent
            # parameter (the others' plots would repeat their plain ones).
            if (
                self.should_plot("corner_anesthetic_latent_components", default=True)
                and len(latent.parameter_indices_by_component) > 1
            ):
                self._corner_per_component(
                    corner_anesthetic,
                    samples,
                    path,
                    groups=latent.parameter_indices_by_latent_component,
                    latent=True,
                )

    @staticmethod
    def _corner_per_component(corner_anesthetic, samples, path, groups, latent=False):
        """
        One corner plot per top-level model component (each galaxy of a lens
        model), as the full plot of a many-parameter model is too dense to read.
        A model with a single component already has it, so none is made (the
        latent plots check that on the model's components, since ``groups``
        then holds only those with a latent parameter).
        """
        if len(groups) < 2 and not latent:
            return

        stem = "corner_anesthetic_latent" if latent else "corner_anesthetic"

        for name, parameter_indices in groups.items():
            corner_anesthetic(
                samples=samples,
                path=path,
                filename=f"{stem}_{name}",
                format="png",
                parameter_indices=parameter_indices,
                latent=latent,
            )


class PointDefaults(PosteriorDefaults):
    kind = PosteriorKind.POINT
    plots_section = "mle"

    def default_initializer(self):
        from autofit.non_linear.initializer import InitializerBall

        return InitializerBall(lower_limit=0.49, upper_limit=0.51)

    @staticmethod
    def clipper_for(clipper=None):
        """
        The search's clipper, the no-op ``ClipperNone`` by default: enforcing prior
        support changes where a search converges, so it is opt-in.
        """
        from autofit.non_linear.clipper import ClipperNone

        return clipper or ClipperNone()

    def plot_results(self, search, samples):
        from autofit.non_linear.plot import (
            figure_of_merit_vs_iteration,
            log_likelihood_vs_iteration,
            subplot_parameters,
        )

        plot_path = search.paths.image_path / "search"

        if self.should_plot("subplot_parameters"):
            subplot_parameters(samples=samples, path=plot_path, format="png")
            subplot_parameters(
                samples=samples, use_log_y=True, path=plot_path, format="png"
            )
            subplot_parameters(
                samples=samples, use_last_50_percent=True, path=plot_path, format="png"
            )

        if self.should_plot("log_likelihood_vs_iteration"):
            log_likelihood_vs_iteration(samples=samples, path=plot_path, format="png")
            log_likelihood_vs_iteration(
                samples=samples, use_log_y=True, path=plot_path, format="png"
            )
            log_likelihood_vs_iteration(
                samples=samples, use_last_50_percent=True, path=plot_path, format="png"
            )

        # No-op unless the samples carry a ``fom_history`` (the auto-convergence
        # gradient searches); LBFGS / Drawer leave it absent. Default-on and
        # tolerant of a config predating this key (older workspaces that shadow
        # the ``mle`` plots section) so it never KeyErrors on an MLE fit.
        try:
            plot_figure_of_merit = self.should_plot("figure_of_merit_vs_iteration")
        except KeyError:
            plot_figure_of_merit = True
        if plot_figure_of_merit:
            figure_of_merit_vs_iteration(samples=samples, path=plot_path, format="png")


CHAIN = ChainDefaults()
WEIGHTED = WeightedDefaults()
POINT = PointDefaults()

_BY_KIND = {defaults.kind: defaults for defaults in (CHAIN, WEIGHTED, POINT)}


def defaults_for(kind) -> Optional[PosteriorDefaults]:
    """The defaults (and plotter) of a ``posterior_kind``, or ``None`` when undeclared."""
    if kind is None:
        return None
    return _BY_KIND[PosteriorKind(kind)]
