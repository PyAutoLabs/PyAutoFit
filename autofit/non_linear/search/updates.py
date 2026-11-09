"""
How a ``NonLinearSearch`` writes its results while and after it runs: the cached
``SearchUpdater`` (samples, latent variables, visualization, profiling, summary) and
the update schedule the chunked searches share.

``SearchUpdatesMixin`` is a base of ``NonLinearSearch`` only; its methods are the
search's own (subclasses override them as before). It defines no ``__init__`` and no
constructor argument, so serialized searches and identifiers are unaffected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from autofit.non_linear.fitness import Fitness
from autofit.non_linear.samples.samples import Samples
from autofit.non_linear.samples.summary import SamplesSummary

if TYPE_CHECKING:
    from autofit.mapper.model import ModelInstance
    from autofit.mapper.prior_model.abstract import AbstractPriorModel
    from autofit.non_linear.analysis import Analysis
    from autofit.non_linear.paths.abstract import AbstractPaths


class SearchUpdatesMixin:
    def _steps_until_full_update(self, iterations_remaining: int) -> int:
        """
        How many iterations to run before the next ``perform_update``
        checkpoint, given how much of the budget is left.

        Searches that chunk their run around ``iterations_per_full_update`` all
        need this same number, and they all need it as an ``int`` — most feed it
        to something that ultimately does ``range(...)``. But
        ``iterations_per_full_update`` is stored as a **float** by
        ``__init__``, because the packaged default is the inf-like ``1e99``
        meaning "one chunk, checkpoint only at the end". That float is kept
        deliberately: ``int(1e99)`` is a 99-digit integer, and writing *that*
        into every saved ``search.json`` in place of a readable ``1e99``
        sentinel would be a poor trade for a conversion each caller can do at
        the point of use.

        So the conversion lives here, once, instead of being re-derived (and
        occasionally forgotten) per search. Forgetting it is not hypothetical:
        it crashed the MultiStart gradient search for every user who set a real
        cadence (PyAutoFit#1420), and left the same latent crash in Emcee and
        BlackJAX NUTS (PyAutoFit#1422).

        There is no second "no intermediate checkpoint" sentinel: the config
        default ``1e99`` already means that, and it flows through the ``min``
        below without a special case. A stored ``0`` is therefore a
        misconfiguration — most plausibly an HPC override — and is rejected
        rather than silently reinterpreted as "checkpoint never".

        Parameters
        ----------
        iterations_remaining
            Iterations left in this search's budget. Validated, because the
            returned chunk can only be a usable positive whole number if this
            is one — searches store their budget under different names
            (``n_steps``, ``nsteps``, ``num_samples``, ``maxiter``) and none of
            them validates it.

        Raises
        ------
        ValueError
            If ``iterations_remaining``, or an explicitly supplied
            ``iterations_per_full_update``, is not a whole number of at
            least 1.
        """
        self._check_step_count(iterations_remaining, "iterations_remaining")
        self._check_step_count(
            self.iterations_per_full_update, "iterations_per_full_update"
        )
        return int(min(self.iterations_per_full_update, iterations_remaining))

    def _check_step_count(self, value, name: str):
        """
        Reject an iteration count that cannot describe a whole number of
        iterations. ``float`` values are allowed — the packaged config default
        ``1e99`` is one — provided they are integral.

        Rejecting rather than clamping is deliberate: a value below 1 truncates
        to a zero-length chunk, so the enclosing ``while`` loop makes no
        progress and spins forever re-running ``perform_update``. Clamping it to
        1 would hide the mistake and silently run a schedule the user never
        asked for, and a silent hang on a cluster is far more expensive than an
        error at the first chunk boundary.
        """
        try:
            is_whole = value == int(value)
        except (TypeError, ValueError, OverflowError):
            is_whole = False

        if not is_whole or value < 1:
            raise ValueError(
                f"{type(self).__name__}: `{name}` must be a whole number of "
                f"iterations and at least 1, but was {value!r}. A fractional "
                "or sub-1 value gives a zero-length chunk, which never advances "
                "the search and would hang it rather than fail."
            )

    @property
    def _updater(self):
        # The cached ``SearchUpdater`` must be invalidated whenever
        # ``self.paths`` is reassigned to a new object — otherwise the
        # updater holds a stale reference to the old paths and writes
        # output (samples, visualizations, profiles) under the wrong
        # directory. This happens routinely when a single search
        # instance is reused across factor optimisations in the EP
        # loop: EP (``SearchFactorOptimiser.optimise``) sets
        # ``self.paths = SubDirectoryPaths(...)`` per factor and per EP
        # iteration, but the updater would otherwise stay pinned to
        # whichever paths were live the first time ``_updater`` was
        # accessed. Identity comparison (``is not``) is the right test:
        # the search instance receives a freshly-constructed
        # ``SubDirectoryPaths`` each time, never an in-place mutation.
        cached = getattr(self, "_search_updater", None)
        if cached is None or cached._paths is not self.paths:
            from autofit.non_linear.search.updater import SearchUpdater

            self._search_updater = SearchUpdater(
                paths=self.paths,
                timer=self.timer,
                search_logger=self.logger,
                plot_results_func=self.plot_results,
                plot_results_during_fit=self.plots_results_during_fit,
                plot_latent_results_func=self.plot_latent_results,
                samples_from_func=self.samples_from,
                disable_output=self.disable_output,
                iterations_per_full_update=self.iterations_per_full_update,
            )
        # Re-applied on every access: EP toggles ``_visualize_fit``
        # per EP factor search, and the cached updater must follow it.
        self._search_updater.visualization_enabled = self._visualize_fit
        return self._search_updater

    def perform_update(
        self,
        model: AbstractPriorModel,
        analysis: Analysis,
        during_analysis: bool,
        fitness: Optional[Fitness] = None,
        search_internal=None,
    ) -> Samples:
        """
        Perform an update of the non-linear search's model-fitting results.

        Delegates to :class:`SearchUpdater` which separates each output
        concern (samples, latent variables, visualization, profiling,
        summary) into its own method.
        """
        return self._updater.update(
            model=model,
            analysis=analysis,
            during_analysis=during_analysis,
            fitness=fitness,
            search_internal=search_internal,
        )

    def perform_visualization(
        self,
        model: AbstractPriorModel,
        analysis: Analysis,
        during_analysis: bool,
        samples_summary: Optional[SamplesSummary] = None,
        instance: Optional[ModelInstance] = None,
        paths_override: Optional[AbstractPaths] = None,
        search_internal=None,
        plot_search: bool = True,
    ):
        """
        Perform visualization of the non-linear search's model-fitting results.

        Delegates to :class:`SearchUpdater.visualize`.
        """
        self._updater.visualize(
            model=model,
            analysis=analysis,
            during_analysis=during_analysis,
            samples_summary=samples_summary,
            instance=instance,
            paths_override=paths_override,
            search_internal=search_internal,
            plot_search=plot_search,
        )

    def plot_results(self, samples):
        """
        The search's own plots of its samples, by its declared ``posterior_kind``
        (``autofit.non_linear.search.posterior_kind``): the chain, weighted or point
        plotter. A search that declares no kind plots nothing of its own.
        """
        from autofit.non_linear.search.posterior_kind import defaults_for

        defaults = defaults_for(type(self).posterior_kind)
        if defaults is None:
            raise NotImplementedError
        defaults.plot_results(self, samples)

    @property
    def plots_results_during_fit(self) -> bool:
        """
        Whether ``plot_results`` runs at every ``iterations_per_full_update`` or
        only once the fit is complete, by the search's ``posterior_kind``
        (``PosteriorDefaults.plots_during_fit``).
        """
        from autofit.non_linear.search.posterior_kind import defaults_for

        defaults = defaults_for(type(self).posterior_kind)
        return True if defaults is None else defaults.plots_during_fit

    def plot_latent_results(self, samples):
        """
        The search's plots of the model's latent parameters (the corner plot of
        a shear's magnitude and angle, say), by its ``posterior_kind``; nothing
        for a search without one. ``SearchUpdater.visualize`` calls this once the
        fit is complete rather than at every ``iterations_per_full_update``, as
        it does for ``latent.csv``.
        """
        from autofit.non_linear.search.posterior_kind import defaults_for

        defaults = defaults_for(type(self).posterior_kind)
        if defaults is None:
            return
        defaults.plot_latent_results(self, samples)
