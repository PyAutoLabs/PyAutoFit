from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import numpy as np
import psutil

from autoconf import conf
from autoconf.test_mode import skip_latents

from autofit import exc
from autofit.non_linear.paths.database import DatabasePaths
from autofit.non_linear.paths.null import NullPaths

if TYPE_CHECKING:
    from autofit.mapper.prior_model.abstract import AbstractPriorModel
    from autofit.mapper.model import ModelInstance
    from autofit.non_linear.analysis import Analysis
    from autofit.non_linear.fitness import Fitness
    from autofit.non_linear.paths.abstract import AbstractPaths
    from autofit.non_linear.samples.samples import Samples
    from autofit.non_linear.samples.summary import SamplesSummary
    from autofit.non_linear.timer import Timer

logger = logging.getLogger(__name__)


class SearchUpdater:
    """
    Handles periodic output updates during a non-linear search.

    Each output concern (samples, latent variables, visualization,
    profiling, summary) is separated into its own method. The search
    passes its type-specific ``plot_results`` and ``samples_from``
    callables via the constructor so no parallel inheritance hierarchy
    is needed.
    """

    def __init__(
        self,
        paths: AbstractPaths,
        timer: Timer,
        search_logger: logging.Logger,
        plot_results_func: Callable[[Samples], None],
        samples_from_func: Callable[[AbstractPriorModel, ...], Samples],
        disable_output: bool,
        iterations_per_full_update: float,
    ):
        self._paths = paths
        self._timer = timer
        self._logger = search_logger
        self._plot_results = plot_results_func
        self._samples_from = samples_from_func
        self._disable_output = disable_output
        self._iterations_per_full_update = iterations_per_full_update
        self._iterations = 0

    @property
    def iterations(self) -> int:
        return self._iterations

    @iterations.setter
    def iterations(self, value: int):
        self._iterations = value

    def update(
        self,
        model: AbstractPriorModel,
        analysis: Analysis,
        during_analysis: bool,
        fitness: Optional[Fitness] = None,
        search_internal=None,
    ) -> Samples:
        """
        Perform a full update of the non-linear search's model-fitting results.

        Called every ``iterations_per_full_update`` during the search and once
        when the search completes.
        """
        self._update_iteration_state()

        samples, samples_summary, instance, samples_save = self._save_samples(
            model=model,
            search_internal=search_internal,
            during_analysis=during_analysis,
        )

        if instance is None:
            return samples

        latent_samples = self._compute_latent_samples(
            samples=samples,
            samples_save=samples_save,
            analysis=analysis,
            fitness=fitness,
            during_analysis=during_analysis,
        )

        start = time.time()

        self.visualize(
            model=model,
            analysis=analysis,
            samples_summary=samples_summary,
            during_analysis=during_analysis,
            search_internal=search_internal,
        )

        visualization_time = time.time() - start

        self._profile_and_summarize(
            samples=samples,
            analysis=analysis,
            fitness=fitness,
            instance=instance,
            latent_samples=latent_samples,
            visualization_time=visualization_time,
        )

        self._log_process_state()

        return samples

    def visualize(
        self,
        model: AbstractPriorModel,
        analysis: Analysis,
        during_analysis: bool,
        samples_summary: Optional[SamplesSummary] = None,
        instance: Optional[ModelInstance] = None,
        paths_override: Optional[AbstractPaths] = None,
        search_internal=None,
    ):
        """
        Perform visualization of the non-linear search's model-fitting results.

        Delegates to the analysis object for model-specific plots and to the
        search's ``plot_results`` for search-specific plots.
        """
        self._logger.debug("Visualizing")

        paths = paths_override or self._paths

        if instance is None and samples_summary is None:
            raise AssertionError(
                """
                The search's perform_visualization method has been called without an input instance or
                samples_summary.

                This should not occur, please ensure one of these inputs is provided.
                """
            )

        if instance is None:
            instance = samples_summary.instance

        if analysis.should_visualize(paths=paths, during_analysis=during_analysis):
            analysis.visualize(
                paths=paths,
                instance=instance,
                during_analysis=during_analysis,
            )
            analysis.visualize_combined(
                paths=paths,
                instance=instance,
                during_analysis=during_analysis,
            )

        if analysis.should_visualize(paths=paths, during_analysis=during_analysis):
            if not isinstance(paths, NullPaths):
                try:
                    samples = self._samples_from(
                        model, search_internal,
                    )

                    self._plot_results(samples=samples)
                except FileNotFoundError:
                    pass

    def _update_iteration_state(self):
        self._iterations += self._iterations_per_full_update

        if not self._disable_output:
            self._logger.info(
                """Fit Running: Updating results (see output folder)."""
            )

        if not isinstance(self._paths, DatabasePaths) and not isinstance(
            self._paths, NullPaths
        ):
            self._timer.update()

    def _save_samples(
        self,
        model: AbstractPriorModel,
        search_internal,
        during_analysis: bool,
    ) -> Tuple[Samples, SamplesSummary, Optional[ModelInstance], Samples]:
        """
        Generate and persist samples.

        Returns (samples, samples_summary, instance, samples_save).
        ``instance`` is ``None`` when the fit has failed, signalling the
        caller to return early.
        """
        samples = self._samples_from(model, search_internal)
        samples_summary = samples.summary()

        try:
            instance = samples_summary.instance
        except exc.FitException:
            return samples, samples_summary, None, samples

        self._paths.save_samples_summary(samples_summary=samples_summary)

        log_message = not during_analysis and not self._disable_output

        samples_save = samples.samples_above_weight_threshold_from(
            log_message=log_message
        )
        self._paths.save_samples(samples=samples_save)

        return samples, samples_summary, instance, samples_save

    def _compute_latent_samples(
        self,
        samples: Samples,
        samples_save: Samples,
        analysis: Analysis,
        fitness: Optional[Fitness],
        during_analysis: bool,
    ) -> Optional[Samples]:
        """Compute and persist latent variable samples if configured."""
        if skip_latents():
            return None
        if not (
            (during_analysis and conf.instance["output"]["latent_during_fit"])
            or (not during_analysis and conf.instance["output"]["latent_after_fit"])
        ):
            return None

        if conf.instance["output"]["latent_draw_via_pdf"]:
            total_draws = conf.instance["output"]["latent_draw_via_pdf_size"]

            logger.info(
                f"Creating latent samples by drawing {total_draws} from the PDF."
            )

            try:
                latent_samples = samples.samples_drawn_randomly_via_pdf_from(
                    total_draws=total_draws
                )
            except AttributeError:
                latent_samples = samples_save
                logger.info(
                    "Drawing via PDF not available for this search, "
                    "using all samples above the samples weight threshold instead."
                )
        else:
            logger.info(
                "Creating latent samples using all samples above "
                "the samples weight threshold."
            )
            latent_samples = samples_save

        latent_samples = analysis.compute_latent_samples(
            latent_samples,
            batch_size=fitness.batch_size,
        )

        if latent_samples:
            if not conf.instance["output"]["latent_draw_via_pdf"]:
                self._paths.save_latent_samples(latent_samples)
            self._paths.save_samples_summary(
                latent_samples.summary(),
                "latent/latent_summary",
            )

        return latent_samples

    def _profile_and_summarize(
        self,
        samples: Samples,
        analysis: Analysis,
        fitness: Optional[Fitness],
        instance: ModelInstance,
        latent_samples: Optional[Samples],
        visualization_time: float,
    ):
        """Write the search summary file."""
        self._logger.debug("Outputting model result")

        try:
            parameters = samples.max_log_likelihood(as_instance=False)

            start = time.time()
            figure_of_merit = fitness.call_wrap(parameters)

            # account for asynchronous JAX calls
            np.array(figure_of_merit)

            log_likelihood_function_time = time.time() - start

            self._paths.save_summary(
                samples=samples,
                latent_samples=latent_samples,
                log_likelihood_function_time=log_likelihood_function_time,
                visualization_time=visualization_time,
            )
        except exc.FitException:
            pass

    @staticmethod
    def _log_process_state():
        total_files = 0

        for process in psutil.process_iter(attrs=["pid"]):
            try:
                proc_info = process.as_dict(attrs=["pid"])
                logger.debug(
                    f"Process ID: {proc_info['pid']} has the following open files:"
                )

                open_files = process.open_files()
                for file in open_files:
                    logger.debug(file)
                    total_files += 1

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if conf.instance["logging"]["total_files_open"]:
            logger.info(f"Total Files Open: {total_files}")
