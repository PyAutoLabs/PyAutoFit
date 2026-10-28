from __future__ import annotations
import copy
import gc
import logging
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, List, Dict, Tuple

if TYPE_CHECKING:
    from autofit.database.sqlalchemy_ import sa
    from autofit.non_linear.result import Result

from autofit.graphical import FactorApproximation, MeanField
from autofit.graphical.utils import Status
from autofit.graphical.expectation_propagation import (
    AbstractFactorOptimiser,
    SearchFactorOptimiser,
)
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.initializer import Initializer

# Re-exported for existing importers: `ITERATIONS_NEVER` (e.g. `multi_start_gradient`)
# and the helpers now in `search/runtime.py` (`settings.py`, Dynesty, Nautilus).
from autofit.non_linear.fitness import ITERATIONS_NEVER, get_timeout_seconds  # noqa: F401
from autofit.non_linear.search.runtime import (  # noqa: F401
    SearchRuntimeSettings,
    configure_handler,
    log_fit_start,
    quick_update_message,
    warn_deprecated_jax_knob,
    warn_if_threads_unpinned,
    warn_unknown_kwargs,
)
from autofit.non_linear.parallel import PoolFactory
from autofit.non_linear.search.updater import log_process_state
from autofit.non_linear.paths import paths_from
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.samples.samples import Samples
from autofit.non_linear.samples.summary import SamplesSummary
from autofit.non_linear.timer import Timer
from autofit.non_linear.analysis import Analysis
from autofit.non_linear.search import capabilities as cap
from autofit.non_linear.search.execution import SearchExecutionMixin, _SearchMeta
from autofit.non_linear.search.internal_state import SearchInternalStateMixin
from autofit.non_linear.search.updates import SearchUpdatesMixin
from autofit.non_linear.status import status_notify, status_recorded
from autofit.non_linear.search.test_mode_bypass import TestModeBypass, TestModeMixin
from autofit.non_linear.test_mode import test_mode_level, skip_fit_output

logger = logging.getLogger(__name__)


class NonLinearSearch(
    SearchExecutionMixin,
    SearchInternalStateMixin,
    SearchUpdatesMixin,
    TestModeMixin,
    AbstractFactorOptimiser,
    ABC,
    metaclass=_SearchMeta,
):
    """
    The base class of every non-linear search: its settings, paths and the fit
    lifecycle. The rest of its methods live in bases without ``__init__`` or
    constructor arguments (``search/execution.py``, ``internal_state.py``,
    ``updates.py``, ``test_mode_bypass.py``); its collaborators are
    ``SearchRuntimeSettings``, ``paths_from``, ``TestModeBypass``, ``PoolFactory``
    and, for expectation propagation, ``SearchFactorOptimiser``.
    """

    # Visualization switches toggled per factor search by EP
    # (``SearchFactorOptimiser``), governed by ``general.yaml -> output ->
    # visualize_ep_factor_searches``. Class-level (not set in ``__init__``) so search
    # doubles that skip ``__init__`` (e.g. the regression suite's ``StaticSearch``)
    # still have them. Outside EP both are always True.
    _visualize_fit = True
    _visualize_before_fit = True

    # Keyword arguments the base class expects in ``**kwargs``: ``save_all_samples`` is
    # read by ``__init__``, and ``initial_values`` / ``inplace`` are the
    # ``AbstractFactorOptimiser`` arguments ``search.json`` records, so a search loaded
    # from it passes them back. Any other key reaching the base class is logged as
    # unknown (a warning, never an error).
    _known_kwargs = ("save_all_samples", "initial_values", "inplace")

    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        unique_tag: Optional[str] = None,
        initializer: Initializer = None,
        iterations_per_quick_update: Optional[int] = None,
        iterations_per_full_update: int = None,
        live_visual_update: Optional[bool] = None,
        number_of_cores: int = 1,
        silence: bool = False,
        session: Optional[sa.orm.Session] = None,
        paths: Optional[AbstractPaths] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Abstract base class for non-linear searches.

        This class sets up the file structure for the non-linear search, which are standardized across all non-linear
        searches.

        Parameters
        ----------
        name
            The name of the search, controlling the last folder results are output.
        path_prefix
            The path of folders prefixing the name folder where results are output.
        unique_tag
            The name of a unique tag for this model-fit, which will be given a unique entry in the sqlite database
            and also acts as the folder after the path prefix and before the search name.
        initializer
            Generates the initialize samples of non-linear parameter space (see autofit.non_linear.initializer).
        silence
            If True, the default print output of the non-linear search is silenced.
        session
            An SQLAlchemy session instance so the results of the model-fit are written to an SQLite database.
        seed
            The search-level random seed: stream 0 of its ``SeedSequence`` seeds the initializer, streams 1+ the
            backend; it enters the identifier only when not None (``autofit.non_linear.search.seed``).
        """
        super().__init__()

        self.disable_output = name is None and path_prefix is None

        if name:
            path_prefix = Path(path_prefix or "")

        self.path_prefix = path_prefix
        self.path_prefix_no_unique_tag = path_prefix
        self._logger = None
        self.unique_tag = unique_tag
        self.seed = seed

        self.paths = paths_from(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            session=session,
            paths=paths,
            save_all_samples=kwargs.get("save_all_samples", False),
        )

        if initializer is not None:
            self.initializer = initializer

        # The config-derived settings, applied as the instance attributes of the same
        # names (`iterations_per_quick_update`, `force_pickle_overwrite`, `silence`, ...).
        SearchRuntimeSettings.from_config(
            iterations_per_quick_update=iterations_per_quick_update,
            iterations_per_full_update=iterations_per_full_update,
            live_visual_update=live_visual_update,
            silence=silence,
        ).apply(self)

        self.kwargs = kwargs

        warn_unknown_kwargs(self, kwargs, known=self._known_kwargs)

        self.number_of_cores = number_of_cores
        warn_if_threads_unpinned(number_of_cores)

        self.optimisation_counter = Counter()

    __identifier_fields__ = tuple()

    # Static capability declarations (``autofit.non_linear.search.capabilities``).
    # These are the defaults for a search that declares nothing; every public search
    # overrides them and the registry (``search/registry.py``) mirrors its values.
    # None of them is ever an identifier field.
    jax_use = cap.JaxUse.NONE
    gradient = cap.Gradient.NONE
    batched = False
    honours_gradient_mode = False
    posterior_kind = None
    produces_evidence = False
    resumable = False
    warm_start = cap.WarmStart.NEUTRAL
    install_extra = ""
    upstream_url = ""
    citation_keys = ()
    status = cap.Status.EXPERIMENTAL
    test_mode_budget = {}
    objective_target = None
    invalid_value = -float("inf")
    # True for a search whose backend is wrong in float32: `fit` then raises, rather
    # than warns, when a JAX analysis runs with `jax_enable_x64` off (A3b preflight).
    requires_fp64 = False

    def optimise(
        self,
        factor_approx: FactorApproximation,
        status: Status = Status(),
    ) -> Tuple[MeanField, Status]:
        """
        Optimise one expectation-propagation factor with this search
        (``autofit.graphical.expectation_propagation.SearchFactorOptimiser``).
        """
        return SearchFactorOptimiser(self).optimise(factor_approx, status)

    @property
    def name(self):
        return self.paths.name

    def __getstate__(self):
        """
        Remove the logger, the paths and the per-fit state for pickling.
        """
        state = self.__dict__.copy()
        state.pop("_logger", None)
        state.pop("paths", None)
        state.pop("_fit_pools", None)
        state.pop("_fit_status", None)
        return state

    @property
    def logger(self):
        if getattr(self, "_logger", None) is None:
            self._logger = logging.getLogger(self.name)
        return self._logger

    @property
    def timer(self) -> Optional[Timer]:
        """
        The timer of the search (how long the search took, how much parallelization
        sped it up), or ``None`` in ``NullPaths`` mode (no output written to disk).
        """
        try:
            return Timer(self.paths.search_internal_path)
        except TypeError:
            pass

    @property
    def paths(self) -> Optional[AbstractPaths]:
        return self._paths

    @paths.setter
    def paths(self, paths: Optional[AbstractPaths]):
        if paths is not None:
            paths.search = self
        self._paths = paths

    @property
    def quick_update_message(self) -> str:
        """The one-line quick-update cadence logged at the start of every fit."""
        return quick_update_message(self.iterations_per_quick_update)

    def copy_with_paths(self, paths):
        self.logger.debug(f"Creating a copy of {self._paths.name}")
        search_instance = copy.copy(self)
        search_instance.paths = paths
        search_instance._logger = None

        return search_instance

    def fit(
        self,
        model: AbstractPriorModel,
        analysis: Analysis,
        info: Optional[Dict] = None,
    ) -> Union[Result, List[Result]]:
        """
        Fit a model, M with some function f that takes instances of the
        class represented by model M and gives a score for their fitness: the search
        produces instances of the model by picking points in its N dimensional space.

        Parameters
        ----------
        analysis
            An object that encapsulates the data and a log likelihood function which fits the model to the data
            via the non-linear search.
        model
            The model that is fitted to the data, which is used by the non-linear search to create instances of
            the model that are fitted to the data via the log likelihood function.
        info
            Optional dictionary containing information about the fit that can be saved in the `files` folder
            (e.g. as `files/info.json`) and can be loaded via the database.

        Returns
        -------
        The ``Result``: how well the model fit the data, the best fit instance and an
        updated model whose priors represent the beliefs produced by this fit. Raises
        ``AssertionError`` if the model has 0 dimensions.
        """
        with status_recorded(self):
            return self._fit_and_output(model=model, analysis=analysis, info=info)

    def _fit_and_output(self, model, analysis, info=None):
        """The body of ``fit``, inside its status record (``status_recorded``)."""
        self.check_model(model=model)

        # Per-fit: set by `start_resume_fit` when the fit's pools are resolved, and read
        # by `search.summary`; cleared so a reused search never reports a stale count.
        self._parallel_summary = None

        log_fit_start(self, analysis)
        log_process_state()

        model = analysis.modify_model(model)
        self.paths.model = model
        self.paths.unique_tag = self.unique_tag

        self.paths.restore()

        model.freeze()
        analysis = analysis.modify_before_fit(paths=self.paths, model=model)
        model.unfreeze()

        if not skip_fit_output():
            self.pre_fit_output(analysis=analysis, model=model, info=info)
        elif hasattr(self.paths, "save_all"):
            # Skip mode still writes `files/search.json` (the sentinel the aggregator
            # scans for) and the identifier files, so downstream aggregator scraping
            # discovers the search directory, but not the expensive
            # `analysis.save_attributes` / `visualize_before_fit` of `pre_fit_output`.
            self.paths.save_all(info=info)

        # `start_resume_fit` finalizes the archive itself, before it writes
        # `.completed`; a completed fit's `post_fit_output` still tidies the folder.
        resumed_or_started = not self.paths.is_complete

        if resumed_or_started:
            result = self.start_resume_fit(analysis=analysis, model=model)
        else:
            result = self.result_via_completed_fit(analysis=analysis, model=model)

        if not skip_fit_output():
            analysis = analysis.modify_after_fit(
                paths=self.paths, model=model, result=result
            )

            self.post_fit_output(
                search_internal=result.search_internal,
                finalize_search_internal=not resumed_or_started,
            )

        gc.collect()

        self.logger.info("Search complete, returning result")

        return result

    def pre_fit_output(
        self, analysis: Analysis, model: AbstractPriorModel, info: Optional[Dict] = None
    ):
        """
        Outputs attributes of fit before the non-linear search begins: the model
        composition (`files/model.json`), the search settings (`files/search.json`),
        the `info` dictionary (`files/info.json`), the analysis' `save_attributes`
        (e.g. `files/data.json`) and its `visualize_before_fit` /
        `visualize_before_fit_combined` visuals (quantities that do not change during
        the model-fit, e.g. the data).
        """
        if not self.disable_output:
            self.logger.info(f"The output path of this fit is {self.paths.output_path}")
        else:
            self.logger.info(
                "Output to hard-disk disabled, input a search name to enable."
            )

        if not self.paths.is_complete or self.force_pickle_overwrite:
            if not self.disable_output:
                self.logger.info(
                    f"Outputting pre-fit files (e.g. model.info, visualization)."
                )

            self.paths.save_all(info=info)
            analysis.save_attributes(paths=self.paths)

        if self._visualize_before_fit and analysis.should_visualize(paths=self.paths):
            analysis.visualize_before_fit(paths=self.paths, model=model)
            analysis.visualize_before_fit_combined(paths=self.paths, model=model)

        timeout_seconds = get_timeout_seconds()

        if timeout_seconds is not None:
            logger.info(
                f"\n\n ***Log Likelihood Function timeout is "
                f"turned on and set to {timeout_seconds} seconds.***\n"
            )

    @configure_handler
    def start_resume_fit(self, analysis: Analysis, model: AbstractPriorModel) -> Result:
        """
        Start a non-linear search from scratch, or resume one which was previously
        terminated mid-way through (from the samples it left off at), and return its
        ``Result``.

        When the search completes a `.completed` file is output, so a rerun is not
        repeated and loads its results via `result_via_completed_fit`; results are also
        output to the `files` folder via the analysis' `save_results`. Under
        ``PYAUTO_TEST_MODE>=2`` the backend is bypassed (``TestModeBypass``); both
        paths end in ``finalize_result``.
        """
        if self.paths.has_timer:
            self.timer.start()

        status_notify(self, "start")

        mode = test_mode_level()
        if mode >= 2:
            return TestModeBypass(self).run(
                model=model,
                analysis=analysis,
                call_likelihood=(mode == 2),
            )

        # The runtime gates, after the test-mode bypass return so `PYAUTO_TEST_MODE>=2`
        # (which never runs a backend, e.g. `PYAUTO_DISABLE_JAX=1` smoke runs) is not
        # affected by them: whole-graph backend agreement, the REQUIRED gate, the x64
        # check and the JAX trace preflight (`autofit.non_linear.search.preflight`).
        from autofit.non_linear.search import preflight

        preflight.check_runtime_gates(self, analysis, model)

        # The fit's pools, with the one JAX fork rule applied once (one INFO line when it
        # downgrades) and the effective worker count recorded for `search.summary`.
        self._fit_pools = PoolFactory(self, analysis)
        self._parallel_summary = self._fit_pools.summary

        model.freeze()
        try:
            search_internal, fitness = self._fit(model=model, analysis=analysis)
        finally:
            self._fit_pools = None

        if hasattr(fitness, "shutdown_quick_update"):
            fitness.shutdown_quick_update()

        samples = self.perform_update(
            model=model,
            analysis=analysis,
            search_internal=search_internal,
            fitness=fitness,
            during_analysis=False,
        )

        samples_summary = None

        if mode == 1:
            samples, samples_summary = TestModeBypass(self).repair_reduced_samples(
                samples
            )

        return self.finalize_result(
            samples=samples,
            model=model,
            analysis=analysis,
            samples_summary=samples_summary,
            search_internal=search_internal,
        )

    def finalize_result(
        self,
        samples: Samples,
        model: AbstractPriorModel,
        analysis: Analysis,
        samples_summary: Optional[SamplesSummary] = None,
        search_internal=None,
    ) -> Result:
        """
        The one ending of a fit that ran (or was bypassed in test mode): build the
        ``Result``, run the analysis' ``save_results`` hooks, unfreeze the model and
        complete the fit (archive, then ``.completed``).
        """
        if samples_summary is None:
            samples_summary = samples.summary()

        result = analysis.make_result(
            samples_summary=samples_summary,
            paths=self.paths,
            samples=samples,
            search_internal=search_internal,
        )

        analysis.save_results(paths=self.paths, result=result)
        analysis.save_results_combined(paths=self.paths, result=result)

        model.unfreeze()

        self._complete_fit(search_internal=search_internal)

        return result

    def _complete_fit(self, search_internal):
        """
        Finalize the fit's retained internal state, then mark the fit complete.

        The order is the transaction: ``.completed`` is written only once the archive
        (and the discard of the resume state it supersedes) has succeeded. An archive
        write that fails (a full disk) therefore leaves no marker, the resume state is
        still on disk, and the next run resumes and finalizes again, rather than
        loading a "completed" fit whose archive never landed.
        """
        if not skip_fit_output():
            self.finalize_search_internal(search_internal=search_internal)

        self.paths.completed()

        status_notify(self, "completed", search_internal)

    def result_via_completed_fit(
        self,
        analysis: Analysis,
        model: AbstractPriorModel,
    ) -> Result:
        """
        Returns the result of the non-linear search of a completed model-fit.

        The result contains the non-linear search samples summary, which contains the maximum log likelihood instance
        that is used for visualization and prior passing via the search chaining API. The full samples are loaded
        too when present (they are optional and can be slow to load).

        Optional tasks update the results on hard-disk depending on the `general.yaml` config file's `output`
        section: `force_visualize_overwrite=True` performs the visualization again, and `force_pickle_overwrite=True`
        recreates the output files of the model-fit.

        """
        status_notify(self, "skipped")

        model.freeze()
        samples_summary = self.paths.load_samples_summary()
        try:
            samples = self.paths.samples
        except FileNotFoundError:
            samples = None

        result = analysis.make_result(
            samples_summary=samples_summary,
            samples=samples,
            paths=self.paths,
        )

        self.logger.info(f"Fit Already Completed: skipping non-linear search.")

        if self.force_visualize_overwrite:
            self.perform_visualization(
                model=model,
                analysis=analysis,
                samples_summary=samples_summary,
                during_analysis=False,
            )

        if self.force_pickle_overwrite:
            self.logger.info("Forcing pickle overwrite")

            analysis.save_results(paths=self.paths, result=result)
            analysis.save_results_combined(paths=self.paths, result=result)

        model.unfreeze()

        return result

    def post_fit_output(self, search_internal, finalize_search_internal: bool = True):
        """
        Cleans up the output folder after a completed non-linear search: removes the
        folder of results so only its `.zip` file is left (for supercomputers with a
        file-count limit), when `remove_files=True` in the `general.yaml` `output`
        section. ``finalize_search_internal=False`` when the fit just ran, since
        ``start_resume_fit`` finalized the internal state before writing ``.completed``.
        """
        if finalize_search_internal:
            self.finalize_search_internal(search_internal=search_internal)

        if not self.disable_output:
            self.logger.info("Removing all files except for .zip file")

        self.paths.zip_remove()

    @abstractmethod
    def _fit(self, model: AbstractPriorModel, analysis: Analysis):
        """
        Run the backend and return ``(search_internal, fitness)``.

        A search either overrides this (every search not yet migrated) or implements
        ``run(ctx)``, in which case this is the bridge (``docs/design/run_ctx.md``,
        ``SearchExecutionMixin._run_bridge``).
        """
        return self._run_bridge(model=model, analysis=analysis)

    def check_model(self, model: AbstractPriorModel):
        if model is not None and model.prior_count == 0:
            raise AssertionError("Model has no priors! Cannot fit a 0 dimension model.")

    def __eq__(self, other):
        return isinstance(other, NonLinearSearch) and self.__dict__ == other.__dict__
