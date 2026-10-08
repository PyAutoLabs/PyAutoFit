from __future__ import annotations
import copy
import gc
import logging
import multiprocessing as mp
import numpy as np
import os
import time
import warnings
from abc import ABC, ABCMeta, abstractmethod
from collections import Counter
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, Tuple, List, Dict


if TYPE_CHECKING:
    from autofit.database.sqlalchemy_ import sa
    from autofit.non_linear.result import Result

from autonerves import conf

from autonerves.output import should_output

from autofit import exc
from autofit.graphical import (
    MeanField,
    AnalysisFactor,
    _HierarchicalFactor,
    FactorApproximation,
)
from autofit.graphical.utils import Status
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.mapper.model import ModelInstance
from autofit.non_linear.initializer import Initializer
from autofit.non_linear.fitness import Fitness
# `ITERATIONS_NEVER` is defined in `fitness.py` (so `Fitness.manage_quick_update` can
# use it) and re-exported here for existing importers, e.g. `multi_start_gradient`.
from autofit.non_linear.fitness import ITERATIONS_NEVER  # noqa: F401
from autofit.non_linear.parallel import (
    PoolFactory,
    SneakyPool,
    check_factor_search_cores,
    fork_context,
)
from autofit.non_linear.search.updater import log_process_state
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.paths.database import DatabasePaths
from autofit.non_linear.paths.directory import DirectoryPaths
from autofit.non_linear.paths.sub_directory_paths import SubDirectoryPaths
from autofit.non_linear.samples.samples import Samples
from autofit.non_linear.samples.summary import SamplesSummary
from autofit.non_linear.checkpoint import Checkpointer, DillCheckpointer
from autofit.non_linear.timer import Timer
from autofit.non_linear.analysis import Analysis
from autofit.non_linear.paths.null import NullPaths
from autofit.graphical.declarative.abstract import PriorFactor
from autofit.graphical.expectation_propagation import AbstractFactorOptimiser

from autofit.non_linear.fitness import get_timeout_seconds
from autofit.non_linear.search import capabilities as cap
from autofit.non_linear.test_mode import (
    test_mode_level,
    test_mode_samples,
    skip_fit_output,
)

logger = logging.getLogger(__name__)


# A reduced test-mode fit needs a valid representative instance for result
# construction and search chaining.  Keep fallback sampling bounded so an
# impossible model fails clearly instead of hanging a smoke-test worker.
TEST_MODE_REPRESENTATIVE_MAX_ATTEMPTS = 100


def warn_deprecated_jax_knob(owner: str, name: str, instead: str):
    """
    Warn, as a ``FutureWarning`` so end users see it, that the deprecated JAX knob
    ``name`` was passed to ``owner``.

    ``use_jax_jit`` and ``use_jax_vmap`` stay accepted for one release (search-
    extensibility phase A2): every search now selects its objective through
    ``Fitness.objective(kind)``, jitting a JAX analysis lazily and batching it where
    the backend has a batched fast path.
    """
    warnings.warn(
        f"{owner}({name}=...) is deprecated and will be removed after one release: "
        f"{instead}",
        FutureWarning,
        stacklevel=3,
    )


def check_cores(func):
    """
    Checks how many cores the search has been configured to
    use and then returns None instead of calling the pool
    creation function in the case that only one core has
    been set.

    Parameters
    ----------
    func
        A function that creates a pool

    Returns
    -------
    None or a pool
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.number_of_cores == 1:
            return None
        return func(self, *args, **kwargs)

    return wrapper


def configure_handler(func):
    """
    Add a file handler for logging during the course of the search.

    Optionally outputs 'search.log' to the search's output directory. Can be
    turned on or off in the output.yaml file.

    Parameters
    ----------
    func
        Some function for which logging should be output to file

    Returns
    -------
    A decorated version of the function
    """
    root_logger = logging.getLogger()

    def decorated(self, *args, **kwargs):
        if not should_output("search_log"):
            return func(self, *args, **kwargs)
        if self.disable_output:
            return func(self, *args, **kwargs)
        try:
            os.makedirs(
                self.paths.output_path,
                exist_ok=True,
            )
            handler = logging.FileHandler(self.paths.output_path / "search.log")
            root_logger.addHandler(handler)
        except AttributeError:
            return func(self, *args, **kwargs)

        try:
            return func(self, *args, **kwargs)
        finally:
            root_logger.removeHandler(handler)
            handler.close()

    return decorated


class _SearchMeta(ABCMeta):
    """
    Keeps ``NonLinearSearch`` abstract until a subclass implements its backend through
    either hook: overriding ``_fit`` (the legacy contract) or ``run(ctx)`` (the bridge,
    ``docs/design/run_ctx.md``). ``_fit`` stays an abstract method; a class that
    defines ``run`` is concrete because the inherited ``_fit`` is the bridge to it.
    """

    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)

        run = getattr(cls, "run", None)

        if (
            "_fit" in cls.__abstractmethods__
            and run is not None
            and not getattr(run, "_is_bridge_default", False)
        ):
            cls.__abstractmethods__ = frozenset(cls.__abstractmethods__ - {"_fit"})

        return cls


class NonLinearSearch(AbstractFactorOptimiser, ABC, metaclass=_SearchMeta):
    # Visualization switches toggled per factor search by ``optimise`` (EP),
    # governed by ``general.yaml -> output -> visualize_ep_factor_searches``.
    # Class-level (not set in ``__init__``) so search doubles that skip
    # ``__init__`` (e.g. the regression suite's ``StaticSearch``) still have
    # them. Outside EP both are always True.
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
        """
        super().__init__()

        if name is None and path_prefix is None:
            self.disable_output = True
        else:
            self.disable_output = False

        from autofit.non_linear.paths.database import DatabasePaths

        if name:
            path_prefix = Path(path_prefix or "")

        self.path_prefix = path_prefix

        self.path_prefix_no_unique_tag = path_prefix

        self._logger = None

        self.unique_tag = unique_tag

        if paths:
            self.paths = paths
        elif session is not None:
            logger.debug("Session found. Using database.")
            self.paths = DatabasePaths(
                name=name,
                path_prefix=path_prefix,
                session=session,
                save_all_samples=kwargs.get("save_all_samples", False),
                unique_tag=unique_tag,
            )
        elif name is not None or path_prefix:
            logger.debug("Session not found. Using directory output.")
            self.paths = DirectoryPaths(
                name=name, path_prefix=path_prefix, unique_tag=unique_tag
            )
        else:
            self.paths = NullPaths()

        self.force_pickle_overwrite = conf.instance["general"]["output"][
            "force_pickle_overwrite"
        ]

        self.force_visualize_overwrite = conf.instance["general"]["output"][
            "force_visualize_overwrite"
        ]

        if initializer is not None:
            self.initializer = initializer

        self.iterations_per_quick_update = float((iterations_per_quick_update or
            conf.instance["general"]["updates"]["iterations_per_quick_update"]))

        self.iterations_per_full_update = float((iterations_per_full_update or
            conf.instance["general"]["updates"]["iterations_per_full_update"]))

        self.quick_update_background = bool(
            conf.instance["general"]["updates"].get(
                "quick_update_background", False,
            )
        )

        self.live_visual_update = bool(
            live_visual_update
            if live_visual_update is not None
            else conf.instance["general"]["updates"].get(
                "live_visual_update", False,
            )
        )

        if conf.instance["general"]["hpc"]["hpc_mode"]:
            self.iterations_per_quick_update = float(conf.instance["general"]["hpc"][
                "iterations_per_quick_update"
            ])
            self.iterations_per_full_update = float(conf.instance["general"]["hpc"][
                "iterations_per_full_update"
            ])

        self.silence = silence

        if conf.instance["general"]["hpc"]["hpc_mode"]:
            self.silence = True

        self.kwargs = kwargs

        for key in kwargs:
            if key not in self._known_kwargs:
                logger.warning(
                    f"{type(self).__name__} received the unknown keyword argument "
                    f"{key!r}, which is ignored. Check its spelling against the "
                    f"search's constructor."
                )

        self.number_of_cores = number_of_cores

        if number_of_cores > 1 and any(
            os.environ.get(key) != "1"
            for key in (
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        ):
            if conf.instance["general"]["parallel"]["warn_environment_variables"]:
                warnings.warn(exc.SearchWarning(""))
                logger.warning(
                    """
                        The non-linear search is using multiprocessing (number_of_cores>1). 
    
                        However, the following environment variables have not been set to 1:
    
                        OPENBLAS_NUM_THREADS
                        MKL_NUM_THREADS
                        OMP_NUM_THREADS
                        VECLIB_MAXIMUM_THREADS
                        NUMEXPR_NUM_THREADS
    
                        This can lead to performance issues, because both the non-linear search and libraries that may be
                        used in your `log_likelihood_function` evaluation (e.g. NumPy, SciPy, scikit-learn) may attempt to
                        parallelize over all cores available.
    
                        This will lead to slow-down, due to overallocation of tasks over the CPUs.
    
                        To mitigate this, set the environment variables to 1 via the following command on your
                        bash terminal / command line:
    
                        export OPENBLAS_NUM_THREADS=1
                        export MKL_NUM_THREADS=1
                        export OMP_NUM_THREADS=1
                        export VECLIB_MAXIMUM_THREADS=1
                        export NUMEXPR_NUM_THREADS=1
    
                        This means only the non-linear search is parallelized over multiple cores.
    
                        If you "know what you are doing" and do not want these environment variables to be set to one, you 
                        can disable this warning by changing the following entry in the config files:
    
                        `config -> general.yaml -> parallel: -> warn_environment_variables=False`
                        """
                )

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
        Perform optimisation for expectation propagation. Currently only
        applicable for ModelFactors created by the declarative interface.

        1. Analysis and model classes are extracted from the factor.
        2. Priors are updated from the mean field.
        3. Analysis and model are fit as usual.
        4. A new mean field is constructed with the (posterior) 'linking' priors.
        5. Projection is performed to produce an updated EPMeanField object.

        Output directories are generated according to the factor and the number
        of the search. For example a factor called "factor" would output:

        factor/optimization_0/<identifier>
        factor/optimization_1/<identifier>
        factor/optimization_2/<identifier>

        For the first, second and third optimizations respectively.

        Parameters
        ----------
        factor_approx
            A collection of messages defining the current best approximation to
            some global model
        status

        Returns
        -------
        An updated approximation to the model having performed optimisation on
        a single factor.
        """

        factor = factor_approx.factor

        _ = status
        if not isinstance(factor, (AnalysisFactor, PriorFactor, _HierarchicalFactor)):
            raise NotImplementedError(
                f"Optimizer {self.__class__.__name__} can only be applied to"
                f" AnalysisFactors, HierarchicalFactors and PriorFactors"
            )

        # A deliberate refusal, not a silent downgrade: EP never forks a factor
        # search (`autofit.non_linear.parallel.pool.check_factor_search_cores`).
        check_factor_search_cores(self)

        model = factor.prior_model.mapper_from_prior_arguments(
            {
                prior: prior.with_message(message)
                for prior, message in factor_approx.cavity_dist.arguments.items()
            }
        )

        analysis = factor.analysis

        uses_jax = analysis.is_jax

        self.logger.info(
            f"EP factor step [{factor.name}]: running the factor search "
            f"{self.__class__.__name__} serially (number_of_cores=1); "
            + (
                "parallelism comes from the vectorised JAX likelihood "
                "(analysis.use_jax=True)."
                if uses_jax
                else "the likelihood is not JAX-vectorised, so this factor step "
                "is single-threaded (set `Analysis(use_jax=True)` for "
                "parallelism)."
            )
        )

        number = self.optimisation_counter[factor.name]

        self.optimisation_counter[factor.name] += 1

        self.paths = SubDirectoryPaths(
            parent=self.paths,
            analysis_name=f"{factor.name}/optimization_{number}",
            is_flat=True,
        )

        # EP re-enters the same factor search once per EP step. By default only
        # the first search of each factor draws its before-fit visuals and no
        # factor search draws per-search visuals (corner plots, model-fit
        # images): at small per-factor fits the plotting dominates the wall
        # time and the images are overwritten every step anyway. Samples and
        # summaries are still written, and the EP optimiser's own visuals
        # (graph.png, ep_history) are unaffected. `.get` with a default so user
        # configs that predate the key keep working.
        visualize_ep_factor_searches = bool(
            conf.instance["general"]["output"].get(
                "visualize_ep_factor_searches", False
            )
        )

        self._visualize_fit = visualize_ep_factor_searches
        self._visualize_before_fit = visualize_ep_factor_searches or number == 0

        try:
            result = self.fit(model=model, analysis=analysis)
        finally:
            self._visualize_fit = True
            self._visualize_before_fit = True

        # Record the sampler's log-evidence of this tilted-distribution fit on
        # the projected mean field — the per-factor Ẑₐ that README §5 documents
        # `MeanField.log_norm` as carrying (#1332 F7(b)). Previously always 0,
        # so `EPMeanField.log_evidence` could not be trusted for model
        # comparison in sampler-driven EP fits. Searches with no evidence
        # estimate (MCMC / MLE) yield None and keep the 0.0 default — evidence-
        # correct model comparison requires nested-sampling factor searches.
        # (Both levels guarded: e.g. StaticResult carries no samples at all.)
        log_evidence = getattr(
            getattr(result, "samples", None), "log_evidence", None
        )

        new_model_dist = MeanField.from_priors(
            result.projected_model.priors,
            log_norm=log_evidence if log_evidence is not None else 0.0,
        )

        # Release the in-memory search internal before the result is pinned on
        # the `Status`.
        #
        # `EPHistory`/`FactorHistory` append every `(approx, status)` pair of
        # the run, so every `Status`, and therefore every `Result`, lives until
        # the EP fit ends. `Result._search_internal` is the sampler, the
        # sampler's likelihood callable is a `Fitness`, and a `Fitness` owns
        # that factor step's compiled JAX executables (its `_vmap` / `_jit` /
        # `_grad` `cached_property` caches). Left attached, an EP run therefore
        # accumulates the compiled code of every factor search it has ever run:
        # RAL job 342410 (`slope_hierarchy_scale`, 25 analysis factors) held 76
        # searches' executables and died with `LLVM ERROR: Unable to allocate
        # section memory!` inside 64 GB, after ~3 EP steps.
        #
        # Nothing in `autofit.graphical` reads the sampler afterwards (the
        # consumers of `latest_result` use `.projected_model`, `.samples` and
        # `.model`), and `Result.search_internal` still falls back to
        # `paths.load_search_internal()`, so the sampler remains reachable from
        # the on-disk dill for anything that does want it.
        #
        # Guarded because `optimise` does not require a `Result`: the regression
        # suite's `StaticSearch` returns a bare object carrying only
        # `projected_model` (as the `getattr` on `samples` above allows for).
        release_search_internal = getattr(result, "release_search_internal", None)

        if release_search_internal is not None:
            release_search_internal()

        status.result = result

        return new_model_dist, status

    @property
    def name(self):
        return self.paths.name

    def __getstate__(self):
        """
        Remove the logger for pickling
        """
        state = self.__dict__.copy()
        if "_logger" in state:
            del state["_logger"]
        if "paths" in state:
            del state["paths"]
        # Per-fit state, never serialized with the search.
        state.pop("_fit_pools", None)
        return state

    @property
    def logger(self):
        if not hasattr(self, "_logger"):
            self._logger = None
        if self._logger is None:
            logger_ = logging.getLogger(self.name)
            self._logger = logger_
        return self._logger

    @property
    def timer(self) -> Optional[Timer]:
        """
        Returns the timer of the search, which is used to output informaiton such as how long the search took and
        how much parallelization sped up the search time.

        If the search is running in `NullPaths` mode, meaning that no output is written to the hard-disk, the timer
        is disabled and a `None` is returned.

        Returns
        -------
        An object which times the non-linear search.
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
        """
        One line, logged at the start of every search, telling the user the real
        cadence of the on-the-fly maximum-likelihood updates.

        The cadence is worth stating because it is the only thing that explains
        the terminal's behaviour during a long fit: either updates appear every
        N iterations, or nothing appears at all until the search finishes. The
        packaged default is the ``ITERATIONS_NEVER`` sentinel, so "nothing at
        all" is what most users get -- and a message that reported that as
        ``1e+99 iterations`` would be technically true and practically useless.
        Hence the two-branch wording, and hence naming the config key: the
        disabled branch is the one a user is most likely to want to act on.
        """
        iterations = self.iterations_per_quick_update

        if not np.isfinite(iterations) or iterations >= ITERATIONS_NEVER:
            return (
                "On-the-fly updates of the maximum likelihood model are disabled. "
                "Set `updates: iterations_per_quick_update` in config/general.yaml "
                "to a finite number of iterations to enable them."
            )

        return (
            "On-the-fly updates of the maximum likelihood model every "
            f"{int(iterations)} iterations."
        )

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
        class represented by model M and gives a score for their fitness.

        A model which represents possible instances with some dimensionality is fit.

        The analysis provides two functions. One visualises an instance of a model and the
        other scores an instance based on how well it fits some data. The search
        produces instances of the model by picking points in an N dimensional space.

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
        An object encapsulating how well the model fit the data, the best fit instance
        and an updated model with free parameters updated to represent beliefs
        produced by this fit.

        Raises
        ------
        AssertionError
            If the model has 0 dimensions.
        """
        self.check_model(model=model)

        # Per-fit: set by `start_resume_fit` when the fit's pools are resolved, and read
        # by `search.summary`; cleared so a reused search never reports a stale count.
        self._parallel_summary = None

        if analysis.is_jax:
            try:
                import jax
                devices = jax.devices()
                device = devices[0]
                backend = device.platform.upper()
                device_name = getattr(device, "device_kind", backend)
                logger.info(
                    f"Starting non-linear search with JAX ({backend}: {device_name})."
                )
            except Exception:
                logger.info("Starting non-linear search with JAX.")
        else:
            logger.info(f"Starting non-linear search with {self.number_of_cores} cores.")
        logger.info(self.quick_update_message)
        log_process_state()

        model = analysis.modify_model(model)
        self.paths.model = model
        self.paths.unique_tag = self.unique_tag

        self.paths.restore()

        model.freeze()
        analysis = analysis.modify_before_fit(paths=self.paths, model=model)
        model.unfreeze()

        if not skip_fit_output():
            self.pre_fit_output(
                analysis=analysis,
                model=model,
                info=info,
            )
        else:
            # Skip mode still needs `save_all` to run so that
            # `files/search.json` — the sentinel the aggregator scans for — and
            # the identifier files exist, letting downstream aggregator
            # scraping discover the search directory. `save_all` is lightweight
            # (a handful of JSON dumps) and skips the expensive
            # `analysis.save_attributes` / `visualize_before_fit` calls that
            # `pre_fit_output` would add.
            if hasattr(self.paths, "save_all"):
                self.paths.save_all(
                    info=info,
                )

        if not self.paths.is_complete:
            result = self.start_resume_fit(
                analysis=analysis,
                model=model,
            )
        else:
            result = self.result_via_completed_fit(
                analysis=analysis,
                model=model,
            )

        if not skip_fit_output():
            analysis = analysis.modify_after_fit(
                paths=self.paths, model=model, result=result
            )

            self.post_fit_output(
                search_internal=result.search_internal,
            )

        gc.collect()

        self.logger.info("Search complete, returning result")

        return result

    def pre_fit_output(
        self, analysis: Analysis, model: AbstractPriorModel, info: Optional[Dict] = None
    ):
        """
        Outputs attributes of fit before the non-linear search begins.

        The following attributes of a fit may be output before the search begins:

        - The model composition, which is output as a .json file (`files/model.json`).

        - The non-linear search settings, which are output as a .json file (`files/search.json`).

        - Custom attributes of the analysis defined via the `save_attributes` method of the analysis class, for
        example the data (e.g. `files/data.json`).

        - Custom Visualization associated with the analysis, defined via the `visualize_before_fit`
        and `visualize_before_fit_combined` methods. This is typically quantities that do not change during the
        model-fit (e.g. the data).

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

            self.paths.save_all(
                info=info,
            )
            analysis.save_attributes(paths=self.paths)

        if self._visualize_before_fit and analysis.should_visualize(
            paths=self.paths
        ):
            analysis.visualize_before_fit(
                paths=self.paths,
                model=model,
            )
            analysis.visualize_before_fit_combined(
                paths=self.paths,
                model=model,
            )

        timeout_seconds = get_timeout_seconds()

        if timeout_seconds is not None:
            logger.info(
                f"\n\n ***Log Likelihood Function timeout is "
                f"turned on and set to {timeout_seconds} seconds.***\n"
            )

    @configure_handler
    def start_resume_fit(self, analysis: Analysis, model: AbstractPriorModel) -> Result:
        """
        Start a non-linear search from scratch, or resumes one which was previously terminated mid-way through.

        If the search is resumed, the model-fit will begin by loading the samples from the previous search and
        from where it left off.

        After the search is completed, a `.completed` file is output so that if the search is resumed in the future
        it is not repeated and results are loaded via the `update_completed_fit` method.

        Results are also output to hard-disk in the `files` folder via the `save_results` method of the analysis class.

        Parameters
        ----------
        analysis
            An object that encapsulates the data and a log likelihood function which fits the model to the data
            via the non-linear search.
        model
            The model that is fitted to the data, which is used by the non-linear search to create instances of
            the model that are fitted to the data via the log likelihood function.

        Returns
        -------
        The result of the non-linear search, which includes the best-fit model instance and best-fit log likelihood
        and errors on the model parameters.
        """
        if not isinstance(self.paths, DatabasePaths) and not isinstance(
            self.paths, NullPaths
        ):
            self.timer.start()

        mode = test_mode_level()
        if mode >= 2:
            return self._fit_bypass_test_mode(
                model=model,
                analysis=analysis,
                call_likelihood=(mode == 2),
            )

        # Runtime gates, after the test-mode bypass return so `PYAUTO_TEST_MODE>=2`
        # (which never runs a backend) is not affected by them.
        from autofit.graphical.declarative.collection import FactorGraphModel
        from autofit.graphical.declarative.factor.analysis import AnalysisFactor
        from autofit.non_linear.analysis.model_analysis import ModelAnalysis

        # Unwrap the analysis wrappers that forward their wrapped analysis's
        # likelihood, so a graph wrapped in `ModelAnalysis` (or an `AnalysisFactor`)
        # is still checked as one whole graph.
        inner = analysis
        while isinstance(inner, (ModelAnalysis, AnalysisFactor)) and not isinstance(
            inner, FactorGraphModel
        ):
            inner = inner.__dict__.get("analysis")
        if isinstance(inner, FactorGraphModel):
            inner.check_backend_agreement()

        # Fail fast: a jax_use='required' search given a numpy analysis raises the one
        # shared SearchException before any backend state exists. `PYAUTO_DISABLE_JAX=1`
        # smoke runs use `PYAUTO_TEST_MODE>=2`, which returned above.
        cap.check_jax_required(self, analysis)

        # The JAX checks (search-extensibility A3b, `autofit.non_linear.search.preflight`),
        # after the test-mode bypass and the REQUIRED gate and before any backend state
        # exists: warn once when x64 is off (raise for `requires_fp64`), then trace the
        # declared objective kind(s) once with `jax.eval_shape` so a non-traceable
        # likelihood fails here, naming the search, rather than inside the backend.
        from autofit.non_linear.search import preflight

        preflight.check_x64(self, analysis)
        preflight.trace_preflight(self, analysis, model)

        # The fit's pools, with the one JAX fork rule applied once (one INFO line when it
        # downgrades) and the effective worker count recorded for `search.summary`.
        self._fit_pools = PoolFactory(self, analysis)
        self._parallel_summary = self._fit_pools.summary

        model.freeze()
        try:
            search_internal, fitness = self._fit(
                model=model,
                analysis=analysis,
            )
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

        samples_summary = samples.summary()

        if mode == 1:
            try:
                samples_summary.max_log_likelihood()
            except (exc.FitException, exc.SamplesException) as error:
                samples = self._test_mode_samples_after_rejected_fit(
                    samples=samples,
                    error=error,
                )
                samples_summary = samples.summary()
                self.paths.save_samples_summary(samples_summary=samples_summary)
                self.paths.save_samples(samples=samples)

        result = analysis.make_result(
            samples_summary=samples_summary,
            paths=self.paths,
            samples=samples,
            search_internal=search_internal,
        )

        analysis.save_results(paths=self.paths, result=result)
        analysis.save_results_combined(paths=self.paths, result=result)

        model.unfreeze()

        self.paths.completed()

        return result

    def _test_mode_valid_parameter_vector(
        self,
        model: AbstractPriorModel,
        failure_prefix: str,
        validate: Optional[Callable[[List[float]], Any]] = None,
    ) -> Tuple[List[float], Any]:
        """The first deterministically-drawn parameter vector the model accepts.

        Test mode has no sampler, so it evaluates the model at a point it picks
        itself — the prior medians.  A model whose components share priors and
        carry an ordering assertion (``trap_0.release_timescale <
        trap_1.release_timescale``, the standard idiom for breaking exchange
        degeneracy) ties *exactly* there, so ``check_assertions`` rejects the
        medians with :class:`FitException`.  A production search absorbs that by
        resampling, and test mode must do the same rather than hard-fail on an
        artifact of its own choice of point.

        Try the prior medians first, then a deterministic sequence of prior
        draws, returning the first candidate ``validate`` accepts.  The fixed
        seed keeps smoke runs reproducible without touching the application's
        global random state.

        Parameters
        ----------
        model
            The model the vector must be valid for.
        failure_prefix
            Opening of the :class:`FitException` message raised once the attempt
            budget is spent; the attempt count is appended to it.
        validate
            Called with each candidate vector.  Raising `FitException` rejects
            that candidate and moves on to the next draw; the return value is
            handed back alongside the accepted vector, so a caller that must
            build something in order to validate it does not build it twice.
            Defaults to instantiating the vector.

        Returns
        -------
        The accepted parameter vector, and whatever ``validate`` returned for it.

        Raises
        ------
        FitException
            If no candidate is accepted within
            ``TEST_MODE_REPRESENTATIVE_MAX_ATTEMPTS`` attempts, chained to the
            final rejection.
        """
        if validate is None:

            def validate(candidate: List[float]):
                return model.instance_from_vector(vector=candidate)

        rng = np.random.default_rng(seed=0)
        last_error = None

        for attempt in range(TEST_MODE_REPRESENTATIVE_MAX_ATTEMPTS):
            unit_vector = (
                [0.5] * model.prior_count
                if attempt == 0
                else rng.random(model.prior_count).tolist()
            )

            try:
                parameter_vector = [
                    float(value)
                    for value in model.vector_from_unit_vector(
                        unit_vector=unit_vector,
                    )
                ]
                validated = validate(parameter_vector)
            except exc.FitException as candidate_error:
                last_error = candidate_error
                continue

            if attempt > 0:
                logger.warning(
                    "TEST MODE: the prior medians are not a valid model "
                    f"instance ({last_error.__cause__ or last_error!r}); using "
                    f"deterministic prior draw {attempt} instead. A model whose "
                    "components share priors and carry an ordering assertion "
                    "ties at its medians, which a real search resamples past."
                )

            return parameter_vector, validated

        raise exc.FitException(
            f"{failure_prefix} after "
            f"{TEST_MODE_REPRESENTATIVE_MAX_ATTEMPTS} attempts."
        ) from last_error

    def _test_mode_samples_after_rejected_fit(
        self,
        samples: Samples,
        error: Exception,
    ) -> Samples:
        """Build valid representative samples after a mode-1 rejected result.

        ``Fitness`` maps :class:`FitException` to the sampler's rejection
        sentinel.  A production search naturally moves on to another point,
        but ``PYAUTO_TEST_MODE=1`` may stop after that first evaluation.  Its
        posterior can therefore contain only the rejected point, which cannot
        be reconstructed while finalizing the result.

        Try the prior medians first, then a deterministic sequence of prior
        draws.  Every synthetic sample is validated before it is returned so
        result construction and downstream chaining cannot reconstruct another
        rejected point.  The fixed seed keeps smoke tests reproducible without
        changing the application's global random state.
        """
        logger.warning(
            "TEST MODE 1: the reduced search's final sample raised "
            f"FitException ({error.__cause__ or error!r}); replacing it with "
            "a valid representative sample for result construction."
        )

        model = samples.model

        def validate(parameter_vector: List[float]):
            """Every synthetic sample, not just the first, must reconstruct."""
            sample_list = self._build_fake_samples(
                model=model,
                parameter_vector=parameter_vector,
                log_likelihood=-1.0e99,
            )

            for sample in sample_list:
                model.instance_from_vector(
                    vector=sample.parameter_lists_for_model(model)
                )

            return sample_list

        _, sample_list = self._test_mode_valid_parameter_vector(
            model=model,
            failure_prefix=(
                "TEST MODE 1 could not construct a valid representative result"
            ),
            validate=validate,
        )

        samples_info = {
            **(samples.samples_info or {}),
            "total_iterations": 1,
            "time": 0.0,
            "log_evidence": -1.0e99,
        }
        samples_info.update(self._test_mode_samples_info())

        return samples.from_list_info_and_model(
            model=model,
            sample_list=sample_list,
            samples_info=samples_info,
        )

    def result_via_completed_fit(
        self,
        analysis: Analysis,
        model: AbstractPriorModel,
    ) -> Result:
        """
        Returns the result of the non-linear search of a completed model-fit.

        The result contains the non-linear search samples summary, which contains the maximum log likelihood instance
        that is used for visualization and prior passing via the search chaining API.

        This funciton may also load the full samples of the completed fit, for example if visualization of the
        seatch chains (e.g. a corner plot) is performed. This task is optional and be slow due to loading times.

        Optional tasks can be performed to update the results of the model-fit on hard-disk depending on the following
        entries of the `general.yaml` config file's `output` section:

        ` `force_visualize_overwrite=True`: the visualization of the model-fit is performed again (e.g. to
        add new visualizations or replot figures with a different source code).

        - `force_pickle_overwrite=True`: the output files of the model-fit are recreated (e.g. to add a new attribute
        that was previously not output).

        Parameters
        ----------
        analysis
            An object that encapsulates the data and a log likelihood function which fits the model to the data
            via the non-linear search.
        model
            The model that is fitted to the data, which is used by the non-linear search to create instances of
            the model that are fitted to the data via the log likelihood function.

        Returns
        -------
        The result of the non-linear search, which includes the best-fit model instance and best-fit log likelihood
        and errors on the model parameters.
        """

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

    def post_fit_output(self, search_internal):
        """
        Cleans up the output folderds after a completed non-linear search.

        The main task this performs is removing the folder containing the results of a non-linear search such that only
        its corresponding `.zip` file is left. This is use for supercomputers, where users often have a file limit on
        the number of files they can store in their home directory, so storing them all in just a .zip file is
        advantageous.

        This only occurs if `remove_files=False` in the `general.yaml` config file's `output` section.

        Parameters
        ----------
        search_internal
            The internal search.
        """
        if not self.retains_search_internal:
            self.logger.info("Removing search internal folder.")
            self.paths.remove_search_internal()
        elif search_internal is not None:
            self.output_search_internal(search_internal=search_internal)

        if not self.disable_output:
            self.logger.info("Removing all files except for .zip file")

        self.paths.zip_remove()

    def _fit_bypass_test_mode(
        self,
        model: AbstractPriorModel,
        analysis: Analysis,
        call_likelihood: bool = True,
    ):
        """
        Bypass the sampler entirely in test mode (levels 2 and 3).

        Generates fake samples and writes all expected output files so that
        downstream code sees a complete result folder.

        Parameters
        ----------
        model
            The model being fitted.
        analysis
            The analysis object with the log likelihood function.
        call_likelihood
            If True (mode 2), call the likelihood function once to verify it
            works. If False (mode 3), skip the likelihood call entirely.
        """
        from autofit.non_linear.samples.pdf import SamplesPDF
        from autofit.non_linear.samples.sample import Sample

        mode = test_mode_level()
        if mode == 2:
            logger.warning(
                "TEST MODE 2 (bypass + likelihood): Skipping sampler, "
                "calling likelihood function once to verify it works."
            )
        else:
            logger.warning(
                "TEST MODE 3 (full bypass): Skipping sampler and likelihood "
                "entirely for maximum speed. No likelihood verification."
            )

        model.freeze()

        # The bypass has no sampler, so it chooses its own evaluation point and
        # must choose one the model accepts. A model whose components share
        # priors and carry an ordering assertion ties exactly at the prior
        # medians, and `check_assertions` rejects that tie with `FitException`.
        # The vector chosen here is also the one written into the fake samples
        # below, so picking a valid one is what keeps
        # `result.max_log_likelihood_instance` reconstructible — which is why
        # mode 3 needs this too, not just mode 2's likelihood call
        # (PyAutoFit #1519).
        parameter_vector, instance = self._test_mode_valid_parameter_vector(
            model=model,
            failure_prefix=(
                "TEST MODE could not find a parameter vector satisfying the "
                "model's assertions"
            ),
        )

        log_likelihood = -1.0e99
        if call_likelihood:
            try:
                log_likelihood = float(
                    analysis.log_likelihood_function(instance)
                )
            except exc.FitException as e:
                # A `FitException` means this particular instance is pathological
                # (e.g. a non-positive-definite inversion, or a degenerate mesh
                # that yields NaN vertices). In a real search the sampler absorbs
                # this by resampling; test mode has no sampler, so a single
                # unlucky verification eval must not hard-fail the run. Keep the
                # `-1.0e99` sentinel — the same effect a resample-to-reject has —
                # and log the cause so a genuinely broken likelihood stays visible.
                # Only `FitException` is caught: real code errors still propagate.
                logger.warning(
                    "TEST MODE 2: likelihood verification raised FitException "
                    f"({e.__cause__ or e!r}); treating as a resample-rejected "
                    "instance and continuing with the sentinel log likelihood."
                )

        sample_list = self._build_fake_samples(
            model=model,
            parameter_vector=parameter_vector,
            log_likelihood=log_likelihood,
        )

        # Stub log_evidence in samples_info so downstream arithmetic
        # (grid search log_evidences, subhalo Bayesian model comparison,
        # scrape aggregator assertions) doesn't crash on None. SamplesPDF
        # reads log_evidence from samples_info.
        samples_info = {
            "total_iterations": 1,
            "time": 0.0,
            "log_evidence": log_likelihood,
        }
        samples_info.update(self._test_mode_samples_info())
        samples = SamplesPDF(
            model=model,
            sample_list=sample_list,
            samples_info=samples_info,
        )

        samples_summary = samples.summary()

        # Persist samples + summary to disk so downstream code that reads
        # from the output folder (database scrape, paths.load_samples_summary)
        # sees a complete result. Matches the docstring's promise. NullPaths
        # and DatabasePaths both handle these calls safely.
        self.paths.save_samples_summary(samples_summary=samples_summary)
        self.paths.save_samples(samples=samples)

        result = analysis.make_result(
            samples_summary=samples_summary,
            paths=self.paths,
            samples=samples,
            search_internal=None,
        )

        # Run the analysis' own output hooks, exactly as start_resume_fit does —
        # a script that writes a file in save_results and reads it back after the
        # fit (e.g. autofit_workspace overview_2's science_summary.json) sees a
        # complete result folder only if the bypass calls them too.
        analysis.save_results(paths=self.paths, result=result)
        analysis.save_results_combined(paths=self.paths, result=result)

        model.unfreeze()

        # Mark the fit complete, exactly as start_resume_fit does — a bypassed
        # fit must be resumable (paths.is_complete -> result_via_completed_fit
        # on the next run), or every rerun re-bypasses the whole pipeline.
        self.paths.completed()

        return result

    def _test_mode_samples_info(self) -> dict:
        """
        Sampler-specific keys to merge into ``samples_info`` when the
        sampler is bypassed via ``PYAUTO_TEST_MODE=2`` or ``=3``.

        **Opt-in, not a per-sampler obligation.** Most searches do not
        override this, and that is correct — no library code reads these
        diagnostic keys under bypass. The properties that read them
        (``SamplesMCMC.total_steps``, ``SamplesNest.total_samples``, …)
        live on ``Samples`` subclasses the bypass never constructs;
        ``_fit_bypass_test_mode`` always builds a ``SamplesPDF``. The only
        consumers are workspace scripts reading ``samples_info[...]``
        directly, so override this only when such a script exists.

        Which fix applies depends on what that script does with the keys:

        - It **prints** them (a tutorial, whose point is the prose, so it
          may legitimately run bypassed) → override here, returning NaN/0
          placeholders. The bypass did not sample; honest empties.
        - It **asserts** on them (a test) → the script must not run
          bypassed at all. Give it an ``ENV: real_search`` declaration
          instead. Do **not** add placeholders for an asserting reader:
          the assert then silently passes on a stub value, which is worse
          than the ``KeyError`` it replaced.

        Both live cases follow that split. ``BlackJAXNUTS`` overrides this
        because ``autofit_workspace/scripts/searches/mcmc.py`` prints
        ``ess_min`` / ``n_divergent``, carries no ``__Env__`` declaration
        and is in that workspace's ``smoke_tests.txt``, so it runs
        bypassed on every PR (PyAutoFit #1260).
        ``AbstractMultiStartGradient`` deliberately does not: its only
        reader asserts on ``total_steps``, so that script declares
        ``ENV: real_search jax`` instead (autofit_workspace_test #83).
        """
        return {}

    @staticmethod
    def _build_fake_samples(model, parameter_vector, log_likelihood):
        """
        Build a list of fake Sample objects for test mode bypass.

        Creates a deterministic sample set: the "best" at the prior median
        and additional slightly perturbed parameters with worse likelihoods.
        The default of four samples keeps bypass mode cheap while allowing
        downstream structural checks to exercise multi-batch sample handling.

        ``PYAUTO_TEST_MODE_SAMPLES=N`` raises the sample count so the
        bypass run's ``samples.csv`` row count and byte size match a
        production sampler stage (N ~ 10k-100k), keeping resume/load
        timings measured against the output honest. The N > 4 samples are
        synthesized vectorized (numpy) then materialised through the same
        ``Sample.from_lists`` path a real sampler run uses, so structure
        and cost are representative by construction. The best (first)
        sample is the unperturbed prior median in both branches.
        """
        from autofit.non_linear.samples.sample import Sample

        total_samples = test_mode_samples()

        if total_samples == 4:
            parameter_lists = [parameter_vector]
            for scale in (1.001, 0.999, 1.002):
                parameter_lists.append(
                    [p * scale if p != 0.0 else scale - 1.0 for p in parameter_vector]
                )

            return Sample.from_lists(
                model=model,
                parameter_lists=parameter_lists,
                log_likelihood_list=[
                    log_likelihood - offset for offset in range(len(parameter_lists))
                ],
                log_prior_list=[0.0] * len(parameter_lists),
                weight_list=[1.0, 0.5, 0.25, 0.125],
            )

        rng = np.random.default_rng(0)
        base = np.asarray(parameter_vector, dtype=float)
        scatter = 1.0e-3 * rng.standard_normal((total_samples, base.shape[0]))
        parameters = np.where(base == 0.0, scatter, base * (1.0 + scatter))
        parameters[0] = base

        # Weights decay over ~N/10 samples so the effective sample size stays
        # a healthy fraction of N and the smallest weight, ~(10/N)e^-10, sits
        # above the output.yaml samples_weight_threshold of 1e-10 for N <= 1e5.
        weights = np.exp(
            -np.arange(total_samples, dtype=float) / (total_samples / 10.0)
        )

        return Sample.from_lists(
            model=model,
            parameter_lists=parameters.tolist(),
            log_likelihood_list=(
                log_likelihood - np.arange(total_samples, dtype=float)
            ).tolist(),
            log_prior_list=[0.0] * total_samples,
            weight_list=(weights / weights.sum()).tolist(),
        )

    # Whether `ctx.start_points` makes the start-point plot. `Drawer` draws its whole
    # result through the initializer, so it has no starting point to plot.
    _plots_start_point = True

    @abstractmethod
    def _fit(self, model: AbstractPriorModel, analysis: Analysis):
        """
        Run the backend and return ``(search_internal, fitness)``.

        A search either overrides this (every search not yet migrated) or implements
        ``run(ctx)``, in which case this is the bridge (``docs/design/run_ctx.md``): it
        builds the fit's ``Fitness`` and ``FitContext`` -- after the test-mode bypass and
        the fail-fast gates, which ``start_resume_fit`` has already run -- calls
        ``run(ctx)`` and cleans the context up on every exit.
        """
        if type(self).run is NonLinearSearch.run:
            raise NotImplementedError(
                f"{type(self).__name__} implements neither run(ctx) nor _fit."
            )

        from autofit.non_linear.search.fit_context import FitContext

        # The REQUIRED gate, repeated for callers that reach `_fit` without `fit`, before
        # any `Fitness` or backend state exists (`fit` has already run it).
        cap.check_jax_required(self, analysis)

        fitness = self.make_fitness(
            analysis=analysis, model=model, **self.fitness_overrides(analysis)
        )

        ctx = FitContext(
            search=self,
            model=model,
            analysis=analysis,
            fitness=fitness,
            pool=self._pools(),
            test_mode_level=test_mode_level(),
        )

        try:
            search_internal = self.run(ctx)
        except BaseException:
            ctx.close(failed=True)
            raise

        ctx.close(failed=False)

        return search_internal, fitness

    def run(self, ctx) -> Any:
        """
        The backend loop of a search on the ``run(ctx)`` hook: run the backend through
        ``ctx`` and return its internal state (``search_internal``). See
        ``docs/design/run_ctx.md``.
        """
        raise NotImplementedError

    run._is_bridge_default = True

    def info_from(self, internal) -> dict:
        """
        The ``samples_info`` entries of a ``run(ctx)`` search's internal state, the
        ``info`` its ``raw_samples_from`` puts on the ``RawSamples``
        (``autofit.non_linear.samples.adapter``).
        """
        return {}

    def fitness_overrides(self, analysis: Analysis) -> dict:
        """
        ``Fitness`` keyword arguments a ``run(ctx)`` search adds to ``make_fitness`` for
        this analysis (e.g. ``batched=True`` for a batched backend). Empty by default.
        """
        return {}

    def make_fitness(
        self, analysis: Analysis, model: AbstractPriorModel, **overrides
    ) -> Fitness:
        """
        The `Fitness` this search evaluates its likelihood through, the one construction
        site every search shares.

        Its figure-of-merit convention comes from the search's declared capabilities
        (`autofit.non_linear.search.capabilities`), never from per-site literals:

        - `objective_target.quantity` picks what the objective returns: the log
          likelihood (`fom_is_log_likelihood=True`), the log posterior, or -2 x the log
          posterior (`convert_to_chi_squared=True`, for the minimizers);
        - `invalid_value` is the sentinel the backend sees for an invalid model. For a
          `neg2_log_posterior` search that is the value *after* the -2 multiply, so the
          `resample_figure_of_merit` substituted before it is `invalid_value / -2`
          (`+inf` -> `-inf`).

        The quick-update cadence, background worker and live-visual settings are always
        forwarded, so no search can silently drop quick updates (what
        `test_quick_update_wiring.py` used to scan for).

        Parameters
        ----------
        analysis
            The analysis whose likelihood is wrapped.
        model
            The model being fitted.
        overrides
            Any other `Fitness` keyword argument, which wins over the derived and
            forwarded ones (e.g. `batched=True`, `store_history=True`).
        """
        target = type(self).objective_target

        kwargs = dict(
            model=model,
            analysis=analysis,
            paths=self.paths,
            iterations_per_quick_update=self.iterations_per_quick_update,
            background_quick_update=self.quick_update_background,
            live_visual_update=self.live_visual_update,
        )

        if target is not None:
            quantity = target.quantity
            neg2 = quantity == cap.ObjectiveQuantity.NEG2_LOG_POSTERIOR

            kwargs.update(
                fom_is_log_likelihood=quantity == cap.ObjectiveQuantity.LOG_LIKELIHOOD,
                convert_to_chi_squared=neg2,
                resample_figure_of_merit=(
                    type(self).invalid_value / -2.0
                    if neg2
                    else type(self).invalid_value
                ),
            )

        kwargs.update(overrides)

        return Fitness(**kwargs)

    def check_model(self, model: AbstractPriorModel):
        if model is not None and model.prior_count == 0:
            raise AssertionError("Model has no priors! Cannot fit a 0 dimension model.")

    def apply_test_mode(self):
        """
        Override in subclasses to reduce sampler iterations for test mode.

        The base class never calls this method: each search that overrides it
        calls it at the end of its own ``__init__`` when ``is_test_mode()`` is
        true (test mode level 1). Overrides directly mutate instance attributes
        to minimize the number of iterations the sampler performs.
        """
        pass

    # --- internal state: archive and resume state (search-extensibility A3, D5) ----
    #
    # See ``autofit.non_linear.checkpoint`` and ``docs/design/checkpointing.md``.

    checkpointer: Checkpointer = DillCheckpointer()
    """
    The archive strategy: where the backend's final state is stored when a fit
    completes, for ``Result.search_internal`` and ``samples_via_internal_from``.
    """

    resume_state: Optional[Checkpointer] = None
    """
    The file an interrupted run resumes from, for searches that genuinely resume
    (``None`` otherwise). Deleted when the fit completes unless it is the archive's
    own file or declares ``retain_after_completion``.
    """

    @property
    def retains_search_internal(self) -> bool:
        """
        Whether ``search_internal/`` survives the end of a fit: always when the
        archive declares ``retain_after_completion`` (its results cannot be rebuilt
        without it), otherwise when the ``output.search_internal`` config is on.
        Constructing or running a search never changes that config.
        """
        return bool(
            self.checkpointer.retain_after_completion
            or conf.instance["output"]["search_internal"]
        )

    def output_search_internal(self, search_internal):
        """
        Archive ``search_internal`` through the search's ``checkpointer`` and, once a
        fit has completed, discard its resume state (there is nothing left to
        resume).

        ``_fit`` of the searches that archive mid-run (BlackJAX NUTS, SMC) also
        calls this after every chunk; their resume state is ``None``.
        """
        self.checkpointer.finalize(self.paths, search_internal)

        resume_state = self.resume_state
        if (
            resume_state is not None
            and not resume_state.retain_after_completion
            and resume_state.filename != self.checkpointer.filename
        ):
            resume_state.discard(self.paths)

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
        # loop: ``AbstractSearch.optimise(factor_approx)`` mutates
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
                samples_from_func=self.samples_from,
                disable_output=self.disable_output,
                iterations_per_full_update=self.iterations_per_full_update,
            )
        # Re-applied on every access: ``optimise`` toggles ``_visualize_fit``
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

    @property
    def should_plot_start_point(self) -> bool:
        return conf.instance["output"]["start_point"]

    def plot_start_point(
        self,
        parameter_vector: List[float],
        model: AbstractPriorModel,
        analysis: Analysis,
    ):
        """
        Visualize the starting point of the non-linear search, using an instance of the model at the starting point
        of the maximum likelihood estimator.

        Plots are output to a folder named `image_start` in the output path, so that the starting point model
        can be compared to the final model inferred by the non-linear search.

        Parameters
        ----------
        model
            The model used by the non-linear search
        analysis
            The analysis which contains the visualization methods which plot the starting point model.

        Returns
        -------

        """

        if not self.should_plot_start_point:
            return

        self.logger.info(f"Visualizing Starting Point Model in image_start folder.")

        instance = model.instance_from_vector(vector=parameter_vector)
        paths = copy.copy(self.paths)
        paths.image_path_suffix = "_start"

        # The search has no samples before it starts, so only the analysis visuals
        # of the starting point are output, not the search's own plots.
        self.perform_visualization(
            model=model,
            analysis=analysis,
            instance=instance,
            during_analysis=False,
            paths_override=paths,
            plot_search=False,
        )

    def start_points(
        self,
        model: AbstractPriorModel,
        fitness: Fitness,
        n: int,
        plot: bool = True,
        **initializer_kwargs,
    ):
        """
        The search's ``n`` initial points, drawn by its initializer, with the start-point
        plot made once, here, from the first of them.

        The one place the searches draw start points: the initializer is evaluated through
        ``fitness`` (so a JAX analysis runs its lazily jitted scalar objective) with the
        fit's effective core count (the JAX fork rule of ``PoolFactory``).

        Parameters
        ----------
        model
            The model being fitted.
        fitness
            The fit's ``Fitness``; its analysis visualizes the start point.
        n
            The number of points.
        plot
            ``False`` skips the start-point plot, for searches whose initial points are
            not a starting point (``Drawer`` draws its whole result this way, Dynesty its
            initial live points).
        initializer_kwargs
            Passed to ``initializer.samples_from_model`` (e.g. ``test_mode_samples``).

        Returns
        -------
        The initializer's ``(unit_parameter_lists, parameter_lists, figure_of_merit_list)``.
        """
        unit_parameter_lists, parameter_lists, figure_of_merit_list = (
            self.initializer.samples_from_model(
                total_points=n,
                model=model,
                fitness=fitness,
                paths=self.paths,
                n_cores=self._pools().number_of_cores,
                **initializer_kwargs,
            )
        )

        if plot:
            self.plot_start_point(
                parameter_vector=parameter_lists[0],
                model=model,
                analysis=fitness.analysis,
            )

        return unit_parameter_lists, parameter_lists, figure_of_merit_list

    # --- samples conversion (search-extensibility phase A3) -------------------------
    #
    # A search maps its backend's internal state onto a ``RawSamples`` in
    # ``raw_samples_from``; ``samples_via_internal_from`` below is implemented once,
    # here, on top of it (``autofit.non_linear.samples.adapter``).

    samples_cls = Samples
    """
    The ``Samples`` class ``samples_via_internal_from`` builds (``SamplesMCMC`` and
    ``SamplesNest`` on the family bases, ``SamplesSMC`` / ``NSSamples`` on those
    searches).
    """

    def samples_from(self, model: AbstractPriorModel, search_internal=None) -> Samples:
        """
        Loads the samples of a non-linear search from its output files.

        The samples can be loaded from one of two files, which are attempted to be loading in the following order:

        1) Load via the internal results of the non-linear search, which are specified to that search's outputs
           (e.g. the .hdf file output by the MCMC method `emcee`).

        2) Load via the `samples.csv` and `samples_info.json` files of the search, which are outputs that are the
           same for all non-linear searches as they are homogenized by autofit.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space.
        """
        try:
            return self.samples_via_internal_from(
                model=model, search_internal=search_internal
            )
        except (FileNotFoundError, NotImplementedError) as e:
            logger.warning(
                f"The samples of {type(self).__name__} could not be loaded from its "
                f"internal results ({type(e).__name__}: {e}), so they are loaded from "
                f"the samples.csv in the output folder instead."
            )
            return self.paths.samples

    def samples_via_internal_from(
        self, model: AbstractPriorModel, search_internal=None
    ) -> Samples:
        """
        The samples of the search, converted from its internal state.

        ``search_internal`` is the backend's in-memory state when given; otherwise
        it is loaded from the output folder (``load_search_internal``). The search's
        ``raw_samples_from`` maps it onto a ``RawSamples`` and ``samples_from_raw``
        builds the ``samples_cls`` object (log priors, the log likelihood from a log
        posterior, length checks, ``time`` / ``class_path``), the same way for every
        search.

        Raises
        ------
        NotImplementedError
            When the search does not implement ``raw_samples_from``.
        FileNotFoundError
            When no internal state is stored.
        """
        from autofit.non_linear.samples.adapter import samples_from_raw

        if search_internal is None:
            search_internal = self.load_search_internal()

        raw = self.raw_samples_from(model, search_internal)

        return samples_from_raw(
            model=model,
            raw=raw,
            samples_cls=self.samples_cls,
            time=self.timer.time if self.timer else None,
        )

    def raw_samples_from(self, model: AbstractPriorModel, search_internal):
        """
        Map the backend's internal state onto a
        ``autofit.non_linear.samples.adapter.RawSamples``: its parameter vectors,
        log likelihoods (or log posteriors), weights and ``samples_info`` entries,
        exactly as the backend stores them. Every concrete search implements it.
        """
        raise NotImplementedError

    def load_search_internal(self):
        """
        The search's internal state as stored in its output folder, or ``None`` when
        its paths store nothing (``NullPaths``, ``DatabasePaths``).
        """
        return self.paths.load_search_internal()

    def _pools(self) -> PoolFactory:
        """
        The current fit's `PoolFactory` (the JAX fork rule applied to its analysis), or,
        outside a fit, one built from `number_of_cores` alone.
        """
        pools = self.__dict__.get("_fit_pools")
        return pools if pools is not None else PoolFactory(self)

    @check_cores
    def make_pool(self):
        """Make the pool instance used to parallelize a `NonLinearSearch` alongside a set of unique ids for every
        process in the pool. If the specified number of cores is 1, a pool instance is not made and None is returned.

        The pool cannot be set as an attribute of the class itself because this prevents pickling, thus it is generated
        via this function before calling the non-linear search.

        The pool instance is also set up with a list of unique pool ids, which are used during model-fitting to
        identify a 'master core' (the one whose id value is lowest) which handles model result output, visualization,
        etc."""
        return self._pools()()

    def make_sneaky_pool(self, fitness: Fitness) -> Optional[SneakyPool]:
        """
        Create a pool for multiprocessing that uses slight-of-hand
        to avoid copying the fitness function between processes
        multiple times.

        Parameters
        ----------
        fitness
            An instance of a fitness class used to evaluate the
            likelihood that a particular model is correct

        Returns
        -------
        An implementation of a multiprocessing pool
        """

        return self._pools().sneaky(fitness)

    def __eq__(self, other):
        return isinstance(other, NonLinearSearch) and self.__dict__ == other.__dict__

    def plot_results(self, samples):
        raise NotImplementedError
