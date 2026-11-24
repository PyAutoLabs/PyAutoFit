"""
Search runtime settings: the global-config reads a ``NonLinearSearch`` makes once, at
construction, plus the logging and deprecation helpers the search module shares.

``SearchRuntimeSettings.from_config`` resolves the ``general.yaml`` output, update,
HPC and parallel settings into one frozen value; ``NonLinearSearch.__init__`` applies
it as the same instance attributes as before (``iterations_per_quick_update``,
``force_pickle_overwrite``, ...), so serialized searches and their identifiers are
unchanged.
"""

import logging
import os

import numpy as np
import warnings
from dataclasses import dataclass
from typing import Optional

from autonerves import conf
from autonerves.output import should_output

from autofit import exc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchRuntimeSettings:
    """
    The configuration-derived settings of one search, resolved once.

    The update cadences are floats because the packaged default is the inf-like
    ``1e99`` ("never"); HPC mode (``general.yaml -> hpc -> hpc_mode``) overrides both
    cadences with its own values and silences the search.
    """

    iterations_per_quick_update: float
    iterations_per_full_update: float
    quick_update_background: bool
    live_visual_update: bool
    force_pickle_overwrite: bool
    force_visualize_overwrite: bool
    silence: bool

    @classmethod
    def from_config(
        cls,
        iterations_per_quick_update: Optional[int] = None,
        iterations_per_full_update: Optional[int] = None,
        live_visual_update: Optional[bool] = None,
        silence: bool = False,
    ) -> "SearchRuntimeSettings":
        general = conf.instance["general"]
        updates = general["updates"]
        hpc = general["hpc"]

        quick = float(iterations_per_quick_update or updates["iterations_per_quick_update"])
        full = float(iterations_per_full_update or updates["iterations_per_full_update"])

        if hpc["hpc_mode"]:
            quick = float(hpc["iterations_per_quick_update"])
            full = float(hpc["iterations_per_full_update"])
            silence = True

        return cls(
            iterations_per_quick_update=quick,
            iterations_per_full_update=full,
            quick_update_background=bool(updates.get("quick_update_background", False)),
            live_visual_update=bool(
                live_visual_update
                if live_visual_update is not None
                else updates.get("live_visual_update", False)
            ),
            force_pickle_overwrite=general["output"]["force_pickle_overwrite"],
            force_visualize_overwrite=general["output"]["force_visualize_overwrite"],
            silence=silence,
        )

    def apply(self, search):
        """
        Set every setting as the instance attribute of the same name on ``search``.
        """
        for name in self.__dataclass_fields__:
            setattr(search, name, getattr(self, name))


_THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def warn_if_threads_unpinned(number_of_cores):
    """
    Warn when a search uses multiprocessing (``number_of_cores > 1``) while the
    numerical libraries' thread-count environment variables are not pinned to 1.
    Disabled by ``general.yaml -> parallel -> warn_environment_variables``.

    Only an int count is checked: an ``"auto"`` request is checked once the fit
    resolves it (``resolve_fit_parallel``).
    """
    if not isinstance(number_of_cores, int) or number_of_cores <= 1 or all(
        os.environ.get(key) == "1" for key in _THREAD_VARIABLES
    ):
        return

    if not conf.instance["general"]["parallel"]["warn_environment_variables"]:
        return

    warnings.warn(exc.SearchWarning(""))
    exports = "\n".join(f"    export {key}=1" for key in _THREAD_VARIABLES)
    logger.warning(
        "The non-linear search is using multiprocessing (number_of_cores>1), but "
        f"the following environment variables are not all set to 1: "
        f"{', '.join(_THREAD_VARIABLES)}.\n\n"
        "This can lead to performance issues, because both the non-linear search and "
        "libraries that may be used in your `log_likelihood_function` evaluation "
        "(e.g. NumPy, SciPy, scikit-learn) may attempt to parallelize over all cores "
        "available, overallocating tasks over the CPUs.\n\n"
        "To mitigate this, set the environment variables to 1 in your terminal, so "
        f"only the non-linear search is parallelized over multiple cores:\n\n{exports}\n\n"
        "If you \"know what you are doing\" and do not want these environment "
        "variables to be set to one, disable this warning in the config files: "
        "`config -> general.yaml -> parallel: -> warn_environment_variables=False`"
    )


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


def _mro_init_parameters(cls) -> frozenset:
    """
    The named parameters of every ``__init__`` in ``cls``'s MRO.
    """
    import inspect

    names = set()
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        try:
            parameters = inspect.signature(init).parameters.values()
        except (TypeError, ValueError):
            continue
        names.update(
            parameter.name
            for parameter in parameters
            if parameter.kind
            not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        )
    return frozenset(names)


def warn_unknown_kwargs(search, kwargs: dict, known=()):
    """
    Warn (a log WARNING, never an error) about each keyword argument that reached
    ``NonLinearSearch.__init__`` through ``**kwargs`` but that no ``__init__`` in the
    search's MRO consumes -- a typo such as ``n_lives=`` that would otherwise be
    silently ignored.

    Not warned about: the ``known`` keys the base class reads or ``search.json``
    records, a key some ``__init__`` in the MRO names (a subclass passing it on on
    purpose), and ``use_jax_vmap=None`` -- the deprecated knob's default, which
    ``SettingsSearch.search_dict`` hands every search and which only the searches that
    accept it read (``SettingsSearch.search_dict_for`` drops it for the others). A
    ``use_jax_vmap`` set to anything else on a search that does not accept it is
    warned about like any other unknown argument.
    """
    consumed = _mro_init_parameters(type(search))

    for key, value in kwargs.items():
        if key in known or key in consumed:
            continue
        if key == "use_jax_vmap" and value is None:
            continue
        logger.warning(
            f"{type(search).__name__} received the unknown keyword argument "
            f"{key!r}, which is ignored. Check its spelling against the "
            f"search's constructor."
        )


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


def log_fit_start(search, analysis):
    """
    Log, once at the start of a fit, the JAX device a JAX analysis runs on and the
    quick-update cadence (``NonLinearSearch.quick_update_message``).

    The worker count is not logged here: it is only known once ``modify_before_fit``
    has resolved the analysis's compute setup, and the fit's one ``Parallel:`` banner
    line reports it (``resolve_fit_parallel``).
    """
    if analysis.is_jax:
        try:
            import jax

            device = jax.devices()[0]
            backend = device.platform.upper()
            device_name = getattr(device, "device_kind", backend)
            logger.info(
                f"Starting non-linear search with JAX ({backend}: {device_name})."
            )
        except Exception:
            logger.info("Starting non-linear search with JAX.")
    else:
        logger.info("Starting non-linear search.")
    logger.info(search.quick_update_message)


def resolve_fit_parallel(search, analysis, model=None):
    """
    Resolve a fit's worker count once ``modify_before_fit`` has resolved the analysis's
    compute setup (``autofit.non_linear.parallel.pool.resolve_number_of_cores``:
    ``"auto"`` becomes the detected cores, 1 for a JAX analysis or an
    expectation-propagation factor search, capped by memory when the analysis declares
    ``resources_per_eval()``), log the fit's one ``Parallel:`` banner line and record
    the resolution for the pools, ``search.summary`` and ``files/parallel.json``.

    ``search.number_of_cores`` keeps the request, so ``search.json`` records it; an
    ``"auto"`` request gets its thread warning here, for the resolved count. Writes
    ``files/parallel.json`` unless the fit is already complete (a completed fit keeps
    the record of the run that produced it).

    After the banner, an analysis that defines ``runtime_estimate(search, model,
    number_of_cores)`` (duck-typed; PyAutoLens implements it) is asked for a one-line
    run-time estimate, which is logged and recorded as ``estimate_line`` in
    ``files/parallel.json``. The hook is advisory: an analysis without it, a hook
    returning anything but a non-empty string, or a hook that raises never affects the fit.
    """
    from autofit.non_linear.parallel.pool import resolve_number_of_cores
    from autofit.non_linear.parallel.resources import is_auto

    resolution = resolve_number_of_cores(
        requested=getattr(search, "number_of_cores", 1),
        analysis=analysis,
        where=type(search).__name__,
        ep_factor_search=bool(getattr(search, "_ep_factor_search", False)),
    )
    search._parallel_resolution = resolution
    search._parallel_summary = resolution.summary

    search_logger = getattr(search, "logger", logger)
    search_logger.info(resolution.banner)

    estimate_line = runtime_estimate_line(
        search=search,
        analysis=analysis,
        model=model,
        number_of_cores=resolution.number_of_cores,
    )
    if estimate_line is not None:
        search_logger.info(estimate_line)

    if is_auto(resolution.requested):
        warn_if_threads_unpinned(resolution.number_of_cores)

    paths = getattr(search, "paths", None)
    if paths is not None and not paths.is_complete:
        record = resolution.to_dict()
        record["estimate_line"] = estimate_line
        paths.save_json("parallel", record)

    return resolution


def runtime_estimate_line(search, analysis, model, number_of_cores):
    """
    The one-line run-time estimate an analysis gives for this fit, or ``None``.

    Calls ``analysis.runtime_estimate(search=..., model=..., number_of_cores=...)`` when
    the analysis defines it. Only a non-empty string is used; any exception is logged at
    debug level and ignored, so an estimate can never stop or change a fit.
    """
    hook = getattr(analysis, "runtime_estimate", None)
    if not callable(hook):
        return None
    try:
        line = hook(search=search, model=model, number_of_cores=number_of_cores)
    except Exception as e:
        logger.debug(f"runtime_estimate failed and was ignored: {e!r}")
        return None
    if isinstance(line, str) and line.strip():
        return line.strip()
    return None


def fit_pools(search, analysis):
    """
    The fit's ``PoolFactory``, built on the resolution ``resolve_fit_parallel`` made
    after ``modify_before_fit`` (or resolved here when there is none), with the one JAX
    fork rule applied once; records the resolution for ``search.summary``.
    """
    from autofit.non_linear.parallel import PoolFactory

    pools = PoolFactory(
        search, analysis, resolution=search.__dict__.get("_parallel_resolution")
    )
    search._parallel_resolution = pools.resolution
    search._parallel_summary = pools.summary
    return pools


def number_of_cores_request(search, number_of_cores):
    """
    The validated ``number_of_cores`` request a search stores (and ``search.json``
    records): an int >= 1 or ``"auto"`` (``validate_number_of_cores``; ``None`` is 1).
    An int count gets its thread warning now; ``"auto"`` gets it once a fit resolves it.
    """
    from autofit.non_linear.parallel.resources import validate_number_of_cores

    number_of_cores = validate_number_of_cores(
        number_of_cores, where=type(search).__name__
    )
    warn_if_threads_unpinned(number_of_cores)
    return number_of_cores


def quick_update_message(iterations: float) -> str:
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
    from autofit.non_linear.fitness import ITERATIONS_NEVER

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
