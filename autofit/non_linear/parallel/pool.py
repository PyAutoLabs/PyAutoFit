"""
The pools PyAutoFit forks, and the one rule that decides whether it may fork at all.

**The JAX fork rule.** A process that has initialised JAX (XLA's thread pools, its
compilation cache, a GPU context) must not ``fork``: a forked child inherits locks held by
threads that do not exist in it, and an XLA compile or execution in the child can deadlock
(the same class of hang as the 27-hour EP run guarded against by
``check_factor_search_cores``). It also defeats the point of JAX: every forked worker
recompiles the likelihood, so ``N`` cores cost ``N`` compiles and run slower than one jitted
process. So:

- a JAX analysis is never evaluated through a multiprocessing pool. A search given one
  with ``number_of_cores > 1`` runs with **one** core, logs exactly one INFO line saying
  so, and records the requested and effective counts in ``search.summary``. Parallelism
  comes from the batched (``jax.vmap``) objective instead;
- the grid-search and sensitivity-mapping job pools apply the same rule to their jobs;
- ``AnalysisPool`` (which always forks) refuses a JAX analysis outright;
- expectation propagation never forks a factor search, JAX or not
  (``check_factor_search_cores``, a deliberate refusal).

``effective_number_of_cores`` is the rule; ``PoolFactory`` applies it to a search's fit and
builds the pools the searches use, with the same pool classes, start method and arguments
as before (``fork_context``), so pool semantics are unchanged wherever forking is allowed.

**``number_of_cores="auto"``.** ``resolve_number_of_cores`` turns a search's request into
the worker count of one fit, after ``Analysis.modify_before_fit`` has resolved the
analysis's compute setup: an int request is honoured (the fork rule aside); ``"auto"`` is
the detected cores (``SLURM_CPUS_PER_TASK`` -> CPU affinity -> ``os.cpu_count()``), 1 for a
JAX analysis or an expectation-propagation factor search, and is capped by
``min(cores // cpus_per_eval, available memory // peak_rss_gb)`` only when the analysis
declares ``resources_per_eval()``. There is deliberately no parallel-efficiency cap: the one
measured curve (imaging, rectangular pixelization, numba) has cores x efficiency rising at
every step up to 32 cores, so throughput favours every core. The result
(``ParallelResolution``) drives the fit's one ``Parallel: ...`` banner line,
``search.summary`` and ``files/parallel.json``; ``search.json`` keeps the request.
"""
import logging
import math
import sys
from dataclasses import dataclass, asdict
from typing import Optional, Union

from autofit import exc

from .context import fork_context
from .resources import (
    AUTO,
    available_memory_gb,
    detected_cores,
    is_auto,
    validate_number_of_cores,
)

logger = logging.getLogger(__name__)

JAX_FORK_REASON = (
    "JAX analyses never fork a multiprocessing pool: a forked child of a "
    "JAX-initialised process can deadlock in XLA and recompiles the likelihood per "
    "worker; parallelism comes from the vectorised (jax.vmap) objective instead"
)


def jax_backend_initialized() -> bool:
    """
    Whether this process has already initialised a JAX (XLA) backend.

    Importing ``jax`` initialises nothing; the first array operation does. Used where the
    analysis that will run is not known before the fork (sensitivity mapping builds its
    analyses inside each job), so the parent's own JAX state is the only signal.
    """
    if "jax" not in sys.modules:
        return False
    try:
        from jax._src import xla_bridge

        return bool(getattr(xla_bridge, "_backends", None))
    except Exception:
        return False


def effective_number_of_cores(requested: int, is_jax: bool, where: str) -> int:
    """
    The JAX fork rule: the number of worker processes ``where`` may fork.

    Parameters
    ----------
    requested
        The ``number_of_cores`` the user asked for.
    is_jax
        Whether the work being parallelised is JAX (``Analysis.is_jax``, or a JAX backend
        already initialised in this process).
    where
        What is asking, as it should read in the one INFO line.

    Returns
    -------
    ``requested``, or 1 when ``is_jax`` and ``requested > 1``, in which case one INFO line
    names the downgrade and why.
    """
    if requested is None:
        return 1

    if is_jax and requested > 1:
        logger.info(
            f"{where}: number_of_cores={requested} requested, running with 1 core. "
            f"{JAX_FORK_REASON}."
        )
        return 1

    return requested


def check_factor_search_cores(search):
    """
    Refuse an expectation-propagation factor search built with ``number_of_cores > 1``.

    A deliberate refusal, not a silent downgrade. ``number_of_cores`` is the user's stated
    intent, and the *same* search instance is re-entered once per factor per EP step — so
    quietly rewriting it to 1 would mutate shared state the caller still owns and would
    hide the misconfiguration rather than fix it. Raising makes the operator change the
    search they built.

    ``AbstractSearch.optimise`` is the single door every factor optimisation passes
    through, so every search type is covered by this one check.

    ``getattr`` with a default of 1: a handful of search doubles subclass
    ``NonLinearSearch`` without running its ``__init__`` (e.g. the regression suite's
    ``StaticSearch``), and they run nothing in parallel anyway.
    """
    number_of_cores = getattr(search, "number_of_cores", 1)

    # "auto" asks PyAutoFit to choose, and a factor search resolves it to 1
    # (`resolve_number_of_cores`), so only an explicit count above 1 is refused.
    if is_auto(number_of_cores):
        return

    if (number_of_cores or 1) > 1:
        raise exc.SearchException(
            f"Expectation propagation never runs a factor search through a "
            f"Python multiprocessing pool (human ruling 2026-09-09), but the "
            f"factor optimiser {search.__class__.__name__} was built with "
            f"number_of_cores={number_of_cores}.\n\n"
            f"Why: a forked likelihood worker that dies (a segfault, an OOM "
            f"kill) is silently replaced by `multiprocessing.Pool`, but the "
            f"task it was running is never re-issued, so the `Pool.map` "
            f"driving the fit blocks forever and the EP run hangs to the wall "
            f"clock rather than failing. RAL job 342351_0 burned 27 hours "
            f"exactly this way.\n\n"
            f"Fix, either of:\n"
            f"  - build the factor search with number_of_cores=1;\n"
            f"  - get parallelism from a vectorised JAX likelihood instead, "
            f"`Analysis(use_jax=True)` — Nautilus then takes `fit_x1_cpu` "
            f"with vectorized=True and no pool at all."
        )

JAX_REASON = "JAX analyses run single-process"
EP_REASON = "expectation-propagation factor searches run single-process"


@dataclass
class ParallelResolution:
    """
    The worker count of one fit and why (``resolve_number_of_cores``).

    Attributes
    ----------
    requested
        The search's ``number_of_cores``: an int or ``"auto"``.
    number_of_cores
        The resolved worker count.
    reason
        Why: ``"requested"`` (an int honoured), ``"detected_cores"`` (auto, uncapped),
        ``"jax"``, ``"expectation_propagation"``, ``"cores_per_eval"`` or ``"memory"``.
    detected_cores
        The cores this process may use (``None`` when not probed).
    available_memory_gb
        The memory available when the memory cap was evaluated (``None`` otherwise).
    setup_id
        The analysis's resolved compute setup (``analysis.setup.setup_id``), or ``None``.
    cpus_per_eval, peak_rss_gb
        The analysis's ``resources_per_eval()`` (``None`` when it declares none).
    is_jax
        Whether the analysis is JAX (the banner then reports the JAX device).
    """

    requested: Union[int, str]
    number_of_cores: int
    reason: str
    detected_cores: Optional[int] = None
    available_memory_gb: Optional[float] = None
    setup_id: Optional[str] = None
    cpus_per_eval: Optional[float] = None
    peak_rss_gb: Optional[float] = None
    is_jax: bool = False

    @property
    def reason_text(self) -> str:
        """
        The reason as the banner and ``search.summary`` say it; empty for an honoured
        int request.
        """
        if self.reason == "detected_cores":
            return "detected cores"
        if self.reason == "jax":
            return JAX_REASON
        if self.reason == "expectation_propagation":
            return EP_REASON
        if self.reason == "cores_per_eval":
            return (
                f"capped by cores: {self.detected_cores} cores / "
                f"{_number(self.cpus_per_eval)} CPUs per evaluation"
            )
        if self.reason == "memory":
            return (
                f"capped by memory: {self.available_memory_gb:.1f} GB available / "
                f"{_number(self.peak_rss_gb)} GB per evaluation"
            )
        return ""

    @property
    def banner(self) -> str:
        """
        The one INFO line a fit logs once its compute setup is resolved, e.g.
        ``Parallel: 64 workers (auto: detected cores)``, ``Parallel: 4 workers (requested 4)``
        or ``Parallel: 1 worker (JAX on GPU: NVIDIA A100-SXM4-80GB)``.

        The compute setup is not repeated here (the fit's ``Setup:`` line reports it);
        ``search.summary`` and ``files/parallel.json`` still record ``setup_id``.
        """
        workers = "worker" if self.number_of_cores == 1 else "workers"
        auto = is_auto(self.requested)

        if self.is_jax or self.reason == "jax":
            detail = jax_device_text()
            if not auto and self.requested != 1:
                detail = f"requested {self.requested}; {detail}"
        elif self.reason == "expectation_propagation":
            detail = "expectation-propagation factor search"
        elif self.reason == "requested" or not auto:
            detail = f"requested {self.requested}"
        else:
            detail = f"auto: {self.reason_text}"

        return f"Parallel: {self.number_of_cores} {workers} ({detail})"

    @property
    def summary(self) -> str:
        """
        The ``search.summary`` lines recording the requested and effective worker
        counts, the reason and the compute setup.
        """
        line = (
            f"Number of cores = {self.number_of_cores} "
            f"(requested {self.requested})"
        )
        if self.reason == "jax" and not is_auto(self.requested):
            line += f": {JAX_FORK_REASON}"
        elif self.reason_text:
            line += f": {self.reason_text}"
        line += "\n"
        if self.setup_id is not None:
            line += f"Setup = {self.setup_id}\n"
        return line

    def to_dict(self) -> dict:
        """
        The ``files/parallel.json`` record.
        """
        record = asdict(self)
        return {
            key: record[key]
            for key in (
                "requested",
                "number_of_cores",
                "reason",
                "detected_cores",
                "available_memory_gb",
                "setup_id",
            )
        }


def jax_device_text() -> str:
    """
    ``"JAX on CPU"`` or ``"JAX on GPU: <device name>"`` for the first JAX device, read
    only when JAX is already imported (this never imports JAX just to report it);
    ``"JAX"`` when it is not imported or the device cannot be read.
    """
    jax = sys.modules.get("jax")
    if jax is None:
        return "JAX"
    try:
        device = jax.devices()[0]
        platform = str(device.platform).upper()
        if platform == "CPU":
            return "JAX on CPU"
        name = getattr(device, "device_kind", None)
        return f"JAX on {platform}: {name}" if name else f"JAX on {platform}"
    except Exception:
        return "JAX"


def _number(value) -> str:
    """
    ``value`` without a trailing ``.0`` when it is whole.
    """
    try:
        if float(value).is_integer():
            return str(int(value))
    except (TypeError, ValueError):
        pass
    return str(value)


def _unwrap_analysis(analysis):
    """
    The analysis a wrapper (``ModelAnalysis``, ``AnalysisFactor``) forwards to; the
    analysis itself when it wraps nothing.
    """
    seen = set()
    while analysis is not None and id(analysis) not in seen:
        seen.add(id(analysis))
        inner = getattr(analysis, "__dict__", {}).get("analysis")
        if inner is None or inner is analysis:
            break
        analysis = inner
    return analysis


def setup_id_of(analysis) -> Optional[str]:
    """
    The resolved compute setup of ``analysis`` (``analysis.setup.setup_id``, set by a
    PyAutoGalaxy / PyAutoLens ``modify_before_fit``), duck-typed so PyAutoFit never
    imports autoarray. A factor graph reports the setup its factors agree on
    (``FactorGraphModel.setup_ids``). ``None`` when there is none.
    """
    analysis = _unwrap_analysis(analysis)
    if analysis is None:
        return None

    setup_ids = getattr(type(analysis), "setup_ids", None)
    if callable(setup_ids):
        ids = set(analysis.setup_ids().values())
        return ids.pop() if len(ids) == 1 else None

    setup = getattr(analysis, "__dict__", {}).get("setup")
    if setup is None:
        try:
            setup = getattr(analysis, "setup", None)
        except Exception:
            setup = None
    setup_id = getattr(setup, "setup_id", None)
    return None if setup_id is None else str(setup_id)


def resources_per_eval_of(analysis):
    """
    ``(cpus_per_eval, peak_rss_gb)`` from the analysis's optional ``resources_per_eval()``
    hook (a dict or an object with those attributes), each ``None`` when not declared.
    """
    analysis = _unwrap_analysis(analysis)
    hook = getattr(analysis, "resources_per_eval", None)
    if not callable(hook):
        return None, None

    resources = hook()
    if resources is None:
        return None, None

    def read(name):
        if isinstance(resources, dict):
            value = resources.get(name)
        else:
            value = getattr(resources, name, None)
        try:
            value = float(value) if value is not None else None
        except (TypeError, ValueError):
            return None
        return value if value is not None and value > 0 else None

    return read("cpus_per_eval"), read("peak_rss_gb")


def resolve_number_of_cores(
    requested,
    analysis=None,
    is_jax: Optional[bool] = None,
    where: str = "search",
    ep_factor_search: bool = False,
) -> ParallelResolution:
    """
    The worker count of one fit: the request, resolved after
    ``Analysis.modify_before_fit``.

    - An int request is honoured, except a JAX analysis runs one process (the fork
      rule; ``effective_number_of_cores`` logs its INFO line where the pools are built).
    - ``"auto"`` is 1 for a JAX analysis or an expectation-propagation factor search,
      otherwise the detected cores, capped by
      ``min(cores // cpus_per_eval, floor(available_gb / peak_rss_gb))`` (never below
      1) only when the analysis declares ``resources_per_eval()``.

    Logs nothing; the fit logs ``ParallelResolution.banner``.

    Parameters
    ----------
    requested
        The search's ``number_of_cores`` (validated here).
    analysis
        The analysis being fitted, after ``modify_before_fit`` (or ``None``).
    is_jax
        Whether the analysis is JAX; defaults to ``analysis.is_jax``.
    where
        What is asking, for the validation error.
    ep_factor_search
        Whether the search is an expectation-propagation factor search.
    """
    requested = validate_number_of_cores(requested, where=where)
    if is_jax is None:
        is_jax = bool(getattr(analysis, "is_jax", False))
    setup_id = setup_id_of(analysis)

    def resolution(n, reason, **kwargs):
        return ParallelResolution(
            requested=requested,
            number_of_cores=n,
            reason=reason,
            setup_id=setup_id,
            is_jax=bool(is_jax),
            **kwargs,
        )

    if requested != AUTO:
        if is_jax and requested > 1:
            return resolution(1, "jax")
        return resolution(requested, "requested")

    if is_jax:
        return resolution(1, "jax")
    if ep_factor_search:
        return resolution(1, "expectation_propagation")

    cores = detected_cores()
    cpus_per_eval, peak_rss_gb = resources_per_eval_of(analysis)

    n, reason = cores, "detected_cores"
    available = None

    if cpus_per_eval is not None:
        by_cores = max(1, int(cores // cpus_per_eval))
        if by_cores < n:
            n, reason = by_cores, "cores_per_eval"

    if peak_rss_gb is not None:
        available = available_memory_gb()
        if available is not None:
            by_memory = max(1, int(math.floor(available / peak_rss_gb)))
            if by_memory < n:
                n, reason = by_memory, "memory"

    return resolution(
        n,
        reason,
        detected_cores=cores,
        available_memory_gb=None if available is None else round(available, 3),
        cpus_per_eval=cpus_per_eval,
        peak_rss_gb=peak_rss_gb,
    )


class PoolFactory:
    def __init__(self, search, analysis=None, resolution=None):
        """
        The pools one fit of ``search`` may use, with the JAX fork rule applied once.

        Built by ``NonLinearSearch`` once per fit, after the test-mode bypass and the
        fail-fast gates, and reached by a ``run(ctx)`` search as ``ctx.pool`` (and by an
        external search that still overrides ``_fit`` as ``self._pools(analysis)``). It
        is the only pool source of a search: every pool a fit uses comes from here.

        Parameters
        ----------
        search
            The search being fitted; its ``number_of_cores`` is the request and its
            ``paths`` go to ``SneakyPool``.
        analysis
            The analysis being fitted, whose ``is_jax`` drives the fork rule. ``None``
            (no fit context) applies no JAX rule.
        resolution
            The fit's ``ParallelResolution``, when ``NonLinearSearch`` already resolved
            it after ``modify_before_fit``; otherwise it is resolved here
            (``resolve_number_of_cores``).
        """
        self.search = search
        self.is_jax = bool(getattr(analysis, "is_jax", False))

        if resolution is None:
            resolution = resolve_number_of_cores(
                requested=getattr(search, "number_of_cores", 1),
                analysis=analysis,
                is_jax=self.is_jax,
                where=type(search).__name__,
                ep_factor_search=bool(getattr(search, "_ep_factor_search", False)),
            )
        self.resolution = resolution
        self.requested = resolution.requested

        # The JAX fork rule's one INFO line for an explicit count above 1 (an "auto"
        # request is explained by the fit's `Parallel:` banner instead).
        if not is_auto(self.requested):
            effective_number_of_cores(
                requested=self.requested,
                is_jax=self.is_jax,
                where=type(search).__name__,
            )

        self.number_of_cores = resolution.number_of_cores

        self._created = []

        # The fit's `Fitness`, set by `NonLinearSearch` once it exists, so that
        # `__call__` can warm the likelihood up before forking (`prepare_for_fork`).
        self.fitness = None

    def prepare_for_fork(self, fitness=None):
        """
        Run the likelihood once in the parent before any pool of this fit forks workers
        (`Fitness.prepare_for_fork`), so lazily initialised native libraries (numba's
        import and first LLVM compile) are initialised once here rather than in every
        forked worker at once, which can segfault them.

        A no-op on one core (no workers are forked), when no fitness is known, or for a
        fitness without the hook (a test double, an external search's own callable).

        Parameters
        ----------
        fitness
            The fitness the workers will call; defaults to the fit's own (`self.fitness`).
        """
        if self.number_of_cores <= 1:
            return

        fitness = fitness if fitness is not None else self.fitness

        prepare = getattr(fitness, "prepare_for_fork", None)

        if prepare is not None:
            prepare()

    @property
    def summary(self) -> str:
        """
        The ``search.summary`` lines recording the requested and effective worker
        counts, the reason and the compute setup (``ParallelResolution.summary``).
        """
        return self.resolution.summary

    def __call__(self, n_cores: Optional[int] = None, **pool_kwargs):
        """
        A ``fork_context().Pool`` of ``n_cores`` workers, or ``None`` when the fit runs
        on one core (no pool at all: a ``Pool(1)`` would push every likelihood call
        through a forked worker).

        Parameters
        ----------
        n_cores
            The worker count; defaults to the fit's effective ``number_of_cores``. A
            larger count than the effective one is clamped to it, so the fork rule
            cannot be bypassed.
        pool_kwargs
            Passed to ``Pool`` (e.g. ``initializer`` / ``initargs``).
        """
        n_cores = self.number_of_cores if n_cores is None else min(
            n_cores, self.number_of_cores
        )

        if n_cores <= 1:
            return None

        self.prepare_for_fork()

        getattr(self.search, "logger", logger).info("...using pool")
        pool = fork_context().Pool(processes=n_cores, **pool_kwargs)
        self._created.append(pool)
        return pool

    def sneaky(self, fitness):
        """
        A ``SneakyPool`` over ``fitness`` (copied to each worker once, at fork time),
        or ``None`` on one core.
        """
        if self.number_of_cores <= 1:
            return None

        from .sneaky import SneakyPool

        getattr(self.search, "logger", logger).warning(
            "...using SneakyPool. This copies the likelihood function "
            "to each process on instantiation to avoid copying multiple "
            "times."
        )
        pool = SneakyPool(
            processes=self.number_of_cores,
            paths=self.search.paths,
            fitness=fitness,
        )
        self._created.append(pool)
        return pool

    def close(self, failed: bool = True):
        """
        Shut down every pool this factory built, once: ``close`` then ``join`` after a
        successful fit (the workers finish and exit), ``terminate`` then ``join`` after a
        failed one. Idempotent, and safe on a pool its backend already shut down.
        """
        created, self._created = self._created, []
        for pool in created:
            for name in ("terminate",) if failed else ("close",):
                method = getattr(pool, name, None)
                if method is None:
                    continue
                try:
                    method()
                except Exception:
                    logger.debug("Pool %s after a fit raised", name, exc_info=True)
            join = getattr(pool, "join", None)
            if join is None:
                continue
            try:
                join()
            except Exception:
                logger.debug("Pool join after a fit raised", exc_info=True)
