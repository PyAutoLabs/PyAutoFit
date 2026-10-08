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
"""
import logging
import sys
from typing import Optional

from autofit import exc

from .context import fork_context

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

    if number_of_cores > 1:
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


class PoolFactory:
    def __init__(self, search, analysis=None):
        """
        The pools one fit of ``search`` may use, with the JAX fork rule applied once.

        Built by ``NonLinearSearch`` once per fit, after the test-mode bypass and the
        fail-fast gates, and reached by a ``run(ctx)`` search as ``ctx.pool`` and by a
        legacy ``_fit`` through ``make_pool`` / ``make_sneaky_pool`` / ``number_of_cores``
        of the fit's factory.

        Parameters
        ----------
        search
            The search being fitted; its ``number_of_cores`` is the request and its
            ``paths`` go to ``SneakyPool``.
        analysis
            The analysis being fitted, whose ``is_jax`` drives the fork rule. ``None``
            (no fit context) applies no JAX rule.
        """
        self.search = search
        self.requested = getattr(search, "number_of_cores", 1) or 1
        self.is_jax = bool(getattr(analysis, "is_jax", False))

        self.number_of_cores = effective_number_of_cores(
            requested=self.requested,
            is_jax=self.is_jax,
            where=type(search).__name__,
        )

    @property
    def summary(self) -> str:
        """
        The ``search.summary`` line recording the requested and effective worker counts.
        """
        line = (
            f"Number of cores = {self.number_of_cores} "
            f"(requested {self.requested})"
        )
        if self.number_of_cores < self.requested:
            line += f": {JAX_FORK_REASON}"
        return line + "\n"

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

        getattr(self.search, "logger", logger).info("...using pool")
        return fork_context().Pool(processes=n_cores, **pool_kwargs)

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
        return SneakyPool(
            processes=self.number_of_cores,
            paths=self.search.paths,
            fitness=fitness,
        )
