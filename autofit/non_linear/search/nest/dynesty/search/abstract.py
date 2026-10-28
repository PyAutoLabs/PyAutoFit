from __future__ import annotations

import logging
import os
import sys
from abc import ABC
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import warnings

from autofit import exc
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.checkpoint import resume_state_usable
from autofit.non_linear.search.abstract_search import NonLinearSearch
from autofit.non_linear.search.posterior_kind import WEIGHTED
from autofit.non_linear.samples.nest import SamplesNest
from autofit.non_linear.samples.adapter import RawSamples
from autofit.non_linear.search import capabilities as cap

if TYPE_CHECKING:
    from autofit.database.sqlalchemy_ import sa

logger = logging.getLogger(__name__)


def _fork_pool_cls(pools):
    """
    dynesty's `Pool` whose worker pool comes from the fit's `PoolFactory` (`pools`, the
    `ctx.pool` of the fit), so it is a "fork" pool (`autofit.non_linear.parallel.fork_context`;
    upstream hardcodes the default multiprocessing context, which Python 3.14 changed to
    "forkserver" on Linux), the JAX fork rule applies and the context shuts it down on
    every exit. `__enter__` mirrors `dynesty.pool.Pool.__enter__` exactly apart from where
    the pool comes from.
    """
    from dynesty import pool as dynesty_pool

    class ForkPool(dynesty_pool.Pool):
        def __enter__(self):
            initargs = (
                self.loglike_0,
                self.prior_transform_0,
                self.logl_args or (),
                self.logl_kwargs or {},
                self.ptform_args or (),
                self.ptform_kwargs or {},
            )
            self.pool = pools(
                self.njobs,
                initializer=dynesty_pool.initializer,
                initargs=initargs,
            )
            dynesty_pool.initializer(*initargs)
            return self

    return ForkPool


class _SingleCoreRun(Exception):
    """
    Control flow inside ``AbstractDynesty.run``: run dynesty without a pool.

    Raised when the fit is single-core by construction (``number_of_cores <= 1``,
    ``force_x1_cpu`` or a JAX analysis) and when the multiprocessing pool cannot be
    created (``RuntimeError`` from its ``__enter__``, e.g. on platforms with weak
    multiprocessing support), which is then its ``__cause__``. It is a private
    ``Exception`` subclass, not a ``RuntimeError``, so the ``except`` that catches it
    never swallows a ``RuntimeError`` raised while dynesty runs (a genuine dynesty
    error, or JAX's ``XlaRuntimeError``, which subclasses ``RuntimeError``): those
    propagate instead of silently restarting the run single-core.
    """


DYNESTY_FILENAME = "savestate.save"
"""
dynesty's checkpoint file, the search's resume state.
"""


def prior_transform(cube, model):
    phys_cube = model.vector_from_unit_vector(
        unit_vector=cube,
    )

    for i in range(len(phys_cube)):
        cube[i] = phys_cube[i]

    return cube


class AbstractDynesty(NonLinearSearch, ABC):
    samples_cls = SamplesNest

    # Static capabilities (see ``autofit.non_linear.search.capabilities``); mirrored
    # by ``search/registry.py``. Never identifier fields.
    jax_use = cap.JaxUse.OPTIONAL
    gradient = cap.Gradient.NONE
    batched = False
    honours_gradient_mode = False
    posterior_kind = cap.PosteriorKind.WEIGHTED
    produces_evidence = True
    resumable = True
    warm_start = cap.WarmStart.PROVIDER
    install_extra = ""
    upstream_url = "https://github.com/joshspeagle/dynesty"
    citation_keys = ("dynesty",)
    status = cap.Status.STABLE
    test_mode_budget = {"maxcall": 1}
    objective_target = cap.ObjectiveTarget(cap.ObjectiveQuantity.LOG_LIKELIHOOD, cap.CoordinateSpace.UNIT_CUBE)
    invalid_value = -1.0e99

    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        unique_tag: Optional[str] = None,
        bound: str = "multi",
        sample: str = "auto",
        bootstrap: Optional[int] = None,
        enlarge: Optional[float] = None,
        walks: int = 5,
        facc: float = 0.2,
        slices: int = 5,
        fmove: float = 0.9,
        max_move: int = 100,
        update_interval: Optional[float] = None,
        first_update: Optional[dict] = None,
        maxcall: Optional[int] = None,
        iterations_per_quick_update: int = None,
        iterations_per_full_update: int = None,
        number_of_cores: int = 1,
        silence: bool = False,
        force_x1_cpu: bool = False,
        use_jax_jit: Optional[bool] = None,
        session: Optional[sa.orm.Session] = None,
        **kwargs,
    ):
        """
        A Dynesty non-linear search.

        For a full description of Dynesty, checkout its GitHub and readthedocs webpages:

        https://github.com/joshspeagle/dynesty
        https://dynesty.readthedocs.io/en/latest/index.html

        Parameters
        ----------
        name
            The name of the search, controlling the last folder results are output.
        path_prefix
            The path of folders prefixing the name folder where results are output.
        unique_tag
            The name of a unique tag for this model-fit, which will be given a unique entry in the sqlite database
            and also acts as the folder after the path prefix and before the search name.
        bound
            Method used to approximately bound the prior using the current set of live points.
        sample
            Method used to sample uniformly within the likelihood constraint.
        maxcall
            Maximum number of likelihood evaluations.
        iterations_per_full_update
            The number of iterations performed between every Dynesty back-up.
        number_of_cores
            The number of cores sampling is performed using a Python multiprocessing Pool instance. A value of 1
            builds no pool and runs serially.
        silence
            If True, the default print output of the non-linear search is silenced.
        force_x1_cpu
            If True, force single-CPU mode even when number_of_cores > 1.
        session
            An SQLalchemy session instance so the results of the model-fit are written to an SQLite database.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            initializer=WEIGHTED.nested_initializer_for(kwargs.pop("initializer", None)),
            iterations_per_quick_update=iterations_per_quick_update,
            iterations_per_full_update=iterations_per_full_update,
            number_of_cores=number_of_cores,
            silence=silence,
            session=session,
            **kwargs,
        )

        self.bound = bound
        self.sample = sample
        self.bootstrap = bootstrap
        self.enlarge = enlarge
        self.walks = walks
        self.facc = facc
        self.slices = slices
        self.fmove = fmove
        self.max_move = max_move
        self.update_interval = update_interval
        self.first_update = first_update

        self.maxcall = maxcall
        self.force_x1_cpu = force_x1_cpu
        if use_jax_jit is not None:
            from autofit.non_linear.search.abstract_search import (
                warn_deprecated_jax_knob,
            )

            warn_deprecated_jax_knob(
                type(self).__name__,
                "use_jax_jit",
                "a JAX analysis is always jitted lazily (use_jax_jit=False still "
                "evaluates it eagerly until the argument is removed).",
            )

        self.use_jax_jit = use_jax_jit

        self.logger.debug(f"Creating {self.__class__.__name__} Search")

    def fitness_overrides(self, analysis) -> dict:
        """
        A JAX analysis is jitted lazily by the scalar objective `__call__` dispatches to;
        the deprecated `use_jax_jit=False` keeps its old meaning (eager JAX) for one
        release.
        """
        return {"compile": getattr(self, "use_jax_jit", None) is not False}

    @property
    def search_kwargs(self) -> Dict:
        """Shared search kwargs passed to both Static and Dynamic Dynesty samplers."""
        return {
            "bound": self.bound,
            "sample": self.sample,
            "bootstrap": self.bootstrap,
            "enlarge": self.enlarge,
            "walks": self.walks,
            "facc": self.facc,
            "slices": self.slices,
            "fmove": self.fmove,
            "max_move": self.max_move,
            "update_interval": self.update_interval,
            "first_update": self.first_update,
        }

    def _rstate_kwargs(self) -> Dict:
        """
        dynesty's ``rstate`` on backend stream 1 of the search-level seed when the search
        has one (``autofit.non_linear.search.seed``); empty otherwise, so an unseeded
        search keeps dynesty's own default generator.
        """
        from autofit.non_linear.search.seed import backend_generator

        rstate = backend_generator(self)
        return {} if rstate is None else {"rstate": rstate}

    def _sampler_kwargs(self) -> Dict:
        """
        The keyword arguments every dynesty sampler constructor (static or dynamic,
        pooled or not) receives beyond its positional set-up: ``search_kwargs`` plus
        ``rstate``, resolved once.

        Precedence: an ``rstate`` already in ``search_kwargs`` (e.g. a subclass that
        seeds dynesty's generator itself) wins; otherwise the search-level seed's
        backend stream 1 (``_rstate_kwargs``); otherwise none, so dynesty keeps its own
        default generator. Prefer the search-level ``seed=``, which also seeds the
        initializer's live points.
        """
        search_kwargs = dict(self.search_kwargs)
        if "rstate" in search_kwargs:
            return search_kwargs
        return {**search_kwargs, **self._rstate_kwargs()}

    @property
    def run_kwargs(self) -> Dict:
        """Run kwargs specific to each subclass, excluding maxcall."""
        raise NotImplementedError()

    def run(self, ctx):
        """
        Run dynesty (the ``run(ctx)`` hook; see ``docs/design/run_ctx.md``) and return the
        sampler.

        A multiprocessing pool is built only when the fit's effective core count is above 1. At
        `number_of_cores=1` the search runs fully serially with no pool, because a `Pool(1)` forces every
        likelihood call through a forked worker, which deadlocks in XLA compilation when the likelihood touches
        JAX and hangs forever if that worker dies (the same fix Nautilus received in #1442 / #1443; see #1630).
        Consequently, a run started with `number_of_cores=1` cannot be resumed with more cores (`check_pool`
        raises a `SearchException`), exactly as on the existing `force_x1_cpu` path.

        However, certain operating systems (e.g. Windows) do not support Python multiprocessing particularly well.
        If creating the pool raises a `RuntimeError`, a Dynesty object without a pool is created and used instead.
        Only the pool's creation falls back: a `RuntimeError` raised while Dynesty runs (including JAX's
        `XlaRuntimeError`) propagates rather than silently restarting the run single-core (`_SingleCoreRun`).

        dynesty resumes from its own native checkpoint (``savestate.save``).

        Parameters
        ----------
        ctx
            The fit's ``FitContext``.
        """
        model = ctx.model
        fitness = ctx.fitness

        # A savestate that cannot be restored (e.g. truncated by a killed job) is
        # moved aside with its search_internal folder and the search starts fresh.
        if ctx.paths.supports_checkpointing:
            checkpoint_exists = resume_state_usable(ctx.paths, self.resume_state)
        else:
            checkpoint_exists = False

        if checkpoint_exists:
            self.logger.info(
                "Resuming Dynesty non-linear search (previous samples found)."
            )
        else:
            self.logger.info(
                "Starting new Dynesty non-linear search (no previous samples found)."
            )

        # The fit's effective core count: the JAX fork rule makes it 1 for a JAX
        # analysis (`autofit.non_linear.parallel.pool`).
        number_of_cores = ctx.pool.number_of_cores
        is_jax = ctx.pool.is_jax

        finished = False

        while not finished:
            try:
                if number_of_cores <= 1 or self.force_x1_cpu or is_jax:
                    raise _SingleCoreRun()

                Pool = _fork_pool_cls(ctx.pool)

                with ExitStack() as stack:
                    # Only a failure to *create* the pool falls back to a single-core
                    # run; a RuntimeError raised while dynesty runs propagates.
                    try:
                        pool = stack.enter_context(
                            Pool(
                                njobs=number_of_cores,
                                loglike=fitness,
                                prior_transform=prior_transform,
                                logl_args=(model, fitness),
                                ptform_args=(model,),
                            )
                        )
                    except RuntimeError as error:
                        raise _SingleCoreRun() from error

                    search_internal = self.search_internal_from(
                        model=model,
                        fitness=fitness,
                        checkpoint_exists=checkpoint_exists,
                        pool=pool,
                        queue_size=number_of_cores,
                    )

                    finished = self.run_search_internal(search_internal=search_internal)

                    checkpoint_exists = True

            except _SingleCoreRun as e:
                if not checkpoint_exists:
                    if is_jax:
                        self.logger.info(
                            "Running Dynesty with JAX-jitted likelihood (single CPU, no pool)."
                        )
                    elif self.force_x1_cpu:
                        self.logger.info(
                            "Running Dynesty single-CPU per `force_x1_cpu=True` (no pool)."
                        )
                    elif number_of_cores <= 1:
                        self.logger.info(
                            "Running Dynesty single-CPU (number_of_cores=1, no pool)."
                        )
                    else:
                        self.logger.info(
                            f"""
                            The Dynesty multiprocessing pool could not be created ({e.__cause__!r}).

                            A single CPU non-multiprocessing Dynesty run is being performed.
                            """
                        )

                search_internal = self.search_internal_from(
                    model=model,
                    fitness=fitness,
                    checkpoint_exists=checkpoint_exists,
                    pool=None,
                    queue_size=None,
                )

                finished = self.run_search_internal(search_internal=search_internal)

                checkpoint_exists = True

            if not finished:
                ctx.update(search_internal)

        return search_internal

    def samples_info_from(self, search_internal=None):
        search_internal = search_internal or self.search_internal

        return {
            "log_evidence": np.max(search_internal.results.logz),
            "total_samples": int(np.sum(search_internal.results.ncall)),
            "total_accepted_samples": len(search_internal.results.logl),
            "time": self.timer.time if self.timer else None,
            "number_live_points": self.number_live_points,
        }

    def load_search_internal(self):
        """
        The archive (``search_internal.dill``, written when the fit completed) or,
        for a run that has not completed, the sampler restored from its resume state
        (``savestate.save``).
        """
        try:
            return super().load_search_internal()
        except FileNotFoundError:
            return self.search_internal

    def raw_samples_from(self, model, search_internal):
        """
        The dynesty run's samples, log likelihoods and nested-sampling weights
        (``exp(logwt - logz[-1])``) as raw samples.

        Parameters
        ----------
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        search_internal
            The dynesty sampler.
        """
        return RawSamples(
            parameters=search_internal.results.samples.tolist(),
            log_likelihood=list(search_internal.results.logl),
            weights=list(
                np.exp(
                    np.asarray(search_internal.results.logwt)
                    - search_internal.results.logz[-1]
                )
            ),
            info=self.samples_info_from(search_internal=search_internal),
            label="Dynesty",
        )

    @property
    def search_internal(self):
        """
        The sampler restored from the resume state (``savestate.save``) of a run that
        has not completed.
        """
        return self.resume_state.load(self.paths)

    def iterations_from(
        self, search_internal: "Union[NestedSampler, DynamicNestedSampler]"
    ) -> Tuple[int, int]:
        """
        Returns the next number of iterations that a dynesty call will use and the total number of iterations
        that have been performed so far.

        This is used so that the `iterations_per_full_update` input leads to on-the-fly output of dynesty results.

        It also ensures dynesty does not perform more samples than the `maxcall` input variable.

        Parameters
        ----------
        search_internal
            The Dynesty sampler (static or dynamic) which is run and performs nested sampling.

        Returns
        -------
        The next number of iterations that a dynesty run sampling will perform and the total number of iterations
        it has performed so far.
        """

        if not self.paths.supports_checkpointing:
            if self.maxcall is not None:
                return self.maxcall, self.maxcall
            # ``sys.maxsize`` rather than ``int(1e99)``: dynesty's dynamic sampler
            # computes ``maxcall - ncall`` where ``ncall`` is a numpy int64, and a
            # 99-digit Python int overflows that subtraction (OverflowError:
            # Python int too large to convert to C long). ``sys.maxsize`` is the
            # "unbounded" sentinel dynesty itself uses when ``maxcall=None``.
            return sys.maxsize, sys.maxsize

        total_iterations = self.total_calls_from(search_internal=search_internal)

        if self.maxcall is not None:
            iterations = self.maxcall - total_iterations

            return int(iterations), int(total_iterations)
        return self.iterations_per_full_update, int(total_iterations)

    def total_calls_from(self, search_internal) -> int:
        """
        The total number of likelihood calls the sampler has made so far, or 0 for a sampler that has not run.

        For the static sampler this is the sum of the per-sample call counts in `results.ncall`.
        `DynestyDynamic` overrides this to use the sampler's own `ncall` counter, which is what dynesty
        compares `maxcall` against: `results.ncall` omits the calls spent initialising each batch's live
        points, so it undercounts and would compare a different quantity with the cumulative budget.
        """
        try:
            return int(np.sum(search_internal.results.ncall))
        except AttributeError:
            return 0

    def maxcall_from(self, iterations: int, total_iterations: int) -> int:
        """
        The `maxcall` handed to `run_nested` for a chunk of `iterations` likelihood calls, given that
        `total_iterations` calls have already been made.

        The static sampler counts calls per `run_nested` call, so the budget is the increment itself.
        `DynestyDynamic` overrides this because the dynamic sampler counts cumulatively.
        """
        return iterations

    def chunk_kwargs(self, total_iterations: int) -> Dict:
        """
        Extra `run_nested` keyword arguments for a chunk, given the number of likelihood calls made before it.

        The static sampler continues a partially run sampler when `run_nested` is simply called again, so
        nothing is needed. `DynestyDynamic` overrides this to run the baseline to completion in the first
        chunk, because the dynamic sampler cannot continue a baseline run that was cut short.
        """
        return {}

    def chunk_is_finished(
        self, iterations: int, total_iterations: int, iterations_after_run: int
    ) -> bool:
        """
        Whether a `run_nested` chunk stopped on dynesty's own termination criterion rather than on its
        budget, given the budget (`iterations`) and the total calls before and after the chunk.

        The static sampler stops on its budget only once its per-run counter is strictly above `maxcall`, so
        a chunk that added no more calls than its budget converged. `DynestyDynamic` overrides this.
        """
        return iterations_after_run - total_iterations <= iterations

    def run_search_internal(
        self, search_internal: "Union[NestedSampler, DynamicNestedSampler]"
    ):
        """
        Run the Dynesty sampler, which could be either the static of dynamic sampler.

        The sampler is handed a budget of `iterations` likelihood calls (`maxcall`), which is either
        `iterations_per_full_update` (so on-the-fly output via `perform_update` happens between chunks) or the
        calls remaining under the search's global `maxcall`. Whether the search is finished is returned, using
        these criteria in order:

        1. No output paths (`NullPaths`): there are no on-the-fly updates, so one pass is always final.
        2. The global `maxcall` has been reached.
        3. Dynesty stopped itself inside the budget (`chunk_is_finished`). The two samplers count `maxcall`
           differently, so the budget handed to `run_nested` (`maxcall_from`) and this criterion are
           per-sampler:

           - The static sampler's per-run call counter resets on every `run_nested` call and it stops on the
             budget only once that counter is *strictly above* `maxcall`, while `sum(results.ncall)` grows by
             at least that counter. So `maxcall` is the per-chunk increment, and a pass that added no more
             calls than its budget stopped on dynesty's own termination criterion (`dlogz`, `maxiter`, ...).
           - The dynamic sampler compares `maxcall` against its *cumulative* call count (`self.ncall` carried
             over from previous `run_nested` calls). Handing it the per-chunk increment again on the second
             chunk is a budget it has already spent, so it returns without sampling, criterion 4 fires and a
             truncated baseline run is returned as the result. Nor can a baseline run that `maxcall` cut
             short be continued (the next call restarts it, and `resume=True` refuses after `RUN_DONE`).
             `DynestyDynamic` therefore runs the whole baseline in the first chunk (`maxbatch=0`, no budget),
             then chunks the batch phase by cumulative budget `total_iterations + iterations`, and a batch
             chunk is finished only when it stopped strictly inside that budget. See `DynestyDynamic.chunk_kwargs`.

           Under the default `iterations_per_full_update` (1e99) every converged run is finished in a single
           pass for both samplers, with no intermediate `perform_update`, checkpoint restore or second no-op
           `run_nested`.
        4. Legacy criterion: the pass performed no new likelihood calls (e.g. re-running an already converged
           sampler restored from its checkpoint).

        A pass that exhausted its budget (a finite `iterations_per_full_update` chunk) is not finished, so the
        search loops through `perform_update` and runs the next chunk.

        Parameters
        ----------
        search_internal
            The Dynesty sampler (static or dynamic) which is run and performs nested sampling.

        Returns
        -------
        True if the search is finished, False if another `run_nested` chunk is required.
        """

        iterations, total_iterations = self.iterations_from(
            search_internal=search_internal
        )

        if iterations > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                search_internal.run_nested(
                    maxcall=self.maxcall_from(
                        iterations=iterations, total_iterations=total_iterations
                    ),
                    print_progress=not self.silence,
                    checkpoint_file=self.checkpoint_file,
                    **self.run_kwargs,
                    **self.chunk_kwargs(total_iterations=total_iterations),
                )

        iterations_after_run = self.total_calls_from(search_internal=search_internal)

        if not self.paths.supports_checkpointing:
            return True

        if self.maxcall is not None and iterations_after_run >= self.maxcall:
            return True

        if iterations > 0 and self.chunk_is_finished(
            iterations=iterations,
            total_iterations=total_iterations,
            iterations_after_run=iterations_after_run,
        ):
            return True

        return bool(total_iterations == iterations_after_run)

    def write_uses_pool(self, uses_pool: bool) -> str:
        """
        If a Dynesty fit does not use a parallel pool, and is then resumed using one,
        this causes significant slow down.

        This file checks the original pool use so an exception can be raised to avoid this.

        If autofit is not outputting results to hard-disk (e.g. paths is `NullPaths`), this function is bypassed.
        """
        try:
            with open(self.paths.search_internal_path / "uses_pool.save", "w+") as f:
                if uses_pool:
                    f.write("True")
                else:
                    f.write("")
        except TypeError:
            pass

    def read_uses_pool(self) -> str:
        """
        If a Dynesty fit does not use a parallel pool, and is then resumed using one,
        this causes significant slow down.

        This file checks the original pool use so an exception can be raised to avoid this.
        """
        with open(self.paths.search_internal_path / "uses_pool.save", "r+") as f:
            return bool(f.read())

    @property
    def checkpoint_file(self) -> str:
        """
        The path to the file used for checkpointing.

        If autofit is not outputting results to hard-disk (e.g. paths is `NullPaths`), this function is bypassed.
        """
        try:
            return str(self.paths.search_internal_path / DYNESTY_FILENAME)
        except TypeError:
            pass

    def live_points_init_from(self, model, fitness):
        """
        By default, dynesty live points are generated via the sampler's in-built initialization.

        However, in test-mode this would take a long time to run, thus we overwrite the initial live points
        with quickly generated samplers from the initializer.

        Parameters
        ----------
        model
        fitness

        Returns
        -------

        """

        (
            unit_parameters,
            parameters,
            log_likelihood_list,
        ) = self.start_points(
            model=model,
            fitness=fitness,
            n=self.number_live_points,
            plot=False,
        )

        init_unit_parameters = np.zeros(
            shape=(self.number_live_points, model.prior_count)
        )
        init_parameters = np.zeros(shape=(self.number_live_points, model.prior_count))
        init_log_likelihood_list = np.zeros(shape=(self.number_live_points))

        for i in range(len(parameters)):
            init_unit_parameters[i, :] = np.asarray(unit_parameters[i])
            init_parameters[i, :] = np.asarray(parameters[i])
            init_log_likelihood_list[i] = np.asarray(log_likelihood_list[i])

        live_points = [
            init_unit_parameters,
            init_parameters,
            init_log_likelihood_list,
        ]

        blobs = np.asarray(self.number_live_points * [False])

        live_points.append(blobs)

        return live_points

    def search_internal_from(
        self,
        model: AbstractPriorModel,
        fitness,
        checkpoint_exists: bool,
        pool: Optional,
        queue_size: Optional[int],
    ):
        raise NotImplementedError()

    def check_pool(self, uses_pool: bool, pool):
        if (uses_pool and pool is None) or (not uses_pool and pool is not None):
            raise exc.SearchException(
                """
                A Dynesty sampler has been loaded and its pool type is not the same as the input pool type.

                This means that the original samples in dynesty were computed with or without a 
                multiprocessing pool, whereas the run is now trying to use a multiprocessing pool.

                This could indiciate the number of cores have change values or Python multiprocessing
                has been disabled and then enabled.
                """
            )

    @property
    def number_live_points(self):
        raise NotImplementedError()
