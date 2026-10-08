from __future__ import annotations

import multiprocessing
import numpy as np
import logging
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Tuple, TYPE_CHECKING


from autofit import exc
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.mapper.prior.vectorized import PriorVectorized
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.parallel import fork_context
from autofit.non_linear.paths.null import NullPaths
from autofit.non_linear.search.nest import abstract_nest
from autofit.non_linear.samples.sample import Sample
from autofit.non_linear.samples.nest import SamplesNest
from autofit.non_linear.test_mode import is_test_mode
from autofit.non_linear.search import capabilities as cap

if TYPE_CHECKING:
    from autofit.database.sqlalchemy_ import sa


logger = logging.getLogger(__name__)


# How often `_LikelihoodWorkerPool.map` wakes up to check that the pool's
# workers are the ones it was built with. Read at call time (not bound at
# import) so tests can monkeypatch the module attribute.
LIKELIHOOD_POOL_POLL_INTERVAL = 1.0


class _LikelihoodWorkerPool:
    """
    A `multiprocessing.Pool` wrapper handed to nautilus as its likelihood pool
    (`pool_l`), whose workers already hold the fitness function in a global set
    by `nautilus.pool.initialize_worker` at fork time.

    Nautilus calls `pool.map(self.likelihood, args)` (`nautilus/sampler.py`),
    which for a plain pool object makes `multiprocessing` pickle the bound
    `Fitness.call_wrap` — and therefore the whole analysis, dataset, PSF
    convolver, sparse operator and adapt images — into every task chunk, for
    every likelihood call. This class ignores the function it is mapped with
    and dispatches `likelihood_worker` instead, which reads the worker-global
    likelihood, exactly as nautilus does for a `pool=<int>` argument (#1547).

    It also carries a **dead-worker watchdog**. `multiprocessing.Pool` silently
    replaces a worker that dies (`Pool._maintain_pool` -> `_repopulate_pool`)
    but never re-issues the task that worker was running, so a plain
    `Pool.map` blocks forever and the whole fit hangs to the wall clock — RAL
    job 342351_0 burned 27 hours after two likelihood workers segfaulted
    (#1608, and the same mechanism as #1547). `map` therefore dispatches via
    `map_async` and polls; on every poll it checks the exit code of every
    worker captured at construction. A replacement restores the worker
    *count*, but the original worker stays dead, so the loss is detectable,
    and the fit fails with a diagnosable `SearchException` (naming the PID and
    the signal) in seconds instead of hanging.
    """

    def __init__(self, pool):
        self._pool = pool
        # The construction-time worker `Process` objects. Watching *these* is
        # what makes a `_repopulate_pool` replacement detectable: the pool
        # restores the number of workers, so counting them proves nothing,
        # while an original worker that died stays dead. Holding the objects
        # (not just their PIDs) matters because the pool's maintenance thread
        # joins and drops a dead worker from `pool._pool` within ~0.1 s; the
        # retained object still reports its cached `exitcode` after that.
        #
        # `getattr` rather than `pool._pool` directly, so a pool-like test
        # double without a `_pool` attribute still works (the watchdog then
        # simply has nothing to watch).
        self._workers = list(getattr(pool, "_pool", []))

    @property
    def size(self):
        """
        The number of worker processes, read by `nautilus.pool.NautilusPool.size`.
        """
        return getattr(self._pool, "_processes", None)

    def map(self, func, iterable):
        # `func` is deliberately ignored: nautilus's likelihood pool is only ever
        # mapped with the likelihood, and dispatching the worker-global instead
        # avoids pickling the `Fitness` (and its analysis and dataset) into every
        # task chunk (#1547).
        from nautilus.pool import likelihood_worker

        result = self._pool.map_async(likelihood_worker, iterable)

        while True:
            try:
                return result.get(timeout=LIKELIHOOD_POOL_POLL_INTERVAL)
            except multiprocessing.TimeoutError:
                # Not done yet — which is the normal case for a long chunk of
                # likelihood evaluations, and also exactly what a hang looks
                # like. Only the worker check can tell them apart.
                self._check_workers_alive()

    def _check_workers_alive(self):
        """
        Raise if any worker the pool was built with is gone.

        `multiprocessing.Pool` replaces a dead worker behind our back and drops
        its in-flight task on the floor, so the `map_async` above would never
        complete. Detect the replacement and fail loudly instead.
        """
        dead = [worker for worker in self._workers if worker.exitcode is not None]

        if not dead:
            return

        def _describe(worker):
            code = worker.exitcode
            if code < 0:
                return f"pid {worker.pid} (exit code {code}, killed by signal {-code})"
            return f"pid {worker.pid} (exit code {code})"

        detail = ", ".join(_describe(worker) for worker in dead)

        self._pool.terminate()

        raise exc.SearchException(
            f"A Nautilus likelihood worker died mid-fit: {detail}.\n\n"
            f"A negative exit code is the signal that killed the worker "
            f"(-11 SIGSEGV, -9 SIGKILL — typically an out-of-memory kill on a "
            f"cluster).\n\n"
            f"`multiprocessing.Pool` has already replaced the worker, but it "
            f"never re-issues the task that worker was running, so the "
            f"`Pool.map` driving this fit would otherwise block forever and the "
            f"fit would hang to the wall clock rather than fail (RAL job "
            f"342351_0 hung 27 hours this way). Failing now instead.\n\n"
            f"Fix, either of:\n"
            f"  - run the search with number_of_cores=1, which builds no pool "
            f"at all;\n"
            f"  - get parallelism from a vectorised JAX likelihood instead, "
            f"`Analysis(use_jax=True)` — Nautilus then takes `fit_x1_cpu` with "
            f"vectorized=True and no pool."
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._pool.close()
        else:
            self._pool.terminate()
        self._pool.join()
        return False


class Nautilus(abstract_nest.AbstractNest):
    # Static capabilities (see ``autofit.non_linear.search.capabilities``); mirrored
    # by ``search/registry.py``. Never identifier fields.
    jax_use = cap.JaxUse.OPTIONAL
    gradient = cap.Gradient.NONE
    batched = True
    honours_gradient_mode = False
    posterior_kind = cap.PosteriorKind.WEIGHTED
    produces_evidence = True
    resumable = True
    warm_start = cap.WarmStart.PROVIDER
    install_extra = "optional"
    upstream_url = "https://github.com/johannesulf/nautilus"
    citation_keys = ("nautilus",)
    status = cap.Status.STABLE
    test_mode_budget = {"n_like_max": 1}
    objective_target = cap.ObjectiveTarget(cap.ObjectiveQuantity.LOG_LIKELIHOOD, cap.CoordinateSpace.UNIT_CUBE)
    invalid_value = -1.0e99

    __identifier_fields__ = (
        "n_live",
        "n_update",
        "enlarge_per_dim",
        "n_points_min",
        "split_threshold",
        "n_networks",
        "n_like_new_bound",
        "seed",
        "n_shell",
        "n_eff",
    )

    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        unique_tag: Optional[str] = None,
        n_live: int = 3000,
        n_update: Optional[int] = None,
        enlarge_per_dim: float = 1.1,
        n_points_min: Optional[int] = None,
        split_threshold: int = 100,
        n_networks: int = 4,
        n_batch: int = 100,
        n_like_new_bound: Optional[int] = None,
        vectorized: bool = False,
        seed: Optional[int] = None,
        f_live: float = 0.01,
        n_shell: int = 1,
        n_eff: int = 500,
        n_like_max: float = float("inf"),
        discard_exploration: bool = False,
        verbose: bool = True,
        iterations_per_quick_update: Optional[int] = None,
        iterations_per_full_update: int = None,
        number_of_cores: int = 1,
        silence: bool = False,
        force_x1_cpu: bool = False,
        session: Optional[sa.orm.Session] = None,
        use_jax_vmap: bool = True,
        **kwargs,
    ):
        """
        A Nautilus non-linear search.

        Nautilus is an optional requirement and must be installed manually via the command `pip install nautilus-sampler`.
        It is optional as it has certain dependencies which are generally straight forward to install.

        For a full description of Nautilus checkout its Github and documentation webpages:

        https://github.com/johannesulf/nautilus
        https://nautilus-sampler.readthedocs.io/en/stable/index.html

        Parameters
        ----------
        name
            The name of the search, controlling the last folder results are output.
        path_prefix
            The path of folders prefixing the name folder where results are output.
        unique_tag
            The name of a unique tag for this model-fit, which will be given a unique entry in the sqlite database
            and also acts as the folder after the path prefix and before the search name.
        n_live
            Number of live points used for sampling.
        n_batch
            Number of likelihood evaluations performed at each step.
        n_like_max
            Maximum number of likelihood evaluations before stopping.
        f_live
            Maximum fraction of evidence in the live set before terminating.
        n_eff
            Minimum effective sample size before stopping.
        iterations_per_full_update
            The number of iterations performed between update (e.g. output latest model to hard-disk, visualization).
        number_of_cores
            The number of cores sampling is performed using a Python multiprocessing Pool instance.

            Not used by expectation-propagation factor searches, which refuse
            `number_of_cores > 1` outright (`AbstractSearch.optimise`, human
            ruling 2026-09-09): a forked worker that dies is replaced by
            `multiprocessing.Pool` while its in-flight task is never re-issued,
            so the fit hangs instead of failing. EP parallelism comes from a
            JAX-vectorised likelihood (`Analysis(use_jax=True)`), which takes
            the pool-free `fit_x1_cpu` path.
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
            iterations_per_full_update=iterations_per_full_update,
            iterations_per_quick_update=iterations_per_quick_update,
            number_of_cores=number_of_cores,
            silence=silence,
            session=session,
            **kwargs,
        )

        self.n_live = n_live
        self.n_update = n_update
        self.enlarge_per_dim = enlarge_per_dim
        self.n_points_min = n_points_min
        self.split_threshold = split_threshold
        self.n_networks = n_networks
        self.n_batch = n_batch
        self.n_like_new_bound = n_like_new_bound
        self.vectorized = vectorized
        self.seed = seed

        self.f_live = f_live
        self.n_shell = n_shell
        self.n_eff = n_eff
        self.n_like_max = n_like_max
        self.discard_exploration = discard_exploration
        self.verbose = verbose

        self.force_x1_cpu = force_x1_cpu
        self.use_jax_vmap = use_jax_vmap

        if is_test_mode():
            self.apply_test_mode()

        self.logger.debug("Creating Nautilus Search")

    def apply_test_mode(self):
        logger.warning(
            "TEST MODE 1 (reduced iterations): Sampler will run with "
            "minimal iterations for faster completion."
        )
        self.n_like_max = 1

    def _fit(self, model: AbstractPriorModel, analysis):
        """
        Fit a model using the search and the Analysis class which contains the data and returns the log likelihood from
        instances of the model, which the `NonLinearSearch` seeks to maximize.

        Parameters
        ----------
        model : ModelMapper
            The model which generates instances for different points in parameter space.
        analysis : Analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the `NonLinearSearch` maximizes.

        Returns
        -------
        A result object comprising the Samples object that includes the maximum log likelihood instance and full
        set of accepted ssamples of the fit.
        """

        if not isinstance(self.paths, NullPaths):
            checkpoint_exists = Path(self.checkpoint_file).exists()
        else:
            checkpoint_exists = False

        if checkpoint_exists:
            self.logger.info(
                "Resuming Nautilus non-linear search (previous samples found)."
            )

        else:
            self.logger.info(
                "Starting new Nautilus non-linear search (no previous samples found)."
            )

        if self.force_x1_cpu or analysis.is_jax:

            fitness = Fitness(
                model=model,
                analysis=analysis,
                paths=self.paths,
                fom_is_log_likelihood=True,
                resample_figure_of_merit=-1.0e99,
                iterations_per_quick_update=self.iterations_per_quick_update,
                background_quick_update=self.quick_update_background,
                live_visual_update=self.live_visual_update,
                use_jax_vmap=self.use_jax_vmap,
                batch_size=self.n_batch,
            )

            search_internal = self.fit_x1_cpu(
                fitness=fitness,
                model=model,
                analysis=analysis,
            )

        else:

            fitness = Fitness(
                model=model,
                analysis=analysis,
                paths=self.paths,
                fom_is_log_likelihood=True,
                resample_figure_of_merit=-1.0e99,
                iterations_per_quick_update=self.iterations_per_quick_update,
                background_quick_update=self.quick_update_background,
                live_visual_update=self.live_visual_update,
            )

            search_internal = self.fit_multiprocessing(
                fitness=fitness,
                model=model,
                analysis=analysis,
            )

        return search_internal, fitness

    @property
    def sampler_cls(self):
        try:
            from nautilus import Sampler

            return Sampler
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "\n--------------------\n"
                "You are attempting to perform a model-fit using Nautilus. \n\n"
                "However, the optional library Nautilus (https://nautilus-sampler.readthedocs.io/en/stable/index.html) is "
                "not installed.\n\n"
                "Install it via the command `pip install nautilus-sampler==1.0.5`.\n\n"
                "----------------------"
            )

    @property
    def checkpoint_file(self):
        """
        The path to the file used for checkpointing.

        If autofit is not outputting results to hard-disk (e.g. paths is `NullPaths`), this function is bypassed.
        """
        try:
            return self.paths.search_internal_path / "checkpoint.hdf5"
        except TypeError:
            pass

    def fit_x1_cpu(self, fitness, model, analysis):
        """
        Perform the non-linear search, using one CPU core.

        This is used if the likelihood function calls external libraries that cannot be parallelized or use
        threading in a way that conflicts with the parallelization of the non-linear search.

        Parameters
        ----------
        fitness
            The function which takes a model instance and returns its log likelihood via the Analysis class
        model
            The model which maps parameters chosen via the non-linear search (e.g. via the priors or sampling) to
            instances of the model, which are passed to the fitness function.
        analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the search maximizes.
        """

        if analysis.is_jax:
            self.logger.info(
                "Running search with JAX vectorization (parallelization handled by JAX)."
            )
        else:
            self.logger.info(
                "Running search where parallelization is disabled."
            )

        search_internal = self.sampler_cls(
            prior=PriorVectorized(model=model),
            likelihood=fitness.call_wrap,
            n_dim=model.prior_count,
            filepath=self.checkpoint_file,
            pool=None,
            vectorized=fitness.use_jax_vmap,
            n_live=self.n_live,
            n_update=self.n_update,
            enlarge_per_dim=self.enlarge_per_dim,
            n_points_min=self.n_points_min,
            split_threshold=self.split_threshold,
            n_networks=self.n_networks,
            n_batch=self.n_batch,
            n_like_new_bound=self.n_like_new_bound,
            seed=self.seed,
        )

        return self.call_search(search_internal=search_internal, model=model, analysis=analysis, fitness=fitness)

    def fit_multiprocessing(self, fitness, model, analysis):
        """
        Perform the non-linear search, using multiple CPU cores parallelized via Python's multiprocessing module.

        This uses PyAutoFit's sneaky pool class, which allows us to use the multiprocessing module in a way that plays
        nicely with the non-linear search (e.g. exception handling, keyboard interupts, etc.).

        Multiprocessing parallelization can only parallelize across multiple cores on a single device, it cannot be
        distributed across multiple devices or computing nodes. For that, use the `fit_mpi` method.

        Parameters
        ----------
        fitness
            The function which takes a model instance and returns its log likelihood via the Analysis class
        model
            The model which maps parameters chosen via the non-linear search (e.g. via the priors or sampling) to
            instances of the model, which are passed to the fitness function.
        analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the search maximizes.
        """
        # A pool object is passed rather than pool=<int> so the pool uses the
        # "fork" start method (see autofit.non_linear.parallel.fork_context) —
        # nautilus builds its internal pools from the default context.
        #
        # For a single core no pool may be created at all: nautilus treats
        # pool=None (and the int 1) as fully serial, whereas a Pool(1) object
        # forces every likelihood call into a forked worker — which deadlocks
        # in XLA compilation when the likelihood touches JAX, since a forked
        # child of a JAX-initialized parent cannot compile.
        #
        # The pool is passed as the tuple `(pool, None)` rather than as a bare
        # `pool`, because nautilus otherwise uses the one pool for likelihood
        # evaluations *and* for its own sampler calculations — in particular
        # neural bound training (`nautilus/neural.py`,
        # `NeuralNetworkEmulator.train` -> `pool.map`). A worker that dies
        # mid-fit (e.g. a per-job OOM kill on a cluster) is silently replaced
        # by `multiprocessing.Pool`, but the task it was running is never
        # re-issued, so `map` blocks forever and the fit hangs — observed on
        # RAL 2026-08-29 (#1547). `(pool, None)` keeps likelihoods parallel
        # (`pool_l`) and makes bound training serial in the parent (`pool_s`),
        # which is negligible: 4 small MLPs fit on at most a few hundred
        # points.
        #
        # The pool is also built with nautilus's own `initialize_worker`, and is
        # handed over wrapped in `_LikelihoodWorkerPool`, so that the likelihood
        # is dispatched as `likelihood_worker`. Before a pool object was passed
        # in place of `pool=<int>` (e6279c53f, #1439), nautilus did this itself;
        # with a pool object it instead maps the bound `Fitness.call_wrap`, so
        # `multiprocessing` pickles the entire analysis (dataset, PSF convolver,
        # sparse operator, adapt images) into every task chunk and each worker
        # unpickles a fresh copy for every likelihood call. Worker memory then
        # grows by a few MB per call (8-core RAL runs reached the 96 GB cgroup
        # limit) while the parent starves the pool serialising it (#1547).
        if self.number_of_cores <= 1:
            pool_context = nullcontext(None)
        else:
            from nautilus.pool import initialize_worker

            pool_context = _LikelihoodWorkerPool(
                fork_context().Pool(
                    self.number_of_cores,
                    initializer=initialize_worker,
                    initargs=(fitness.call_wrap,),
                )
            )

        with pool_context as pool:
            search_internal = self.sampler_cls(
                prior=PriorVectorized(model=model),
                likelihood=fitness.call_wrap,
                n_dim=model.prior_count,
                filepath=self.checkpoint_file,
                pool=(pool, None) if pool is not None else None,
                n_live=self.n_live,
                n_update=self.n_update,
                enlarge_per_dim=self.enlarge_per_dim,
                n_points_min=self.n_points_min,
                split_threshold=self.split_threshold,
                n_networks=self.n_networks,
                n_batch=self.n_batch,
                n_like_new_bound=self.n_like_new_bound,
                vectorized=self.vectorized,
                seed=self.seed,
            )

            search_internal = self.call_search(
                search_internal=search_internal,
                model=model,
                analysis=analysis,
                fitness=fitness
            )

        # Drop the pool references so their finalizers don't fire at interpreter
        # shutdown (after pickle has been torn down, causing AttributeError on
        # Pool.__del__).
        for pool_attr in ("pool_l", "pool_s"):
            pool = getattr(search_internal, pool_attr, None)
            if pool is not None:
                try:
                    pool.close()
                    pool.join()
                except Exception:
                    pass
                setattr(search_internal, pool_attr, None)

        return search_internal

    def call_search(self, search_internal, model, analysis, fitness):
        """
        The x1 CPU and multiprocessing searches both call this function to perform the non-linear search.

        This function calls the search a reduced number of times, corresponding to the `iterations_per_full_update` of the
        search. This allows the search to output results on-the-fly, for example writing to the hard-disk the latest
        model and samples.

        It tracks how often to do this update alongside the maximum number of iterations the search will perform.
        This ensures that on-the-fly output is performed at regular intervals and that the search does not perform more
        iterations than the `n_like_max` input variable.

        Parameters
        ----------
        search_internal
            The single CPU or multiprocessing search which is run and performs nested sampling.
        model
            The model which maps parameters chosen via the non-linear search (e.g. via the priors or sampling) to
            instances of the model, which are passed to the fitness function.
        analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the search maximizes.
        """

        finished = False

        minimum_iterations_per_full_updates = 3 * self.n_live

        # The effective update interval is local to this call: the search's own
        # `iterations_per_full_update` is a constructor argument and is never mutated.
        iterations_per_full_update = self.iterations_per_full_update

        if iterations_per_full_update < minimum_iterations_per_full_updates:

            iterations_per_full_update = minimum_iterations_per_full_updates

            logger.info(
                f"""
                The number of iterations_per_full_update is less than 3 times the number of live points, which can cause
                issues where Nautilus loses sampling information due to stopping to output results. The number of
                iterations per update has been increased to 3 times the number of live points, therefore a value
                of {minimum_iterations_per_full_updates}.

                To remove this warning, increase the number of iterations_per_full_update to three or more times the
                number of live points.
                """
            )

        while not finished:

            iterations, total_iterations = self.iterations_from(
                search_internal=search_internal,
                iterations_per_full_update=iterations_per_full_update,
            )

            converged = search_internal.run(
                f_live=self.f_live,
                n_shell=self.n_shell,
                n_eff=self.n_eff,
                discard_exploration=self.discard_exploration,
                verbose=self.verbose,
                n_like_max=iterations,
            )

            iterations_after_run = self.iterations_from(search_internal=search_internal)[1]

            if (
                converged
                or total_iterations == iterations_after_run
                or (
                    self.n_like_max is not None
                    and iterations_after_run >= self.n_like_max
                )
            ):
                finished = True

            if not finished:

                self.perform_update(
                    model=model,
                    analysis=analysis,
                    during_analysis=True,
                    fitness=fitness,
                    search_internal=search_internal
                )

        return search_internal

    def iterations_from(
        self, search_internal, iterations_per_full_update: Optional[float] = None
    ) -> Tuple[int, int]:
        """
        Return the next cumulative likelihood-call budget and calls performed.

        Nautilus budgets likelihood evaluations, not posterior samples. Batches
        can overshoot a budget, so advance from the sampler's actual call count.

        Parameters
        ----------
        search_internal
            The Nautilus sampler.
        iterations_per_full_update
            The likelihood calls between full updates; defaults to the search's
            `iterations_per_full_update`. `call_search` passes its effective value,
            which is raised to at least three times the number of live points.

        Returns
        -------
        The next cumulative likelihood-call limit and current likelihood count.
        """

        total_iterations = search_internal.n_like

        if isinstance(self.paths, NullPaths):
            if self.n_like_max is not None and self.n_like_max != float("inf"):
                return int(self.n_like_max), total_iterations
            return int(1e99), total_iterations

        if iterations_per_full_update is None:
            iterations_per_full_update = self.iterations_per_full_update

        iterations = total_iterations + iterations_per_full_update

        if self.n_like_max is not None and self.n_like_max != float("inf"):
            if iterations > self.n_like_max:
                iterations = int(self.n_like_max)

        return iterations, total_iterations

    def output_search_internal(self, search_internal):
        """
        Output the sampler results to hard-disk in their internal format.

        The multiprocessing `Pool` object cannot be pickled and thus the sampler cannot be saved to hard-disk. This
        function therefore extracts the necessary information from the sampler and saves it to hard-disk.

        Parameters
        ----------
        sampler
            The nautilus sampler object containing the results of the model-fit.
        """

        pool_l = search_internal.pool_l
        pool_s = search_internal.pool_s

        search_internal.pool_l = None
        search_internal.pool_s = None

        self.paths.save_search_internal(
            obj=search_internal,
        )

        search_internal.pool_l = pool_l
        search_internal.pool_s = pool_s

        try:
            os.remove(self.checkpoint_file)
        except (TypeError, FileNotFoundError):
            pass

    def samples_info_from(self, search_internal=None):
        return {
            "log_evidence": search_internal.log_z,
            "total_samples": int(search_internal.n_like),
            "total_accepted_samples": int(search_internal.n_like),
            "time": self.timer.time if self.timer else None,
            "number_live_points": int(search_internal.n_live),
        }

    def samples_via_internal_from(
        self, model: AbstractPriorModel, search_internal=None
    ):
        """
        Returns a `Samples` object from the nautilus internal results.

        The samples contain all information on the parameter space sampling (e.g. the parameters,
        log likelihoods, etc.).

        The internal search results are converted from the native format used by the search to lists of values
        (e.g. `parameter_lists`, `log_likelihood_list`).

        Parameters
        ----------
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        """

        if search_internal is None:
            search_internal = self.paths.load_search_internal()

        parameters, log_weights, log_likelihoods = search_internal.posterior()

        parameter_lists = parameters.tolist()
        log_likelihood_list = log_likelihoods.tolist()
        weight_list = np.exp(log_weights).tolist()

        log_prior_list = [
            sum(model.log_prior_list_from_vector(vector=vector))
            for vector in parameter_lists
        ]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameter_lists,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list,
        )

        return SamplesNest(
            model=model,
            sample_list=sample_list,
            samples_info=self.samples_info_from(search_internal=search_internal),
        )

    @property
    def batch_size(self):
        return self.n_batch
