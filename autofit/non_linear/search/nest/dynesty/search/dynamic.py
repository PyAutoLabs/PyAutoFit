from __future__ import annotations

import sys
from typing import Dict, Optional

from autofit.mapper.prior_model.abstract import AbstractPriorModel

from .abstract import AbstractDynesty, prior_transform


class DynestyDynamic(AbstractDynesty):
    __identifier_fields__ = (
        "bound",
        "sample",
        "enlarge",
        "bootstrap",
        "walks",
        "facc",
        "slices",
        "fmove",
        "max_move"
    )

    def __init__(
            self,
            name: Optional[str] = None,
            path_prefix: Optional[str] = None,
            unique_tag: Optional[str] = None,
            nlive_init: int = 500,
            dlogz_init: float = 0.01,
            logl_max_init: float = float("inf"),
            maxcall_init: Optional[int] = None,
            maxiter: Optional[int] = None,
            maxiter_init: Optional[int] = None,
            n_effective: Optional[int] = None,
            iterations_per_quick_update: int = None,
            iterations_per_full_update: int = None,
            number_of_cores: int = 1,
            silence: bool = False,
            **kwargs
    ):
        """
        A Dynesty non-linear search, using a dynamically changing number of live points.

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
        nlive_init
            Number of live points used during the initial exploration phase.
        dlogz_init
            Stopping criterion for the initial baseline run.
        n_effective
            The minimum effective posterior sample size (ESS) of the whole run. After the baseline run, dynesty
            keeps adding batches of live points until its default stopping function judges the estimated ESS to
            have reached this value (`target_n_effective`). If `None` (the default), the argument is not passed
            and dynesty's own default applies, which in dynesty 2.x is `max(10000, ndim ** 2)`. It has no effect
            if a custom `stop_function` is used. It is not an identifier field, so changing it alone does not
            give the search a new output folder.
        iterations_per_full_update
            The number of iterations performed between update (e.g. output latest model to hard-disk, visualization).
        number_of_cores
            The number of cores sampling is performed using a Python multiprocessing Pool instance.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            iterations_per_quick_update=iterations_per_quick_update,
            iterations_per_full_update=iterations_per_full_update,
            number_of_cores=number_of_cores,
            silence=silence,
            **kwargs
        )

        self.nlive_init = nlive_init
        self.dlogz_init = dlogz_init
        self.logl_max_init = logl_max_init
        self.maxcall_init = maxcall_init
        self.n_effective = n_effective
        self.maxiter = maxiter
        self.maxiter_init = maxiter_init

        from autofit.non_linear.test_mode import is_test_mode
        if is_test_mode():
            self.apply_test_mode()

        self.logger.debug("Creating DynestyDynamic Search")

    @property
    def run_kwargs(self) -> Dict:
        run_kwargs = {
            "dlogz_init": self.dlogz_init,
            "logl_max_init": self.logl_max_init,
            "maxcall_init": self.maxcall_init,
            "maxiter": self.maxiter,
            "maxiter_init": self.maxiter_init,
            "nlive_init": self.nlive_init,
        }
        if self.n_effective is not None:
            run_kwargs["n_effective"] = self.n_effective
        return run_kwargs

    @property
    def chunked(self) -> bool:
        """
        Whether a run is split into `run_nested` chunks for on-the-fly output, i.e. whether
        `iterations_per_full_update` is finite rather than the inf-like default (1e99).
        """
        return self.iterations_per_full_update is not None and self.iterations_per_full_update < sys.maxsize

    def total_calls_from(self, search_internal) -> int:
        """
        The dynamic sampler's own cumulative call counter, the quantity dynesty compares the cumulative
        `maxcall` with. `results.ncall` undercounts it (it omits each batch's live-point initialisation),
        which would make a budget-limited batch chunk look converged. Falls back to the `results` count for
        objects without the counter.
        """
        try:
            return int(search_internal.ncall)
        except AttributeError:
            return super().total_calls_from(search_internal=search_internal)

    def maxcall_from(self, iterations: int, total_iterations: int) -> int:
        """
        The `maxcall` for a chunk of the dynamic sampler.

        dynesty's `DynamicNestedSampler.run_nested` compares `maxcall` with its *cumulative* call count
        (`self.ncall`, carried over from earlier calls), unlike the static sampler whose counter resets per
        call. The budget for a chunk is therefore the calls made so far plus the chunk size, capped at
        `sys.maxsize` (dynesty's own "unbounded" value) so an inf-like value cannot overflow the C long
        dynesty subtracts from.

        The first chunk of a chunked run is the exception: it is given the whole budget (`maxcall` if set,
        otherwise unbounded) because the baseline run must never be cut short, see `chunk_kwargs`.
        """
        if self.chunked and total_iterations == 0:
            return self.maxcall if self.maxcall is not None else sys.maxsize
        return int(min(total_iterations + iterations, sys.maxsize))

    def chunk_kwargs(self, total_iterations: int) -> Dict:
        """
        Extra `run_nested` keyword arguments for a chunk of the dynamic sampler.

        A dynamic run is a baseline nested sampling run followed by batches of live points. dynesty cannot
        continue a baseline run that `maxcall` cut short: the next `run_nested` call restarts it from scratch
        (`sample_initial` calls `reset()`), and `resume=True` refuses because the previous call ended in the
        `RUN_DONE` state. Batches, however, can be added across `run_nested` calls: once `self.base` is set the
        baseline is skipped and the batch loop continues from `self.batch`, evaluating the stopping criterion
        (including `n_effective`) first.

        A chunked run therefore completes the baseline in its first chunk (`maxbatch=0`, no call budget) and
        chunks only the batch phase, by cumulative `maxcall` (`maxcall_from`). Intermediate output starts
        after the baseline run.
        """
        if self.chunked and total_iterations == 0:
            return {"maxbatch": 0}
        return {}

    def chunk_is_finished(
        self, iterations: int, total_iterations: int, iterations_after_run: int
    ) -> bool:
        """
        The baseline-only first chunk of a chunked run is never finished: the batch phase (and so the
        `n_effective` stopping criterion) has not run yet. If it turns out to be satisfied already, the next
        chunk adds no calls and `run_search_internal`'s no-new-calls criterion finishes the search.

        A batch chunk stops sampling once its cumulative call count reaches the cumulative `maxcall` (it may
        land exactly on it), so it converged only if it stopped strictly inside that budget.
        """
        if self.chunked and total_iterations == 0:
            return False
        return iterations_after_run < self.maxcall_from(
            iterations=iterations, total_iterations=total_iterations
        )

    @property
    def search_internal(self):
        from dynesty.dynesty import DynamicNestedSampler
        return DynamicNestedSampler.restore(self.checkpoint_file)

    def search_internal_from(
            self,
            model: AbstractPriorModel,
            fitness,
            checkpoint_exists : bool,
            pool: Optional,
            queue_size: Optional[int]
    ):
        """
        Returns an instance of the Dynesty dynamic sampler set up using the input variables of this class.

        If no existing dynesty sampler exist on hard-disk (located via a `checkpoint_file`) a new instance is
        created with which sampler is performed. If one does exist, the dynesty `restore()` function is used to
        create the instance of the sampler.

        Dynesty samplers with a multiprocessing pool may be created by inputting a dynesty `Pool` object, however
        non pooled instances can also be created by passing `pool=None` and `queue_size=None`.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space.
        fitness
            An instance of the fitness class used to evaluate the likelihood of each model.
        pool
            A dynesty Pool object which performs likelihood evaluations over multiple CPUs.
        queue_size
            The number of CPU's over which multiprocessing is performed, determining how many samples are stored
            in the dynesty queue for samples.
        """
        from dynesty.dynesty import DynamicNestedSampler

        try:

            search_internal = DynamicNestedSampler.restore(
                fname=self.checkpoint_file,
                pool=pool
            )

            uses_pool = self.read_uses_pool()

            self.check_pool(uses_pool=uses_pool, pool=pool)

            return search_internal

        except (FileNotFoundError, TypeError):

            if pool is not None:

                self.write_uses_pool(uses_pool=True)

                return DynamicNestedSampler(
                    loglikelihood=pool.loglike,
                    prior_transform=pool.prior_transform,
                    ndim=model.prior_count,
                    queue_size=queue_size,
                    pool=pool,
                    **self.search_kwargs,
                )

            self.write_uses_pool(uses_pool=False)

            return DynamicNestedSampler(
                loglikelihood=fitness,
                prior_transform=prior_transform,
                ndim=model.prior_count,
                logl_args=[model, fitness],
                ptform_args=[model],
                **self.search_kwargs,
            )

    @property
    def number_live_points(self):
        return self.nlive_init
