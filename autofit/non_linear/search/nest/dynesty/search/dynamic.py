from __future__ import annotations
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
        self.maxiter = maxiter
        self.maxiter_init = maxiter_init

        from autofit.non_linear.test_mode import is_test_mode
        if is_test_mode():
            self.apply_test_mode()

        self.logger.debug("Creating DynestyDynamic Search")

    @property
    def run_kwargs(self) -> Dict:
        return {
            "dlogz_init": self.dlogz_init,
            "logl_max_init": self.logl_max_init,
            "maxcall_init": self.maxcall_init,
            "maxiter": self.maxiter,
            "maxiter_init": self.maxiter_init,
            "nlive_init": self.nlive_init,
        }

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
