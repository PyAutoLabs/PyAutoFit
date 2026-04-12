from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union


from autofit.database.sqlalchemy_ import sa

from autofit.mapper.prior_model.abstract import AbstractPriorModel

from .abstract import AbstractDynesty, prior_transform

class DynestyStatic(AbstractDynesty):
    __identifier_fields__ = (
        "nlive",
        "bound",
        "sample",
        "bootstrap",
        "enlarge",
        "walks",
        "facc",
        "slices",
        "fmove",
        "max_move",
    )

    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[Union[str, Path]] = None,
        unique_tag: Optional[str] = None,
        nlive: int = 50,
        dlogz: Optional[float] = None,
        maxiter: Optional[int] = None,
        logl_max: float = float("inf"),
        iterations_per_quick_update: int = None,
        iterations_per_full_update: int = None,
        number_of_cores: int = 1,
        silence: bool = False,
        session: Optional[sa.orm.Session] = None,
        **kwargs,
    ):
        """
        A Dynesty `NonLinearSearch` using a static number of live points.

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
        nlive
            Number of live points used for sampling.
        dlogz
            Stopping criterion: iteration stops when the estimated contribution of the remaining prior volume
            to the total evidence falls below this threshold.
        maxiter
            Maximum number of iterations.
        logl_max
            Maximum log-likelihood value allowed.
        iterations_per_full_update
            The number of iterations performed between update (e.g. output latest model to hard-disk, visualization).
        number_of_cores
            The number of cores sampling is performed using a Python multiprocessing Pool instance.
        session
            An SQLalchemy session instance so the results of the model-fit are written to an SQLite database.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            iterations_per_quick_update=iterations_per_quick_update,
            iterations_per_full_update=iterations_per_full_update,
            number_of_cores=number_of_cores,
            silence=silence,
            session=session,
            **kwargs,
        )

        self.nlive = nlive
        self.dlogz = dlogz
        self.maxiter = maxiter
        self.logl_max = logl_max

        from autofit.non_linear.test_mode import is_test_mode
        if is_test_mode():
            self.apply_test_mode()

    @property
    def run_kwargs(self) -> Dict:
        return {
            "dlogz": self.dlogz,
            "maxiter": self.maxiter,
            "logl_max": self.logl_max,
        }

    @property
    def search_internal(self):
        from dynesty import NestedSampler as StaticSampler
        return StaticSampler.restore(self.checkpoint_file)

    def search_internal_from(
        self,
        model: AbstractPriorModel,
        fitness,
        checkpoint_exists: bool,
        pool: Optional,
        queue_size: Optional[int],
    ):
        """
        Returns an instance of the Dynesty static sampler set up using the input variables of this class.

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
        from dynesty import NestedSampler as StaticSampler

        if checkpoint_exists:
            search_internal = StaticSampler.restore(
                fname=self.checkpoint_file, pool=pool
            )

            uses_pool = self.read_uses_pool()

            self.check_pool(uses_pool=uses_pool, pool=pool)

            return search_internal

        else:
            live_points = self.live_points_init_from(model=model, fitness=fitness)

            if pool is not None:
                self.write_uses_pool(uses_pool=True)
                return StaticSampler(
                    loglikelihood=pool.loglike,
                    prior_transform=pool.prior_transform,
                    ndim=model.prior_count,
                    live_points=live_points,
                    queue_size=queue_size,
                    pool=pool,
                    nlive=self.nlive,
                    **self.search_kwargs,
                )

            self.write_uses_pool(uses_pool=False)
            return StaticSampler(
                loglikelihood=fitness,
                prior_transform=prior_transform,
                ndim=model.prior_count,
                logl_args=[model, fitness],
                ptform_args=[model],
                live_points=live_points,
                nlive=self.nlive,
                **self.search_kwargs,
            )

    @property
    def number_live_points(self):
        return self.nlive
