import numpy as np
import logging
import os
import sys
from typing import Dict, Optional, Tuple

from autofit.database.sqlalchemy_ import sa

from autoconf import conf
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.mapper.prior.vectorized import PriorVectorized
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.paths.null import NullPaths
from autofit.non_linear.search.nest import abstract_nest
from autofit.non_linear.samples.sample import Sample
from autofit.non_linear.samples.nest import SamplesNest


logger = logging.getLogger(__name__)

class Nautilus(abstract_nest.AbstractNest):
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
        iterations_per_quick_update: Optional[int] = None,
        iterations_per_full_update: int = None,
        number_of_cores: int = None,
        session: Optional[sa.orm.Session] = None,
        **kwargs
    ):
        """
        A Nautilus non-linear search.

        Nautilus is an optional requirement and must be installed manually via the command `pip install ultranest`.
        It is optional as it has certain dependencies which are generally straight forward to install (e.g. Cython).

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
        iterations_per_full_update
            The number of iterations performed between update (e.g. output latest model to hard-disk, visualization).
        number_of_cores
            The number of cores sampling is performed using a Python multiprocessing Pool instance.
        session
            An SQLalchemy session instance so the results of the model-fit are written to an SQLite database.
        """

        number_of_cores = (
            self._config("parallel", "number_of_cores")
            if number_of_cores is None
            else number_of_cores
        )

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            iterations_per_full_update=iterations_per_full_update,
            iterations_per_quick_update=iterations_per_quick_update,
            number_of_cores=number_of_cores,
            session=session,
            **kwargs,
        )

        self.logger.debug("Creating Nautilus Search")

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
            checkpoint_exists = os.path.exists(self.checkpoint_file)
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

        if (
            self.config_dict.get("force_x1_cpu")
            or self.kwargs.get("force_x1_cpu")
            or analysis._use_jax
        ):

            fitness = Fitness(
                model=model,
                analysis=analysis,
                paths=self.paths,
                fom_is_log_likelihood=True,
                resample_figure_of_merit=-1.0e99,
                iterations_per_quick_update=self.iterations_per_quick_update,
                use_jax_vmap=True,
                batch_size=self.config_dict_search["n_batch"],
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
                iterations_per_quick_update=self.iterations_per_quick_update
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

        self.logger.info(
            """
            Running search where parallelization is disabled.
            """
        )

        config_dict = self.config_dict_search
        try:
            config_dict.pop("vectorized")
        except KeyError:
            pass

        search_internal = self.sampler_cls(
            prior=PriorVectorized(model=model),
            likelihood=fitness.call_wrap,
            n_dim=model.prior_count,
            filepath=self.checkpoint_file,
            pool=None,
            vectorized=True,
            **config_dict,
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
        search_internal = self.sampler_cls(
            prior=PriorVectorized(model=model),
            likelihood=fitness.call_wrap,
            n_dim=model.prior_count,
            filepath=self.checkpoint_file,
            pool=self.number_of_cores,
            **self.config_dict_search,
        )

        search_internal = self.call_search(
            search_internal=search_internal,
            model=model,
            analysis=analysis,
            fitness=fitness
        )

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

        minimum_iterations_per_full_updates = 3 * self.config_dict_search["n_live"]

        if self.iterations_per_full_update < minimum_iterations_per_full_updates:

            self.iterations_per_full_update = minimum_iterations_per_full_updates

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
                search_internal=search_internal
            )

            config_dict_run = {
                key: value
                for key, value in self.config_dict_run.items()
                if key != "n_like_max"
            }

            search_internal.run(
                **config_dict_run,
                n_like_max=iterations,
            )

            iterations_after_run = self.iterations_from(search_internal=search_internal)[1]

            if (
                    total_iterations == iterations_after_run
                    or iterations_after_run == self.config_dict_run["n_like_max"]
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
        self, search_internal
    ) -> Tuple[int, int]:
        """
        Returns the next number of iterations that a dynesty call will use and the total number of iterations
        that have been performed so far.

        This is used so that the `iterations_per_full_update` input leads to on-the-fly output of dynesty results.

        It also ensures dynesty does not perform more samples than the `n_like_max` input variable.

        Parameters
        ----------
        search_internal
            The Dynesty sampler (static or dynamic) which is run and performs nested sampling.

        Returns
        -------
        The next number of iterations that a dynesty run sampling will perform and the total number of iterations
        it has performed so far.
        """

        if isinstance(self.paths, NullPaths):
            n_like_max = self.config_dict_run.get("n_like_max")

            if n_like_max is not None:
                return n_like_max, n_like_max
            return int(1e99), int(1e99)

        try:
            total_iterations = len(search_internal.posterior()[1])
        except ValueError:
            total_iterations = 0

        iterations = total_iterations + self.iterations_per_full_update

        if self.config_dict_run["n_like_max"] is not None:
            if iterations > self.config_dict_run["n_like_max"]:
                iterations = self.config_dict_run["n_like_max"]

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
            "log_evidence": search_internal.evidence(),
            "total_samples": int(search_internal.n_like),
            "total_accepted_samples": int(search_internal.n_like),
            "time": self.timer.time if self.timer else None,
            "number_live_points": int(search_internal.n_live),
        }

    def samples_via_internal_from(
        self, model: AbstractPriorModel, search_internal=None
    ):
        """
        Returns a `Samples` object from the ultranest internal results.

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
        return self.config_dict_search.get("n_batch")

    @property
    def config_dict(self):
        return conf.instance["non_linear"]["nest"][self.__class__.__name__]

    def config_dict_test_mode_from(self, config_dict : Dict) -> Dict:
        """
        Returns a configuration dictionary for test mode meaning that the sampler terminates as quickly as possible.

        Entries which set the total number of samples of the sampler (e.g. maximum calls, maximum likelihood
        evaluations) are reduced to low values meaning it terminates nearly immediately.

        Parameters
        ----------
        config_dict
            The original configuration dictionary for this sampler which includes entries controlling how fast the
            sampler terminates.

        Returns
        -------
        A configuration dictionary where settings which control the sampler's number of samples are reduced so it
        terminates as quickly as possible.
        """
        return {
            **config_dict,
            "n_like_max": 1,
        }