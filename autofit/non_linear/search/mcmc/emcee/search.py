import os
from typing import Dict, Optional

import numpy as np

from autoconf import conf

from autofit.database.sqlalchemy_ import sa
from autofit.mapper.model_mapper import ModelMapper
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.initializer import Initializer
from autofit.non_linear.search.mcmc.abstract_mcmc import AbstractMCMC
from autofit.non_linear.search.mcmc.auto_correlations import AutoCorrelationsSettings
from autofit.non_linear.search.mcmc.auto_correlations import AutoCorrelations
from autofit.non_linear.test_mode import is_test_mode
from autofit.non_linear.samples.sample import Sample
from autofit.non_linear.samples.mcmc import SamplesMCMC


class Emcee(AbstractMCMC):
    __identifier_fields__ = ("nwalkers",)

    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        unique_tag: Optional[str] = None,
        initializer: Optional[Initializer] = None,
        auto_correlation_settings=AutoCorrelationsSettings(),
        iterations_per_quick_update: int = None,
        iterations_per_full_update: int = None,
        number_of_cores: int = None,
        session: Optional[sa.orm.Session] = None,
        **kwargs,
    ):
        """
        An Emcee non-linear search.

        For a full description of Emcee, checkout its Github and readthedocs webpages:

        https://github.com/dfm/emcee

        https://emcee.readthedocs.io/en/stable/

        If you use `Emcee` as part of a published work, please cite the package following the instructions under the
        *Attribution* section of the GitHub page.

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
        auto_correlation_settings
            Customizes and performs auto correlation calculations performed during and after the search.
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
            initializer=initializer,
            auto_correlation_settings=auto_correlation_settings,
            iterations_per_quick_update=iterations_per_quick_update,
            iterations_per_full_update=iterations_per_full_update,
            number_of_cores=number_of_cores,
            session=session,
            **kwargs,
        )

        self.logger.debug("Creating Emcee Search")

        # TODO : Emcee visualization tools rely on the .hdf file and thus require that the search internal is
        # TODO : On hard-disk, which this forces to occur.

        conf.instance["output"]["search_internal"] = True

    def _fit(self, model: AbstractPriorModel, analysis):
        """
        Fit a model using Emcee and the Analysis class which contains the data and returns the log likelihood from
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
        A result object comprising the Samples object that inclues the maximum log likelihood instance and full
        chains used by the fit.
        """
        import emcee

        fitness = Fitness(
            model=model,
            analysis=analysis,
            paths=self.paths,
            fom_is_log_likelihood=False,
            resample_figure_of_merit=-np.inf,
        )

        pool = self.make_sneaky_pool(fitness)

        try:
            backend = emcee.backends.HDFBackend(filename=self.backend_filename)
        except TypeError:
            backend = None

        search_internal = emcee.EnsembleSampler(
            nwalkers=self.config_dict_search["nwalkers"],
            ndim=model.prior_count,
            log_prob_fn=fitness.call_wrap,
            backend=backend,
            pool=pool,
        )

        try:
            state = search_internal.get_last_sample()
            samples = self.samples_from(model=model, search_internal=search_internal)

            total_iterations = search_internal.iteration

            if samples.converged:
                iterations_remaining = 0
            else:
                iterations_remaining = self.config_dict_run["nsteps"] - total_iterations

                self.logger.info(
                    "Resuming Emcee non-linear search (previous samples found)."
                )

        except AttributeError:
            (
                unit_parameter_lists,
                parameter_lists,
                log_posterior_list,
            ) = self.initializer.samples_from_model(
                total_points=search_internal.nwalkers,
                model=model,
                fitness=fitness,
                paths=self.paths,
                n_cores=self.number_of_cores,
            )

            self.plot_start_point(
                parameter_vector=parameter_lists[0],
                model=model,
                analysis=analysis,
            )

            state = np.zeros(shape=(search_internal.nwalkers, model.prior_count))

            self.logger.info(
                "Starting new Emcee non-linear search (no previous samples found)."
            )

            for index, parameters in enumerate(parameter_lists):
                state[index, :] = np.asarray(parameters)

            total_iterations = 0
            iterations_remaining = self.config_dict_run["nsteps"]

        while iterations_remaining > 0:
            if self.iterations_per_full_update > iterations_remaining:
                iterations = iterations_remaining
            else:
                iterations = self.iterations_per_full_update

            for sample in search_internal.sample(
                initial_state=state,
                iterations=iterations,
                progress=True,
                skip_initial_state_check=True,
                store=True,
            ):
                pass

            state = search_internal.get_last_sample()

            total_iterations += iterations
            iterations_remaining = self.config_dict_run["nsteps"] - total_iterations

            samples = self.samples_from(model=model, search_internal=search_internal)

            if self.auto_correlation_settings.check_for_convergence:
                if (
                    search_internal.iteration
                    > self.auto_correlation_settings.check_size
                ):
                    if samples.converged:
                        iterations_remaining = 0

            if iterations_remaining > 0:
                self.perform_update(
                    model=model,
                    analysis=analysis,
                    search_internal=search_internal,
                    fitness=fitness,
                    during_analysis=True,
                )

        return search_internal, fitness

    def output_search_internal(self, search_internal):
        """
        Output the sampler results to hard-disk in their internal format.

        Emcee uses a backend to store and load results, therefore the outputting of the search internal to a
        dill file is disabled.

        Parameters
        ----------
        sampler
            The nautilus sampler object containing the results of the model-fit.
        """
        pass

    def samples_info_from(self, search_internal=None):
        search_internal = search_internal or self.backend

        auto_correlations = self.auto_correlations_from(search_internal=search_internal)

        return {
            "check_size": auto_correlations.check_size,
            "required_length": auto_correlations.required_length,
            "change_threshold": auto_correlations.change_threshold,
            "total_walkers": len(search_internal.get_chain()[0, :, 0]),
            "total_steps": len(search_internal.get_log_prob()),
            "time": self.timer.time if self.timer else None,
        }

    def samples_via_internal_from(self, model, search_internal=None):
        """
        Returns a `Samples` object from the emcee internal results.

        The samples contain all information on the parameter space sampling (e.g. the parameters,
        log likelihoods, etc.).

        The internal search results are converted from the native format used by the search to lists of values
        (e.g. `parameter_lists`, `log_likelihood_list`).

        Parameters
        ----------
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        """

        search_internal = search_internal or self.backend

        if is_test_mode():
            samples_after_burn_in = search_internal.get_chain(
                discard=5, thin=5, flat=True
            )

        else:
            auto_correlations = self.auto_correlations_from(
                search_internal=search_internal
            )

            discard = int(3.0 * np.max(auto_correlations.times))
            thin = int(np.max(auto_correlations.times) / 2.0)
            samples_after_burn_in = search_internal.get_chain(
                discard=discard, thin=thin, flat=True
            )

        parameter_lists = samples_after_burn_in.tolist()

        log_prior_list = model.log_prior_list_from(parameter_lists=parameter_lists)

        total_samples = len(parameter_lists)

        log_posterior_list = search_internal.get_log_prob(flat=True)[
            -total_samples - 1 : -1
        ].tolist()

        log_likelihood_list = [
            log_posterior - log_prior
            for log_posterior, log_prior in zip(log_posterior_list, log_prior_list)
        ]

        weight_list = len(log_likelihood_list) * [1.0]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameter_lists,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list,
        )

        return SamplesMCMC(
            model=model,
            sample_list=sample_list,
            samples_info=self.samples_info_from(search_internal=search_internal),
            auto_correlation_settings=self.auto_correlation_settings,
            auto_correlations=self.auto_correlations_from(
                search_internal=search_internal
            ),
        )

    def auto_correlations_from(self, search_internal=None):
        import emcee

        search_internal = search_internal or self.backend

        times = search_internal.get_autocorr_time(tol=0)

        previous_auto_correlation_times = emcee.autocorr.integrated_time(
            x=search_internal.get_chain()[
                : -self.auto_correlation_settings.check_size, :, :
            ],
            tol=0,
        )

        return AutoCorrelations(
            check_size=self.auto_correlation_settings.check_size,
            required_length=self.auto_correlation_settings.required_length,
            change_threshold=self.auto_correlation_settings.change_threshold,
            times=times,
            previous_times=previous_auto_correlation_times,
        )

    def config_dict_test_mode_from(self, config_dict: Dict) -> Dict:
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
            "nwalkers": 20,
            "nsteps": 10,
        }

    @property
    def backend_filename(self):
        return self.paths.search_internal_path / "search_internal.hdf"

    @property
    def backend(self) -> "emcee.backends.HDFBackend":
        """
        The `Emcee` hdf5 backend, which provides access to all samples, likelihoods, etc. of the non-linear search.

        The sampler is described in the "Results" section at https://dynesty.readthedocs.io/en/latest/quickstart.html
        """
        import emcee

        if os.path.isfile(self.backend_filename):
            return emcee.backends.HDFBackend(filename=str(self.backend_filename))
        else:
            raise FileNotFoundError(
                f"The file search_internal.hdf does not exist at the path {self.paths.search_internal_path}"
            )
