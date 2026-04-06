import logging
from typing import Dict, Optional

import numpy as np
import os

from autofit.database.sqlalchemy_ import sa
from autofit.mapper.model_mapper import ModelMapper
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.initializer import Initializer
from autofit.non_linear.search.mcmc.abstract_mcmc import AbstractMCMC
from autofit.non_linear.search.mcmc.auto_correlations import AutoCorrelationsSettings
from autofit.non_linear.search.mcmc.auto_correlations import AutoCorrelations
from autofit.non_linear.samples.sample import Sample
from autofit.non_linear.test_mode import is_test_mode
from autofit.non_linear.samples.mcmc import SamplesMCMC


class Zeus(AbstractMCMC):
    __identifier_fields__ = (
        "nwalkers",
        "tune",
        "tolerance",
        "patience",
        "mu",
        "light_mode",
    )

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
        **kwargs
    ):
        """
        An Zeus non-linear search.

        For a full description of Zeus, checkout its Github and readthedocs webpages:

        https://github.com/minaskar/zeus

        https://zeus-mcmc.readthedocs.io/en/latest/

        If you use `Zeus` as part of a published work, please cite the package following the instructions under the
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
        nwalkers
            The number of walkers in the ensemble used to sample parameter space.
        nsteps
            The number of steps that must be taken by every walker. The `NonLinearSearch` will thus run for nwalkers *
            nsteps iterations.
        initializer
            Generates the initialize samples of non-linear parameter space (see autofit.non_linear.initializer).
        auto_correlation_settings : AutoCorrelationsSettings
            Customizes and performs auto correlation calculations performed during and after the search.
        number_of_cores
            The number of cores Zeus sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.
        session
            An SQLalchemy session instance so the results of the model-fit are written to an SQLite database.
        """

        number_of_cores = number_of_cores or self._config("parallel", "number_of_cores")

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
            **kwargs
        )

        self.logger.debug("Creating Zeus Search")

    def _fit(self, model: AbstractPriorModel, analysis):
        """
        Fit a model using Zeus and the Analysis class which contains the data and returns the log likelihood from
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

        try:
            import zeus
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "\n--------------------\n"
                "You are attempting to perform a model-fit using Zeus. \n\n"
                "However, the optional library Zeus (https://zeus-mcmc.readthedocs.io/en/latest/) is "
                "not installed.\n\n"
                "Install it via the command `pip install zeus-mcmc==2.5.4`.\n\n"
                "----------------------"
            )

        pool = self.make_pool()

        fitness = Fitness(
            model=model,
            analysis=analysis,
            paths=self.paths,
            fom_is_log_likelihood=False,
            resample_figure_of_merit=-np.inf,
        )

        try:
            search_internal = self.paths.load_search_internal()

            state = search_internal.get_last_sample()
            log_posterior_list = search_internal.get_last_log_prob()

            samples = self.samples_from(model=model, search_internal=search_internal)

            total_iterations = search_internal.iteration

            if samples.converged:
                iterations_remaining = 0
            else:
                iterations_remaining = self.config_dict_run["nsteps"] - total_iterations

                self.logger.info(
                    "Resuming Zeus non-linear search (previous samples found)."
                )

        except (FileNotFoundError, AttributeError):
            search_internal = zeus.EnsembleSampler(
                nwalkers=self.config_dict_search["nwalkers"],
                ndim=model.prior_count,
                logprob_fn=fitness.call_wrap,
                pool=pool,
            )

            search_internal.ncall_total = 0

            (
                unit_parameter_lists,
                parameter_lists,
                log_posterior_list,
            ) = self.initializer.samples_from_model(
                total_points=search_internal.nwalkers,
                model=model,
                fitness=fitness,
                test_mode_samples=False,
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
                "Starting new Zeus non-linear search (no previous samples found)."
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
                start=state,
                log_prob0=log_posterior_list,
                iterations=iterations,
                progress=True,
            ):
                pass

            search_internal.ncall_total += search_internal.ncall

            self.paths.save_search_internal(
                obj=search_internal,
            )

            state = search_internal.get_last_sample()
            log_posterior_list = search_internal.get_last_log_prob()

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

            auto_correlation_time = zeus.AutoCorrTime(
                samples=search_internal.get_chain()
            )

            discard = int(3.0 * np.max(auto_correlation_time))
            thin = int(np.max(auto_correlation_time) / 2.0)
            chain = search_internal.get_chain(discard=discard, thin=thin, flat=True)

            if "maxcall" in self.kwargs:
                if search_internal.ncall_total > self.kwargs["maxcall"]:
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

    def samples_info_from(self, search_internal=None):
        search_internal = search_internal or self.paths.load_search_internal()

        auto_correlations = self.auto_correlations_from(search_internal=search_internal)

        return {
            "check_size": auto_correlations.check_size,
            "required_length": auto_correlations.required_length,
            "change_threshold": auto_correlations.change_threshold,
            "total_walkers": len(search_internal.get_chain()[0, :, 0]),
            "total_steps": int(search_internal.ncall_total),
            "time": self.timer.time if self.timer else None,
        }

    def samples_via_internal_from(self, model, search_internal=None):
        """
        Returns a `Samples` object from the zeus internal results.

        The samples contain all information on the parameter space sampling (e.g. the parameters,
        log likelihoods, etc.).

        The internal search results are converted from the native format used by the search to lists of values
        (e.g. `parameter_lists`, `log_likelihood_list`).

        Parameters
        ----------
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        """

        search_internal = search_internal or self.paths.load_search_internal()

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

            if len(samples_after_burn_in) == 0:

                logging.info(
                    """
                    After thinnng the Zeus samples in order to remove burn-in, no samples were left.
                    
                    To create a samples object containing samples, so that the code can continue and results
                    can be inspected, the full list of samples before removing burn-in has been used. This may 
                    indicate that the sampler has not converged and therefore your results may not be reliable.
                    
                    To fix this, run Zeus with more steps to ensure convergence is achieved or change the auto
                    correlation settings to be less aggressive in thinning samples.                
                    """
                )

                samples_after_burn_in = search_internal.get_chain(flat=True)

        parameter_lists = samples_after_burn_in.tolist()
        log_posterior_list = search_internal.get_log_prob(flat=True).tolist()
        log_prior_list = model.log_prior_list_from(parameter_lists=parameter_lists)

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
        import zeus

        search_internal = search_internal or self.paths.load_search_internal()

        times = zeus.AutoCorrTime(samples=search_internal.get_chain())
        try:
            previous_auto_correlation_times = zeus.AutoCorrTime(
                samples=search_internal.get_chain()[
                    : -self.auto_correlation_settings.check_size, :, :
                ],
            )
        except IndexError:
            self.logger.debug("Unable to compute previous auto correlation times.")
            previous_auto_correlation_times = None

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
