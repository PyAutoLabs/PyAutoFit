from __future__ import annotations

import logging
from typing import Dict, Optional, TYPE_CHECKING

import numpy as np
import os

from autofit import exc
from autofit.mapper.model_mapper import ModelMapper
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.initializer import Initializer
from autofit.non_linear.search.mcmc.abstract_mcmc import AbstractMCMC
from autofit.non_linear.search.mcmc.auto_correlations import AutoCorrelationsSettings
from autofit.non_linear.search.mcmc.auto_correlations import AutoCorrelations
from autofit.non_linear.test_mode import is_test_mode
from autofit.non_linear.samples.adapter import ChainPosterior, RawSamples
from autofit.non_linear.search import capabilities as cap

if TYPE_CHECKING:
    from autofit.database.sqlalchemy_ import sa

logger = logging.getLogger(__name__)


class Zeus(AbstractMCMC):
    # Static capabilities (see ``autofit.non_linear.search.capabilities``); mirrored
    # by ``search/registry.py``. Never identifier fields.
    jax_use = cap.JaxUse.NONE
    gradient = cap.Gradient.NONE
    batched = False
    honours_gradient_mode = False
    posterior_kind = cap.PosteriorKind.CHAIN
    produces_evidence = False
    resumable = False
    warm_start = cap.WarmStart.CONSUMER
    install_extra = "optional"
    upstream_url = "https://github.com/minaskar/zeus"
    citation_keys = ("zeus1", "zeus2")
    status = cap.Status.STABLE
    test_mode_budget = {"nwalkers": 20, "nsteps": 10}
    objective_target = cap.ObjectiveTarget(cap.ObjectiveQuantity.LOG_POSTERIOR, cap.CoordinateSpace.PHYSICAL)
    invalid_value = -float("inf")

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
        nwalkers: int = 50,
        nsteps: int = 2000,
        tune: bool = True,
        tolerance: float = 0.05,
        patience: int = 5,
        mu: float = 1.0,
        light_mode: bool = False,
        maxsteps: int = 10000,
        maxiter: int = 10000,
        vectorize: bool = False,
        shuffle_ensemble: bool = True,
        check_walkers: bool = True,
        maxcall: Optional[int] = None,
        initializer: Optional[Initializer] = None,
        auto_correlation_settings=AutoCorrelationsSettings(),
        iterations_per_quick_update: int = None,
        iterations_per_full_update: int = None,
        number_of_cores: int = 1,
        silence: bool = False,
        session: Optional[sa.orm.Session] = None,
        **kwargs
    ):
        """
        A Zeus non-linear search.

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
            The number of steps that must be taken by every walker.
        initializer
            Generates the initialize samples of non-linear parameter space (see autofit.non_linear.initializer).
        auto_correlation_settings
            Customizes and performs auto correlation calculations performed during and after the search.
        number_of_cores
            The number of cores Zeus sampling is performed using a Python multiprocessing Pool instance.
        silence
            If True, the default print output of the non-linear search is silenced.
        session
            An SQLalchemy session instance so the results of the model-fit are written to an SQLite database.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            initializer=initializer,
            auto_correlation_settings=auto_correlation_settings,
            iterations_per_quick_update=iterations_per_quick_update,
            iterations_per_full_update=iterations_per_full_update,
            number_of_cores=number_of_cores,
            silence=silence,
            session=session,
            **kwargs
        )

        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.tune = tune
        self.tolerance = tolerance
        self.patience = patience
        self.mu = mu
        self.light_mode = light_mode
        self.maxsteps = maxsteps
        self.maxiter = maxiter
        self.vectorize = vectorize
        self.shuffle_ensemble = shuffle_ensemble
        self.check_walkers = check_walkers
        self.maxcall = maxcall

        if is_test_mode():
            self.apply_test_mode()

        self.logger.debug("Creating Zeus Search")

    def apply_test_mode(self):
        logger.warning(
            "TEST MODE 1 (reduced iterations): Sampler will run with "
            "minimal iterations for faster completion."
        )
        self.nwalkers = 20
        self.nsteps = 10

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

        fitness = self.make_fitness(analysis=analysis, model=model)

        try:
            search_internal = self.paths.load_search_internal()

            state = search_internal.get_last_sample()
            log_posterior_list = search_internal.get_last_log_prob()

            samples = self.samples_from(model=model, search_internal=search_internal)

            total_iterations = search_internal.iteration

            if samples.converged:
                iterations_remaining = 0
            else:
                iterations_remaining = self.nsteps - total_iterations

                self.logger.info(
                    "Resuming Zeus non-linear search (previous samples found)."
                )

        except (FileNotFoundError, AttributeError):
            search_internal = zeus.EnsembleSampler(
                nwalkers=self.nwalkers,
                ndim=model.prior_count,
                logprob_fn=fitness.call_wrap,
                pool=pool,
            )

            search_internal.ncall_total = 0

            (
                unit_parameter_lists,
                parameter_lists,
                log_posterior_list,
            ) = self.start_points(
                model=model,
                fitness=fitness,
                n=search_internal.nwalkers,
                test_mode_samples=False,
            )

            state = np.zeros(shape=(search_internal.nwalkers, model.prior_count))

            self.logger.info(
                "Starting new Zeus non-linear search (no previous samples found)."
            )

            for index, parameters in enumerate(parameter_lists):
                state[index, :] = np.asarray(parameters)

            total_iterations = 0
            iterations_remaining = self.nsteps

        while iterations_remaining > 0:
            # zeus casts internally (``self.nsteps = int(iterations)``), so a
            # float never crashes it — but ``total_iterations`` below is
            # incremented by this value, so an uncast float would drift the
            # bookkeeping away from the samples zeus actually drew
            # (PyAutoFit#1422).
            iterations = self._steps_until_full_update(iterations_remaining)

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
            iterations_remaining = self.nsteps - total_iterations

            samples = self.samples_from(model=model, search_internal=search_internal)

            if self.auto_correlation_settings.check_for_convergence:
                if (
                    search_internal.iteration
                    > self.auto_correlation_settings.check_size
                ):
                    if samples.converged:
                        iterations_remaining = 0

            if self.maxcall is not None:
                if search_internal.ncall_total > self.maxcall:
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

    def samples_info_from(self, search_internal=None, auto_correlations=None):
        search_internal = search_internal or self.paths.load_search_internal()

        if auto_correlations is None:
            auto_correlations = self.auto_correlations_from(
                search_internal=search_internal
            )

        return {
            "check_size": auto_correlations.check_size,
            "required_length": auto_correlations.required_length,
            "change_threshold": auto_correlations.change_threshold,
            "total_walkers": len(search_internal.get_chain()[0, :, 0]),
            "total_steps": int(search_internal.ncall_total),
            "time": self.timer.time if self.timer else None,
        }

    def raw_samples_from(self, model, search_internal):
        """
        The zeus chain after burn-in and thinning, as raw samples.

        The parameters and log posteriors come out of zeus under the *same*
        ``discard`` / ``thin`` (PyAutoFit#1628), through ``ChainPosterior.thin``,
        which falls back to the whole chain when burn-in removal leaves no draws; the log likelihood is the log
        posterior minus the log prior (``samples_from_raw``), and every draw has
        weight 1.

        Parameters
        ----------
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        search_internal
            The ``zeus.EnsembleSampler``.
        """
        # Computed once per conversion and shared by the burn-in, `samples_info` and
        # `SamplesMCMC`.
        auto_correlations = self.auto_correlations_from(
            search_internal=search_internal
        )

        if is_test_mode():
            discard = 5
            thin = 5

        else:
            discard = int(3.0 * np.max(auto_correlations.times))
            thin = int(np.max(auto_correlations.times) / 2.0)

        parameters, log_posterior = ChainPosterior.from_sampler(
            search_internal, label="Zeus"
        ).thin(discard=discard, thin=thin)

        return RawSamples(
            parameters=parameters,
            log_posterior=log_posterior,
            info=self.samples_info_from(
                search_internal=search_internal,
                auto_correlations=auto_correlations,
            ),
            samples_kwargs=dict(
                auto_correlation_settings=self.auto_correlation_settings,
                auto_correlations=auto_correlations,
            ),
            label="Zeus",
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

