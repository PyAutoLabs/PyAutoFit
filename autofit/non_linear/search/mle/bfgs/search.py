from typing import Optional

from autofit.database.sqlalchemy_ import sa

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.search.mle.abstract_mle import AbstractMLE
from autofit.non_linear.analysis import Analysis
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.initializer import AbstractInitializer
from autofit.non_linear.samples.sample import Sample
from autofit.non_linear.samples.samples import Samples

import numpy as np


class AbstractBFGS(AbstractMLE):

    method = None

    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        unique_tag: Optional[str] = None,
        tol: Optional[float] = None,
        disp: bool = False,
        eps: float = 1.0e-08,
        ftol: float = 2.220446049250313e-09,
        gtol: float = 1.0e-05,
        iprint: float = -1.0,
        maxcor: int = 10,
        maxfun: int = 15000,
        maxiter: int = 15000,
        maxls: int = 20,
        initializer: Optional[AbstractInitializer] = None,
        iterations_per_full_update: int = None,
        iterations_per_quick_update: int = None,
        silence: bool = False,
        session: Optional[sa.orm.Session] = None,
        **kwargs
    ):
        """
        Abstract wrapper for the BFGS and L-BFGS scipy non-linear searches.

        Parameters
        ----------
        tol
            Tolerance for termination.
        disp
            Set to True to print convergence messages.
        maxiter
            Maximum number of iterations.
        maxfun
            Maximum number of function evaluations.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            initializer=initializer,
            iterations_per_quick_update=iterations_per_quick_update,
            iterations_per_full_update=iterations_per_full_update,
            silence=silence,
            session=session,
            **kwargs
        )

        self.tol = tol
        self.disp = disp
        self.eps = eps
        self.ftol = ftol
        self.gtol = gtol
        self.iprint = iprint
        self.maxcor = maxcor
        self.maxfun = maxfun
        self.maxiter = maxiter
        self.maxls = maxls

        self.logger.debug(f"Creating {self.method} Search")

    @property
    def options(self):
        return {
            "disp": self.disp,
            "eps": self.eps,
            "ftol": self.ftol,
            "gtol": self.gtol,
            "iprint": self.iprint,
            "maxcor": self.maxcor,
            "maxfun": self.maxfun,
            "maxiter": self.maxiter,
            "maxls": self.maxls,
        }

    def _fit(
        self,
        model: AbstractPriorModel,
        analysis: Analysis,
    ):
        """
        Fit a model using the scipy L-BFGS method and the Analysis class which contains the data and returns the log
        likelihood from instances of the model, which the `NonLinearSearch` seeks to maximize.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space.
        analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data,
            returning the log likelihood the `NonLinearSearch` maximizes.

        Returns
        -------
        A result object comprising the Samples object that inclues the maximum log likelihood instance and full
        chains used by the fit.
        """
        from scipy import optimize

        fitness = Fitness(
            model=model,
            analysis=analysis,
            paths=self.paths,
            fom_is_log_likelihood=False,
            resample_figure_of_merit=-np.inf,
            convert_to_chi_squared=True,
            store_history=self.should_plot_start_point
        )

        try:
            search_internal_dict = self.paths.load_search_internal()

            x0 = search_internal_dict["x0"]
            total_iterations = search_internal_dict["total_iterations"]

            self.logger.info(
                "Resuming LBFGS non-linear search (previous samples found)."
            )

        except (FileNotFoundError, TypeError):

            (
                unit_parameter_lists,
                parameter_lists,
                log_posterior_list,
            ) = self.initializer.samples_from_model(
                total_points=1,
                model=model,
                fitness=fitness,
                paths=self.paths,
                n_cores=self.number_of_cores,
            )

            x0 = np.asarray(parameter_lists[0])

            total_iterations = 0

            self.logger.info(
               f"Starting new {self.method} non-linear search (no previous samples found)."
            )

            self.plot_start_point(
                parameter_vector=x0,
                model=model,
                analysis=analysis,
            )

        while total_iterations < self.maxiter:

            iterations_remaining = self.maxiter - total_iterations
            iterations = min(self.iterations_per_full_update, iterations_remaining)

            if iterations > 0:
                options = dict(self.options)
                options["maxiter"] = iterations

                if analysis._use_jax:

                    search_internal = optimize.minimize(
                        fun=fitness._jit,
                        x0=x0,
                        method=self.method,
                        options=options,
                        tol=self.tol,
                    )
                else:

                    search_internal = optimize.minimize(
                        fun=fitness.__call__,
                        x0=x0,
                        method=self.method,
                        options=options,
                        tol=self.tol,
                    )

                total_iterations += search_internal.nit

                search_internal.log_posterior_list = -0.5 * fitness(
                    parameters=search_internal.x
                )

                if self.should_plot_start_point:

                    search_internal.parameters_history_list = fitness.parameters_history_list
                    search_internal.log_likelihood_history_list = fitness.log_likelihood_history_list

                self.paths.save_search_internal(
                    obj=search_internal,
                )

                x0 = search_internal.x

                if search_internal.nit < iterations:
                    return search_internal, fitness

                self.perform_update(
                    model=model,
                    analysis=analysis,
                    during_analysis=True,
                    fitness=fitness,
                    search_internal=search_internal,
                )

        self.logger.info(f"{self.method} sampling complete.")

        return search_internal, fitness

    def samples_via_internal_from(
        self, model: AbstractPriorModel, search_internal=None
    ):
        """
        Returns a `Samples` object from the LBFGS internal results.

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

        x0 = search_internal.x
        total_iterations = search_internal.nit

        if self.should_plot_start_point:

            parameter_lists = search_internal.parameters_history_list
            log_prior_list = model.log_prior_list_from(parameter_lists=parameter_lists)
            log_likelihood_list = search_internal.log_likelihood_history_list

        else:

            parameter_lists = [list(x0)]
            log_prior_list = model.log_prior_list_from(parameter_lists=parameter_lists)
            log_posterior_list = np.array([search_internal.log_posterior_list])
            log_likelihood_list = [
                lp - prior for lp, prior in zip(log_posterior_list, log_prior_list)
            ]

        weight_list = len(log_likelihood_list) * [1.0]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameter_lists,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list,
        )

        samples_info = {
            "total_iterations": total_iterations,
            "time": self.timer.time if self.timer else None,
        }

        return Samples(
            model=model,
            sample_list=sample_list,
            samples_info=samples_info,
        )


class BFGS(AbstractBFGS):
    """
    The BFGS non-linear search, which wraps the scipy Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm.

    For a full description of the scipy BFGS method, checkout its documentation:

    https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html#optimize-minimize-bfgs
    """

    method = "BFGS"


class LBFGS(AbstractBFGS):
    """
    The L-BFGS non-linear search, which wraps the scipy Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS)
    algorithm.

    For a full description of the scipy L-BFGS method, checkout its documentation:

    https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
    """

    method = "L-BFGS-B"
