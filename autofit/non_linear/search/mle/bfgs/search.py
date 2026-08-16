from typing import Optional

from autofit import exc
from autofit.database.sqlalchemy_ import sa

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.search.mle.abstract_mle import AbstractMLE
from autofit.non_linear.analysis import Analysis
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.clipper import AbstractClipper, ClipperNone
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
        clipper: Optional[AbstractClipper] = None,
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
        clipper
            Enforcement of prior support (see :mod:`autofit.non_linear.clipper`).
            Unlike the multi-start gradient searches, which project the parameters
            themselves after every update, this search declares the box to scipy
            and lets scipy enforce it — ``L-BFGS-B`` supports box bounds natively.

            Default ``ClipperNone``, under which no ``bounds=`` is passed at all
            and behaviour is unchanged. Only bound-supporting methods accept a
            real clipper; see ``_bounds_from``.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            initializer=initializer,
            clipper=clipper,
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

    # The scipy methods that accept box bounds. Plain ``BFGS`` is deliberately
    # absent: scipy does not reject bounds it cannot use, it *ignores* them behind
    # a ``UserWarning`` and returns the unconstrained optimum. A user who asked for
    # prior-support enforcement and received an unbounded fit plus a log line is
    # exactly the silent-wrong-answer this class is meant to prevent, so
    # ``_bounds_from`` raises instead.
    _BOUND_SUPPORTING_METHODS = ("L-BFGS-B", "TNC", "SLSQP")

    def _bounds_from(self, model):
        """
        The ``scipy.optimize.Bounds`` for ``model``, or ``None`` when no clipper is
        configured (in which case no ``bounds=`` is passed at all and behaviour is
        unchanged).

        The conversion is not incidental. ``AbstractClipper.bounds_from_model``
        returns ``(lower, upper)`` as two arrays, which is the shape ``project``
        broadcasts against — but ``optimize.minimize`` reads a ``(lower, upper)``
        tuple as a *sequence of ``(min, max)`` pairs*. For a two-parameter model
        that is a valid-looking pair sequence, so scipy pins each parameter to a
        constant and returns a wrong fit with no error and no warning; at every
        other dimensionality it raises. Building an explicit ``Bounds`` is what
        makes the intent unambiguous.
        """
        # Imported lazily, as everywhere else in autofit -- no module in the
        # package pulls scipy in at import time.
        from scipy import optimize

        if isinstance(self.clipper, ClipperNone):
            return None

        if self.method not in self._BOUND_SUPPORTING_METHODS:
            raise exc.SearchException(
                f"A {type(self.clipper).__name__} was passed to a search using "
                f"method '{self.method}', which does not support box bounds. "
                f"SciPy would silently ignore the bounds and return an "
                f"unconstrained fit. Use one of "
                f"{', '.join(self._BOUND_SUPPORTING_METHODS)} (e.g. af.LBFGS) or "
                f"leave the clipper as the default ClipperNone."
            )

        lower, upper = self.clipper.bounds_from_model(model)
        return optimize.Bounds(lower, upper)

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

        # Resolved once, before any stepping: an unsupported method should fail
        # immediately rather than after the first chunk of iterations.
        bounds = self._bounds_from(model=model)

        fitness = Fitness(
            model=model,
            analysis=analysis,
            paths=self.paths,
            fom_is_log_likelihood=False,
            resample_figure_of_merit=-np.inf,
            convert_to_chi_squared=True,
            store_history=self.should_plot_start_point,
            iterations_per_quick_update=self.iterations_per_quick_update,
            background_quick_update=self.quick_update_background,
            live_visual_update=self.live_visual_update,
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
            # SciPy tolerates an integral float ``maxiter``, but a fractional one
            # would quietly acquire ceiling semantics instead of being rejected
            # (PyAutoFit#1422).
            iterations = self._steps_until_full_update(iterations_remaining)

            if iterations > 0:
                options = dict(self.options)
                options["maxiter"] = iterations

                # ``bounds`` is ``None`` under the default ``ClipperNone``, and
                # ``minimize(bounds=None)`` is the same call as omitting it, so the
                # default path is unchanged.
                #
                # Only the JAX branch is actually *exposed* to the prior-support
                # problem: ``UniformPrior.log_prior_from_value`` returns ``0.0``
                # unconditionally on the NumPy path, with no bound test, so there
                # is no hard wall to fall off there. Bounds are still passed on
                # both branches — they are correct on both, and having them
                # diverge by branch would be a trap of its own.
                if analysis._use_jax:

                    search_internal = optimize.minimize(
                        fun=fitness._jit,
                        x0=x0,
                        method=self.method,
                        options=options,
                        tol=self.tol,
                        bounds=bounds,
                    )
                else:

                    search_internal = optimize.minimize(
                        fun=fitness.__call__,
                        x0=x0,
                        method=self.method,
                        options=options,
                        tol=self.tol,
                        bounds=bounds,
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
            # Prior-support enforcement (PyAutoFit#1477). The name only, with
            # deliberately NO ``n_clipped_lane_steps`` alongside it: this search
            # is declarative, handing ``optimize.Bounds`` to scipy and letting
            # scipy enforce, so ``Clipper.project`` is never called, no mask is
            # produced and there is nothing to count. Writing a ``0`` here would
            # read as "the clipper never fired" when it means "this search
            # cannot know"; the summary renders the absent key as "not measured"
            # instead.
            "clipper": type(self.clipper).__name__,
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
