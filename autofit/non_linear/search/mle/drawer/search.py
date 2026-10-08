from __future__ import annotations

import numpy as np
from typing import Optional, TYPE_CHECKING


from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.search.mle.abstract_mle import AbstractMLE
from autofit.non_linear.initializer import AbstractInitializer
from autofit.non_linear.samples import Samples, Sample
from autofit.non_linear.search import capabilities as cap

if TYPE_CHECKING:
    from autofit.database.sqlalchemy_ import sa


class Drawer(AbstractMLE):
    # Static capabilities (see ``autofit.non_linear.search.capabilities``); mirrored
    # by ``search/registry.py``. Never identifier fields.
    jax_use = cap.JaxUse.NONE
    gradient = cap.Gradient.NONE
    batched = False
    honours_gradient_mode = False
    posterior_kind = cap.PosteriorKind.POINT
    produces_evidence = False
    resumable = False
    warm_start = cap.WarmStart.NEUTRAL
    install_extra = ""
    upstream_url = "https://github.com/PyAutoLabs/PyAutoFit"
    citation_keys = ()
    status = cap.Status.STABLE
    test_mode_budget = {}
    objective_target = cap.ObjectiveTarget(cap.ObjectiveQuantity.LOG_POSTERIOR, cap.CoordinateSpace.PHYSICAL)
    invalid_value = -float("inf")

    # The draws are the result, not a starting point: nothing to plot.
    _plots_start_point = False

    __identifier_fields__ = ("total_draws",)

    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        unique_tag: Optional[str] = None,
        total_draws: int = 50,
        initializer: Optional[AbstractInitializer] = None,
        iterations_per_full_update: int = None,
        iterations_per_quick_update: int = None,
        silence: bool = False,
        session: Optional[sa.orm.Session] = None,
        **kwargs,
    ):
        """
        A Drawer non-linear search, which simply draws a fixed number of samples from the model uniformly from the
        priors.

        Therefore, it does not seek to determine model parameters which maximize the likelihood or map out the
        posterior of the overall parameter space.

        Whilst this is not the typical use case of a non-linear search, it has certain niche applications, for example:

        - Given a model one can determine how much variation there is in the log likelihood / log posterior values.
          By visualizing this as a histogram one can therefore quantify the behaviour of that
          model's `log_likelihood_function`.

        - If the `log_likelihood_function` of a model is stochastic (e.g. different values of likelihood may be
          computed for an identical model due to randomness in the likelihood evaluation) this search can quantify
          the behaviour of that stochasticity.

        - For advanced modeling tools, for example sensitivity mapping performed via the `Sensitivity` object,
          the `Drawer` search may be sufficient to perform the overall modeling task, without the need of performing
          an actual parameter space search.

        The drawer search itself is performed by simply reusing the functionality of the `AbstractInitializer` object.
        Whereas this is normally used to initialize a non-linear search, for the drawer it performed all log
        likelihood evluations.

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
        session
            An SQLalchemy session instance so the results of the model-fit are written to an SQLite database.
        """

        # Drawer is single-core only; drop any saved number_of_cores so a
        # round-tripped search.json (which records the resolved value) can be
        # deserialized without colliding with the hardcoded kwarg below.
        kwargs.pop("number_of_cores", None)

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            initializer=initializer,
            iterations_per_quick_update=iterations_per_quick_update,
            iterations_per_full_update=iterations_per_full_update,
            number_of_cores=1,
            silence=silence,
            session=session,
            **kwargs,
        )

        self.total_draws = total_draws

        self.logger.debug("Creating Drawer Search")

    def run(self, ctx):
        """
        Draw ``total_draws`` points from the initializer and return them (the ``run(ctx)``
        hook; see ``docs/design/run_ctx.md``).

        Every log likelihood evaluation happens in the initializer, through the fit's
        objective (lazily jitted on a JAX analysis).

        Parameters
        ----------
        ctx
            The fit's ``FitContext``.

        Returns
        -------
        The internal state: the drawn parameter lists, their log posteriors and the run time.
        """
        self.logger.info(
            f"Performing DrawerSearch for a total of {self.total_draws} points."
        )

        parameters, log_posteriors = ctx.start_points(self.total_draws)

        search_internal = {
            "parameter_lists": parameters.tolist(),
            "log_posterior_list": log_posteriors.tolist(),
            "time": self.timer.time if self.timer else None,
        }

        ctx.paths.save_search_internal(
            obj=search_internal,
        )

        self.logger.info("Drawer complete")

        return search_internal

    def raw_samples_from(self, model, internal):
        """
        The drawn points as ``RawSamples``: log posteriors, unit weights, and the internal
        dictionary itself as ``samples_info``.
        """
        from autofit.non_linear.search.fit_context import RawSamples

        return RawSamples(
            parameters=internal["parameter_lists"],
            log_posterior=internal["log_posterior_list"],
            info=self.info_from(internal),
        )

    def info_from(self, internal):
        return internal
