"""
A toy search written from ``autofit/non_linear/search/_template/search.py`` (the
template copied and renamed, nothing else changed), kept as the fixture of the
template smoke: ``test_template_search.py`` runs both layers of the search
conformance suite on it. It is not exported, registered or documented.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from autofit.non_linear.initializer import AbstractInitializer
from autofit.non_linear.samples.adapter import RawSamples
from autofit.non_linear.samples.samples import Samples
from autofit.non_linear.search import capabilities as cap
from autofit.non_linear.search.abstract_search import NonLinearSearch
from autofit.non_linear.search.posterior_kind import POINT


class ToyHillClimb(NonLinearSearch):
    # 1. Static capabilities: class attributes, never identifier fields.
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
    status = cap.Status.EXPERIMENTAL
    # What ``PYAUTO_TEST_MODE=1`` reduces (``attribute: value``).
    test_mode_budget = {"n_steps": 10}
    # What the objective returns, in which coordinates, and its value for an invalid model.
    objective_target = cap.ObjectiveTarget(
        cap.ObjectiveQuantity.LOG_POSTERIOR, cap.CoordinateSpace.PHYSICAL
    )
    invalid_value = -float("inf")

    samples_cls = Samples

    # The constructor arguments that change the result, and so key the output folder.
    __identifier_fields__ = ("n_steps", "step_size")

    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        unique_tag: Optional[str] = None,
        n_steps: int = 1000,
        step_size: float = 0.1,
        initializer: Optional[AbstractInitializer] = None,
        iterations_per_quick_update: int = None,
        iterations_per_full_update: int = None,
        number_of_cores: Union[int, str] = "auto",
        silence: bool = False,
        session=None,
        **kwargs,
    ):
        """
        A random-walk hill climber: from a start point, propose a Gaussian step of
        ``step_size`` times each parameter's interquartile prior width and keep it when it raises
        the log posterior. Every visited point is a sample.

        Parameters
        ----------
        n_steps
            The number of proposals.
        step_size
            The proposal width, as a fraction of each parameter's interquartile prior
            width.
        """
        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            initializer=POINT.initializer_for(initializer),
            iterations_per_quick_update=iterations_per_quick_update,
            iterations_per_full_update=iterations_per_full_update,
            number_of_cores=number_of_cores,
            silence=silence,
            session=session,
            **kwargs,
        )

        self.n_steps = n_steps
        self.step_size = step_size

    # 2. The backend loop.
    def run(self, ctx):
        """
        Climb for ``n_steps`` proposals, updating the fit every
        ``iterations_per_full_update`` steps, and return the visited points.
        """
        objective = ctx.fitness.call_wrap
        rng = np.random.default_rng(ctx.rng.spawn(1)[0])

        # Each parameter's interquartile prior width scales its proposals.
        n_dim = ctx.model.prior_count
        widths = np.asarray(
            ctx.model.vector_from_unit_vector([0.75] * n_dim)
        ) - np.asarray(ctx.model.vector_from_unit_vector([0.25] * n_dim))

        parameters, log_posteriors = ctx.start_points(1)
        current, current_log_posterior = parameters[0], float(log_posteriors[0])

        internal = {
            "parameter_lists": [current.tolist()],
            "log_posterior_list": [current_log_posterior],
            "total_steps": 0,
            "n_accepted": 0,
        }

        for chunk in ctx.schedule.chunks(self.n_steps):
            for _ in range(chunk):
                proposal = current + self.step_size * widths * rng.standard_normal(
                    current.shape
                )
                log_posterior = float(objective(proposal))

                if np.isfinite(log_posterior) and log_posterior > current_log_posterior:
                    current, current_log_posterior = proposal, log_posterior
                    internal["n_accepted"] += 1

                internal["parameter_lists"].append(current.tolist())
                internal["log_posterior_list"].append(current_log_posterior)

            internal["total_steps"] += chunk

            if internal["total_steps"] < self.n_steps:
                ctx.update(internal)

        return internal

    # 3. The samples adapter.
    def raw_samples_from(self, model, internal):
        """
        The visited points and their log posteriors, equally weighted.
        """
        return RawSamples(
            parameters=internal["parameter_lists"],
            log_posterior=internal["log_posterior_list"],
            info=self.info_from(internal),
            label=type(self).__name__,
        )

    def info_from(self, internal) -> dict:
        """
        The search's own ``samples_info`` entries.
        """
        return {
            "total_steps": internal["total_steps"],
            "n_accepted": internal["n_accepted"],
        }
