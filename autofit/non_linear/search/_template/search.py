"""
The template for a new non-linear search (search-extensibility phase A5).

Copy this file to ``autofit/non_linear/search/<family>/<name>/search.py``, rename
``TemplateSearch``, replace the hill-climbing loop in ``run`` with your backend and
fill in the class attributes. The search below is complete and runnable: a random-walk
hill climber that keeps every point it visits. A copy of it,
``test_autofit/non_linear/search/template_toy_search.py``, passes both layers of the
search conformance suite (``test_template_search.py``).

The contract (``docs/design/run_ctx.md``)
-----------------------------------------
A search implements three things and declares the rest:

1. **Class attributes**: the static capabilities (``autofit.non_linear.search.capabilities``:
   ``jax_use``, ``gradient``, ``batched``, ``posterior_kind``, ``objective_target``,
   ``invalid_value``, ``resumable``, ``test_mode_budget``, ...), the ``samples_cls`` the
   samples are built as, the ``checkpointer`` / ``resume_state`` archive strategy
   (``autofit.non_linear.checkpoint``; the default archives a dill) and
   ``__identifier_fields__``, the constructor arguments that change the result and so
   key the output folder. Every constructor argument is stored as ``self.<arg>``,
   because ``search.json`` reads them back.
2. ``run(self, ctx) -> internal``: the backend loop. ``ctx`` (``FitContext``) gives
   the objective (``ctx.objective(kind)`` / ``ctx.fitness``), pools (``ctx.pool``;
   never build one yourself), start points (``ctx.start_points(n)``), randomness
   (``ctx.rng``, a ``SeedSequence`` of the search's ``seed``; ``ctx.rng.spawn(k)``),
   the resume state (``ctx.resume``), the update cadence (``ctx.schedule``) and the
   mid-run update (``ctx.update(internal)``). ``run`` never builds a ``Fitness``, a
   pool or an initializer, never reads ``PYAUTO_TEST_MODE`` (level 1 is already applied
   from ``test_mode_budget``), never calls ``perform_update`` and never writes
   ``.completed``. It returns the backend's internal state, which is archived and
   read back by ``raw_samples_from``.
3. ``raw_samples_from(self, model, internal) -> RawSamples``: the parameters, the log
   likelihoods or log posteriors and the weights of the internal state, plus the
   ``samples_info`` entries (``info_from``).

The six touch points of a new search in PyAutoFit
-------------------------------------------------
1. the search file (this template);
2. its registry entry (``autofit/non_linear/search/registry.py``), which generates the
   capability matrix, API page and citations (``python docs/_generate_searches.py``);
3. the export line in ``autofit/__init__.py``;
4. a unit test (and a roster entry in ``test_autofit/non_linear/search/conformance_roster.py``);
5. an integration script, ``autofit_workspace_test/scripts/searches/<Name>.py``;
6. a section in ``autofit_workspace/scripts/searches/``.
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


class TemplateSearch(NonLinearSearch):
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
