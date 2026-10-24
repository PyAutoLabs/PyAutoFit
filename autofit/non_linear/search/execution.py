"""
How a ``NonLinearSearch`` executes its backend: the ``run(ctx)`` bridge
(``docs/design/run_ctx.md``), the one ``Fitness`` construction site, start points and
pools.

``SearchExecutionMixin`` is a base of ``NonLinearSearch`` only; its methods are the
search's own (subclasses override them as before). It defines no ``__init__`` and no
constructor argument, so serialized searches and identifiers are unaffected.
"""

from __future__ import annotations

import copy
import logging
from abc import ABCMeta
from typing import TYPE_CHECKING, Any, List, Optional

from autonerves import conf

from autofit.non_linear.fitness import Fitness
from autofit.non_linear.parallel import PoolFactory
from autofit.non_linear.search import capabilities as cap
from autofit.non_linear.search.seed import initializer_rng
from autofit.non_linear.test_mode import test_mode_level

if TYPE_CHECKING:
    from autofit.mapper.prior_model.abstract import AbstractPriorModel
    from autofit.non_linear.analysis import Analysis

logger = logging.getLogger(__name__)


class _SearchMeta(ABCMeta):
    """
    The metaclass of ``NonLinearSearch``. It

    - keeps a search abstract until it implements its backend through either hook:
      overriding ``_fit`` (the legacy contract) or ``run(ctx)`` (the bridge,
      ``docs/design/run_ctx.md``); a class that defines ``run`` is concrete because
      the inherited ``_fit`` is the bridge to it;
    - runs the post-init hook ``_post_init`` once, after the outermost ``__init__`` of
      a construction returns (so a subclass's own attributes are set before it).
    """

    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)

        run = getattr(cls, "run", None)

        if (
            "_fit" in cls.__abstractmethods__
            and run is not None
            and not getattr(run, "_is_bridge_default", False)
        ):
            cls.__abstractmethods__ = frozenset(cls.__abstractmethods__ - {"_fit"})

        return cls

    def __call__(cls, *args, **kwargs):
        search = super().__call__(*args, **kwargs)
        search._post_init()
        return search


class SearchExecutionMixin:
    # Whether `ctx.start_points` makes the start-point plot. `Drawer` draws its whole
    # result through the initializer, so it has no starting point to plot.
    _plots_start_point = True
    # Whether `ctx.start_points` lets the initializer return its test-mode samples under
    # `PYAUTO_TEST_MODE=1` (`test_mode_samples`); `Zeus` draws real start points.
    _start_points_in_test_mode = True

    def _run_bridge(self, model: AbstractPriorModel, analysis: Analysis):
        """
        The ``run(ctx)`` bridge behind ``NonLinearSearch._fit``: build the fit's
        ``Fitness`` and ``FitContext`` -- after the test-mode bypass and the fail-fast
        gates, which ``start_resume_fit`` has already run -- call ``run(ctx)`` and clean
        the context up on every exit. Returns ``(search_internal, fitness)``.
        """
        from autofit.non_linear.search.abstract_search import NonLinearSearch
        from autofit.non_linear.search.fit_context import FitContext

        if type(self).run is NonLinearSearch.run:
            raise NotImplementedError(
                f"{type(self).__name__} implements neither run(ctx) nor _fit."
            )

        # The REQUIRED gate, repeated for callers that reach `_fit` without `fit`, before
        # any `Fitness` or backend state exists (`fit` has already run it).
        cap.check_jax_required(self, analysis)

        fitness = self.make_fitness(
            analysis=analysis,
            model=model,
            **self.execution_settings(analysis).fitness_overrides,
        )

        pools = self._pools(analysis)
        # Every `pools(...)` pool then warms this fitness up before it forks
        # (`PoolFactory.prepare_for_fork`).
        pools.fitness = fitness

        ctx = FitContext(
            search=self,
            model=model,
            analysis=analysis,
            fitness=fitness,
            pool=pools,
            test_mode_level=test_mode_level(),
        )

        try:
            search_internal = self.run(ctx)
        except BaseException:
            ctx.close(failed=True)
            raise

        ctx.close(failed=False)

        return search_internal, fitness

    def _identifier_seed(self):
        """
        The seed the identifier walk hashes when the search's ``__identifier_fields__``
        do not name ``"seed"``: the search-level seed, hashed only when not None
        (``autofit.non_linear.search.seed``).
        """
        return getattr(self, "seed", None)

    def run(self, ctx) -> Any:
        """
        The backend loop of a search on the ``run(ctx)`` hook: run the backend through
        ``ctx`` and return its internal state (``search_internal``). See
        ``docs/design/run_ctx.md``.
        """
        raise NotImplementedError

    run._is_bridge_default = True

    def info_from(self, internal) -> dict:
        """
        The ``samples_info`` entries of a ``run(ctx)`` search's internal state, the
        ``info`` its ``raw_samples_from`` puts on the ``RawSamples``
        (``autofit.non_linear.samples.adapter``).
        """
        return {}

    def fitness_overrides(self, analysis: Analysis) -> dict:
        """
        ``Fitness`` keyword arguments a ``run(ctx)`` search adds to ``make_fitness`` for
        this analysis (e.g. ``batched=True`` for a batched backend). Empty by default.
        """
        return {}

    def execution_settings(self, analysis: Analysis):
        """
        How this search executes its objective on ``analysis``, resolved once: batching,
        eager or jitted execution, gradient use and the resolved gradient mode, plus the
        ``Fitness`` keyword arguments the backend builds its ``Fitness`` with
        (``autofit.non_linear.search.preflight.ExecutionSettings``).

        The JAX trace preflight validates exactly this object and the backend builds its
        ``Fitness`` from it, so the two cannot disagree.
        """
        from autofit.non_linear.search.preflight import ExecutionSettings

        return ExecutionSettings.resolve(self, analysis)

    def make_fitness(
        self, analysis: Analysis, model: AbstractPriorModel, **overrides
    ) -> Fitness:
        """
        The `Fitness` this search evaluates its likelihood through, the one construction
        site every search shares.

        Its figure-of-merit convention comes from the search's declared capabilities
        (`autofit.non_linear.search.capabilities`), never from per-site literals:

        - `objective_target.quantity` picks what the objective returns: the log
          likelihood (`fom_is_log_likelihood=True`), the log posterior, or -2 x the log
          posterior (`convert_to_chi_squared=True`, for the minimizers);
        - `invalid_value` is the sentinel the backend sees for an invalid model. For a
          `neg2_log_posterior` search that is the value *after* the -2 multiply, so the
          `resample_figure_of_merit` substituted before it is `invalid_value / -2`
          (`+inf` -> `-inf`).

        The quick-update cadence, background worker and live-visual settings are always
        forwarded, so no search can silently drop quick updates (what
        `test_quick_update_wiring.py` used to scan for).

        Parameters
        ----------
        analysis
            The analysis whose likelihood is wrapped.
        model
            The model being fitted.
        overrides
            Any other `Fitness` keyword argument, which wins over the derived and
            forwarded ones (e.g. `batched=True`, `store_history=True`).
        """
        target = type(self).objective_target

        kwargs = dict(
            model=model,
            analysis=analysis,
            paths=self.paths,
            iterations_per_quick_update=self.iterations_per_quick_update,
            background_quick_update=self.quick_update_background,
            live_visual_update=self.live_visual_update,
        )

        if target is not None:
            quantity = target.quantity
            neg2 = quantity == cap.ObjectiveQuantity.NEG2_LOG_POSTERIOR

            kwargs.update(
                fom_is_log_likelihood=quantity == cap.ObjectiveQuantity.LOG_LIKELIHOOD,
                convert_to_chi_squared=neg2,
                resample_figure_of_merit=(
                    type(self).invalid_value / -2.0
                    if neg2
                    else type(self).invalid_value
                ),
            )

        kwargs.update(overrides)

        return Fitness(**kwargs)

    @property
    def should_plot_start_point(self) -> bool:
        return conf.instance["output"]["start_point"]

    def plot_start_point(
        self,
        parameter_vector: List[float],
        model: AbstractPriorModel,
        analysis: Analysis,
    ):
        """
        Visualize the starting point of the non-linear search, using an instance of the model at the starting point
        of the maximum likelihood estimator.

        Plots are output to a folder named `image_start` in the output path, so that the starting point model
        can be compared to the final model inferred by the non-linear search.

        Parameters
        ----------
        model
            The model used by the non-linear search
        analysis
            The analysis which contains the visualization methods which plot the starting point model.
        """
        if not self.should_plot_start_point:
            return

        self.logger.info(f"Visualizing Starting Point Model in image_start folder.")

        instance = model.instance_from_vector(vector=parameter_vector)
        paths = copy.copy(self.paths)
        paths.image_path_suffix = "_start"

        # The search has no samples before it starts, so only the analysis visuals
        # of the starting point are output, not the search's own plots.
        self.perform_visualization(
            model=model,
            analysis=analysis,
            instance=instance,
            during_analysis=False,
            paths_override=paths,
            plot_search=False,
        )

    def start_points(
        self,
        model: AbstractPriorModel,
        fitness: Fitness,
        n: int,
        plot: bool = True,
        **initializer_kwargs,
    ):
        """
        The search's ``n`` initial points, drawn by its initializer, with the start-point
        plot made once, here, from the first of them.

        The one place the searches draw start points, from stream 0 of the search's seed
        (``autofit.non_linear.search.seed``): the initializer is evaluated through
        ``fitness`` (so a JAX analysis runs its lazily jitted scalar objective) with the
        fit's effective core count (the JAX fork rule of ``PoolFactory``).

        Parameters
        ----------
        model
            The model being fitted.
        fitness
            The fit's ``Fitness``; its analysis visualizes the start point.
        n
            The number of points.
        plot
            ``False`` skips the start-point plot, for searches whose initial points are
            not a starting point (``Drawer`` draws its whole result this way, Dynesty its
            initial live points).
        initializer_kwargs
            Passed to ``initializer.samples_from_model`` (e.g. ``test_mode_samples``).

        Returns
        -------
        The initializer's ``(unit_parameter_lists, parameter_lists, figure_of_merit_list)``.
        """
        unit_parameter_lists, parameter_lists, figure_of_merit_list = (
            self.initializer.samples_from_model(
                total_points=n,
                model=model,
                fitness=fitness,
                paths=self.paths,
                n_cores=self._pools(fitness.analysis).number_of_cores,
                rng=initializer_rng(self),
                **initializer_kwargs,
            )
        )

        if plot:
            self.plot_start_point(
                parameter_vector=parameter_lists[0],
                model=model,
                analysis=fitness.analysis,
            )

        return unit_parameter_lists, parameter_lists, figure_of_merit_list

    def _pools(self, analysis: Optional[Analysis] = None) -> PoolFactory:
        """
        The current fit's `PoolFactory` (the JAX fork rule applied to its analysis), or,
        outside a fit, one built for `analysis` (so a direct `_fit` caller still gets
        the fork rule), or from `number_of_cores` alone when no analysis is given.

        The only pool source of a search (search-extensibility A5 removed the legacy
        pool shims of the `_fit` contract): a `run(ctx)` search reaches it as
        `ctx.pool`, an external search that still overrides `_fit` as
        `self._pools(analysis)` (`PoolFactory.__call__` / `PoolFactory.sneaky`).
        """
        pools = self.__dict__.get("_fit_pools")
        return pools if pools is not None else PoolFactory(self, analysis)
