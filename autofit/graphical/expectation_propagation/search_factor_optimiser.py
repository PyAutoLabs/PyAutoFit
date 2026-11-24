"""
The expectation-propagation adapter of a non-linear search.

``SearchFactorOptimiser(search)`` runs one EP factor step with any
``NonLinearSearch``. ``NonLinearSearch`` stays an ``AbstractFactorOptimiser`` (so
``EPOptimiser`` accepts ``af.Emcee()`` directly) and its ``optimise`` delegates here.
"""

import logging
from typing import Tuple

from autonerves import conf

from autofit.graphical.mean_field import FactorApproximation, MeanField, Status
from autofit.graphical.expectation_propagation.factor_optimiser import (
    AbstractFactorOptimiser,
)

logger = logging.getLogger(__name__)


class SearchFactorOptimiser(AbstractFactorOptimiser):
    def __init__(self, search):
        """
        Optimise single EP factors with ``search``.

        The per-factor state lives on the search, which EP re-enters once per factor
        step: its ``optimisation_counter`` (the ``optimization_<n>`` folder number) and
        the ``_visualize_fit`` / ``_visualize_before_fit`` switches its updater reads.
        """
        super().__init__()
        self.search = search

    def optimise(
        self,
        factor_approx: FactorApproximation,
        status: Status = Status(),
    ) -> Tuple[MeanField, Status]:
        """
        Perform optimisation for expectation propagation. Currently only
        applicable for ModelFactors created by the declarative interface.

        1. Analysis and model classes are extracted from the factor.
        2. Priors are updated from the mean field.
        3. Analysis and model are fit as usual.
        4. A new mean field is constructed with the (posterior) 'linking' priors.
        5. Projection is performed to produce an updated EPMeanField object.

        Output directories are generated according to the factor and the number
        of the search. For example a factor called "factor" would output:

        factor/optimization_0/<identifier>
        factor/optimization_1/<identifier>
        factor/optimization_2/<identifier>

        For the first, second and third optimizations respectively.

        Parameters
        ----------
        factor_approx
            A collection of messages defining the current best approximation to
            some global model
        status

        Returns
        -------
        An updated approximation to the model having performed optimisation on
        a single factor.
        """
        from autofit.graphical.declarative.abstract import PriorFactor
        from autofit.graphical.declarative.factor.analysis import AnalysisFactor
        from autofit.graphical.declarative.factor.hierarchical import (
            _HierarchicalFactor,
        )
        from autofit.non_linear.parallel import check_factor_search_cores
        from autofit.non_linear.paths.sub_directory_paths import SubDirectoryPaths

        search = self.search
        factor = factor_approx.factor

        if not isinstance(factor, (AnalysisFactor, PriorFactor, _HierarchicalFactor)):
            raise NotImplementedError(
                f"Optimizer {search.__class__.__name__} can only be applied to"
                f" AnalysisFactors, HierarchicalFactors and PriorFactors"
            )

        # A deliberate refusal, not a silent downgrade: EP never forks a factor
        # search (`autofit.non_linear.parallel.pool.check_factor_search_cores`).
        check_factor_search_cores(search)

        model = factor.prior_model.mapper_from_prior_arguments(
            {
                prior: prior.with_message(message)
                for prior, message in factor_approx.cavity_dist.arguments.items()
            }
        )

        analysis = factor.analysis

        search.logger.info(
            f"EP factor step [{factor.name}]: running the factor search "
            f"{search.__class__.__name__} serially (number_of_cores=1); "
            + (
                "parallelism comes from the vectorised JAX likelihood "
                "(analysis.use_jax=True)."
                if analysis.is_jax
                else "the likelihood is not JAX-vectorised, so this factor step "
                "is single-threaded (set `Analysis(use_jax=True)` for "
                "parallelism)."
            )
        )

        number = search.optimisation_counter[factor.name]

        search.optimisation_counter[factor.name] += 1

        # A fresh paths object per factor step: the search's cached updater compares
        # paths by identity (`NonLinearSearch._updater`) and rebuilds itself, so it
        # never writes into a previous step's folder.
        search.paths = SubDirectoryPaths(
            parent=search.paths,
            analysis_name=f"{factor.name}/optimization_{number}",
            is_flat=True,
        )

        # EP re-enters the same factor search once per EP step. By default only
        # the first search of each factor draws its before-fit visuals and no
        # factor search draws per-search visuals (corner plots, model-fit
        # images): at small per-factor fits the plotting dominates the wall
        # time and the images are overwritten every step anyway. Samples and
        # summaries are still written, and the EP optimiser's own visuals
        # (graph.png, ep_history) are unaffected. `.get` with a default so user
        # configs that predate the key keep working.
        visualize_ep_factor_searches = bool(
            conf.instance["general"]["output"].get(
                "visualize_ep_factor_searches", False
            )
        )

        search._visualize_fit = visualize_ep_factor_searches
        search._visualize_before_fit = visualize_ep_factor_searches or number == 0

        # Marks the fit as an EP factor search, so a `number_of_cores="auto"` request
        # resolves to 1 (`autofit.non_linear.parallel.pool.resolve_number_of_cores`).
        search._ep_factor_search = True

        try:
            result = search.fit(model=model, analysis=analysis)
        finally:
            search._visualize_fit = True
            search._visualize_before_fit = True
            search._ep_factor_search = False

        # Record the sampler's log-evidence of this tilted-distribution fit on
        # the projected mean field — the per-factor Ẑₐ that README §5 documents
        # `MeanField.log_norm` as carrying (#1332 F7(b)). Previously always 0,
        # so `EPMeanField.log_evidence` could not be trusted for model
        # comparison in sampler-driven EP fits. Searches with no evidence
        # estimate (MCMC / MLE) yield None and keep the 0.0 default — evidence-
        # correct model comparison requires nested-sampling factor searches.
        # (Both levels guarded: e.g. StaticResult carries no samples at all.)
        log_evidence = getattr(getattr(result, "samples", None), "log_evidence", None)

        new_model_dist = MeanField.from_priors(
            result.projected_model.priors,
            log_norm=log_evidence if log_evidence is not None else 0.0,
        )

        # Release the in-memory search internal before the result is pinned on
        # the `Status`.
        #
        # `EPHistory`/`FactorHistory` append every `(approx, status)` pair of
        # the run, so every `Status`, and therefore every `Result`, lives until
        # the EP fit ends. `Result._search_internal` is the sampler, the
        # sampler's likelihood callable is a `Fitness`, and a `Fitness` owns
        # that factor step's compiled JAX executables (its `_vmap` / `_jit` /
        # `_grad` `cached_property` caches). Left attached, an EP run therefore
        # accumulates the compiled code of every factor search it has ever run:
        # RAL job 342410 (`slope_hierarchy_scale`, 25 analysis factors) held 76
        # searches' executables and died with `LLVM ERROR: Unable to allocate
        # section memory!` inside 64 GB, after ~3 EP steps.
        #
        # Nothing in `autofit.graphical` reads the sampler afterwards (the
        # consumers of `latest_result` use `.projected_model`, `.samples` and
        # `.model`), and `Result.search_internal` still falls back to
        # `paths.load_search_internal()`, so the sampler remains reachable from
        # the on-disk dill for anything that does want it.
        #
        # Guarded because `optimise` does not require a `Result`: the regression
        # suite's `StaticSearch` returns a bare object carrying only
        # `projected_model` (as the `getattr` on `samples` above allows for).
        release_search_internal = getattr(result, "release_search_internal", None)

        if release_search_internal is not None:
            release_search_internal()

        status.result = result

        return new_model_dist, status
