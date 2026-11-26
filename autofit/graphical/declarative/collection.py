from typing import Dict, List, Union, Optional

from autofit.graphical.declarative.factor.analysis import AnalysisFactor
from autofit.graphical.declarative.factor.hierarchical import HierarchicalFactor

from autofit.tools.namer import namer
from .abstract import AbstractDeclarativeFactor
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.samples.pdf import SamplesPDF
from autofit.non_linear.samples.summary import SamplesSummary

from autofit.mapper.model import ModelInstance
from autofit.mapper.prior_model.prior_model import Model

from ...non_linear.combined_result import CombinedResult

class FactorGraphModel(AbstractDeclarativeFactor):
    def __init__(
        self,
        *model_factors: Union[AbstractDeclarativeFactor, HierarchicalFactor],
        name=None,
        include_prior_factors=True,
        use_jax: Optional[bool] = None,
    ):
        """
        A collection of factors that describe models, which can be
        used to create a graph and messages.

        If the models have shared priors then the graph has shared variables

        Parameters
        ----------
        model_factors
            Factors which are hierarchical or associated with a specific analysis
        use_jax
            Whether the whole graph is evaluated on JAX when it is fitted as one
            analysis (``search.fit(model=graph.global_prior_model, analysis=graph)``).
            ``None`` (the default) derives it from the factors: the graph is JAX when
            every factor is. An explicit bool is honoured, but whole-graph fitting
            requires every factor to agree with it and raises ``SearchException``
            naming the factors that do not (``check_backend_agreement``). Per-factor
            expectation propagation (``graph.optimise``) fits each factor with its own
            analysis, so mixed numpy/JAX factors are allowed there.
        """
        super().__init__(
            include_prior_factors=include_prior_factors,
            use_jax=use_jax,
        )
        self._model_factors = list(model_factors)
        self._name = name or namer(self.__class__.__name__)

    def tree_flatten(self):
        return (
            (self._model_factors,),
            (self._name, self._include_prior_factors, self._explicit_use_jax),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            *children[0],
            name=aux_data[0],
            include_prior_factors=aux_data[1],
            use_jax=aux_data[2],
        )

    def __setstate__(self, state):
        """
        Restore a pickled graph, migrating pickles written before ``use_jax`` could be
        derived from the factors.

        Those stored the graph's backend as a plain ``_use_jax`` attribute, which the
        ``_use_jax`` property below now shadows; it is moved to ``_explicit_use_jax``
        so the graph keeps the backend it was built with (then always an explicit
        bool, ``False`` by default) and ``tree_flatten`` finds the attribute.
        """
        state = dict(state)
        if "_explicit_use_jax" not in state:
            state["_explicit_use_jax"] = state.pop("_use_jax", None)
        self.__dict__.update(state)

    @property
    def _use_jax(self) -> bool:
        """
        The graph's backend: the explicit ``use_jax`` when one was given, otherwise
        ``True`` exactly when every factor is JAX (``is_jax`` reads this).

        The factors are the flattened ones (a ``HierarchicalFactor`` contributes the
        factors it generates), the same set ``check_backend_agreement`` reads, so a
        derived backend always agrees with its own factors.
        """
        explicit = self.__dict__.get("_explicit_use_jax")
        if explicit is not None:
            return bool(explicit)
        flags = [factor.is_jax for factor in self._flat_factors()]
        return bool(flags) and all(flags)

    @_use_jax.setter
    def _use_jax(self, value: Optional[bool]):
        # Set by `Analysis.__init__`: `None` means "derive from the factors".
        self._explicit_use_jax = value

    def _flat_factors(self):
        model_factors = list()
        for model_factor in self.__dict__.get("_model_factors", []):
            if isinstance(model_factor, HierarchicalFactor):
                model_factors.extend(model_factor.factors)
            else:
                model_factors.append(model_factor)
        return model_factors

    def factors_disagreeing_on_backend(self) -> List[str]:
        """
        The names of the factors whose ``is_jax`` differs from the graph's.
        """
        is_jax = self.is_jax
        return [
            factor.name for factor in self._flat_factors() if factor.is_jax != is_jax
        ]

    def check_backend_agreement(self):
        """
        Raise ``SearchException`` unless every factor agrees with the graph's backend.

        Whole-graph fitting evaluates every factor inside one objective, so a JAX graph
        traces numpy factors (which fail on tracers) and a numpy graph never compiles a
        JAX factor. ``NonLinearSearch`` calls this before fitting a graph as one
        analysis; per-factor expectation propagation never does.
        """
        disagreeing = self.factors_disagreeing_on_backend()
        if disagreeing:
            from autofit import exc

            raise exc.SearchException(
                f"The FactorGraphModel {self.name} is fitted as one analysis with "
                f"is_jax={self.is_jax}, but these factors have "
                f"is_jax={not self.is_jax}: {', '.join(disagreeing)}. Whole-graph "
                f"fitting requires every factor to use the same backend: build every "
                f"factor's analysis with the same use_jax (and HierarchicalFactor with "
                f"the same use_jax), or omit FactorGraphModel(use_jax=...) so the graph "
                f"derives it from its factors. Per-factor expectation propagation "
                f"(graph.optimise) allows mixed factors."
            )

    def modify_before_fit(self, paths: AbstractPaths, model):
        """
        Forward ``modify_before_fit`` to every factor's analysis before a whole-graph fit,
        so each factor resolves its own compute setup (e.g. a PyAutoGalaxy / PyAutoLens
        ``setup=``) exactly as it would fitted alone.

        Each analysis factor gets its own sub-paths (``analyses/analysis_<i>``, as
        ``_for_each_analysis`` names them) and its own model: the ``i``-th child of the
        fitted ``model`` (which may carry replacement priors, e.g. in a grid-search
        cell), else the factor's ``prior_model``. ``AnalysisFactor.modify_before_fit``
        rebinds both of the factor's analysis references to the object returned.
        Hierarchical factors carry no analysis and are skipped.

        Factors whose setup is the implicit default (``setup_is_default``, e.g. a
        PyAutoGalaxy / PyAutoLens analysis built with neither ``setup`` nor ``use_jax``)
        then adopt one shared backend when their default setups disagree
        (``adopt_shared_backend``), so a graph mixing dataset types fits without pins.
        The factors must then agree on their resolved setup (``check_setup_agreement``);
        setups the user passed are never changed, so disagreeing explicit setups raise.

        Returns
        -------
        This graph, its factors now holding their modified analyses.
        """
        factors = self.model_factors
        for i, factor in enumerate(factors):
            if not isinstance(factor, (AnalysisFactor, FactorGraphModel)):
                continue
            child_paths = paths.for_sub_analysis(analysis_name=f"analyses/analysis_{i}")
            factor.modify_before_fit(child_paths, self._factor_model(model, i, factor))

        self.adopt_shared_backend()
        self.check_setup_agreement()
        return self

    def _setup_factors(self) -> List:
        """
        The factors that may carry a compute setup: analysis factors and nested graphs
        (the flattened factors, so a ``HierarchicalFactor``'s generated factors, which
        have no analysis, report no default setup).
        """
        return [
            factor
            for factor in self._flat_factors()
            if isinstance(factor, (AnalysisFactor, FactorGraphModel))
        ]

    @property
    def setup_is_default(self) -> bool:
        """
        Whether this graph, as a factor of an enclosing graph, has only default setups:
        ``True`` when at least one of its factors has a default setup and none has a
        setup the user passed.
        """
        ids = self.setup_ids()
        factors = self._setup_factors()
        defaulted = [factor for factor in factors if factor.setup_is_default]
        explicit = [
            factor
            for factor in factors
            if factor.name in ids and not factor.setup_is_default
        ]
        return bool(defaulted) and not explicit

    def adopt_backend(self, use_jax: bool):
        """
        Move every factor of this graph with a default setup onto ``use_jax``
        (``Analysis.adopt_backend``), when this graph is a factor of an enclosing graph.
        """
        for factor in self._setup_factors():
            if factor.setup_is_default:
                factor.adopt_backend(use_jax)

    def _shared_backend(self, explicit_factors) -> Optional[bool]:
        """
        The backend factors with a default setup adopt: the graph's explicit ``use_jax``
        when one was given, else the backend of the factors the user configured (an
        explicit ``setup=`` or ``use_jax=``), else JAX, unless JAX is disabled
        (``PYAUTO_DISABLE_JAX=1``) or not installed. ``None`` when the configured
        factors disagree on their backend (``check_backend_agreement`` then raises).
        """
        explicit = self.__dict__.get("_explicit_use_jax")
        if explicit is not None:
            return bool(explicit)
        if explicit_factors:
            backends = {bool(factor.is_jax) for factor in explicit_factors}
            return backends.pop() if len(backends) == 1 else None

        import importlib.util
        import os

        if os.environ.get("PYAUTO_DISABLE_JAX") == "1":
            return False
        return importlib.util.find_spec("jax") is not None

    def adopt_shared_backend(self) -> Optional[bool]:
        """
        Put every factor whose compute setup is the implicit default on one shared
        backend, when the factors would otherwise disagree.

        Each factor resolves its own setup (``modify_before_fit``), so a graph mixing
        dataset types (e.g. imaging and interferometer, or imaging and a point source)
        resolves a different default setup per factor, which a whole-graph fit cannot
        run. When the resolved setups disagree, or a factor with a default setup runs on
        a different backend from the graph's ``use_jax`` or from the factors the user
        configured, every factor with a default setup (``setup_is_default``) drops it
        and runs with no setup on the shared backend (``adopt_backend``, see
        ``_shared_backend``). Factors the user configured (``setup=`` or ``use_jax=``)
        are never changed: when their setups or backends disagree with one another
        nothing is adopted and ``check_setup_agreement`` / ``check_backend_agreement``
        raise.

        Returns
        -------
        The backend adopted (``True`` for JAX), or ``None`` when nothing changed.
        """
        factors = self._setup_factors()
        defaulted = [factor for factor in factors if factor.setup_is_default]
        if not defaulted:
            return None

        configured = [factor for factor in factors if not factor.setup_is_default]

        ids = self.setup_ids()
        configured_ids = {ids[f.name] for f in configured if f.name in ids}
        if len(configured_ids) > 1:
            return None

        use_jax = self._shared_backend(explicit_factors=configured)
        if use_jax is None:
            return None

        backend_is_set = (
            self.__dict__.get("_explicit_use_jax") is not None or bool(configured)
        )
        setups_disagree = len(set(ids.values())) > 1
        backends_disagree = backend_is_set and any(
            bool(factor.is_jax) != use_jax for factor in defaulted
        )

        if not (setups_disagree or backends_disagree):
            return None

        for factor in defaulted:
            factor.adopt_backend(use_jax)

        return use_jax

    def _factor_model(self, model, index: int, factor):
        """
        The model of factor ``index`` within the fitted graph ``model``, or the factor's
        own ``prior_model`` when ``model`` does not have one child per factor.
        """
        try:
            if model is not None and len(model) == len(self.model_factors):
                return model[index]
        except (TypeError, KeyError, IndexError, AttributeError):
            pass
        return factor.prior_model

    def setup_ids(self) -> Dict[str, str]:
        """
        The resolved compute setup of each factor that has one, as {factor name:
        setup_id} (``analysis.setup.setup_id``, duck-typed: factors whose analysis has
        no ``setup`` are left out).
        """
        from autofit.non_linear.parallel.pool import setup_id_of

        ids = {}
        for factor in self._flat_factors():
            setup_id = setup_id_of(factor)
            if setup_id is not None:
                ids[factor.name] = setup_id
        return ids

    def check_setup_agreement(self):
        """
        Raise ``SearchException`` when the factors resolved different compute setups.

        A whole-graph fit evaluates every factor in one process, so every factor that
        has a setup must have resolved the same one (factors without a setup are
        ignored). Called at the end of ``modify_before_fit`` (after factors with a
        default setup have adopted a shared backend, ``adopt_shared_backend``) and by
        the runtime gates beside ``check_backend_agreement``.
        """
        ids = self.setup_ids()
        if len(set(ids.values())) <= 1:
            return

        from autofit import exc

        by_setup = {}
        for name, setup_id in ids.items():
            by_setup.setdefault(setup_id, []).append(name)
        listing = "; ".join(
            f"{setup_id}: {', '.join(names)}" for setup_id, names in by_setup.items()
        )
        raise exc.SearchException(
            f"The FactorGraphModel {self.name} is fitted as one analysis, but its "
            f"factors resolved different compute setups ({listing}). Whole-graph "
            f"fitting evaluates every factor in one process, so every factor must use "
            f"the same setup. Factors left on the default setup (neither setup= nor "
            f"use_jax= passed) adopt one shared backend automatically; setups passed "
            f"explicitly are never changed, so build every factor's analysis with the "
            f"same setup=, or pass none so the graph chooses the backend "
            f"(FactorGraphModel(use_jax=...) sets it)."
        )

    @property
    def gradient_mode(self) -> str:
        """
        The factors' common ``gradient_mode``; ``"reverse"`` (the ``Analysis`` default)
        when the factors declare different modes. Both modes compute the same gradient,
        so a mixed graph falls back to the mode whose memory does not scale with the
        number of free parameters.
        """
        modes = {
            getattr(factor, "gradient_mode", "reverse")
            for factor in self._flat_factors()
        }
        if len(modes) == 1:
            return modes.pop()
        return "reverse"

    @property
    def prior_model(self):
        """
        Construct a Collection comprising the prior models described
        in each model factor
        """
        from autofit.mapper.prior_model.collection import Collection

        return Collection(
            {factor.name: factor.prior_model for factor in self.model_factors}
        )

    @property
    def optimiser(self):
        raise NotImplemented()

    @property
    def info(self) -> str:
        """
        Info describing this collection.
        """
        return self.graph.info

    @property
    def name(self):
        return self._name

    def add(self, model_factor: AbstractDeclarativeFactor):
        """
        Add another factor to this collection.
        """
        self._model_factors.append(model_factor)

    def log_likelihood_function(self, instance: ModelInstance) -> float:
        """
        Compute the combined likelihood of each factor from a collection of instances
        with the same ordering as the factors.

        Before the per-factor loop, the lead factor is asked to compute a `shared`
        object via `Analysis.shared_state_from` (see that method). If it returns a
        non-`None` value — the opt-in case — that object is forwarded as the `shared`
        keyword argument to every factor's `log_likelihood_function`, so that work
        which is identical for all factors at this point in parameter space is computed
        once and reused rather than recomputed for each factor.

        When no factor provides a shared object (the default) the loop calls each
        factor's `log_likelihood_function` exactly as it would without this mechanism,
        so existing graphs are unchanged.

        Parameters
        ----------
        instance
            A collection of instances, one corresponding to each factor

        Returns
        -------
        The combined likelihood of all factors
        """
        shared = self._shared_state_from(instance)

        log_likelihood = 0
        for model_factor, instance_ in zip(self.model_factors, instance):
            if shared is None:
                log_likelihood += model_factor.log_likelihood_function(instance_)
            else:
                log_likelihood += model_factor.log_likelihood_function(
                    instance_, shared=shared
                )

        return log_likelihood

    def _shared_state_from(self, instance: ModelInstance):
        """
        Compute the per-evaluation object shared across factors, by asking each factor's
        `Analysis` in turn (via `Analysis.shared_state_from`) until one returns a
        non-`None` value — the "lead" factor. Returns `None` if no factor opts in, in
        which case no state is shared this evaluation.
        """
        for model_factor, instance_ in zip(self.model_factors, instance):
            analysis = getattr(model_factor, "analysis", None)
            shared_state_from = getattr(analysis, "shared_state_from", None)
            if shared_state_from is None:
                continue
            shared = shared_state_from(instance_)
            if shared is not None:
                return shared

        return None

    @property
    def model_factors(self):
        return self._flat_factors()

    def make_result(
        self,
        samples_summary: SamplesSummary,
        paths: AbstractPaths,
        samples: Optional[SamplesPDF] = None,
        search_internal: Optional[object] = None,
        analysis: Optional[object] = None,
    ) -> CombinedResult:
        """
        Make a result from the samples summary and paths.

        The top level result accounts for the combined model.
        There is one child result for each model factor.

        Parameters
        ----------
        samples_summary
            A summary of the samples
        paths
            Handles saving and loading data
        samples
            The full list of samples
        search_internal
        analysis

        Returns
        -------
        A result with child results for each model factor
        """
        child_results = []
        for index, model_factor in enumerate(self.model_factors):
            # Use the fitted submodel, which may carry replacement priors in grid cells.
            fitted_model = samples_summary.model[index]
            child_results.append(
                model_factor.analysis.make_result(
                    samples_summary=samples_summary.subsamples(fitted_model),
                    paths=paths,
                    samples=samples.subsamples(fitted_model) if samples else None,
                    search_internal=search_internal,
                    analysis=model_factor,
                )
            )
        return CombinedResult(
            child_results,
            samples_summary=samples_summary,
            paths=paths,
            samples=samples,
            search_internal=search_internal,
            analysis=analysis,
        )

    def _for_each_analysis(
        self,
        name,
        paths,
        *args,
        **kwargs,
    ):
        """
        Convenience function to call an underlying function for each
        analysis with a paths object with an integer attached to the
        end.

        Parameters
        ----------
        func
            Some function of the analysis class
        paths
            An object describing the paths for saving data (e.g. hard-disk directories or entries in sqlite database).
        """
        results = []
        for (i, analysis), *args in zip(
            enumerate(self.model_factors),
            *args,
        ):
            child_paths = paths.for_sub_analysis(analysis_name=f"analyses/analysis_{i}")
            func = getattr(analysis, name)
            results.append(
                func(
                    child_paths,
                    *args,
                    **kwargs,
                )
            )

        return results

    def visualize(
        self,
        paths: AbstractPaths,
        instance: ModelInstance,
        during_analysis: bool,
    ):
        """
        Visualise the instances provided using each factor.

        Instances in the ModelInstance must have the same order as the factors.

        Parameters
        ----------
        paths
            Object describing where data should be saved to
        instance
            A collection of instances, each corresponding to a factor
        during_analysis
            Is this visualisation during analysis?
        """
        self._for_each_analysis(
            "visualize",
            paths,
            instance,
            during_analysis=during_analysis,
        )

    def visualize_before_fit(
        self,
        paths: AbstractPaths,
        model: Model,
    ):
        """
        Visualise the model provided using each factor.

        Models in the ModelInstance must have the same order as the factors.

        Parameters
        ----------
        paths
            Object describing where data should be saved to
        model
            A collection of models, each corresponding to a factor
        """
        self._for_each_analysis(
            "visualize_before_fit",
            paths,
            model,
        )

    def save_attributes(self, paths: AbstractPaths):
        """
        Save the attributes of the analysis to the paths object.
        """
        self._for_each_analysis("save_attributes", paths)

    def save_results(self, paths: AbstractPaths, result):
        """
        Save the results of the analysis to the paths object.
        """
        self._for_each_analysis("save_results", paths, result)

    def _factors_grouped_by_visualizer(self, instance):
        """
        ``(factors, instances)`` pairs grouped by each factor's analysis ``Visualizer``
        class, order preserved within groups.

        A combined visualization can only draw analyses its ``Visualizer`` understands,
        so a mixed-dataset graph (e.g. imaging + weak-lensing factors) must dispatch one
        combined call per visualizer type rather than routing every factor into the lead
        factor's visualizer. A homogeneous graph produces exactly one group, making the
        dispatch identical to the previous lead-factor-takes-all behaviour.
        """
        groups = {}
        for factor, single_instance in zip(self.model_factors, instance):
            analysis = getattr(factor, "analysis", factor)
            key = getattr(type(analysis), "Visualizer", type(analysis))
            groups.setdefault(key, ([], []))
            groups[key][0].append(factor)
            groups[key][1].append(single_instance)
        return groups.values()

    def visualize_combined(
        self,
        instance,
        paths: AbstractPaths,
        during_analysis,
    ):
        for factors, instances in self._factors_grouped_by_visualizer(instance):
            factors[0].visualize_combined(
                factors,
                paths,
                instances,
                during_analysis=during_analysis,
            )

    def perform_quick_update(self, paths, instance):

        try:
            for factors, instances in self._factors_grouped_by_visualizer(instance):
                factors[0].visualize_combined(
                    analyses=factors,
                    paths=paths,
                    instance=instances,
                    during_analysis=True,
                    quick_update=True,
                )
        except Exception as e:
            pass
