from abc import ABC, abstractmethod
from collections import Counter
from typing import Set, List, Dict, Optional
from typing import Tuple

from autofit.graphical.declarative.factor.prior import PriorFactor
from autofit.graphical.declarative.graph import DeclarativeFactorGraph
from autofit.graphical.expectation_propagation import (
    AbstractFactorOptimiser,
    ApproxUpdater,
)
from autofit.graphical.expectation_propagation import EPMeanField, EPOptimiser
from autofit.mapper.prior.abstract import Prior
from autofit.mapper.prior_model.collection import Collection
from autofit.mapper.variable import Plate
from autofit.messages.normal import NormalMessage
from autofit.non_linear.analysis import Analysis
from autofit.non_linear.paths.abstract import AbstractPaths


class AbstractDeclarativeFactor(Analysis, ABC):
    optimiser: AbstractFactorOptimiser
    _plates: Tuple[Plate, ...] = ()

    def __init__(self, include_prior_factors=False, use_jax : bool = False):
        self._include_prior_factors = include_prior_factors

        super().__init__(use_jax=use_jax)

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    def model_factors(self) -> List["AbstractDeclarativeFactor"]:
        return [self]

    @property
    @abstractmethod
    def prior_model(self):
        pass

    @property
    @abstractmethod
    def info(self):
        pass

    @property
    def priors(self) -> Set[Prior]:
        """
        A set of all priors encompassed by the contained likelihood models
        """
        return {
            prior for model in self.model_factors for prior in model.prior_model.priors
        }

    @property
    def prior_counts(self) -> List[Tuple[Prior, int]]:
        """
        A dictionary mapping unique priors in the graph to the number of factors
        in which each prior occurs.
        """
        counter = Counter()
        for factor in self.model_factors:
            for prior in factor.prior_model.priors:
                counter[prior] += 1
        return [
            (prior, count + 1 if self._include_prior_factors else count)
            for prior, count in counter.items()
        ]

    @property
    def prior_factors(self) -> List[PriorFactor]:
        """
        A list of factors that act as priors on latent variables. One factor exists
        for each unique prior.
        """
        return list(map(PriorFactor, sorted(self.priors)))

    @property
    def message_dict(self) -> Dict[Prior, NormalMessage]:
        """
        Dictionary mapping priors to messages.

        Messages are computed to the power of one over the number of factors
        a prior is shared by. This means that when the cavity distribution
        is computed for the first optimisation the message has the same sigma
        as the user defined prior.
        """
        return {
            prior: prior.message ** (1 / (count - 1)) if count > 1 else prior.message
            for prior, count in self.prior_counts
        }

    @property
    def graph(self) -> DeclarativeFactorGraph:
        """
        The complete graph made by combining all factors and priors
        """
        factors = [model for model in self.model_factors]
        if self._include_prior_factors:
            factors += self.prior_factors
        # noinspection PyTypeChecker
        return DeclarativeFactorGraph(factors)

    def draw_graph(self, **kwargs):
        """
        Visualise the graph.

        Variables and Factors are nodes labelled according to their corresponding
        prior or model respectively.

        Parameters
        ----------
        kwargs
            Arguments passed to visualisation
        """
        graph = self.graph

        factor_labels = {
            factor: factor.name if factor.label is None else factor.label
            for factor in graph.factors
        }
        variable_labels = {
            variable: variable.name if variable.label is None else variable.label
            for variable in graph.all_variables
        }

        import matplotlib.pyplot as plt

        if "draw_labels" not in kwargs:
            kwargs["draw_labels"] = True
        if "variable_labels" not in kwargs:
            kwargs["variable_labels"] = variable_labels
        if "factor_labels" not in kwargs:
            kwargs["factor_labels"] = factor_labels
        graph.draw_graph(**kwargs)
        plt.show()
        plt.close()

    @property
    def plates(self):
        return self._plates

    def mean_field_approximation(self) -> EPMeanField:
        """
        Returns a EPMeanField of the factor graph
        """
        return EPMeanField.from_approx_dists(self.graph, self.message_dict)

    def _make_ep_optimiser(
        self,
        optimiser: AbstractFactorOptimiser,
        paths: Optional[AbstractPaths] = None,
        ep_history: Optional = None,
        updater: Optional[ApproxUpdater] = None,
    ) -> EPOptimiser:
        return EPOptimiser(
            self.graph,
            default_optimiser=optimiser,
            factor_optimisers={
                factor: factor.optimiser
                for factor in self.model_factors
                if factor.optimiser is not None
            },
            ep_history=ep_history,
            paths=paths,
            updater=updater,
        )

    def optimise(
        self,
        optimiser: AbstractFactorOptimiser,
        paths: Optional[AbstractPaths] = None,
        ep_history: Optional = None,
        updater: Optional[ApproxUpdater] = None,
        **kwargs,
    ):
        """
        Use an EP Optimiser to optimise the graph associated with this collection
        of factors and create a Collection to represent the results.

        Parameters
        ----------
        paths
            Optionally define how data should be output. This paths
            object is copied to every optimiser.
        optimiser
            An optimiser that acts on graphs
        updater
            An optional policy controlling how strongly EP factor messages are
            updated. If omitted, EP keeps its existing undamped update policy.

        Returns
        -------
        ep_result: EPResult
            A collection of prior models
        """
        from autofit.graphical.declarative.result import EPResult

        opt = self._make_ep_optimiser(
            optimiser,
            paths=paths,
            ep_history=ep_history,
            updater=updater,
        )
        updated_ep_mean_field = opt.run(self.mean_field_approximation(), **kwargs)

        return EPResult(
            ep_history=opt.ep_history,
            declarative_factor=self,
            updated_ep_mean_field=updated_ep_mean_field,
        )

    @property
    def global_prior_model(self) -> Collection:
        """
        A collection of prior models, with one model for each factor.
        """
        return GlobalPriorModel(self)


class GlobalPriorModel(Collection):
    def __init__(self, factor: AbstractDeclarativeFactor):
        """
        A global model comprising all factors which can be used to compare
        results between global optimisation and expectation propagation.

        Parameters
        ----------
        factor
            A factor comprising one or more factors, usually a graph
        """
        super().__init__(
            [model_factor.prior_model for model_factor in factor.model_factors]
        )
        self.factor = factor

    @property
    def graph_info(self) -> str:
        """
        A string describing the collection of factors in the graphical style
        """
        return self.factor.graph.info
