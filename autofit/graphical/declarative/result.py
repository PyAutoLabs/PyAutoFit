from typing import Union, List

from autofit.graphical.declarative.factor.hierarchical import HierarchicalFactor
from autofit.graphical.expectation_propagation import EPHistory
from autofit.graphical.expectation_propagation.ep_mean_field import EPMeanField
from autofit.graphical.factor_graphs.factor import Factor
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.mapper.prior_model.collection import Collection
from autofit.non_linear.result import Result, AbstractResult
from autofit.non_linear.samples.samples import Samples


class HierarchicalResult(AbstractResult):
    def __init__(self, results: List[AbstractResult]):
        """
        One true factor is created per each variable drawn from the declarative
        hierarchical factor. One result for each of those factors is in this
        collection.

        Parameters
        ----------
        results
            Results from hierarchical factor optimisations
        """
        super().__init__(samples_summary=None, paths=None)
        self.results = results

    @property
    def samples(self) -> Samples:
        """
        All samples from all underlying optimisations are summed together.
        """
        return sum(result.samples for result in self.results)

    @property
    def model(self) -> AbstractPriorModel:
        """
        Priors with the same path in the model are combined by taking the
        product of their underlying messages. i.e. the product of the
        posteriors.

        The distribution model is returned.
        """
        return AbstractPriorModel.product(
            result.model for result in self.results
        ).distribution_model

    @property
    def instance(self):
        """
        Return the instance (e.g. prior) describing the distribution
        from which samples are drawn.
        """
        return self.samples.instance.distribution_model


class EPResult:
    def __init__(
        self,
        ep_history: EPHistory,
        declarative_factor,
        updated_ep_mean_field: EPMeanField,
        factor_graph=None,
    ):
        """
        The result of an EP Optimisation including its history, a declarative
        representation of the optimised graph and the resultant EPMeanField
        which comprises factors and resultant approximations of variables.

        Parameters
        ----------
        ep_history
            A history of the optimisation
        declarative_factor: AbstractDeclarativeFactor
            A declarative representation of the factor being optimised
        updated_ep_mean_field
            An updated mean field; effectively the result of the optimisation
        factor_graph
            The factor graph the EP optimiser actually swept
            (``EPOptimiser.factor_graph``), which is the object to hand to
            ``af.EPPlotter`` for ``kind="state"``. Never
            ``FactorGraphModel.graph``: that property builds a fresh graph,
            renaming every prior factor, on every access, so the history's
            keys would not match it.
        """
        self.ep_history = ep_history
        self.declarative_factor = declarative_factor
        self.updated_ep_mean_field = updated_ep_mean_field
        self.factor_graph = factor_graph

    @property
    def model(self) -> Collection:
        """
        A collection populated with messages representing the posteriors of
        the EP Optimisation. Each item in the collection represents a single
        factor in the optimisation.
        """
        collection = Collection(
            {
                factor.name: factor.prior_model
                for factor in self.declarative_factor.model_factors
            }
        )
        arguments = {
            prior: prior.with_message(self.updated_ep_mean_field.mean_field[prior])
            for prior in collection.priors
        }

        return collection.gaussian_prior_model_for_arguments(arguments)

    @property
    def latest_results(self) -> List[Result]:
        """
        A list of results for all analysis and hierarchical factors.
        """
        return [
            self.ep_history[factor].latest_result
            for factor in self.declarative_factor.model_factors
        ]

    @property
    def instance(self):
        """
        The median instance taken from the updated model
        """
        return self.model.instance_from_prior_medians()

    def latest_for(self, factor: Union[Factor, HierarchicalFactor]) -> AbstractResult:
        """
        Return the latest result for a factor.

        If the factor is hierarchical return the latest result for
        the first true factor (i.e. for one of the drawn variable)

        Parameters
        ----------
        factor
            A factor from the graph

        Returns
        -------
        The latest result for that factor
        """
        if isinstance(factor, HierarchicalFactor):
            results = [
                self.ep_history[child_factor].latest_result
                for child_factor in factor.factors
            ]
            return HierarchicalResult(results)
        return self.ep_history[factor].latest_result
