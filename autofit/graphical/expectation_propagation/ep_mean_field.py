import logging
from typing import Dict, Tuple, Optional, List

import numpy as np

from autonerves import cached_property
from autofit.graphical.factor_graphs.factor import Factor
from autofit.graphical.factor_graphs.graph import FactorGraph
from autofit.graphical.mean_field import MeanField, FactorApproximation
from autofit.graphical.utils import Status
from autofit.mapper.variable import Variable, Plate, VariableData
from autofit.messages.abstract import AbstractMessage

logger = logging.getLogger(__name__)


class EPMeanField(FactorGraph):
    """
    this class encode the EP mean-field approximation to a factor graph


    Attributes
    ----------
    factor_graph: FactorGraph
        the base factor graph being approximated

    factor_mean_field: Dict[Factor, MeanField]
        the mean-field approximation for each factor in the factor graph

    mean_field: MeanField
        the mean-field approximation of the full factor graph
        i.e. the product of the factor mean-field approximations

    variables: Set[Variable]
        the variables of the approximation

    deterministic_variables: Set[Variable]
        the deterministic variables

    log_evidence: float
        the approximate log evidence of the approximation

    is_valid: bool
        returns whether the factor mean-field approximations are all valid

    Methods
    -------
    from_approx_dists(factor_graph, approx_dists)
        create a EPMeanField object from the passed factor_graph
        using approx_dists to initialise the factor mean-field approximations

    factor_approximation(factor)
        create the FactorApproximation for the factor

    project_factor_approx(factor_approximation)
        given the passed FactorApproximation, return a new `EPMeanField`
        object encoding the updated mean-field approximation
    """

    def __init__(
            self, factor_graph: FactorGraph, factor_mean_field: Dict[Factor, MeanField]
    ):
        self._factor_graph = factor_graph
        self._factor_mean_field = factor_mean_field

        super().__init__(self.factor_graph.factors)

    @property 
    def parameters(self):
        return {
            'factor_graph': self._factor_graph, 
            'factor_mean_field': self._factor_mean_field
        }

    def copy(self):
        return type(self)(**{k: val.copy() for k, val in self.parameters.items()})

    def subset(self, plates_index: Dict[Plate, List[int]]) -> "EPMeanFieldSubset":
        """Given a dictionary of Plates with a subset of indexes 
        returns the EPMeanFieldSubset that corresponds to the subset of indexes.
        """
        factor_subset_factor = {}
        factor_mean_field_subset = {}
        factor_mean_field_rescale = {}
        for factor, mean_field in self.factor_mean_field.items():
            plate_sizes = VariableData.plate_sizes(mean_field)
            factor_subset_factor[factor] = subset_factor = factor.subset(
                plates_index, plate_sizes
            )

            mean_field_subset = mean_field.subset(plates_index=plates_index)
            factor_mean_field_subset[subset_factor] = mean_field_subset
            mean_field_size = VariableData.prod(plate_sizes)
            subset_size = VariableData.prod(VariableData.plate_sizes(mean_field_subset))
            scale_factor = subset_size / mean_field_size
            factor_mean_field_rescale[subset_factor] = VariableData({
                v: scale_factor * mean_field[v].size / message.size
                for v, message in mean_field_subset.items()
            })

        subset_factor_graph = FactorGraph(factor_subset_factor.values())
        return EPMeanFieldSubset(
            subset_factor_graph,
            factor_mean_field_subset,
            factor_mean_field_rescale,
            factor_subset_factor,
            self,
            plates_index,
        )

    __getitem__ = subset

    def __setitem__(self, index, subset_approx):
        subset_fac_meanfield = subset_approx.factor_mean_field 
        for factor, subset_fac  in subset_approx._factor_subset_factor.items():
            new_dist = subset_fac_meanfield[subset_fac]
            self.update_factor_mean_field(factor, new_dist, index)

    def update(self, subset_approx):
        self[subset_approx.plates_index] = subset_approx
        return self 

    def merge(self, index, subset_approx):
        factor_mean_field = self.factor_mean_field
        subset_fac_mean_field = subset_approx.factor_mean_field 
        for factor, subset_fac  in subset_approx._factor_subset_factor.items():
            new_dist = subset_fac_mean_field[subset_fac]
            old_dist = factor_mean_field[factor] 
            factor_mean_field[factor] = old_dist.merge(index, new_dist)

        return type(self)(
            self.factor_graph, 
            factor_mean_field, 
        )

    def update_factor_mean_field(
            self,
            factor: Factor,
            new_dist: MeanField,
            index: Optional[Dict[Plate, List[int]]] = None, 
    ):
        if index:
            self._factor_mean_field[factor][index] = new_dist
        else:
            self._factor_mean_field[factor] = new_dist

    @property
    def name(self):
        return f"EP_{self.factor_graph.name}"

    @cached_property
    def fixed_values(self):
        return {
            k: v for mf in self._factor_mean_field.values() 
            for k, v in mf.fixed_values.items()
        }

    @property
    def variables(self):
        return self.factor_graph.variables

    @property
    def deterministic_variables(self):
        return self.factor_graph.deterministic_variables

    @property
    def factor_mean_field(self) -> Dict[Factor, MeanField]:
        return self._factor_mean_field.copy()

    @property
    def factor_graph(self) -> FactorGraph:
        return self._factor_graph

    @classmethod
    def from_approx_dists(
            cls,
            factor_graph: FactorGraph,
            approx_dists: Dict[Variable, AbstractMessage],
    ) -> "EPMeanField":
        factor_mean_field = {
            factor: MeanField({v: approx_dists[v].copy() for v in factor.all_variables})
            for factor in factor_graph.factors
        }

        return cls(factor_graph, factor_mean_field)

    from_kws = from_approx_dists

    def factor_approximation(self, factor: Factor) -> FactorApproximation:
        """
        Create an approximation for one factor.

        This comprises:
        - The factor
        - The factor's variable distributions
        - The cavity distribution, which is the product of the distributions
          for each variable for all other factors
        - The model distribution, which is the product of the distributions
          for each variable for all factors

        Parameters
        ----------
        factor
            Some factor

        Returns
        -------
        An object comprising distributions with a specific distribution excluding
        that factor
        """
        factor_mean_field = self._factor_mean_field.copy()
        factor_dist = factor_mean_field.pop(factor)
        cavity_dist = MeanField({
            v: 1.0
            for v in factor_dist.all_variables
        }).prod(
            *factor_mean_field.values(),
        )

        model_dist = factor_dist.prod(cavity_dist, default=factor)

        return FactorApproximation(factor, cavity_dist, factor_dist, model_dist)

    get_factor_approx = factor_approximation

    def project_mean_field(
        self,
        new_dist: MeanField,
        factor_approx: FactorApproximation,
        delta: float = 1.0,
        status: Status = Status(),
    ) -> Tuple["EPMeanField", Status]:

        new_factor_dist, status = new_dist.update_factor_mean_field(
            factor_approx.cavity_dist,
            factor_approx.factor_dist,
            delta=delta,
            status=status,
        )
        factor_mean_field = self.factor_mean_field
        factor_mean_field[factor_approx.factor] = new_factor_dist

        new_approx = type(self)(
            factor_graph=self._factor_graph, 
            factor_mean_field=factor_mean_field,
        )
        return new_approx, status

    def project_factor_approx(
            self,
            projection: FactorApproximation,
            status: Optional[Status] = None,
    ) -> Tuple["EPMeanField", Status]:
        """ """
        factor_mean_field = self.factor_mean_field
        factor_mean_field[projection.factor] = projection.factor_dist

        new_approx = type(self)(
            factor_graph=self._factor_graph, 
            factor_mean_field=factor_mean_field
        )
        return new_approx, status

    project = project_factor_approx

    @property
    def mean_field(self) -> MeanField:
        return MeanField({v: 1.0 for v in self.all_variables}).prod(
            *self._factor_mean_field.values()
        )

    model_dist = mean_field

    @property
    def variable_messages(self) -> Dict[Variable, List[AbstractMessage]]:
        variable_messages = {v: [] for v in self.all_variables}
        for meanfield in self.factor_mean_field.values():
            for v, message in meanfield.items():
                variable_messages[v].append(message)

        return variable_messages

    @property
    def variable_message_count(self) -> Dict[Variable, int]:
        """
        The number of messages (i.e. the number of factors) associated with each
        variable
        """
        return {
            variable: len(messages)
            for variable, messages
            in self.variable_messages.items()
        }

    @property
    def variable_evidence(self) -> Dict[Variable, np.ndarray]:
        return {
            v: AbstractMessage.log_normalisation(*ms)
            for v, ms in self.variable_messages.items()
        }

    @property
    def factor_evidence(self) -> Dict[Factor, np.ndarray]:
        return {
            factor: meanfield.log_norm
            for factor, meanfield in self.factor_mean_field.items()
        }

    @property
    def log_evidence(self):
        """
        Calculates evidence for the EP approximation

        Evidence for a variable, xᵢ,

        Zᵢ = ∫ ∏ₐ m_{a → i} (xᵢ) dxᵢ

        Evidence for a factor, f_a,

                ∫ ∏_{j ∈ a} m_{i → a} (xᵢ) fₐ(xₐ) dxₐ
        Zₐ = -----------------------------------------
                             ∏_{j ∈ a} Zⱼ

        Evidence for model

        Z = ∏ᵢ Zᵢ ∏ₐ Zₐ
        """
        variable_evidence = {
            v: np.sum(logz) for v, logz in self.variable_evidence.items()
        }
        factor_evidence = sum(
            np.sum(meanfield.log_norm)
            - sum(variable_evidence[v] for v in factor.all_variables)
            for factor, meanfield in self.factor_mean_field.items()
        )
        return factor_evidence + sum(variable_evidence.values())

    def __repr__(self) -> str:
        clsname = type(self).__name__
        try:
            log_evidence = self.log_evidence
        except Exception as e:
            logger.exception(e)
            log_evidence = float("nan")
        return f"{clsname}({self.factor_graph}, " f"log_evidence={log_evidence})"


class EPMeanFieldSubset(EPMeanField):
    """
    this class encode the EP mean-field approximation to a factor graph


    Attributes
    ----------
    factor_graph: FactorGraph
        the base factor graph being approximated

    factor_mean_field: Dict[Factor, MeanField]
        the mean-field approximation for each factor in the factor graph

    mean_field: MeanField
        the mean-field approximation of the full factor graph
        i.e. the product of the factor mean-field approximations

    variables: Set[Variable]
        the variables of the approximation

    deterministic_variables: Set[Variable]
        the deterministic variables

    log_evidence: float
        the approximate log evidence of the approximation

    is_valid: bool
        returns whether the factor mean-field approximations are all valid

    Methods
    -------
    from_approx_dists(factor_graph, approx_dists)
        create a EPMeanField object from the passed factor_graph
        using approx_dists to initialise the factor mean-field approximations

    factor_approximation(factor)
        create the FactorApproximation for the factor

    project_factor_approx(factor_approximation)
        given the passed FactorApproximation, return a new `EPMeanField`
        object encoding the updated mean-field approximation
    """

    def __init__(
        self,
        factor_graph: FactorGraph,
        factor_mean_field: Dict[Factor, MeanField],
        factor_rescale: Dict[Factor, Dict[Variable, float]],
        factor_subset_factor: Dict[Factor, Factor],
        ep_mean_field: EPMeanField,
        plates_index: Dict[Plate, List[int]],
    ):
        super().__init__(factor_graph, factor_mean_field)

        self._factor_rescale = factor_rescale
        self._factor_subset_factor = factor_subset_factor
        self._ep_mean_field = ep_mean_field
        self._plates_index = plates_index

    
    @property 
    def parameters(self):
        return {
            "factor_graph": self._factor_graph, 
            "factor_mean_field": self._factor_mean_field,
            "factor_rescale": self._factor_rescale,
            "factor_subset_factor": self._factor_subset_factor,
            "ep_mean_field": self._ep_mean_field,
            "plates_index": self._plates_index,
        }

    @property 
    def plates_index(self):
        return self._plates_index.copy()

    @property 
    def factor_rescale(self):
        return self._factor_rescale.copy()

    @property 
    def factor_subset_factor(self):
        return self._factor_subset_factor.copy()

    def project_mean_field(
        self,
        new_model_dist: MeanField,
        factor_approx: FactorApproximation,
        delta: float = 1.0,
        status: Status = Status(),
    ) -> Tuple["EPMeanField", Status]:

        factor_mean_field = self.factor_mean_field
        
        # We're fitting the full factor_dist, not the subset factor_dist
        rescale1 = {
            v: 1 - scale 
            for v, scale in self._factor_rescale[factor_approx.factor].items()
        }
        last_factor_dist = factor_mean_field.pop(factor_approx.factor)
        subset_cavity_dist = last_factor_dist.rescale(rescale1)
        
        if delta < 1:
            # This is a more stable update that effectively performs an
            # Exponential moving average of the natural parameters of the
            # factor message with delta as the EMA coefficient
            factor_mean_field[factor_approx.factor], status = (
                new_model_dist * subset_cavity_dist
            ).update_factor_mean_field(
                factor_approx.cavity_dist,
                factor_approx.factor_dist * subset_cavity_dist,
                delta=delta,
                status=status,
            )
        else:
            new_factor_dist, status = new_model_dist.update_factor_mean_field(
                factor_approx.cavity_dist,
                factor_approx.factor_dist,
                delta=delta,
                status=status,
            )
            factor_mean_field[factor_approx.factor] = new_factor_dist  * subset_cavity_dist
            
        new_approx = type(self)(
            factor_graph=self._factor_graph,
            factor_mean_field=factor_mean_field,
            factor_rescale=self._factor_rescale,
            factor_subset_factor=self._factor_subset_factor, 
            ep_mean_field=self._ep_mean_field,
            plates_index=self._plates_index,
        )
        return new_approx, status

    def factor_approximation(self, factor: Factor) -> FactorApproximation:
        """
        Create an approximation for one factor.

        This comprises:
        - The factor
        - The factor's variable distributions
        - The cavity distribution, which is the product of the distributions
          for each variable for all other factors
        - The model distribution, which is the product of the distributions
          for each variable for all factors

        Parameters
        ----------
        factor
            Some factor

        Returns
        -------
        An object comprising distributions with a specific distribution excluding
        that factor
        """
        factor = self._factor_subset_factor.get(factor, factor)

        factor_mean_field = self._factor_mean_field.copy()

        factor_dist = factor_mean_field.pop(factor)
        cavity_dist = dict(
            MeanField({v: 1.0 for v in factor_dist.all_variables}).prod(
                *factor_mean_field.values()
            )
        )
        model_dist = factor_dist.prod(cavity_dist)

        factor_dist = dict(factor_dist)
        for v, scale in self._factor_rescale[factor].items():
            if scale < 1:
                cavity_dist[v] = cavity_dist[v] * factor_dist[v] ** (1 - scale)
                factor_dist[v] = factor_dist[v] ** scale

        return FactorApproximation(
            factor, MeanField(cavity_dist), MeanField(factor_dist), model_dist
        )
