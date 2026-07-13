from typing import Set, Optional, Type, List, Tuple, Dict

import numpy as np

from autofit import exc
from autofit.mapper.model import ModelInstance
from autofit.mapper.prior.abstract import Prior
from autofit.mapper.prior_model.collection import Collection
from autofit.mapper.prior_model.prior_model import Model
from autofit.mapper.variable import Plate
from autofit.messages import NormalMessage
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.tools.namer import namer

from .abstract import AbstractModelFactor


class HierarchicalFactor(Model):
    _plates: Tuple[Plate, ...] = ()

    def __init__(
        self,
        distribution: Type[Prior],
        optimiser=None,
        name: Optional[str] = None,
        use_jax : bool = False,
        **kwargs,
    ):
        """
        Associates variables in the graph with a distribution. That is,
        the optimisation prefers instances of the variables which better
        match the distribution. Both the distribution and variables sampled
        from it are optimised during a factor optimisation. This allows
        expectations from other factors to influence the optimisation of
        the distribution and vice-versa.

        Each HierarchicalFactor actually generates multiple factors - one
        for each associated variables - as this avoids optimisation of a
        high dimensionality factor.

        Question: would it make sense to experiment with a hierarchical
        factor that optimises all variables samples from a distribution
        simultaneously?

        Parameters
        ----------
        distribution
            A distribution from which variables are drawn
        optimiser
            An optional optimiser for this factor
        name
            An optional name for this factor
        kwargs
            Constants or Priors passed to the distribution to parameterize
            it. For example, a GaussianPrior requires mean and sigma arguments

        Examples
        --------
        factor = g.HierarchicalFactor(
            af.GaussianPrior,
            mean=af.GaussianPrior(
                mean=100,
                sigma=10
            ),
            sigma=af.GaussianPrior(
                mean=10,
                sigma=5
            )
        )
        factor.add_sampled_variable(
            prior
        )
        """
        super().__init__(distribution, **kwargs)
        self._name = name or namer(self.__class__.__name__)
        self._factors = list()
        self.optimiser = optimiser
        self._use_jax = use_jax

    @property
    def name(self):
        return self._name

    @property
    def prior_model(self):
        return self

    @property
    def plates(self):
        return self._plates

    def add_drawn_variable(self, prior: Prior):
        """
        Add a variable which is drawn from the distribution. This
        is likely the attribute of a FactorModel in the graph.

        Parameters
        ----------
        prior
            A variable which is sampled from the distribution.
        """
        self._factors.append(_HierarchicalFactor(self, prior))

    @property
    def factors(self) -> List["_HierarchicalFactor"]:
        """
        One factor is generated for each variable sampled from the
        distribution.
        """
        return self._factors


class Factor:
    def __init__(self, distribution_model: HierarchicalFactor):
        """
        Provides the function called by the factor in the graph.

        Parameters
        ----------
        distribution_model
            A prior model which parameterizes a distribution from which it
            is assumed the variable is drawn
        """
        self.distribution_model = distribution_model

    def __call__(self, **kwargs):
        """
        Call the factor

        Parameters
        ----------
        kwargs
            Maps names containing prior ids as well as a name 'argument' to values
            for each priors and the argument with which the distribution is called.

        Returns
        -------
        The probability that the argument was drawn from the distribution
        """
        argument = kwargs.pop("argument")
        arguments = dict()
        for name_, array in kwargs.items():
            prior_id = int(name_.split("_")[1])
            prior = self.distribution_model.prior_with_id(prior_id)
            arguments[prior] = array
        try:
            return self.distribution_model.instance_for_arguments(arguments).message(
                argument
            )
        except exc.MessageException:
            # The distribution was parameterised outside its support — e.g. an EP
            # factor optimiser proposing a (transiently) negative scale for a
            # ``GaussianPrior`` distribution, just below a truncated hyper-prior's
            # ``lower_limit=0``. Such a point has zero probability density, so the
            # factor's log-value is ``-inf`` rather than a hard error. The strict
            # ``NormalMessage`` sigma guard stays intact for genuine prior-passing
            # misuse; here we translate it into the correct EP semantics.
            return -np.inf


class _HierarchicalFactor(AbstractModelFactor):
    def __init__(
        self, distribution_model: HierarchicalFactor, drawn_prior: Prior,
    ):
        """
        A factor that links a variable to a parameterised distribution.

        Parameters
        ----------
        distribution_model
            A prior model which parameterizes a distribution from which it
            is assumed the variable is drawn
        drawn_prior
            A prior representing a variable which was drawn from the distribution
        """
        self.distribution_model = distribution_model
        self.drawn_prior = drawn_prior
        self._use_jax = distribution_model._use_jax

        prior_variable_dict = {prior.name: prior for prior in distribution_model.priors}

        prior_variable_dict["argument"] = drawn_prior

        super().__init__(
            prior_model=Collection(
                distribution_model=distribution_model, drawn_prior=drawn_prior
            ),
            factor=Factor(distribution_model),
            optimiser=distribution_model.optimiser,
            prior_variable_dict=prior_variable_dict,
            name=distribution_model.name,
        )

    @property
    def message_dict(self) -> Dict[Prior, NormalMessage]:
        """
        Dictionary mapping priors to messages. Does not account for inverse cavity
        behaviour as this caused bugs for hierarchical factors.
        """
        return {prior: prior.message for prior in self.priors}

    @property
    def variable(self):
        return self.drawn_prior

    def log_likelihood_function(self, instance, shared=None):
        return instance.distribution_model.message(instance.drawn_prior, xp=self._xp)

    @property
    def priors(self) -> Set[Prior]:
        """
        The set of priors associated with this factor. This is the priors used
        to parameterize the distribution, plus an additional prior for the
        variable drawn from the distribution.
        """
        priors = super().priors
        priors.add(self.drawn_prior)
        return priors

    @property
    def analysis(self):
        return self

    def visualize(
        self, paths: AbstractPaths, instance: ModelInstance, during_analysis: bool
    ):
        pass
