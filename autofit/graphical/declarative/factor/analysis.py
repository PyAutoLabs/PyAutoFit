from typing import Optional

import numpy as np

from autofit.mapper.prior_model.prior_model import Model
from autofit.graphical.expectation_propagation import AbstractFactorOptimiser
from autofit.mapper.model import ModelInstance
from autofit.mapper.prior_model.prior_model import AbstractPriorModel
from autofit.non_linear.analysis import Analysis
from autofit.non_linear.paths.abstract import AbstractPaths
from .abstract import AbstractModelFactor


class FactorCallable:
    def __init__(
        self,
        prior_model: AbstractPriorModel,
        analysis: Analysis,
    ):
        self.prior_model = prior_model
        self.analysis = analysis

    def __call__(self, **kwargs: np.ndarray) -> float:
        """
        Creates an instance of the prior model and evaluates it, forming
        a factor.

        Parameters
        ----------
        kwargs
            Arguments with names that are unique for each prior.

        Returns
        -------
        Calculated likelihood
        """
        arguments = dict()
        for name_, array in kwargs.items():
            prior_id = int(name_.split("_")[1])
            prior = self.prior_model.prior_with_id(prior_id)
            arguments[prior] = array
        # noinspection PyTypeChecker
        instance = self.prior_model.instance_for_arguments(arguments)
        return self.analysis.log_likelihood_function(instance)

class AnalysisFactor(AbstractModelFactor):
    @property
    def prior_model(self):
        return self._prior_model

    def __init__(
        self,
        prior_model: AbstractPriorModel,
        analysis: Analysis,
        optimiser: Optional[AbstractFactorOptimiser] = None,
        name=None,
    ):
        """
        A factor in the graph that actually computes the likelihood of a model
        given values for each variable that model contains

        Parameters
        ----------
        prior_model
            A model with some dimensionality
        analysis
            A class that implements a function which evaluates how well an
            instance of the model fits some data
        optimiser
            A custom optimiser that will be used to fit this factor specifically
            instead of the default optimiser
        """
        self.label = prior_model.label
        self.analysis = analysis

        prior_variable_dict = {prior.name: prior for prior in prior_model.priors}

        super().__init__(
            prior_model=prior_model,
            factor=FactorCallable(
                prior_model=prior_model,
                analysis=analysis,
            ),
            optimiser=optimiser,
            prior_variable_dict=prior_variable_dict,
            name=name,
        )

    def tree_flatten(self):
        return (
            (self.prior_model,),
            (
                self.analysis,
                self.optimiser,
                self.name,
            ),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            children[0],
            analysis=aux_data[0],
            optimiser=aux_data[1],
            name=aux_data[2],
        )

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.prior_model, item)

    def name_for_variable(self, variable):
        path = ".".join(self.prior_model.path_for_prior(variable))
        return f"{self.name}.{path}"

    def visualize(
        self, paths: AbstractPaths, instance: ModelInstance, during_analysis: bool
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
        self.analysis.visualize(
            paths=paths, instance=instance, during_analysis=during_analysis
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
        self.analysis.visualize_before_fit(paths=paths, model=model)

    def save_attributes(self, paths: AbstractPaths):
        """
        Save the attributes of the analysis object to a file.

        Parameters
        ----------
        paths
            Object describing where data should be saved to
        """
        self.analysis.save_attributes(paths=paths)

    def save_results(self, paths: AbstractPaths, result):
        """
        Save the results of the analysis to a file.

        Parameters
        ----------
        paths
            Object describing where data should be saved to
        result
            The result of the analysis
        """
        self.analysis.save_results(paths=paths, result=result)

    def log_likelihood_function(self, instance: ModelInstance) -> float:
        return self.analysis.log_likelihood_function(instance)


class EPAnalysisFactor(AnalysisFactor):
    """
    An ``AnalysisFactor`` that exposes the EP cavity distribution to its
    ``Analysis`` on each likelihood evaluation.

    On every iteration of the EP optimiser, the cavity distribution
    ``q⁻ᵃ`` — the product of the posterior approximations from all
    *other* factors over the variables this factor shares with them —
    is computed in
    :class:`autofit.graphical.mean_field.FactorApproximation`. For most
    factors that distribution is consumed implicitly: it becomes the
    prior the search samples from.

    Some hierarchical / population-level analyses want to read those
    cavity messages directly. A canonical example is a "global"
    Analysis whose log-likelihood compares model predictions to the
    per-dataset Gaussian posterior summaries produced by upstream
    local fits, e.g.::

        log L = -0.5 * sum_i || (pred_i - cavity_mean_i) / cavity_sigma_i ||^2

    To support that, ``EPAnalysisFactor`` attaches the current cavity
    distribution to its ``Analysis`` immediately before optimisation,
    via the attribute ``_cavity_mean_field``. The user's
    ``log_likelihood_function`` can then read each shared variable's
    cavity message (``.mean`` and ``.scale`` on the
    ``AbstractMessage`` value) out of the dict.

    The hook is invoked from
    :func:`autofit.graphical.expectation_propagation.optimiser.factor_step`
    via duck-typing (``hasattr(factor, "set_cavity_dist")``), so the
    behaviour of plain ``AnalysisFactor`` is unaffected.
    """

    def set_cavity_dist(self, cavity_dist):
        """
        Store the cavity distribution on the wrapped ``Analysis``.

        Called by :func:`factor_step` once per EP iteration, before this
        factor's local search runs. The Analysis can read the messages
        inside ``log_likelihood_function`` by inspecting
        ``self._cavity_mean_field`` — a ``MeanField`` mapping each
        shared :class:`Variable` (i.e. ``Prior``) to an
        ``AbstractMessage`` whose ``.mean`` and ``.scale`` give the
        cavity Gaussian summary.
        """
        self.analysis._cavity_mean_field = cavity_dist
