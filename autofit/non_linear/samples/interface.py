from abc import ABC, abstractmethod
from functools import wraps
from typing import Union, List, Tuple, Dict

from autofit.mapper.model import ModelInstance
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.mapper.prior_model.abstract import Path


def to_instance(func):
    """
    Decorator for methods that return a vector of parameters, which can be converted to a model instance.

    Parameters
    ----------
    func
        A method that returns a vector of parameters

    Returns
    -------
    A wrapper that converts the vector to a model instance
    """

    @wraps(func)
    def wrapper(
        self,
        *args,
        as_instance: bool = True,
        as_dict: bool = False,
        **kwargs,
    ) -> Union[List, Dict, ModelInstance]:
        vector = func(self, *args, **kwargs)

        if as_dict:
            return {".".join(path[0]): value for path, value in zip(self.paths, vector)}

        if as_instance:
            return self._instance_from_vector(vector)

        return vector

    return wrapper


class SamplesInterface(ABC):
    def __init__(self, model: AbstractPriorModel):
        """
        An interface for the samples of a `NonLinearSearch` that has been run, including the maximum log likelihood
        """
        self.model = model

        self._paths = None
        self._names = None
        self._instance = None

    def _rebind_model(self, model: AbstractPriorModel) -> None:
        """Rebind a copied samples object without retaining model-derived caches."""
        self.model = model
        self._paths = None
        self._names = None
        self._instance = None

    @property
    def instance(self):
        if self._instance is None:
            self._instance = self.max_log_likelihood()
        return self._instance

    @property
    def paths(self) -> List[Tuple[Path]]:
        """
        A list of paths to unique priors in the same order as prior
        ids (and therefore sample columns)

        Uses hasattr to make backwards compatible
        """
        if not hasattr(self, "_paths") or self._paths is None:
            self._paths = self.model.all_paths
        return self._paths

    @property
    def names(self) -> List[Tuple[str]]:
        """
        A list of names of unique priors in the same order as prior
        ids (and therefore sample columns)

        Uses hasattr to make backwards compatible
        """
        if not hasattr(self, "_names") or self._names is None:
            self._names = self.model.all_names
        return self._names

    @to_instance
    def max_log_likelihood(self, as_instance: bool = True) -> List[float]:
        """
        The parameters of the maximum log likelihood sample of the `NonLinearSearch` returned as a model instance or
        list of values.
        """

        sample = self.max_log_likelihood_sample

        return sample.parameter_lists_for_paths(
            self.paths if sample.is_path_kwargs else self.names
        )

    @property
    @abstractmethod
    def max_log_likelihood_sample(self):
        pass

    @property
    def log_likelihood(self):
        return self.max_log_likelihood_sample.log_likelihood

    @property
    @abstractmethod
    def log_evidence(self):
        pass

    @property
    def model_centred(self) -> AbstractPriorModel:
        """
        Returns a model where every free parameter is a `GaussianPrior` with `mean` the previous result's
        inferred maximum log likelihood parameter values and `sigma` values determined from the errors on
        each parameter or the prior config files

        For example, a previous result may infer a parameter to have a maximum log likelihood value of 2.

        If this result is used for search chaining, `model_centred` will assign this free parameter
        `GaussianPrior(mean=2.0, sigma=0.1)` in the new model, where `sigma` is determined from the errors on
        each parameter or the prior config files.

        Returns
        -------
        A model mapper created by taking results from this search.
        """
        return self.model.mapper_from_prior_means(means=self.prior_means)

    def model_centred_absolute(self, a: float) -> AbstractPriorModel:
        """
        Returns a model where every free parameter is a `GaussianPrior` with `mean` the previous result's
        inferred maximum log likelihood parameter values and `sigma` the input absolute value `a`.

        For example, a previous result may infer a parameter to have a maximum log likelihood value of 2.

        If this result is used for search chaining, `model_centred_absolute(a=0.1)` will assign this free parameter
        `GaussianPrior(mean=2.0, sigma=0.1)` in the new model, where `sigma` is linked to the input `a`.

        Parameters
        ----------
        a
            The absolute width of gaussian priors

        Returns
        -------
        A model mapper created by taking results from this search and creating priors with the defined absolute
        width.
        """
        return self.model.mapper_from_prior_means(means=self.prior_means, a=a)

    def model_centred_relative(self, r: float) -> AbstractPriorModel:
        """
        Returns a model where every free parameter is a `GaussianPrior` with `mean` the previous result's
        inferred maximum log likelihood parameter values and `sigma` a relative value from the result `r`.

        For example, a previous result may infer a parameter to have a maximum log likelihood value of 2 and
        an error at the input `sigma` of 0.5.

        If this result is used for search chaining, `model_centred_relative(r=0.1)` will assign this free parameter
        `GaussianPrior(mean=2.0, sigma=0.5*0.1)` in the new model, where `sigma` is the inferred error times `r`.

        Parameters
        ----------
        r
            The relative width of gaussian priors

        Returns
        -------
        A model mapper created by taking results from this search and creating priors with the defined relative
        width.
        """
        return self.model.mapper_from_prior_means(means=self.prior_means, r=r)

    def model_centred_max_lh_bounded(self, b: float) -> AbstractPriorModel:
        """
        Returns a model where every free parameter is a `UniformPrior` with `lower_limit` and `upper_limit the previous
        result's inferred maximum log likelihood parameter values minus and plus the bound `b`.

        For example, a previous result may infer a parameter to have a maximum log likelihood value of 2.

        If this result is used for search chaining, `model_bound(b=0.1)` will assign this free parameter
        `UniformPrior(lower_limit=1.9, upper_limit=2.1)` in the new model.

        Parameters
        ----------
        b
            The size of the bounds of the uniform prior

        Returns
        -------
        A model mapper created by taking results from this search and creating priors with the defined bounded
        uniform priors.
        """
        return self.model.mapper_from_uniform_floats(
            floats=self.max_log_likelihood(as_instance=False), b=b
        )

    def _instance_from_vector(self, vector: List[float]) -> ModelInstance:
        return self.model.instance_from_vector(vector=vector)

    @property
    def prior_means(self) -> [List]:
        """
        The mean of every parameter used to link its inferred values and errors to priors used to sample the
        same (or similar) parameters in a subsequent search, where:

        - The mean is given by their most-probable values median PDF values if using a sampler that provides this
          information(using median_pdf(as_instance=False)).
        - The man is given by the maximum log likelihood values otherwise (e.g. for a maximum likelihood estimator).
        """

        try:
            return self.median_pdf(as_instance=False)
        except (AttributeError, IndexError):
            return self.max_log_likelihood(as_instance=False)

    def path_map_for_model(self, model):
        return {
            tuple(self.model.all_paths_for_prior(prior)): path
            for path, prior in model.path_priors_tuples
        }
