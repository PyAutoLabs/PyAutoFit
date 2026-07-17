from __future__ import annotations
import logging
from abc import ABC, abstractmethod
import numpy as np
from typing import TYPE_CHECKING, Optional
import warnings

if TYPE_CHECKING:
    from autofit.non_linear.analysis.analysis import Analysis

from autofit import exc
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.samples import Samples
from autofit.non_linear.samples.summary import SamplesSummary
from autofit.text import text_util


class Placeholder:
    def __getattr__(self, item):
        """
        Placeholders return None to represent the missing result's value
        """
        return None

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass

    def __gt__(self, other):
        return False

    def __lt__(self, other):
        return True

    @property
    def samples(self):
        return self

    @property
    def log_likelihood(self):
        return -np.inf

    def summary(self):
        return self


class AbstractResult(ABC):
    """
    @DynamicAttrs
    """

    def __init__(self, samples_summary, paths):
        """
        Abstract result of a non-linear search.

        Parameters
        ----------
        samples_summary
            A summary of the most important samples of the non-linear search (e.g. maximum log likelihood, median PDF).
        paths
            The paths to the results of the search.
        """

        self._samples_summary = samples_summary
        self.paths = paths

    @property
    def samples_summary(self):
        return self._samples_summary

    @property
    @abstractmethod
    def samples(self):
        pass

    @property
    @abstractmethod
    def model(self):
        pass

    @property
    def info(self) -> str:
        return text_util.result_info_from(
            samples=self.samples,
        )

    def __gt__(self, other):
        """
        Results are sorted by their associated log_likelihood.

        Placeholders are always low.
        """
        if isinstance(other, Placeholder):
            return True
        return self.log_likelihood > other.log_likelihood

    def __lt__(self, other):
        """
        Results are sorted by their associated log_likelihood.

        Placeholders are always low.
        """
        if isinstance(other, Placeholder):
            return False
        return self.log_likelihood < other.log_likelihood

    @property
    def log_likelihood(self):
        return self.samples_summary.max_log_likelihood_sample.log_likelihood

    @property
    def instance(self):
        try:
            return self.samples_summary.instance
        except AttributeError as e:
            logging.warning(e)
            return None

    @property
    def max_log_likelihood_instance(self):
        return self.instance

    @property
    def model_centred(self) -> AbstractPriorModel:
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
        return self.samples_summary.model_centred

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
        return self.samples_summary.model_centred_absolute(a)

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
        return self.samples_summary.model_centred_relative(r)

    def model_centred_max_lh_bounded(self, b: float) -> AbstractPriorModel:
        """
        Returns a model where every free parameter is a `UniformPrior` with `lower_limit` and `upper_limit` the previous
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
        return self.samples_summary.model_centred_max_lh_bounded(b)


class Result(AbstractResult):
    def __init__(
        self,
        samples_summary: SamplesSummary,
        paths: Optional[AbstractPaths] = None,
        samples: Optional[Samples] = None,
        search_internal: Optional[object] = None,
        analysis: Optional[Analysis] = None,
    ):
        """
        The result of a non-linear search.

        The default behaviour is for all key results to be in the `samples_summary` attribute, which is a concise
        summary of the results of the non-linear search. The reasons for this to be the main attribute are:

        - It is concise and therefore has minimal I/O overhead, which is important because when runs are resumed
        the results are loaded often, which can become very slow for large results via a `samples.csv`.

        - The `output.yaml` config files can be used to disable the output of the `samples.csv` file
        and `search_internal.dill` files. This means in order for results to be loaded in a way that allows a run to
        resume, the `samples_summary` must contain all results necessary to resume the run.

        For this reason, the `samples` and `search_internal` attributes are optional. On the first run of a model-fit,
        they will always contain values as they are passed in via memory from the results of the search. However, if
        a run is resumed they are no longer available in memory, and they will only be available if their corresponding
        `samples.csv` and `search_internal.dill` files are output on disk and available to load.

        This object includes:

        - The `samples_summary` attribute, which is a summary of the results of the non-linear search.

        - The `paths` attribute, which contains the path structure to the results of the search on the hard-disk and
        is used to load the samples and search internal attributes if they are required and not available in memory.

        - The samples of the non-linear search (E.g. MCMC chains, nested sampling samples) which are used to compute
        the maximum likelihood model, posteriors and other properties.

        - The non-linear search used to perform the model fit in its internal format (e.g. the Dynesty sampler used
        by dynesty itself as opposed to PyAutoFit abstract classes).

        Parameters
        ----------
        samples_summary
            A summary of the most important samples of the non-linear search (e.g. maximum log likelihood, median PDF).
        paths
            The paths to the results of the search, used to load the samples and search internal attributes if they are
            required and not available in memory.
        samples
            The samples of the non-linear search, for example the MCMC chains.
        search_internal
            The non-linear search used to perform the model fit in its internal format.
        analysis
            The `Analysis` object that was used to perform the model-fit from which this result is inferred.
        """
        super().__init__(samples_summary=samples_summary, paths=paths)

        self._samples = samples
        self._search_internal = search_internal

        self.analysis = analysis

        self.__model = None

        self.child_results = None

    def dict(self) -> dict:
        """
        Human-readable dictionary representation of the results
        """
        return {
            "max_log_likelihood": self.samples_summary.max_log_likelihood_sample.model_dict(),
            "median pdf": self.samples_summary.median_pdf_sample.model_dict(),
        }

    @property
    def samples(self) -> Optional[Samples]:
        """
        Returns the samples of the non-linear search, for example the MCMC chains or nested sampling samples.

        When a model-fit is run the first time, the samples are passed into the result via memory and therefore
        always available.

        However, if a model-fit is resumed the samples are not available in memory and they only way to load them is
        via the `samples.csv` file output on the hard-disk. This property handles the loading of the samples from
        the `samples.csv` file if they are not available in memory.

        Returns
        -------
        The samples of the non-linear search.
        """

        if self._samples is not None:
            return self._samples

        try:
            return self.paths.samples
        except FileNotFoundError:
            return None

    @property
    def search_internal(self):
        """
        Returns the non-linear search used to perform the model fit in its internal sampler format.

        When a model-fit is run the first time, the search internal is passed into the result via memory and therefore
        always available.

        However, if a model-fit is resumed the search internal is not available in memory and they only way to load
        it is via the `search_internal.dill` file output on the hard-disk. This property handles the loading of
        the search internal from the `search_internal.dill` file if it is not available in memory.

        Returns
        -------
        The non-linear search used to perform the model fit in its internal sampler format.
        """
        if self._search_internal is not None:
            return self._search_internal

        try:
            return self.paths.load_search_internal()
        except FileNotFoundError:
            pass

    @property
    def projected_model(self) -> AbstractPriorModel:
        """
        Create a new model with the same structure as the previous model,
        replacing each prior with a new prior created by calculating sufficient
        statistics from samples and corresponding weights for that prior.

        `Samples.weight_list` holds *linear* importance weights; the message
        projection is an importance-weighted moment match over *log* weights,
        so they are converted here. Zero-weight samples map to -inf and drop
        out of the moments.
        """
        weights = np.asarray(self.samples.weight_list)
        if not np.any(weights > 0.0):
            raise ValueError(
                "Cannot project a model from samples whose weights are all zero."
            )
        with np.errstate(divide="ignore"):
            log_weight_list = np.log(weights)
        arguments = {
            prior: prior.project(
                samples=np.array(self.samples.values_for_path(path)),
                log_weight_list=log_weight_list,
            )
            for path, prior in self.samples.model.path_priors_tuples
        }
        return self.samples.model.mapper_from_prior_arguments(arguments)

    @property
    def model(self):
        if self.__model is None:
            self.__model = self.samples_summary.model.mapper_via_defaults_from()

        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    def __str__(self):
        return "Analysis Result:\n{}".format(
            "\n".join(
                ["{}: {}".format(key, value) for key, value in self.__dict__.items()]
            )
        )

    def __getitem__(self, item):
        return self.child_results[item]

    def __iter__(self):
        return iter(self.child_results)

    def __len__(self):
        return len(self.child_results)


class ResultsCollection:
    def __init__(self, result_list=None):
        """
        A collection of results from previous searches. Results can be obtained using an index or the name of the search
        from whence they came.
        """
        self.__result_list = []
        self.__result_dict = {}

        if result_list is not None:
            for result in result_list:
                self.add(name="", result=result)

    def copy(self):
        collection = ResultsCollection()
        collection.__result_dict = self.__result_dict
        collection.__result_list = self.__result_list
        return collection

    @property
    def reversed(self):
        return reversed(self.__result_list)

    @property
    def last(self):
        """
        The result of the last search
        """
        if len(self.__result_list) > 0:
            return self.__result_list[-1]
        return None

    @property
    def first(self):
        """
        The result of the first search
        """
        if len(self.__result_list) > 0:
            return self.__result_list[0]
        return None

    def add(self, name, result):
        """
        Add the result of a search.

        Parameters
        ----------
        name: str
            The name of the search
        result
            The result of that search
        """
        try:
            self.__result_list[self.__result_list.index(result)] = result
        except ValueError:
            self.__result_list.append(result)
        self.__result_dict[name] = result

    def __getitem__(self, item):
        """
        Get the result of a previous search by index

        Parameters
        ----------
        item: int
            The index of the result

        Returns
        -------
        result: Result
            The result of a previous search
        """
        return self.__result_list[item]

    def __len__(self):
        return len(self.__result_list)

    def from_name(self, name):
        """
        Returns the result of a previous search by its name

        Parameters
        ----------
        name: str
            The name of a previous search

        Returns
        -------
        result: Result
            The result of that search

        Raises
        ------
        exc.PipelineException
            If no search with the expected result is found
        """
        try:
            return self.__result_dict[name]
        except KeyError:
            raise exc.PipelineException(
                "No previous search named {} found in results ({})".format(
                    name, ", ".join(self.__result_dict.keys())
                )
            )

    def __contains__(self, item):
        return item in self.__result_dict
