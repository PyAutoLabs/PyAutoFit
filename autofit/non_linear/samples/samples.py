from abc import ABC

import json
from copy import copy
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pathlib import Path

from autoconf import conf
from autoconf.class_path import get_class_path
from autofit import exc
from autofit.mapper.model import ModelInstance
from autofit.non_linear.test_mode import is_test_mode
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.samples.sample import Sample

from .summary import SamplesSummary
from .interface import SamplesInterface, to_instance
from ...text.formatter import write_table

logger = logging.getLogger(__name__)


class Samples(SamplesInterface, ABC):
    def __init__(
        self,
        model: AbstractPriorModel,
        sample_list: List[Sample],
        samples_info: Optional[Dict] = None,
    ):
        """
        Contains the samples of the non-linear search, including parameter values, log likelihoods,
        weights and other quantites.

        For example, the output class can be used to load an instance of the best-fit model, get an instance of any
        individual sample by the `NonLinearSearch` and return information on the likelihoods, errors, etc.

        This class stores samples of searches which provide maximum likelihood estimates of the  model-fit (e.g.
        PySwarms, LBFGS).

        Parameters
        ----------
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        sample_list
            The list of `Samples` which contains the paramoeters, likelihood, weights, etc. of every sample taken
            by the non-linear search.
        samples_info
            Contains information on the samples (e.g. total iterations, time to run the search, etc.).
        """

        super().__init__(model=model)

        self.sample_list = sample_list
        self.samples_info = {
            **(samples_info or {}),
            "class_path": get_class_path(self.__class__),
        }

    def __str__(self):
        return f"{self.__class__.__name__}({len(self.sample_list)})"

    def __repr__(self):
        return str(self)

    @property
    def instances(self):
        """
        One model instance for each sample
        """
        return [
            self.model.instance_from_vector(
                sample.parameter_lists_for_paths(
                    self.paths if sample.is_path_kwargs else self.names
                ),
            )
            for sample in self.sample_list
        ]

    @property
    def log_evidence(self):
        return None

    @classmethod
    def from_list_info_and_model(
        cls,
        sample_list,
        samples_info,
        model: AbstractPriorModel,
    ):
        return cls(
            model=model,
            sample_list=sample_list,
            samples_info=samples_info,
        )

    def summary(self):
        return SamplesSummary(
            model=self.model,
            max_log_likelihood_sample=self.max_log_likelihood_sample,
        )

    def __add__(self, other: "Samples") -> "Samples":
        """
        Samples can be added together, which combines their `sample_list` meaning that inferred parameters are
        computed via their joint PDF.

        Parameters
        ----------
        other
            The Samples to be added to this Samples instance.

        Returns
        -------
        A class that combined the samples of the two Samples objects.
        """

        self._check_addition(other=other)

        return self.__class__(
            model=self.model,
            sample_list=self.sample_list + other.sample_list,
        )

    def __radd__(self, other):
        """
        Samples can be added together, which combines their `sample_list` meaning that inferred parameters are
        computed via their joint PDF.

        Overwriting `__radd__` enables the sum function to be used on a list of samples, e.g.:

        `samples = sum([samples_x5, samples_x5, samples_x5])`

        Parameters
        ----------
        other
            The Samples to be added to this Samples instance.

        Returns
        -------
        A class that combines the samples of a list of Samples objects.
        """
        return self

    def __len__(self):
        return len(self.sample_list)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result._names = None
        result._paths = None
        return result

    def _check_addition(self, other: "Samples"):
        """
        When adding samples together, perform the following checks to make sure it is valid to add the two objects
        together:

        - That both objects being added are `Samples` objects.
        - That both models have the same prior count, else the dimensionality does not allow for valid addition.
        - That both `Samples` objects use an identical model, such that we are adding together the same parameters.

        Parameters
        ----------
        other
            The Samples to be added to this Samples instance.
        """

        def raise_exc():
            raise exc.SamplesException(
                "Cannot add together two Samples objects which have different models."
            )

        if not isinstance(self, Samples):
            raise_exc()

        if not isinstance(other, Samples):
            raise_exc()

        if self.model.prior_count != other.model.prior_count:
            raise_exc()

        for path_self, path_other in zip(self.model.paths, other.model.paths):
            if path_self != path_other:
                raise_exc()

    def values_for_path(self, path: Tuple[str]) -> List[float]:
        """
        Returns the value for a variable with a given path
        for each sample in the model
        """
        return [sample.kwargs[path] for sample in self.sample_list]

    @property
    def total_iterations(self) -> int:
        return self.samples_info["total_iterations"]

    @property
    def time(self) -> Optional[float]:
        return self.samples_info["time"]

    @property
    def parameter_lists(self):
        result = list()
        for sample in self.sample_list:
            tuples = self.paths if sample.is_path_kwargs else self.names
            result.append(sample.parameter_lists_for_paths(tuples))

        return result

    @property
    def total_samples(self):
        return len(self.sample_list)

    @property
    def weight_list(self):
        return [sample.weight for sample in self.sample_list]

    @property
    def log_likelihood_list(self):
        return [sample.log_likelihood for sample in self.sample_list]

    @property
    def log_posterior_list(self):
        return [sample.log_posterior for sample in self.sample_list]

    @property
    def log_prior_list(self):
        return [sample.log_prior for sample in self.sample_list]

    @property
    def parameters_extract(self):
        return np.asarray(self.parameter_lists).T

    @property
    def _headers(self) -> List[str]:
        """
        Headers for the samples table
        """

        return self.model.joined_paths + [
            "log_likelihood",
            "log_prior",
            "log_posterior",
            "weight",
        ]

    @property
    def _rows(self) -> List[List[float]]:
        """
        Rows in the samples table
        """

        log_likelihood_list = self.log_likelihood_list
        log_prior_list = self.log_prior_list
        log_posterior_list = self.log_posterior_list
        weight_list = self.weight_list

        for index, row in enumerate(self.parameter_lists):
            yield row + [
                log_likelihood_list[index],
                log_prior_list[index],
                log_posterior_list[index],
                weight_list[index],
            ]

    def write_table(self, filename: Union[str, Path]):
        """
        Write a table of parameters, posteriors, priors and likelihoods.

        Parameters
        ----------
        filename
            Where the table is to be written
        """
        write_table(
            filename=filename,
            headers=list(self._headers),
            rows=list(self._rows),
        )

    def info_to_json(self, filename):
        with open(filename, "w") as outfile:
            json.dump(self.samples_info, outfile)

    @property
    def max_log_likelihood_sample(self) -> Sample:
        """
        The index of the sample with the highest log likelihood.
        """
        most_likely_sample = None
        for sample in self.sample_list:
            if (
                most_likely_sample is None
                or sample.log_likelihood > most_likely_sample.log_likelihood
            ):
                most_likely_sample = sample
        return most_likely_sample

    @property
    def max_log_likelihood_index(self) -> int:
        """
        The index of the sample with the highest log likelihood.
        """
        return int(np.argmax(self.log_likelihood_list))

    @to_instance
    def max_log_likelihood(self) -> List[float]:
        """
        The parameters of the maximum log likelihood sample of the `NonLinearSearch` returned as a model instance or
        list of values.
        """

        sample = self.max_log_likelihood_sample

        return sample.parameter_lists_for_paths(
            self.paths if sample.is_path_kwargs else self.names
        )

    @property
    def max_log_posterior_sample(self) -> Sample:
        return self.sample_list[self.max_log_posterior_index]

    @property
    def max_log_posterior_index(self) -> int:
        """
        The index of the sample with the highest log posterior.
        """
        return int(np.argmax(self.log_posterior_list))

    @to_instance
    def max_log_posterior(self) -> ModelInstance:
        """
        The parameters of the maximum log posterior sample of the `NonLinearSearch` returned as a model instance.
        """
        return self.parameter_lists[self.max_log_posterior_index]

    @to_instance
    def from_sample_index(self, sample_index: int) -> ModelInstance:
        """
        The parameters of an individual sample of the non-linear search, returned as a model instance.

        Parameters
        ----------
        sample_index
            The sample index of the weighted sample to return.
        """
        return self.parameter_lists[sample_index]

    def samples_above_weight_threshold_from(
        self, weight_threshold: Optional[float] = None, log_message: bool = False
    ) -> "Samples":
        """
        Returns a new `Samples` object containing only the samples with a weight above the input threshold.

        This function can be used after a non-linear search is complete, to reduce the samples to only the high weight
        values. The benefit of this is that the corresponding `samples.csv` file will be reduced in hard-disk size.

        For large libraries of results can significantly reduce the overall hard-disk space used and speed up the
        time taken to load the samples from a .csv file and perform analysis on them.

        For a sufficiently low threshold, this has a neglible impact on the numerical accuracy of the results, and
        even higher values can be used for aggresive use cases where hard-disk space is at a premium.

        Parameters
        ----------
        weight_threshold
            The threshold of weight at which a sample is included in the new `Samples` object.
        """

        if weight_threshold is None:
            weight_threshold = conf.instance["output"]["samples_weight_threshold"]

        if is_test_mode():
            weight_threshold = None

        if weight_threshold is None:
            return self

        sample_list = []

        for sample in self.sample_list:
            if sample.weight > weight_threshold:
                sample_list.append(sample)

        if log_message:
            logger.info(
                f"Samples with weight less than {weight_threshold} removed from samples.csv."
            )

        return self.__class__(
            model=self.model,
            sample_list=sample_list,
            samples_info=self.samples_info,
        )

    def minimise(self) -> "Samples":
        """
        A copy of this object with only important samples retained
        """
        samples = copy(self)
        samples.sample_list = list(
            {self.max_log_likelihood_sample, self.max_log_posterior_sample}
        )
        return samples

    def with_paths(self, paths: Union[List[Tuple[str, ...]], List[str]]) -> "Samples":
        """
        Create a copy of this object with only attributes specified
        by a list of paths.

        Parameters
        ----------
        paths
            A list of paths to attributes. Only kwargs and model components
            specified by these paths are retained.

            All children of a given path are retained.

        Returns
        -------
        A set of samples with a reduced set of attributes
        """
        with_paths = copy(self)
        with_paths.model = self.model.with_paths(paths)
        with_paths.sample_list = [
            sample.with_paths(paths) for sample in self.sample_list
        ]
        return with_paths

    def without_paths(
        self, paths: Union[List[Tuple[str, ...]], List[str]]
    ) -> "Samples":
        """
        Create a copy of this object with only attributes not specified
        by a list of paths.

        Parameters
        ----------
        paths
            A list of paths to attributes. kwargs and model components
            specified by these paths are removed.

            All children of a given path are removed.

        Returns
        -------
        A set of samples with a reduced set of attributes
        """
        with_paths = copy(self)
        with_paths.model = self.model.without_paths(paths)
        with_paths.sample_list = [
            sample.without_paths(paths) for sample in self.sample_list
        ]
        return with_paths

    def subsamples(self, model):
        if self.model is None:
            return None

        path_map = self.path_map_for_model(model)
        copied = copy(self)
        copied._paths = None
        copied._names = None
        copied.model = model

        copied.sample_list = [sample.subsample(path_map) for sample in self.sample_list]
        return copied
