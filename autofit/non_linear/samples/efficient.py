from typing import List, cast, Type

import numpy as np

from autonerves.class_path import get_class
from .sample import Sample
from .samples import Samples
from .pdf import SamplesPDF


class EfficientSamples:
    def __init__(self, samples: SamplesPDF):
        """
        A representation of samples where values are stored in numpy arrays. This is more
        efficient for storage and computation.

        Parameters
        ----------
        samples
            A representation of samples where values are in instances of the Sample class.
        """
        self.model = samples.model
        self.samples_info = samples.samples_info

        sample_list = samples.sample_list
        try:
            self._keys = list(sample_list[0].kwargs.keys())
        except IndexError:
            self._keys = []
        self._values = np.asarray(
            [[sample.kwargs[key] for key in self._keys] for sample in sample_list]
        )
        self._log_likelihoods = np.asarray(
            [sample.log_likelihood for sample in sample_list]
        )
        self._log_priors = np.asarray([sample.log_prior for sample in sample_list])
        self._weights = np.asarray([sample.weight for sample in sample_list])

    @property
    def samples(self) -> Samples:
        """
        Convert the efficient samples back to a SamplesPDF instance.
        """
        cls = cast(Type[Samples], get_class(self.samples_info["class_path"]))
        return cls(
            model=self.model,
            samples_info=self.samples_info,
            sample_list=self.sample_list,
        )

    @property
    def sample_list(self) -> List[Sample]:
        """
        Convert the efficient samples back to a list of Sample instances.
        """
        return [
            Sample(
                log_likelihood=log_likelihood,
                log_prior=log_prior,
                weight=weight,
                kwargs=kwargs,
            )
            for log_likelihood, log_prior, weight, kwargs in zip(
                self._log_likelihoods,
                self._log_priors,
                self._weights,
                self._kwargs,
            )
        ]

    @property
    def _kwargs(self) -> List[dict]:
        """
        Convert the efficient samples back to a list of dictionaries of kwargs.
        """
        return [
            {key: value for key, value in zip(self._keys, values)}
            for values in self._values
        ]
