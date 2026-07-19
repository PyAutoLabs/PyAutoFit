import csv
import inspect
from copy import copy
from pathlib import Path
from typing import List, Tuple, Union

from autonerves.class_path import get_class_path
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.tools.util import split_paths


class Sample:
    def __init__(
        self,
        log_likelihood: float,
        log_prior: float,
        weight: float,
        kwargs=None,
    ):
        """
        One sample taken during a search

        Parameters
        ----------
        log_likelihood
            The likelihood associated with this instance
        log_prior
            A logarithmic prior of the instance
        weight
            The weight this sample contributes to the PDF.
        kwargs
            Dictionary mapping model paths to values for the sample
        """
        self.log_likelihood = float(log_likelihood)
        self.log_prior = log_prior
        self.weight = weight
        self.kwargs = {
            tuple(key.split(".")) if isinstance(key, str) else key: value
            for key, value in (kwargs or dict()).items()
        }

    def dict(self):
        return {
            "type": "instance",
            "class_path": get_class_path(type(self)),
            "arguments": {
                "log_likelihood": self.log_likelihood,
                "log_prior": self.log_prior,
                "weight": self.weight,
                "kwargs": {
                    "type": "dict",
                    "arguments": {
                        ".".join(key) if isinstance(key, tuple) else key: value
                        for key, value in self.kwargs.items()
                    },
                },
            },
        }

    def model_dict(self) -> dict:
        """
        A dictionary mapping model paths to values for the sample
        """
        model_dict = dict()
        for key, value in self.kwargs.items():
            current = model_dict
            if not isinstance(key, tuple):
                key = (key,)
            for part in key[:-1]:
                if part not in current:
                    current[part] = dict()
                current = current[part]
            current[key[-1]] = value
        return model_dict

    @property
    def log_posterior(self) -> float:
        """
        Compute the posterior
        """
        return self.log_likelihood + self.log_prior

    def parameter_lists_for_model(self, model: AbstractPriorModel) -> List[float]:
        """
        Values for instantiating a model, in the same order as priors
        from the model.

        Parameters
        ----------
        model
            The model from which this was a sample

        Returns
        -------
        A list of physical values
        """
        if self.is_path_kwargs:
            paths = model.all_paths
        else:
            paths = model.all_names

        return self.parameter_lists_for_paths(paths)

    def parameter_lists_for_paths(self, paths):
        result = list()
        for keys in paths:
            is_found = False
            for key in keys:
                if key in self.kwargs:
                    result.append(self.kwargs[key])
                    is_found = True
                    break
            if not is_found:
                raise KeyError(
                    f"Could not find any of the following keys in kwargs {keys}"
                )
        return result

    @property
    def is_path_kwargs(self) -> bool:
        """
        Are the keys in the kwargs dictionary tuples? If they
        are this indicates that they are explicit paths through
        the model.
        """
        for key in self.kwargs:
            return isinstance(key, tuple)
        return False

    def subsample(self, path_map):
        arg_dict = {}
        for paths, new_path in path_map.items():
            for path in paths:
                if path in self.kwargs:
                    arg_dict[new_path] = self.kwargs[path]
                    break
            else:
                raise KeyError(f"No path from {paths} in sample")

        return type(self)(
            log_likelihood=self.log_likelihood,
            log_prior=self.log_prior,
            weight=self.weight,
            kwargs=arg_dict,
        )

    @classmethod
    def from_lists(
        cls,
        model: AbstractPriorModel,
        parameter_lists: List[List[float]],
        log_likelihood_list: List[float],
        log_prior_list: List[float],
        weight_list: List[float],
    ) -> List["Sample"]:
        """
        Convenience method to create a list of samples from lists of contained values
        """
        samples = list()

        paths = model.unique_prior_paths

        for params, log_likelihood, log_prior, weight in zip(
            parameter_lists, log_likelihood_list, log_prior_list, weight_list
        ):
            arg_dict = {t: param for t, param in zip(paths, params)}

            samples.append(
                cls(
                    log_likelihood=log_likelihood,
                    log_prior=log_prior,
                    weight=weight,
                    kwargs=arg_dict,
                )
            )
        return samples

    def instance_for_model(
        self,
        model: AbstractPriorModel,
        ignore_assertions: bool = False,
    ):
        """
        Create an instance from this sample for a model

        Parameters
        ----------
        model
            The model the this sample was taken from
        ignore_assertions
            If True, do not check that the instance is valid

        Returns
        -------
        The instance corresponding to this sample
        """
        try:
            if self.is_path_kwargs:
                return model.instance_from_path_arguments(
                    self.kwargs,
                    ignore_assertions=ignore_assertions,
                )
            else:
                return model.instance_from_prior_name_arguments(
                    self.kwargs,
                    ignore_assertions=ignore_assertions,
                )

        except KeyError:
            return model.instance_from_vector(
                self.parameter_lists_for_model(model),
                ignore_assertions=ignore_assertions,
            )

    @split_paths
    def with_paths(self, paths: List[Tuple[str, ...]]) -> "Sample":
        """
        Create a copy of this object retaining only the kwargs for which
        there is a matching path in paths.

        Parameters
        ----------
        paths
            A list of paths for which attributes should be retained.

        Returns
        -------
        A reduced sample
        """
        with_paths = copy(self)
        with_paths.kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if any(
                all(first == second for first, second in zip(key, path))
                for path in paths
            )
        }
        return with_paths

    @split_paths
    def without_paths(self, paths: List[Tuple[str, ...]]) -> "Sample":
        """
        Create a copy of this object retaining only the kwargs for which
        there is no matching path in paths.

        Parameters
        ----------
        paths
            A list of paths for which attributes should be removed.

        Returns
        -------
        A reduced sample
        """
        without_paths = copy(self)
        without_paths.kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if not any(
                all(first == second for first, second in zip(key, path))
                for path in paths
            )
        }
        return without_paths


sample_args = set(inspect.getfullargspec(Sample.__init__).args)


def samples_from_iterator(iterator):
    samples = list()

    headers = [header.strip() for header in next(iterator)]

    # Classify and split the headers once — doing it per row (and letting Sample
    # re-split every string key) dominates load time for large samples.csv files.
    excluded = sample_args | {"log_posterior"}
    arg_indices = {
        header: index for index, header in enumerate(headers) if header in sample_args
    }
    kwarg_indices = [
        (index, tuple(header.split(".")))
        for index, header in enumerate(headers)
        if header not in excluded
    ]

    for row in iterator:
        samples.append(
            Sample(
                **{name: float(row[index]) for name, index in arg_indices.items()},
                kwargs={key: float(row[index]) for index, key in kwarg_indices},
            )
        )
    return samples


def load_from_table(filename: Union[str, Path]) -> List[Sample]:
    """
    Load samples from a table

    Parameters
    ----------
    filename
        The path to a CSV file

    Returns
    -------
    A list of samples, one for each row in the CSV
    """
    with open(filename, "r+", newline="") as f:
        reader = csv.reader(f)
        return samples_from_iterator(reader)
