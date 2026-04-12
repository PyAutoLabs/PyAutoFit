from typing import List, Optional, Union, Iterable, Tuple

import numpy as np

from autofit import exc
from autofit.non_linear.search.abstract_search import NonLinearSearch
from autofit.non_linear.grid.grid_list import GridList, as_grid_list
from autofit.mapper import model_mapper as mm
from autofit.mapper.prior.abstract import Prior

from autofit.non_linear.samples.interface import SamplesInterface


class AbstractGridSearchResult:
    def __init__(self, samples: GridList):
        self.samples = samples

    # noinspection PyTypeChecker
    @as_grid_list
    def physical_centres_lists_from(
        self,
        path: Union[str, Tuple[str, ...]],
    ) -> GridList:
        """
        Get the physical centres of the grid search from a path to an attribute of the instance in the samples.

        Parameters
        ----------
        path
            The path to the attribute to get from the instance

        Returns
        -------
        A list of lists of physical values
        """
        return self._physical_centres_lists_from(self.samples, path)

    # noinspection PyTypeChecker
    @as_grid_list
    def _physical_centres_lists_from(
        self,
        samples: GridList,
        path: Union[str, Tuple[str, ...]],
    ) -> GridList:
        """
        Get the physical centres of the grid search from a path to an attribute of the instance in the samples.

        Parameters
        ----------
        path
            The path to the attribute to get from the instance

        Returns
        -------
        A list of lists of physical values
        """
        if isinstance(path, str):
            path = path.split(".")

            def value_for_samples(samples):
                return samples.model.object_for_path(path).mean

        else:
            paths = [p.split(".") for p in path]

            def value_for_samples(samples):
                return tuple(samples.model.object_for_path(p).mean for p in paths)

        return [value_for_samples(samples) for samples in samples]


# noinspection PyTypeChecker
class GridSearchResult(AbstractGridSearchResult):
    def __init__(
        self,
        samples: List[SamplesInterface],
        lower_limits_lists: Union[List, GridList],
        grid_priors: List[Prior],
        parent: Optional[NonLinearSearch] = None,
    ):
        """
        The sample of a grid search.

        Parameters
        ----------
        samples
            The samples of the non linear optimizations performed at each grid step
        lower_limits_lists
            A list of lists of values representing the lower bounds of the grid searched values at each step
        """
        self.no_dimensions = len(lower_limits_lists[0])
        self.no_steps = len(lower_limits_lists)

        self.lower_limits_lists = GridList(lower_limits_lists, self.shape)
        self.side_length = int(self.no_steps ** (1 / self.no_dimensions))
        self.step_size = 1 / self.side_length
        self.grid_priors = grid_priors

        self.parent = parent

        super().__init__(GridList(samples, self.shape) if samples is not None else None)

    @property
    @as_grid_list
    def physical_lower_limits_lists(self) -> GridList:
        """
        The lower physical values for each grid square
        """
        return self._physical_values_for(self.lower_limits_lists)

    @property
    @as_grid_list
    def physical_centres_lists(self) -> GridList:
        """
        The middle physical values for each grid square
        """
        return self._physical_values_for(self.centres_lists)

    @property
    @as_grid_list
    def physical_upper_limits_lists(self) -> GridList:
        """
        The upper physical values for each grid square
        """
        return self._physical_values_for(self.upper_limits_lists)

    @property
    @as_grid_list
    def upper_limits_lists(self) -> GridList:
        """
        The upper values for each grid square
        """
        return [
            [limit + self.step_size for limit in limits]
            for limits in self.lower_limits_lists
        ]

    @property
    @as_grid_list
    def centres_lists(self) -> List:
        """
        The centre values for each grid square
        """
        return [
            [(upper + lower) / 2 for upper, lower in zip(upper_limits, lower_limits)]
            for upper_limits, lower_limits in zip(
                self.lower_limits_lists, self.upper_limits_lists
            )
        ]

    def _physical_values_for(self, unit_lists: GridList) -> List:
        """
        Compute physical values for lists of lists of unit hypercube
        values.

        Parameters
        ----------
        unit_lists
            A list of lists of hypercube values

        Returns
        -------
        A list of lists of physical values
        """
        return [
            [prior.value_for(limit) for prior, limit in zip(self.grid_priors, limits)]
            for limits in unit_lists
        ]

    def __setstate__(self, state):
        return self.__dict__.update(state)

    def __getstate__(self):
        return self.__dict__

    def __getattr__(self, item: str) -> object:
        """
        We default to getting attributes from the best sample. This allows promises to reference best samples.
        """
        return getattr(self.best_samples, item)

    @property
    def shape(self):
        return self.no_dimensions * (int(self.no_steps ** (1 / self.no_dimensions)),)

    @property
    def best_samples(self):
        """
        The best sample of the grid search. That is, the sample output by the non linear search that had the highest
        maximum figure of merit.

        Returns
        -------
        best_sample: sample
        """
        return max(
            self.samples,
            key=lambda sample: sample.log_likelihood,
        )

    @property
    def best_model(self):
        """
        Returns
        -------
        best_model: mm.ModelMapper
            The model mapper instance associated with the highest figure of merit from the grid search
        """
        return self.best_sample.model

    @property
    def all_models(self):
        """
        Returns
        -------
        all_models: [mm.ModelMapper]
            All model mapper instances used in the grid search
        """
        return [sample.model for sample in self.samples]

    @property
    def physical_step_sizes(self):
        physical_step_sizes = []

        # TODO : Make this work for all dimensions in a less ugly way.

        for dim in range(self.no_dimensions):
            values = [value[dim] for value in self.physical_lower_limits_lists]
            diff = [abs(values[n] - values[n - 1]) for n in range(1, len(values))]

            if dim == 0:
                physical_step_sizes.append(np.max(diff))
            elif dim == 1:
                physical_step_sizes.append(np.min(diff))
            else:
                raise exc.GridSearchException(
                    "This feature does not support > 2 dimensions"
                )

        return tuple(physical_step_sizes)

    @as_grid_list
    def attribute_grid(self, attribute_path: Union[str, Iterable[str]]) -> GridList:
        """
        Get a list of the attribute of the best instance from every search in a numpy array with the native dimensions
        of the grid search.

        Parameters
        ----------
        attribute_path
            The path to the attribute to get from the instance

        Returns
        -------
        A numpy array of the attribute of the best instance from every search in the grid search.
        """
        if isinstance(attribute_path, str):
            attribute_path = attribute_path.split(".")

        attribute_list = []
        for sample in self.samples:
            attribute = sample.instance
            for attribute_name in attribute_path:
                attribute = getattr(attribute, attribute_name)
            attribute_list.append(attribute)

        return attribute_list

    @as_grid_list
    def log_likelihoods(
        self,
        relative_to_value: float = 0.0,
    ) -> GridList:
        """
        The maximum log likelihood of every grid search on a NumPy array whose shape is the native dimensions of the
        grid search.

        For example, for a 2x2 grid search the shape of the Numpy array is (2,2) and it is numerically ordered such
        that the first search's maximum likelihood (corresponding to unit priors (0.0, 0.0)) are in the first
        value (E.g. entry [0, 0]) of the NumPy array.

        Parameters
        ----------
        relative_to_value
            The value to subtract from every log likelihood, for example if Bayesian model comparison is performed
            on the grid search and the subtracted value is the maximum log likelihood of a previous search.
        """
        return [sample.log_likelihood - relative_to_value for sample in self.samples]

    @as_grid_list
    def log_evidences(self, relative_to_value: float = 0.0) -> GridList:
        """
        The maximum log evidence of every grid search on a NumPy array whose shape is the native dimensions of the
        grid search.

        For example, for a 2x2 grid search the shape of the Numpy array is (2,2) and it is numerically ordered such
        that the first search's maximum evidence (corresponding to unit priors (0.0, 0.0)) are in the first
        value (E.g. entry [0, 0]) of the NumPy array.

        Parameters
        ----------
        relative_to_value
            The value to subtract from every log likelihood, for example if Bayesian model comparison is performed
            on the grid search and the subtracted value is the maximum log likelihood of a previous search.
        """
        return [
            sample.log_evidence - relative_to_value
            if sample.log_evidence is not None
            else None
            for sample in self.samples
        ]

    def figure_of_merits(
        self, use_log_evidences: bool, relative_to_value: float = 0.0
    ) -> GridList:
        """
        Convenience method to get either the log likelihoods or log evidences of the grid search.

        Parameters
        ----------
        use_log_evidences
            If true, the log evidences are returned, otherwise the log likelihoods are returned.
        relative_to_value
            The value to subtract from every log likelihood, for example if Bayesian model comparison is performed
            on the grid search and the subtracted value is the maximum log likelihood of a previous search.
        """

        if use_log_evidences:
            return self.log_evidences(relative_to_value)
        return self.log_likelihoods(relative_to_value)
