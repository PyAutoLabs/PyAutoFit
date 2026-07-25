from functools import wraps
from typing import List, Tuple

import numpy as np


def as_grid_list(func):
    """
    Wrap functions with a function which converts the output list of grid search results to a `GridList` object.

    Parameters
    ----------
    func
        A function which computes and retrusn a list of grid search results.

    Returns
    -------
        A function which converts a list of grid search results to a `GridList` object.
    """

    @wraps(func)
    def wrapper(grid_search_result, *args, **kwargs) -> List:
        """
        This decorator converts the output of a function which computes a list of grid search results to a `GridList`.

        Parameters
        ----------
        grid_search_result
            The instance of the `GridSearchResult` which is being operated on.

        Returns
        -------
            The function output converted to a `GridList`.
        """

        values = func(grid_search_result, *args, **kwargs)

        return GridList(values=values, shape=grid_search_result.shape)

    return wrapper


class GridList(list):
    def __init__(self, values: List, shape: Tuple):
        """
        Many quantities of a `GridSearchResult` are stored as lists of lists.

        The number of lists corresponds to the dimensionality of the grid search and the number of elements
        in each list corresponds to the number of steps in that grid search dimension.

        This class provides a wrapper around lists of lists to provide some convenience methods for accessing
        the values in the lists. For example, it provides a conversion of the list of list structure to a ndarray.

        For example, for a 2x2 grid search the shape of the Numpy array is (2,2) and it is numerically ordered such
        that the first search's entries(corresponding to unit priors (0.0, 0.0)) are in the first
        value (E.g. entry [0, 0]) of the NumPy array.

        Parameters
        ----------
        values
        """
        super().__init__(values)

        self.shape = shape

    @property
    def as_list(self) -> List:
        return self

    @property
    def native(self) -> np.ndarray:
        """
        The list of lists as an ndarray.
        """
        return np.reshape(np.array(self), self.shape)
