import copy
from abc import ABC, abstractmethod
from typing import List, Dict, cast

from autofit.mapper.model import ModelInstance

from .query import InterpolatorPath, Equality
from .. import Model


class AbstractInterpolator(ABC):
    def __init__(self, instances: List[ModelInstance]):
        """
        A TimeSeries allows interpolation on any variable.

        For example, each instance may have an attribute t. Other attributes
        can be determined for any given value of t by interpolating their
        values for each instance in the time series.

        Parameters
        ----------
        instances
            A list of instances (e.g. best fits) each with all the same attributes
            and a value on which a time series may be built
        """
        self.instances = instances

    def __getattr__(self, item: str) -> InterpolatorPath:
        """
        Used to indicate which attribute is the time attribute.

        This attribute may be at any path in the model.

        e.g.
        instance.t
        instance.something.t

        Parameters
        ----------
        item
            The name of an attribute of the instance

        Returns
        -------
        A class that keeps track of which attributes have been addressed
        """
        return InterpolatorPath([item])

    def _value_map(self, path: InterpolatorPath) -> Dict[float, ModelInstance]:
        """
        Maps know values to corresponding instances for a given path

        Parameters
        ----------
        path
            A path to an attribute, e.g. time

        Returns
        -------
        A dictionary mapping values of that attribute for each instance to the corresponding
        instance
        """
        return {path.get_value(instance): instance for instance in self.instances}

    def __getitem__(self, item: Equality) -> ModelInstance:
        """
        Create an artificial model instance which has values interpolated
        for a given interpolation value.

        Parameters
        ----------
        item
            Indicates a value for a given attribute to which the instance should
            be interpolated

        Returns
        -------
        An artificial instance with values interpolated

        Examples
        --------
        # Each instance in the time_series has an attribute 't'
        time_series = af.LinearTimeSeries([instance_1, instance_2, instance_3])

        # We can now create an instance which has a value of t = 3.5 by interpolating
        instance = time_series[time_series.t == 3.5)

        # We can also interpolate on any arbitrary attribute of the instance
        instance = time_series[time_series.some.arbitrary.attribute == -1.0]
        """
        value_map = self._value_map(item.path)
        try:
            return value_map[item.value]
        except KeyError:
            value_map = self._value_map(item.path)
            x = sorted(value_map)

            if not self.instances:
                raise IndexError(
                    "Cannot interpolate: no instances have been added to the interpolator."
                )

            instance = self.instances[0]
            new_instance = copy.copy(instance)

            for path, _ in instance.path_instance_tuples_for_class(float):
                y = [value_map[value].object_for_path(path) for value in x]

                new_instance = new_instance.replacing_for_path(
                    path,
                    self._interpolate(x, cast(List[float], y), item.value),
                )

        # noinspection PyTypeChecker
        return new_instance.replacing_for_path(tuple(item.path.keys), item.value)

    def relationships(self, path: InterpolatorPath) -> Model:
        instance = self.instances[0]
        new_instance = copy.copy(instance)

        value_map = self._value_map(path)
        x = sorted(value_map)

        for path, _ in instance.path_instance_tuples_for_class(float):
            y = [value_map[value].object_for_path(path) for value in x]

            new_instance = new_instance.replacing_for_path(
                path,
                self._relationship(x, cast(List[float], y)),
            )

        return new_instance

    @classmethod
    def _interpolate(cls, x: List[float], y: List[float], value: float) -> float:
        """
        Interpolate a given attribute to find its effective value at some time

        Parameters
        ----------
        x
            A list of times (or another series)
        y
            A list which one value for each time
        value
            The time for which we want an interpolated value for the attribute

        Returns
        -------
        An interpolated value for the attribute
        """
        return cls._relationship(x, y)(value)

    @staticmethod
    @abstractmethod
    def _relationship(x: List[float], y: List[float]):
        """
        Calculate the relationship between two values

        Parameters
        ----------
        x
            The first value
        y
            The second value

        Returns
        -------
        The relationship between the two values
        """
