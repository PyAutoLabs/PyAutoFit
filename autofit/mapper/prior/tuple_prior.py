import copy
from typing import List, Tuple, Union, Dict

from autofit.mapper.model_object import ModelObject
from autofit.mapper.prior_model.attribute_pair import (
    cast_collection,
    PriorNameValue,
    InstanceNameValue,
)
from .abstract import Prior
from .constant import Constant

NameValue = Tuple[str, Union[Prior, float]]


class TuplePrior(ModelObject):
    """
    A prior comprising one or more priors in a tuple
    """

    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def tree_flatten(self):
        """
        Flatten this tuple prior as a JAX PyTree.

        The prior members are the children; their names and any fixed
        (non-prior) members go in the aux data, so ``tree_unflatten``
        rebuilds the same mix of free and fixed tuple entries.
        """
        prior_tuples = self.prior_tuples
        names = tuple(name for name, _ in prior_tuples)
        children = tuple(prior for _, prior in prior_tuples)
        fixed = tuple(
            (key, value)
            for key, value in self.__dict__.items()
            if key != "id" and not isinstance(value, Prior)
        )
        return children, (names, fixed)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Rebuild a tuple prior from its flattened PyTree.
        """
        names, fixed = aux_data
        return cls(**dict(zip(names, children)), **dict(fixed))

    @property
    @cast_collection(PriorNameValue)
    def prior_tuples(self):
        """
        Returns
        -------
        priors: [(String, Prior)]
            A list of priors contained in this tuple
        """
        return list(filter(lambda t: isinstance(t[1], Prior), self.__dict__.items()))

    @property
    def priors(self):
        return [prior for _, prior in self.prior_tuples]

    @property
    def unique_prior_tuples(self):
        return self.prior_tuples

    @property
    @cast_collection(InstanceNameValue)
    def instance_tuples(self):
        """
        Returns
        -------
        instances: [(String, instance)]
            A list of instances
        """
        return [
            (key, value.value if isinstance(value, Constant) else value)
            for key, value in sorted(
                [
                    (key, value)
                    for key, value in self.__dict__.items()
                    if key != "id" and isinstance(value, (int, float))
                ],
                key=lambda tup: tup[0],
            )
        ]

    def value_for_arguments(self, arguments):
        """
        Parameters
        ----------
        arguments: {Prior: float}
            A dictionary of arguments

        Returns
        -------
        tuple: (float,...)
            A tuple of float values
        """

        def convert(tup):
            if hasattr(tup, "prior"):
                return arguments[tup.prior]
            return tup.instance

        return tuple(
            map(
                convert,
                sorted(
                    self.prior_tuples + self.instance_tuples, key=lambda tup: tup.name
                ),
            )
        )

    def gaussian_tuple_prior_for_arguments(self, arguments):
        """
        Parameters
        ----------
        arguments: {Prior: float}
            A dictionary of arguments

        Returns
        -------
        tuple_prior: TuplePrior
            A new tuple prior with gaussian priors
        """
        tuple_prior = TuplePrior()
        for name, prior in self.prior_tuples:
            setattr(tuple_prior, name, arguments[prior])
        for name, value in self.instance_tuples:
            setattr(tuple_prior, name, value)
        return tuple_prior

    @property
    def tuples(self) -> List[NameValue]:
        """
        The names and instances of all priors and constants ordered
        by their name.

        This means they are in the order they should be in the tuple.
        """
        return sorted(self.prior_tuples + self.instance_tuples, key=lambda t: t[0])

    def _with_paths(self, tree: Dict[str, dict]) -> "TuplePrior":
        """
        An instance of this tuple prior with only tuples with positions
        indicated in the tree dictionary.

        Note applying this twice will give unexpected results.
        """
        new = TuplePrior()
        for key in tree:
            key, value = self._get_key_value(key)
            setattr(new, key, value)
        return new

    def _without_paths(self, tree: Dict[str, dict]) -> "TuplePrior":
        """
        An instance of this tuple prior without tuples with positions
        indicated in the tree dictionary.

        Note applying this twice will give unexpected results.
        """
        new = copy.deepcopy(self)
        for key in tree:
            key, value = self._get_key_value(key)
            delattr(new, key)
        return new

    def _get_key_value(self, key: Union[str, int]) -> NameValue:
        """
        Retrieve a key and value by an attribute name or index which may
        be expressed as a string or integer.
        """
        try:
            return self.tuples[int(key)]
        except ValueError:
            return key, getattr(self, key)

    def __getitem__(self, item):
        return self._get_key_value(item)[1]
