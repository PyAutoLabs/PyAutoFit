import numpy as np

from collections.abc import Iterable

from autofit.mapper.model import ModelInstance, assert_not_frozen
from autofit.mapper.prior.abstract import Prior
from autofit.mapper.prior.constant import Constant
from autofit.mapper.prior_model.abstract import AbstractPriorModel


class Collection(AbstractPriorModel):
    def name_for_prior(self, prior: Prior) -> str:
        """
        Construct a name for the prior. This is the path taken
        to get to the prior.

        Parameters
        ----------
        prior

        Returns
        -------
        A string of object names joined by underscores
        """
        for name, prior_model in self.prior_model_tuples:
            prior_name = prior_model.name_for_prior(prior)
            if prior_name is not None:
                return "{}_{}".format(name, prior_name)
        for name, direct_prior in self.direct_prior_tuples:
            if prior == direct_prior:
                return name

    def tree_flatten(self):
        """Flatten this collection into a JAX-compatible PyTree representation.

        Returns
        -------
        tuple
            A (children, aux_data) pair where children are the values and
            aux_data are the corresponding keys.
        """
        keys, values = zip(*self.items())
        return values, keys

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct a Collection from a flattened PyTree.

        Parameters
        ----------
        aux_data
            The keys of the collection items.
        children
            The values of the collection items.
        """
        instance = cls()

        for key, value in zip(aux_data, children):
            setattr(instance, key, value)
        return instance

    def __contains__(self, item):
        return item in self._dict or item in self._dict.values()

    def __getitem__(self, item):
        """Retrieve an item by string key or integer index.

        Parameters
        ----------
        item : str or int
            A string key for dict-style access, or an integer index
            for positional access into the values list.
        """
        if isinstance(item, str):
            return self._dict[item]
        return self.values[item]

    def __len__(self):
        return len(self.values)

    def __str__(self):
        return "\n".join(f"{key} = {value}" for key, value in self.items())

    def __hash__(self):
        return self.id

    def __repr__(self):
        return f"<{self.__class__.__name__} {self}>"

    @property
    def values(self):
        """The model components in this collection as a list."""
        return list(self._dict.values())

    def items(self):
        """The (key, model_component) pairs in this collection."""
        return self._dict.items()

    def with_prefix(self, prefix: str):
        """
        Filter members of the collection, only returning those that start
        with a given prefix as a new collection.
        """
        return Collection(
            {key: value for key, value in self.items() if key.startswith(prefix)}
        )

    def as_model(self):
        """Convert all prior models in this collection to Model instances.

        Returns a new Collection where each AbstractPriorModel child has
        been converted via its own as_model() method.
        """
        return Collection(
            {
                key: value.as_model()
                if isinstance(value, AbstractPriorModel)
                else value
                for key, value in self.dict().items()
            }
        )

    def __init__(
        self,
        *arguments,
        **kwargs,
    ):
        """
        The object multiple Python classes are input into to create model-components, which has free parameters that
        are fitted by a non-linear search.

        Multiple Python classes can be input into a `Collection` in order to compose high dimensional models made of
        multiple model-components.

        The ``Collection`` object is highly flexible, and can create models from many input Python data structures
        (e.g. a list of classes, dictionary of classes, hierarchy of classes).

        For a complete description of the model composition API, see the **PyAutoFit** model API cookbooks:

        https://pyautofit.readthedocs.io/en/latest/cookbooks/cookbook_1_basics.html

        The Python class input into a ``Model`` to create a model component is written using the following format:

        - The name of the class is the name of the model component (e.g. ``Gaussian``).
        - The input arguments of the constructor are the parameters of the mode (e.g. ``centre``, ``normalization`` and ``sigma``).
        - The default values of the input arguments tell PyAutoFit whether a parameter is a single-valued float or a
        multi-valued tuple.

        [Rich document more clearly]

        A prior model used to represent a list of prior models for convenience.

        Arguments are flexibly converted into a collection.

        Parameters
        ----------
        arguments
            Classes, prior models, instances or priors

        Examples
        --------

        class Gaussian:

            def __init__(
                self,
                centre=0.0,        # <- PyAutoFit recognises these
                normalization=0.1, # <- constructor arguments are
                sigma=0.01,        # <- the Gaussian's parameters.
            ):
                self.centre = centre
                self.normalization = normalization
                self.sigma = sigma

        model = af.Collection(gaussian_0=Gaussian, gaussian_1=Gaussian)
        """
        super().__init__()
        self.item_number = 0
        arguments = list(arguments)
        if len(arguments) == 0:
            self.add_dict_items(kwargs)
        elif len(arguments) == 1:
            arguments = arguments[0]

            if isinstance(arguments, dict):
                self.add_dict_items(arguments)
            elif isinstance(arguments, Iterable):
                for argument in arguments:
                    self.append(argument)
            else:
                self.append(arguments)
        else:
            self.__init__(arguments)

    @assert_not_frozen
    def add_dict_items(self, item_dict):
        """Add all entries from a dictionary, converting values to prior models.

        Parameters
        ----------
        item_dict
            A dictionary mapping string keys to classes, instances, or prior models.
        """
        for key, value in item_dict.items():
            if isinstance(key, tuple):
                key = ".".join(key)
            setattr(self, key, AbstractPriorModel.from_object(value))

    def __eq__(self, other):
        if other is None:
            return False
        if len(self) != len(other):
            return False
        for i, item in enumerate(self):
            if item != other[i]:
                return False
        return True

    @assert_not_frozen
    def append(self, item):
        """Append an item to the collection with an auto-incremented numeric key.

        The item is converted to an AbstractPriorModel if it is not already one.
        """
        setattr(self, str(self.item_number), AbstractPriorModel.from_object(item))
        self.item_number += 1

    @assert_not_frozen
    def __setitem__(self, key, value):
        """Set an item by key, converting the value to a prior model.

        Preserves the id of any existing item at the same key so that
        prior identity is maintained across replacements.
        """
        obj = AbstractPriorModel.from_object(value)
        try:
            obj.id = getattr(self, str(key)).id
        except AttributeError:
            pass
        setattr(self, str(key), obj)

    @assert_not_frozen
    def __setattr__(self, key, value):
        """Set an attribute, automatically converting values to prior models.

        Private attributes (starting with ``_``) are set directly. All other
        values are wrapped via ``AbstractPriorModel.from_object`` so that
        plain classes become ``Model`` instances and floats become fixed values.
        """
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            try:
                super().__setattr__(key, AbstractPriorModel.from_object(value))
            except AttributeError:
                pass

    def remove(self, item):
        """Remove an item from the collection by value equality.

        Parameters
        ----------
        item
            The item to remove. All entries whose value equals this item
            are deleted.
        """
        for key, value in self.__dict__.copy().items():
            if value == item:
                del self.__dict__[key]

    def _instance_for_arguments(
        self,
        arguments,
        ignore_assertions=False,
        xp=np,
    ):
        """
        Parameters
        ----------
        arguments: {Prior: float}
            A dictionary of arguments

        Returns
        -------
        model_instances: [object]
            A list of instances constructed from the list of prior models.
        """
        result = ModelInstance()
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, AbstractPriorModel):
                value = value.instance_for_arguments(
                    arguments,
                    ignore_assertions=ignore_assertions,
                    xp=xp
                )
            elif isinstance(value, Prior):
                value = arguments[value]
            elif isinstance(value, Constant):
                value = value.value
            setattr(result, key, value)
        return result

    def gaussian_prior_model_for_arguments(self, arguments):
        """
        Create a new collection, updating its priors according to the argument
        dictionary.

        Parameters
        ----------
        arguments
            A dictionary of arguments

        Returns
        -------
        A new collection
        """
        collection = Collection()

        for key, value in self.items():
            if key in ("component_number", "item_number", "id") or key.startswith("_"):
                continue

            if isinstance(value, AbstractPriorModel):
                collection[key] = value.gaussian_prior_model_for_arguments(arguments)
            elif isinstance(value, Prior):
                collection[key] = arguments[value]
            else:
                collection[key] = value

        return collection

    @property
    def prior_class_dict(self):
        """Map each prior to the class it will produce when instantiated.

        For child prior models, delegates to their own prior_class_dict.
        Direct priors on the collection itself map to ModelInstance.
        """
        return {
            **{
                prior: cls
                for prior_model in self.direct_prior_model_tuples
                for prior, cls in prior_model[1].prior_class_dict.items()
            },
            **{prior: ModelInstance for _, prior in self.direct_prior_tuples},
        }
