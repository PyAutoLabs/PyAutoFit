import copy
import logging
from functools import wraps
from typing import Optional, Union, Tuple, List, Iterable, Type, Dict

from autofit.mapper.model_object import ModelObject
from autofit.mapper.prior_model.recursion import DynamicRecursionCache

logger = logging.getLogger(__name__)


def frozen_cache(func):
    """
    Decorator that caches results from function calls when
    a model is frozen.

    Value is cached by function name, instance and arguments.

    Parameters
    ----------
    func
        Some function attached to a freezable, hashable object
        that takes hashable arguments

    Returns
    -------
    Function with cache
    """

    @wraps(func)
    def cache(self, *args, **kwargs):
        if hasattr(self, "_is_frozen") and self._is_frozen:
            key = (
                func.__name__,
                self,
                *args,
            ) + tuple(kwargs.items())

            if key not in self._frozen_cache:
                self._frozen_cache[key] = func(self, *args, **kwargs)
            return self._frozen_cache[key]
        return func(self, *args, **kwargs)

    return cache


def assert_not_frozen(func):
    """
    Decorator that asserts a function is not called when an object
    is frozen. For example, it should not be possible to set an
    attribute on a frozen model as that might invalidate the results
    in the cache.

    Parameters
    ----------
    func
        Some function

    Raises
    ------
    AssertionError
        If the function is called when the object is frozen
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        string_args = list(filter(lambda arg: isinstance(arg, str), args))
        if (
            "_is_frozen" not in string_args
            and "_frozen_cache" not in string_args
            and hasattr(self, "_is_frozen")
            and self._is_frozen
        ):
            raise AssertionError("Frozen models cannot be modified")
        return func(self, *args, **kwargs)

    return wrapper


class AbstractModel(ModelObject):
    def __init__(self, label=None, id_=None):
        self._is_frozen = False
        self._frozen_cache = dict()
        super().__init__(label=label, id_=id_)

    @classmethod
    def _cached_property_names(cls) -> frozenset:
        """
        Return the names of every ``cached_property``-style descriptor
        declared anywhere in ``cls``'s MRO.

        Used by the ``__dict__``-iteration sites in this module and in
        ``autofit/mapper/prior_model/`` to exclude cached descriptor values
        from instance construction, ``ModelInstance.dict``, pickling, and
        downstream JAX pytree flattening. See PyAutoFit#1300 for the
        diagnosed leak this defends against.
        """
        from autoconf.tools.decorators import cached_property_names

        return cached_property_names(cls)

    def __getstate__(self):
        excluded = type(self)._cached_property_names()
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "_frozen_cache" and key not in excluded
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._frozen_cache = {}

    def freeze(self):
        """
        Freeze this object.

        A frozen object caches results for some function calls
        and does not allow its state to be modified.
        """
        logger.debug("Freezing model")
        tuples = self.direct_tuples_with_type(AbstractModel)
        for _, model in tuples:
            if model is not self:
                model.freeze()
        self._is_frozen = True

    def unfreeze(self):
        """
        Unfreeze this object. Allows modification and removes
        caches associated with some functions.
        """
        logger.debug("Thawing model")
        self._is_frozen = False
        tuples = self.direct_tuples_with_type(AbstractModel)
        for _, model in tuples:
            if model is not self:
                model.unfreeze()
        self._frozen_cache = dict()

    def __add__(self, other):
        instance = self.__class__()

        def add_items(item_dict):
            for key, value in item_dict.items():
                if isinstance(value, list) and hasattr(instance, key):
                    setattr(instance, key, getattr(instance, key) + value)
                else:
                    setattr(instance, key, value)

        add_items(self.__dict__)
        add_items(other.__dict__)
        return instance

    def copy(self):
        """
        Create a copy of the model. All priors remain equivalent - i.e. two
        copies of a model in a collection has the same prior count as a single
        model.
        """
        return copy.deepcopy(self)

    def object_for_path(
        self, path: Iterable[Union[str, int, type]]
    ) -> Union[object, List]:
        """
        Get the object at a given path.

        The path describes the location of some object in the model.

        String entries get an attribute.
        Int entries index an attribute.
        Type entries product a new ModelInstance which collates all of the instances
        of a given type in the path.

        Parameters
        ----------
        path
            A tuple describing the path to an object in the model tree

        Returns
        -------
        An object or Instance collating a collection of objects with a given type.
        """
        instance = self
        for name in path:
            if isinstance(name, int):
                instance = instance[name]
            elif isinstance(name, type):
                from autofit.mapper.prior_model.prior_model import Model

                instances = [
                    instance
                    for _, instance in self.path_instance_tuples_for_class(name)
                ]
                instances += [
                    instance
                    for _, instance in self.path_instance_tuples_for_class(Model)
                    if issubclass(instance.cls, name)
                ]
                instance = ModelInstance(instances)
            else:
                instance = getattr(instance, name)
        return instance

    @frozen_cache
    def path_instance_tuples_for_class(
        self,
        cls: Union[Tuple, Type],
        ignore_class: Optional[type] = None,
        ignore_children: bool = True,
    ):
        """
        Tuples containing the path tuple and instance for every instance of the class
        in the model tree.

        Parameters
        ----------
        ignore_class
            Children of instances of this class are ignored
        ignore_children
            If true do not continue to recurse the children of an object once found
        cls
            The type to find instances of

        Returns
        -------
        path_instance_tuples: [((str,), object)]
            Tuples containing the path to and instance of objects of the given type.
        """
        return path_instances_of_class(
            self, cls, ignore_class=ignore_class, ignore_children=ignore_children
        )

    @frozen_cache
    def direct_tuples_with_type(self, class_type):
        return list(
            filter(
                lambda t: t[0] != "id"
                and not t[0].startswith("_")
                and isinstance(t[1], class_type),
                self.__dict__.items(),
            )
        )

    @frozen_cache
    def models_with_type(
        self,
        cls: Union[Type, Tuple[Type, ...]],
        include_zero_dimension=False,
    ) -> List["AbstractModel"]:
        """
        Return all models of a given type in the model tree.

        Parameters
        ----------
        cls
            The type to find instances of
        include_zero_dimension
            If true, include models with zero dimensions

        Returns
        -------
        A list of models of the given type
        """
        # noinspection PyTypeChecker
        return [
            t[1]
            for t in self.model_tuples_with_type(
                cls, include_zero_dimension=include_zero_dimension
            )
        ]

    @frozen_cache
    def model_tuples_with_type(
        self, cls: Union[Type, Tuple[Type, ...]], include_zero_dimension=False
    ):
        """
        All models of the class in this model which have at least
        one free parameter, recursively.

        Parameters
        ----------
        cls
            The type of the model
        include_zero_dimension
            If true, include models with 0 free parameters

        Returns
        -------
        Models with free parameters
        """
        from .prior_model.prior_model import Model

        return [
            (path, model)
            for path, model in self.attribute_tuples_with_type(
                Model, ignore_children=False
            )
            if issubclass(model.cls, cls)
            and (include_zero_dimension or model.prior_count > 0)
        ]

    @frozen_cache
    def attribute_tuples_with_type(
        self,
        class_type,
        ignore_class=None,
        ignore_children=True,
    ) -> List[tuple]:
        """
        Tuples describing the name and instance for attributes in the model
        with a given type, recursively.

        Parameters
        ----------
        ignore_children
            If True then recursion stops at instances with the type
        class_type
            The type of the objects to find
        ignore_class
            Any classes which should not be recursively searched

        Returns
        -------
        Tuples containing the name and instance of each attribute with the type
        """
        return [
            (path[-1] if len(path) > 0 else "", value)
            for path, value in self.path_instance_tuples_for_class(
                class_type,
                ignore_class=ignore_class,
                ignore_children=ignore_children,
            )
        ]


@DynamicRecursionCache()
def path_instances_of_class(
    obj,
    cls: type,
    ignore_class: Optional[Union[type, Tuple[type]]] = None,
    ignore_children: bool = False,
):
    """
    Recursively search the object for instances of a given class

    Parameters
    ----------
    obj
        The object to recursively search
    cls
        The type to search for
    ignore_class
        A type or tuple of classes to skip
    ignore_children
        If true stop recursion at found objects

    Returns
    -------
    instance of type
    """
    if ignore_class is not None and isinstance(obj, ignore_class):
        return []

    results = []
    if isinstance(obj, cls):
        results.append((tuple(), obj))
        if ignore_children:
            return results

    if isinstance(obj, list):
        for i, item in enumerate(obj):
            for path, instance in path_instances_of_class(
                item, cls, ignore_class=ignore_class, ignore_children=ignore_children
            ):
                results.append(((i,) + path, instance))
        return results

    try:
        from autofit.mapper.prior_model.annotation import AnnotationPriorModel

        if isinstance(obj, dict):
            d = obj
        else:
            d = obj.__dict__

        for key, value in d.items():
            if key.startswith("_"):
                continue
            for item in path_instances_of_class(
                value, cls, ignore_class=ignore_class, ignore_children=ignore_children
            ):
                if isinstance(value, AnnotationPriorModel):
                    path = (key,)
                else:
                    path = (key, *item[0])
                results.append((path, item[1]))
        return results
    except (AttributeError, TypeError):
        return results


class ModelInstance(AbstractModel):
    """
    An instance of a Collection or Model. This is created by optimisers and correspond
    to a point in the parameter space.

    @DynamicAttrs
    """

    __dictable_type__ = "instance"

    def __init__(self, child_items: Optional[Union[List, Dict]] = None, id_=None):
        """
        An instance of a Collection or Model. This is created by optimisers and correspond
        to a point in the parameter space.

        Parameters
        ----------
        child_items
            The child items of the instance. This can be a list or dict.

            If a list, the items are assigned to the instance in order.
            If a dict, the items are assigned to the instance by key and accessed by attribute.
        """
        super().__init__()
        self.child_items = child_items
        self.id = id_

    def __eq__(self, other):
        try:
            return self.__dict__ == other.__dict__
        except AttributeError:
            return False

    def __getitem__(self, item):
        if isinstance(item, int):
            return list(self.values())[item]
        if isinstance(item, slice):
            return ModelInstance(list(self.values())[item])
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    @property
    def child_items(self):
        return self.dict

    @child_items.setter
    def child_items(self, child_items):
        if isinstance(child_items, list):
            for i, item in enumerate(child_items):
                self[i] = item
        if isinstance(child_items, dict):
            for key, value in child_items.items():
                self[key] = value

    def items(self):
        return self.dict.items()

    def __hash__(self):
        return self.id

    @property
    def dict(self):
        excluded = type(self)._cached_property_names()
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("id", "component_number", "item_number")
            and not (isinstance(key, str) and key.startswith("_"))
            and key not in excluded
        }

    def tree_flatten(self) -> Tuple[List, Tuple]:
        """
        Flatten the instance into a PyTree
        """
        keys, values = zip(*self.dict.items())
        return values, (
            *keys,
            self.id,
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Tuple,
        children: List,
    ):
        """
        Create an instance from a flattened PyTree

        Parameters
        ----------
        aux_data
            Auxiliary information that remains unchanged including
            the keys of the dict
        children
            Child objects subject to change

        Returns
        -------
        An instance of this class
        """
        *keys, id_ = aux_data

        instance = cls(id_=id_)

        for key, value in zip(keys, children):
            instance[key] = value
        return instance

    def values(self):
        return self.dict.values()

    def __len__(self):
        return len(self.values())

    def as_model(
        self,
        model_classes: Union[type, Iterable[type]] = tuple(),
        excluded_classes: Union[type, Iterable[type]] = tuple(),
    ):
        """
        Convert this instance to a model

        Parameters
        ----------
        model_classes
            The classes to convert to models
        excluded_classes
            The classes to exclude from conversion

        Returns
        -------
        A model
        """

        from autofit.mapper.prior_model.abstract import AbstractPriorModel

        return AbstractPriorModel.from_instance(
            self,
            model_classes,
            excluded_classes,
        )
