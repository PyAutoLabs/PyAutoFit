import copy
import inspect
import json
import logging
import random
import types
from collections import defaultdict
from typing import Tuple, Optional, Dict, List, Iterable, Generator, Union, Type

import numpy as np

from autoconf import conf
from autoconf.exc import ConfigException
from autofit import exc
from autofit.mapper import model
from autofit.mapper.model import AbstractModel, frozen_cache
from autofit.mapper.prior import TruncatedGaussianPrior
from autofit.mapper.prior import UniformPrior
from autofit.mapper.prior.abstract import Prior
from autofit.mapper.prior.constant import Constant
from autofit.mapper.prior.deferred import DeferredArgument
from autofit.mapper.prior.tuple_prior import TuplePrior
from autofit.mapper.prior.width_modifier import WidthModifier
from autofit.mapper.prior_model.attribute_pair import DeferredNameValue
from autofit.mapper.prior_model.attribute_pair import (
    cast_collection,
    PriorNameValue,
    InstanceNameValue,
)
from autofit.mapper.prior_model.recursion import DynamicRecursionCache
from autofit.mapper.prior_model.representative import find_groups
from autofit.mapper.prior_model.util import PriorModelNameValue
from autofit.text import formatter as frm
from autofit.text.formatter import TextFormatter
from autofit.tools.util import info_whitespace
from autofit.tools.util import split_paths

logger = logging.getLogger(__name__)


class Limits:
    @staticmethod
    def for_class_and_attributes_name(cls, attribute_name):
        """Look up the (lower, upper) limits for a class attribute from config.

        Parameters
        ----------
        cls
            The model class.
        attribute_name
            The name of the attribute on that class.

        Returns
        -------
        tuple
            A (lower, upper) pair of limit values.
        """
        limit_dict = conf.instance.prior_config.for_class_and_suffix_path(
            cls, [attribute_name, "limits"]
        )
        return limit_dict["lower"], limit_dict["upper"]


class TuplePathModifier:
    def __init__(self, model_: "AbstractPriorModel"):
        """
        Modifies paths to priors contained in tuples.

        When a tuple is found in a signature a PriorTuple is created.

        The true path to a variable in the tuple is
        ("some", "preamble", "tuple_name", "tuple_name_0")

        Where 0 is the index of the tuple member. "tuple_name" must be
        removed from this path for some uses.

        When called, instances of this class remove the name of the tuple
        i.e:
        -> ("some", "preamble", "tuple_name_0")

        Parameters
        ----------
        model_
        """
        tuple_priors = model_.path_instance_tuples_for_class(TuplePrior)
        try:
            self.tuple_paths, _ = zip(*tuple_priors)
        except ValueError:
            self.tuple_paths = None

    def __call__(self, path):
        if self.tuple_paths is not None:
            if path[:-1] in self.tuple_paths:
                return path[:-2] + (path[-1],)
        return path


Path = Tuple[str, ...]


class MeanField:
    def __init__(self, prior_model: "AbstractPriorModel"):
        """
        Implements same interface as graphical code

        Parameters
        ----------
        prior_model
        """
        self.prior_model = prior_model

    def __getitem__(self, item):
        """
        Retrieve a prior by a prior with the same id
        """
        for prior in self.prior_model.priors:
            if prior == item:
                return prior
        raise KeyError(f"Could not find {item} in model")


def paths_to_tree(paths: List[Tuple[str, ...]], tree: Optional[dict] = None) -> dict:
    """
    Recursively convert a list of paths to a tree structure where common paths
    are matched.

    Parameters
    ----------
    paths
        A list of paths to attributes in the model.
    tree
        A tree already embedded in a parent tree.

    Returns
    -------
    A tree with depth max(map(len, paths))

    Examples
    --------
    paths_to_tree([
        ("one", "two", "three"),
        ("one", "two", "four"),
    ])

    gives

    {
        "one": {
            "two": {
                "three": {},
                "four": {}
            }
        }
    }
    """
    tree = tree or dict()
    for path in paths:
        if len(path) == 0:
            return tree
        first, *rest = path
        if first not in tree:
            child = dict()
            tree[first] = child
        tree[first] = paths_to_tree([rest], tree=tree[first])
    return tree


class AbstractPriorModel(AbstractModel):
    """
    Abstract model that maps a set of priors to a particular class. Must be
    overridden by any prior model so that the model mapper recognises its prior \
    model attributes.
    @DynamicAttrs
    """

    def __init__(self, label=None):
        super().__init__(label=label)
        self._assertions = list()

    @property
    def assertions(self):
        """The list of assertion constraints attached to this model.

        Assertions are checked when creating instances; a failed assertion
        raises FitException, causing the non-linear search to resample.
        """
        return self._assertions

    @assertions.setter
    def assertions(self, assertions):
        self._assertions = assertions

    def check_assertions(self, arguments: Dict[Prior, float]):
        """
        Check that all assertions are satisfied by the given arguments.

        Parameters
        ----------
        arguments
            A dictionary mapping priors to values

        Raises
        ------
        FitException
            If any assertion is not satisfied
        """
        failed_assertions = [
            assertion
            for assertion in self._assertions
            if assertion is False
            or assertion is not True
            and not assertion.instance_for_arguments(
                arguments,
            )
        ]
        number_of_failed_assertions = len(failed_assertions)
        if number_of_failed_assertions > 0:
            name_string = "\n".join(
                [
                    assertion.name
                    for assertion in failed_assertions
                    if hasattr(assertion, "name") and assertion.name is not None
                ]
            )
            raise exc.FitException(
                f"{number_of_failed_assertions} assertions failed!\n{name_string}"
            )

    def set_item_at_path(self, path: Tuple[str, ...], value):
        """
        Set an item at a path in the model.

        Parameters
        ----------
        path
            A tuple of strings representing a path to an attribute
        value
            The value to be set at the path
        """
        obj = self
        for attribute in path[:-1]:
            obj = getattr(obj, attribute)

        setattr(obj, path[-1], value)

    def cast(
        self,
        value_dict: Dict["AbstractModel", dict],
        new_class: type,
    ) -> "AbstractPriorModel":
        """
        Cast models to a new type. Allows selected models in within this
        model to be given a new type and new arguments.

        Parameters
        ----------
        value_dict
            A dictionary mapping models to dictionaries of argument overrides
        new_class
            A new class to which specified models should be converted

        Returns
        -------
        A model where specified child models have been updated to a new class
        and new arguments
        """
        from .prior_model import Model

        updated = self

        for path, prior_model in self.path_instance_tuples_for_class(Model):
            try:
                model_value_dict = value_dict[prior_model]
                argument_dict = {
                    **dict(prior_model.direct_prior_tuples),
                    **dict(prior_model.direct_tuples_with_type(float)),
                    **model_value_dict,
                }
                updated = updated.replacing_for_path(
                    path, Model(new_class, **argument_dict)
                )

            except KeyError:
                pass

        return updated

    def without_attributes(self) -> "AbstractModel":
        """
        Returns a copy of this object with all priors, prior models and
        constants removed.
        """
        without_attributes = copy.copy(self)
        for key in self.__dict__:
            if not (key.startswith("_") or key in ("cls", "id")):
                delattr(without_attributes, key)
        return without_attributes

    def _with_paths(self, tree: dict) -> "AbstractModel":
        """
        Recursively generate a copy of this model retaining only objects
        specified by the tree.

        Parameters
        ----------
        tree
            A tree formed of dictionaries describing which components of the
            model should be retained.

        Returns
        -------
        A copy of this model with a subset of attributes
        """
        if len(tree) == 0:
            return self

        with_paths = self.without_attributes()
        for name, subtree in tree.items():
            # noinspection PyProtectedMember
            new_value = getattr(self, name)
            if isinstance(
                new_value,
                (
                    AbstractPriorModel,
                    TuplePrior,
                ),
            ):
                new_value = new_value._with_paths(subtree)
            setattr(with_paths, name, new_value)
        return with_paths

    @split_paths
    def with_paths(self, paths: List[Tuple[str, ...]]) -> "AbstractModel":
        """
        Recursively generate a copy of this model retaining only objects
        specified by the list of paths.

        Parameters
        ----------
        paths
            A list of tuples of strings each of which points to a retained attribute.
            All children of a given path are retained.

        Returns
        -------
        A copy of this model with a subset of attributes
        """
        return self._with_paths(paths_to_tree(paths))

    def _without_paths(self, tree: dict) -> "AbstractModel":
        """
        Recursively generate a copy of this model removing objects
        specified by the tree.

        Parameters
        ----------
        tree
            A tree formed of dictionaries describing which components of the
            model should be removed.

        Returns
        -------
        A copy of this model with a subset of attributes
        """
        without_paths = copy.deepcopy(self)
        for name, subtree in tree.items():
            # noinspection PyProtectedMember
            if len(subtree) == 0:
                delattr(without_paths, name)
            else:
                new_value = getattr(without_paths, name)
                if isinstance(
                    new_value,
                    (
                        AbstractPriorModel,
                        TuplePrior,
                    ),
                ):
                    new_value = new_value._without_paths(subtree)
                setattr(without_paths, name, new_value)
        return without_paths

    @split_paths
    def without_paths(self, paths: List[Tuple[str, ...]]) -> "AbstractModel":
        """
        Recursively generate a copy of this model retaining only objects
        not specified by the list of paths.

        Parameters
        ----------
        paths
            A list of tuples of strings each of which points to removed attribute.
            All children of a given path are removed.

        Returns
        -------
        A copy of this model with a subset of attributes
        """
        return self._without_paths(paths_to_tree(paths))

    def index(self, path: Tuple[str, ...]) -> int:
        """
        Retrieve the index of a given path in the model
        """
        return self.paths.index(path)

    @property
    def mean_field(self) -> MeanField:
        """
        Implements the same interface as the graphical code
        """
        return MeanField(self)

    @classmethod
    def from_json(cls, file: str):
        """
        Loads the model from a .json file, which was written using the model's dictionary (`dict`) attribute as
        follows:

        import json

        with open(filename, "w+") as f:
            json.dump(model.dict, f, indent=4)

        Parameters
        ----------
        file
            The path and name of the .json file the model is loaded from.

        Returns
        -------
        object
            The model, which may be a `Collection` of `Model` objects or a single `Model`.
        """

        with open(file) as json_file:
            model_dict = json.load(json_file)

        return cls.from_dict(d=model_dict)

    def add_assertion(self, assertion, name=""):
        """
        Assert that some relationship holds between physical values associated with
        priors at the point an instance is created. If this fails a FitException is
        raised causing the model to be re-sampled.
        Parameters
        ----------
        assertion
            An assertion that one prior must be greater than another.
        name
            A name describing the assertion that is logged when it is violated.
        """
        if assertion is True:
            return
        try:
            assertion.name = name
        except AttributeError:
            pass
        self._assertions.append(assertion)

    @property
    def name(self):
        """The class name of this prior model (e.g. ``"Model"`` or ``"Collection"``)."""
        return self.__class__.__name__

    # noinspection PyUnusedLocal
    @staticmethod
    def from_object(t, *args, **kwargs):
        """Convert an arbitrary object into an appropriate prior model representation.

        - Classes become ``Model`` instances.
        - Lists and dicts become ``Collection`` instances.
        - Floats become ``Constant`` instances.
        - Existing prior models and other objects are returned as-is.

        Parameters
        ----------
        t
            A class, list, dict, float, or existing prior model.

        Returns
        -------
        An AbstractPriorModel, Constant, or the original object.
        """
        if inspect.isclass(t):
            from .prior_model import Model

            obj = object.__new__(Model)
            obj.__init__(t, **kwargs)
        elif isinstance(t, list) or isinstance(t, dict):
            from autofit.mapper.prior_model import collection

            obj = object.__new__(collection.Collection)
            obj.__init__(t)
        else:
            obj = t

        if isinstance(obj, float):
            return Constant(obj)
        return obj

    def take_attributes(self, source: object):
        """
        Take all attributes with a matching path from the source prior model.

        For example, if this prior model has a prior "one" and a matching prior
        is found associated with the source model then that attribute is attached
        to this model.

        If no matching attribute is found nothing happens.

        Parameters
        ----------
        source
            An instance or prior model from a previous search from which attributes
            are passed to this model.
        """

        def assert_no_assertions(obj):
            if len(getattr(obj, "_assertions", [])) > 0:
                raise AssertionError(
                    "take_attributes cannot be called once assertions have been added to the target"
                )

        assert_no_assertions(self)

        for path, _ in sum(
            map(self.path_instance_tuples_for_class, (Prior, float, TuplePrior)), []
        ):
            try:
                item = copy.copy(source)
                if isinstance(item, dict):
                    from autofit.mapper.prior_model.collection import Collection

                    item = Collection(item)
                for attribute in path:
                    if isinstance(attribute, int):
                        item = item[attribute]
                    else:
                        item = copy.copy(getattr(item, attribute))

                target = self
                for attribute in path[:-1]:
                    if isinstance(attribute, int):
                        target = target[attribute]
                    else:
                        target = getattr(target, attribute)
                    assert_no_assertions(target)

                attribute = path[-1]
                if isinstance(attribute, int):
                    target[attribute] = item
                else:
                    setattr(target, path[-1], item)
            except AttributeError:
                pass

    def instance_from_unit_vector(self, unit_vector, ignore_assertions : bool = False):
        """
        Returns a ModelInstance, which has an attribute and class instance corresponding
        to every `Model` attributed to this instance.
        This method takes as input a unit vector of parameter values, converting each to
        physical values via their priors.
        Parameters
        ----------
        unit_vector: [float]
            A unit hypercube vector that is mapped to an instance of physical values via the priors.
        Returns
        -------
        model_instance : autofit.mapper.model.ModelInstance
            An object containing reconstructed model_mapper instances
        ignore_assertions
            If True, the assertions attached to this model (e.g. that one parameter > another parameter) are ignored.

        Raises
        ------
        exc.FitException
            If any assertion attached to this object returns False.
        """
        exception_tuples = self.attribute_tuples_with_type(ConfigException)
        if len(exception_tuples) > 0:
            for name, exception in exception_tuples:
                logger.exception(f"Could not load {name} because:\n\n{exception}")
            names = [name for name, _ in exception_tuples]
            raise ConfigException(
                f"No configuration was found for some attributes ({', '.join(names)})"
            )

        if self.prior_count != len(unit_vector):
            raise AssertionError(
                f"prior_count ({self.prior_count}) != len(unit_vector) {len(unit_vector)}"
            )

        arguments = dict(
            map(
                lambda prior_tuple, unit: (
                    prior_tuple.prior,
                    prior_tuple.prior.value_for(
                        unit,
                    ),
                ),
                self.prior_tuples_ordered_by_id,
                unit_vector,
            )
        )

        return self.instance_for_arguments(
            arguments,
            ignore_assertions=ignore_assertions
        )

    @property
    @cast_collection(PriorNameValue)
    @frozen_cache
    def unique_prior_tuples(self):
        """
        Returns
        -------
        prior_tuple_dict: [(Prior, PriorTuple)]
            The set of all priors associated with this mapper
        """
        return list(
            {
                prior_tuple[1]: prior_tuple
                for prior_tuple in self.attribute_tuples_with_type(Prior)
            }.values()
        )

    @property
    @cast_collection(PriorNameValue)
    @frozen_cache
    def prior_tuples_ordered_by_id(self):
        """
        Returns
        -------
        priors: [Prior]
            An ordered list of unique priors associated with this mapper
        """
        return sorted(
            list(self.unique_prior_tuples), key=lambda prior_tuple: prior_tuple.prior.id
        )

    @property
    def priors_ordered_by_id(self):
        """Unique priors sorted by their id, defining the canonical parameter ordering."""
        return [prior for _, prior in self.prior_tuples_ordered_by_id]

    def vector_from_unit_vector(self, unit_vector):
        """
        Parameters
        ----------
        unit_vector: [float]
            A unit hypercube vector

        Returns
        -------
        values: [float]
            A vector with values output by priors
        """
        return list(
            map(
                lambda prior_tuple, unit: prior_tuple.prior.value_for(
                    unit,
                ),
                self.prior_tuples_ordered_by_id,
                unit_vector,
            )
        )

    def random_unit_vector_within_limits(
        self, lower_limit: float = 0.0, upper_limit: float = 1.0
    ) -> List[float]:
        """
        Generate a random vector of unit values by drawing uniform random values between 0 and 1.

        Returns
        -------
        unit_values
            A list of unit values constructed by taking random values from each prior.
        """
        return [
            random.uniform(
                lower_limit,
                upper_limit,
            )
            for prior in self.priors_ordered_by_id
        ]

    def random_vector_from_priors_within_limits(
        self, lower_limit: float = 0.0, upper_limit: float = 1.0
    ) -> List[float]:
        """
        Returns a random vector of physical values by drawing uniform random values between lower and upper limits
        defined as unit values, using the model priors to map them from unit values to physical values.

        This function guesses infinite unit vectors, until one is randomly sampled whose physical parameters are within
        the model's phyiscal limits.

        This is used for MCMC initialization, whereby the starting points of a walker(s) are confined to a restricted
        range of prior space. In particular, it is used for generate the "ball" initialization of Emcee.

        Parameters
        ----------
        lower_limit
            The lower limit as a unit value within which unit values (which are converted to physical values via the
            priors) are generated.
        upper_limit
            The upper limit as a unit value within which unit values (which are converted to physical values via the
            priors) are generated.

        Returns
        -------
        physical_values
            A list of physical values constructed by taking the mean possible value from
            each prior.
        """
        vector = []

        for prior in self.priors_ordered_by_id:
            vector.append(
                prior.random(
                    lower_limit=lower_limit,
                    upper_limit=upper_limit,
                )
            )

        return vector

    def random_instance_from_priors_within_limits(
        self, lower_limit: float = 0.0, upper_limit: float = 1.0
    ):
        """
        Returns a random instance of physical values by drawing uniform random values between lower and upper limits
        defined as unit values, using the model priors to map them from unit values to physical values.

        This function guesses infinite unit vectors, until one is randomly sampled whose physical parameters are within
        the model's phyiscal limits. The final unit vector is then mapped to an instance.

        Parameters
        ----------
        lower_limit
            The lower limit as a unit value within which unit values (which are converted to physical values via the
            priors) are generated.
        upper_limit
            The upper limit as a unit value within which unit values (which are converted to physical values via the
            priors) are generated.

        Returns
        -------
        physical_values
            A list of physical values constructed by taking the mean possible value from
            each prior.
        """

        vector = self.random_vector_from_priors_within_limits(
            lower_limit=lower_limit, upper_limit=upper_limit
        )

        return self.instance_from_vector(vector=vector)

    @property
    def random_vector_from_priors(self):
        """Generate a random vector of physical values by drawing uniform random values between 0 and 1 and using
        the model priors to map them from unit values to physical values.
        Returns
        -------
        physical_values: [float]
            A list of physical values constructed by taking random values from each prior.
        """
        return self.random_vector_from_priors_within_limits(
            lower_limit=0.0, upper_limit=1.0
        )

    @property
    def physical_values_from_prior_medians(self):
        """
        Returns
        -------
        physical_values: [float]
            A list of physical values constructed by taking the mean possible value from
            each prior.
        """
        return self.vector_from_unit_vector([0.5] * len(self.unique_prior_tuples))

    def instance_from_vector(self, vector, ignore_assertions: bool = False, xp=np):
        """
        Returns a ModelInstance, which has an attribute and class instance corresponding
        to every `Model` attributed to this instance.
        This method takes as input a physical vector of parameter values, thus omitting
        the use of priors.
        Parameters
        ----------
        vector: [float]
            A vector of physical parameter values that is mapped to an instance.
        ignore_assertions
            If True, any assertions attached to this object are ignored and not checked.

        Returns
        -------
        model_instance : autofit.mapper.model.ModelInstance
            An object containing reconstructed model_mapper instances
        """
        if len(vector) != self.prior_count:
            raise AssertionError(
                f"Vector length {len(vector)} != prior count {self.prior_count}"
            )
        arguments = dict(
            map(
                lambda prior_tuple, physical_unit: (prior_tuple.prior, physical_unit),
                self.prior_tuples_ordered_by_id,
                vector,
            )
        )

        return self.instance_for_arguments(
            arguments,
            ignore_assertions=ignore_assertions,
            xp=xp
        )

    def has(self, cls: Union[Type, Tuple[Type, ...]]) -> bool:
        """
        Parameters
        ----------
        cls
            The type to check for

        Returns
        -------
        True iff this model contains an instance or model with the given
        type recursively. Includes models which have zero priors.
        """
        return self.has_instance(cls) or self.has_model(
            cls, include_zero_dimension=True
        )

    def has_instance(self, cls) -> bool:
        """
        True iff this model contains an instance of type
        cls, recursively.
        """
        return len(self.attribute_tuples_with_type(cls)) > 0

    def has_model(self, cls, include_zero_dimension=False) -> bool:
        """
        True iff this model contains a Model of type
        cls, recursively.
        """
        return (
            len(
                self.model_tuples_with_type(
                    cls,
                    include_zero_dimension=include_zero_dimension,
                )
            )
            > 0
        )

    def is_only_model(self, cls) -> bool:
        """
        True iff this model contains at least one Model
        of type cls and contains no PriorModels that are not
        of type cls, recursively.
        """
        from .prior_model import Model

        cls_models = self.model_tuples_with_type(cls)
        other_models = [
            value
            for _, value in self.attribute_tuples_with_type(Model)
            if value.prior_count > 0
        ]
        return len(cls_models) > 0 and len(cls_models) == len(other_models)

    def replacing(self, arguments):
        """Return a new model with some priors replaced.

        This is a convenience alias for ``mapper_from_partial_prior_arguments``.
        Priors not present in the arguments dict are kept unchanged.

        Parameters
        ----------
        arguments : dict
            A dictionary mapping existing Prior objects to new priors or fixed values.
        """
        return self.mapper_from_partial_prior_arguments(arguments)

    @classmethod
    def product(cls, models: Iterable["AbstractPriorModel"]) -> "AbstractPriorModel":
        """
        Combine multiple models with the same structure by replacing priors with
        priors that contain a message which is the product of the messages of the
        priors with the same path in each model.

        Parameters
        ----------
        models
            A list of models to be combined

        Returns
        -------
        A model where each prior has a message which is the product of the messages
        associated with that prior across the models.
        """
        first, *rest = models

        arguments = dict()

        for path, prior in first.path_priors_tuples:
            for other in rest:
                prior = prior.with_message(
                    prior.message * other.object_for_path(path).message
                )
            arguments[prior] = prior
        return first.mapper_from_prior_arguments(arguments)

    def mapper_from_partial_prior_arguments(self, arguments):
        """
        Returns a new model mapper from a dictionary mapping existing priors to
        new priors, keeping existing priors where no mapping is provided.
        Parameters
        ----------
        arguments: {Prior: Prior}
            A dictionary mapping priors to priors
        Returns
        -------
        model_mapper: ModelMapper
            A new model mapper with updated priors.
        """
        original_prior_dict = {prior: prior for prior in self.priors}
        return self.mapper_from_prior_arguments({**original_prior_dict, **arguments})

    def mapper_from_prior_arguments(self, arguments):
        """
        Returns a new model mapper from a dictionary mapping existing priors to
        new priors.
        Parameters
        ----------
        arguments: {Prior: Prior}
            A dictionary mapping priors to priors
        Returns
        -------
        model_mapper: ModelMapper
            A new model mapper with updated priors.
        """
        logger.debug(f"Creating a new mapper from arguments")

        return self.gaussian_prior_model_for_arguments(arguments)

    def gaussian_prior_model_for_arguments(self, arguments):
        raise NotImplementedError()

    def mapper_via_defaults_from(self):
        """
        The widths of the new priors are taken from the
        width_config. The new gaussian priors must be provided in the same order as
        the priors associated with model.
        If a is not None then all priors are created with an absolute width of a.
        If r is not None then all priors are created with a relative width of r.
        Parameters
        ----------
        means
            The median PDF value of every Gaussian, which centres each `GaussianPrior`.
        r
            The relative width to be assigned to gaussian priors
        a
            print(tuples[i][1], width)
            The absolute width to be assigned to gaussian priors
        tuples
            A list of tuples each containing the mean and width of a prior
        Returns
        -------
        mapper: ModelMapper
            A new model mapper with all priors replaced by gaussian priors.
        """

        prior_tuples = self.prior_tuples_ordered_by_id
        arguments = {}

        for prior_tuple in prior_tuples:

            prior = prior_tuple.prior

            arguments[prior] = prior

        return self.mapper_from_prior_arguments(arguments)

    def mapper_from_prior_means(self, means, a=None, r=None, no_limits=False):
        """
        The widths of the new priors are taken from the
        width_config. The new gaussian priors must be provided in the same order as
        the priors associated with model.
        If a is not None then all priors are created with an absolute width of a.
        If r is not None then all priors are created with a relative width of r.
        Parameters
        ----------
        means
            The median PDF value of every Gaussian, which centres each `GaussianPrior`.
        r
            The relative width to be assigned to gaussian priors
        a
            print(tuples[i][1], width)
            The absolute width to be assigned to gaussian priors
        tuples
            A list of tuples each containing the mean and width of a prior
        Returns
        -------
        mapper: ModelMapper
            A new model mapper with all priors replaced by gaussian priors.
        """

        prior_tuples = self.prior_tuples_ordered_by_id
        prior_class_dict = self.prior_class_dict
        arguments = {}

        for prior_tuple, mean in zip(prior_tuples, means):
            prior = prior_tuple.prior
            cls = prior_class_dict[prior]

            name = prior_tuple.name
            # Use the name of the collection for configuration when a prior's name
            # is just a number (i.e. its position in a collection)
            if name.isdigit():
                name = self.path_for_prior(prior_tuple.prior)[-2]

            width_modifier = (
                prior.width_modifier
                or WidthModifier.for_class_and_attribute_name(cls, name)
            )

            if a is not None and r is not None:
                raise exc.PriorException(
                    "Width of new priors cannot be both relative and absolute."
                )
            if a is not None:
                width = a
            elif r is not None:
                width = r * mean
            else:
                width = width_modifier(mean)

            if no_limits:
                limits = (float("-inf"), float("inf"))
            else:
                try:
                    limits = Limits.for_class_and_attributes_name(cls, name)
                except ConfigException:
                    limits = prior.limits

            sigma = width

            new_prior = TruncatedGaussianPrior(mean, sigma, *limits)
            new_prior.id = prior.id
            new_prior.width_modifier = prior.width_modifier
            arguments[prior] = new_prior

        return self.mapper_from_prior_arguments(arguments)

    def with_limits(self, limits: List[Tuple[float, float]]) -> "AbstractPriorModel":
        """
        Create a new instance of this model where each prior is updated to
        lie between new limits.

        Parameters
        ----------
        limits
            A list of pairs of lower and upper limits, one for each prior.

        Returns
        -------
        A new model with updated limits
        """
        return self.mapper_from_prior_arguments(
            {
                prior: prior.with_limits(*prior_limits)
                for prior, prior_limits in zip(self.priors_ordered_by_id, limits)
            }
        )

    def mapper_from_uniform_floats(self, floats, b):
        """
        The widths of the new priors are the `floats` value minus and plus the input bound `b`.

        Parameters
        ----------
        floats
            A list of floats each containing the centre of the new uniform priors.
        b
            The bound value which is subtracted from each float to calculate the `lower_limit` and `upper_limit`
            of each uniform prior.

        Returns
        -------
        mapper: ModelMapper
            A new model mapper with all priors replaced by uniform priors.
        """

        prior_tuples = self.prior_tuples_ordered_by_id
        arguments = {}

        for i, prior_tuple in enumerate(prior_tuples):
            prior = prior_tuple.prior

            new_prior = UniformPrior(
                lower_limit=floats[i] - b, upper_limit=floats[i] + b
            )
            new_prior.id = prior.id
            arguments[prior] = new_prior

        return self.mapper_from_prior_arguments(arguments)

    def instance_from_prior_medians(self, ignore_assertions : bool = False):
        """
        Returns a list of physical values from the median values of the priors.
        Returns
        -------
        ignore_assertions
            If True, the assertions attached to this model (e.g. that one parameter > another parameter) are ignored
            and not checked.
        """
        return self.instance_from_unit_vector(
            unit_vector=[0.5] * self.prior_count,
            ignore_assertions=ignore_assertions
        )

    def log_prior_list_from_vector(
        self,
        vector: [float],
        xp=np,
    ):
        """
        Compute the log priors of every parameter in a vector, using the Prior of every parameter.
        The log prior values are used by Emcee to map the log likelihood to the poserior of the model.
        Parameters
        ----------
        vector : [float]
            A vector of physical parameter values.
        Returns
        -------
        log_prior_list : []
            An list of the log prior value of every parameter.
        """
        return list(
            map(
                lambda prior_tuple, value: prior_tuple.prior.log_prior_from_value(
                    value=value, xp=xp
                ),
                self.prior_tuples_ordered_by_id,
                vector,
            )
        )

    def random_instance(self, ignore_assertions : bool  = False):
        """
        Returns a random instance of the model.

        Parameters
        ----------
        ignore_assertions
            If True, the assertions attached to this model (e.g. that one parameter > another parameter) are ignored.
        """
        if ignore_assertions:
            return self.instance_from_unit_vector(
                unit_vector=[random.random() for _ in range(self.prior_count)],
                ignore_assertions=ignore_assertions,
            )
        return self.instance_from_unit_vector(
            unit_vector=[random.random() for _ in range(self.prior_count)],
        )

    @staticmethod
    @DynamicRecursionCache()
    def from_instance(
        instance,
        model_classes: Union[type, Iterable[type]] = tuple(),
        exclude_classes: Union[type, Iterable[type]] = tuple(),
    ):
        """
        Recursively create a prior object model from an object model.

        Parameters
        ----------
        model_classes
            A tuple of classes that should be converted to a prior model
        exclude_classes
            A tuple of classes that should not be converted to a prior model
        instance
            A dictionary, list, class instance or model instance
        Returns
        -------
        abstract_prior_model
            A concrete child of an abstract prior model
        """
        from autofit.mapper.prior_model import collection

        if isinstance(instance, exclude_classes):
            return instance
        if isinstance(instance, (Prior, AbstractPriorModel)):
            return instance
        elif isinstance(instance, list):
            result = collection.Collection(
                [
                    AbstractPriorModel.from_instance(
                        item,
                        model_classes=model_classes,
                        exclude_classes=exclude_classes,
                    )
                    for item in instance
                ]
            )
        elif isinstance(instance, model.ModelInstance):
            from autofit.mapper import model_mapper

            result = model_mapper.ModelMapper()
            for key, value in instance.dict.items():
                setattr(
                    result,
                    key,
                    AbstractPriorModel.from_instance(
                        value,
                        model_classes=model_classes,
                        exclude_classes=exclude_classes,
                    ),
                )
        elif isinstance(instance, dict):
            result = collection.Collection(
                {
                    key: AbstractPriorModel.from_instance(
                        value,
                        model_classes=model_classes,
                        exclude_classes=exclude_classes,
                    )
                    for key, value in instance.items()
                }
            )
        elif isinstance(instance, (np.ndarray, types.FunctionType)):
            return instance
        else:
            from .prior_model import Model

            try:
                result = Model(
                    instance.__class__,
                    **{
                        key: AbstractPriorModel.from_instance(
                            value,
                            model_classes=model_classes,
                            exclude_classes=exclude_classes,
                        )
                        for key, value in instance.__dict__.items()
                        if key != "cls"
                    },
                )
            except AttributeError:
                return instance
        if isinstance(instance, model_classes):
            return result.as_model()
        return result

    def items(self):
        """Return (name, value) pairs for all public, non-internal attributes.

        Excludes private attributes (prefixed with ``_``), ``cls``, and ``id``.
        """
        return [
            (key, value)
            for key, value in self.__dict__.items()
            if not key.startswith("_") and key not in ("cls", "id")
        ]

    @property
    @cast_collection(PriorNameValue)
    def direct_prior_tuples(self):
        return self.direct_tuples_with_type(Prior)

    @property
    @cast_collection(InstanceNameValue)
    def direct_instance_tuples(self):
        """(name, value) tuples for direct float and Constant attributes."""
        return self.direct_tuples_with_type(float) + self.direct_tuples_with_type(
            Constant
        )

    @property
    @cast_collection(PriorModelNameValue)
    def prior_model_tuples(self):
        """(name, prior_model) tuples for direct child AbstractPriorModel attributes."""
        return self.direct_tuples_with_type(AbstractPriorModel)

    @property
    @cast_collection(PriorModelNameValue)
    def direct_prior_model_tuples(self):
        """(name, prior_model) tuples for immediate child prior models (non-recursive)."""
        return self.direct_tuples_with_type(AbstractPriorModel)

    @property
    @cast_collection(PriorModelNameValue)
    def direct_tuple_priors(self):
        """(name, tuple_prior) tuples for direct TuplePrior attributes."""
        return self.direct_tuples_with_type(TuplePrior)

    @property
    @cast_collection(PriorNameValue)
    def tuple_prior_tuples(self):
        """
        Returns
        -------
        tuple_prior_tuples: [(String, TuplePrior)]
        """
        return self.direct_tuples_with_type(TuplePrior)

    @property
    @cast_collection(PriorNameValue)
    def direct_prior_tuples(self):
        """
        Returns
        -------
        direct_priors: [(String, Prior)]
        """
        return self.direct_tuples_with_type(Prior)

    @property
    @cast_collection(DeferredNameValue)
    def direct_deferred_tuples(self):
        """(name, deferred_argument) tuples for direct DeferredArgument attributes."""
        return self.direct_tuples_with_type(DeferredArgument)

    @property
    @cast_collection(PriorNameValue)
    def prior_tuples(self):
        """
        Returns
        -------
        priors: [(String, Prior))]
        """
        # noinspection PyUnresolvedReferences
        return self.attribute_tuples_with_type(Prior, ignore_children=True)

    def __eq__(self, other):
        return (
            isinstance(other, AbstractPriorModel)
            and self.direct_prior_model_tuples == other.direct_prior_model_tuples
        )

    @property
    @cast_collection(InstanceNameValue)
    def instance_tuples(self):
        """
        Returns
        -------
        instances: [(String, instance)]
        """
        return self.attribute_tuples_with_type(float, ignore_class=Prior)

    @property
    def prior_class_dict(self):
        """Map each prior to the class it will produce when instantiated.

        Direct priors on this model map to ``self.cls``. Child prior models
        contribute their own mappings recursively.
        """
        from autofit.mapper.prior_model.annotation import AnnotationPriorModel

        d = {prior[1]: self.cls for prior in self.prior_tuples}
        for prior_model in self.prior_model_tuples:
            if not isinstance(prior_model[1], AnnotationPriorModel):
                d.update(prior_model[1].prior_class_dict)
        return d

    def _instance_for_arguments(
        self,
        arguments: Dict[Prior, float],
        ignore_assertions: bool = False,
        xp=np,
    ):
        raise NotImplementedError()

    def instance_for_arguments(
        self,
        arguments: Dict[Prior, float],
        ignore_assertions: bool = False,
        xp=np,
    ):
        """
        Returns an instance of the model for a set of arguments

        Parameters
        ----------
        arguments
            Dictionary mapping priors to attribute analysis_path and value pairs
        ignore_assertions
            If True, the assertions attached to this model (e.g. that one parameter > another parameter) are ignored.

        Returns
        -------
            An instance of the class
        """
        if not (
            conf.instance["general"]["test"]["exception_override"] or ignore_assertions
        ):
            self.check_assertions(arguments)

        return self._instance_for_arguments(
            arguments,
            ignore_assertions=ignore_assertions,
            xp=xp
        )

    def path_for_name(self, name: str) -> Tuple[str, ...]:
        """
        Find the path to a prior in the model that matches
        a given name.

        For example, name_of_model_name_of_prior could match
        (name_of_model, name_of_prior). Unfortunately it is
        possible for ambiguity to occur. For example, another
        valid path could be (name, of_model, name_of_prior)
        in which case there is no way to determine which path
        actually matches the name.

        Parameters
        ----------
        name
            A name for a prior where names of models and the
            name of the prior have been joined by underscores.

        Returns
        -------
        A path to that model

        Raises
        ------
        AssertionError
            Iff no matching path is found
        """

        def _explode_path(path_):
            return tuple(string for part in path_ for string in part.split("_"))

        exploded = tuple(name.split("_"))
        for path, _ in self.path_priors_tuples:
            exploded_path = _explode_path(path)
            if exploded_path == exploded:
                return path

        for path, prior_tuple in self.path_instance_tuples_for_class(TuplePrior):
            for name, prior in prior_tuple.prior_tuples:
                total_path = path[:-1] + (name,)
                exploded_path = _explode_path(total_path)
                if exploded_path == exploded:
                    return path + (name,)
        raise AssertionError(f"No path was found matching {name}")

    def instance_from_prior_name_arguments(
        self,
        prior_name_arguments: Dict[str, float],
        ignore_assertions: bool = False,
    ):
        """
        Instantiate the model from the names of priors and
        corresponding values.

        Parameters
        ----------
        prior_name_arguments
            The names of priors where names of models and the
            name of the prior have been joined by underscores,
            mapped to corresponding values.
        ignore_assertions
            If True, the assertions attached to this model (e.g. that one parameter > another parameter) are ignored.

        Returns
        -------
        An instance of the model
        """
        return self.instance_from_path_arguments(
            {
                self.path_for_name(name): value
                for name, value in prior_name_arguments.items()
            },
            ignore_assertions=ignore_assertions,
        )

    def instance_from_path_arguments(
        self,
        path_arguments: Dict[Tuple[str], float],
        ignore_assertions: bool = False,
    ):
        """
        Create an instance from a dictionary mapping paths to tuples
        to corresponding values.

        Parameters
        ----------
        path_arguments
            A dictionary mapping paths to priors to corresponding values.
            Note that, for linked priors, each path only needs to be
            specified once. If multiple paths for the same prior are
            specified then the value for the last path will be used.
        ignore_assertions
            If True, the assertions attached to this model (e.g. that one parameter > another parameter) are ignored.

        Returns
        -------
        An instance of the model
        """
        arguments = {
            self.object_for_path(path): value for path, value in path_arguments.items()
        }
        return self._instance_for_arguments(
            arguments,
            ignore_assertions=ignore_assertions,
        )

    @property
    def prior_count(self) -> int:
        """
        How many unique priors does this model contain?
        """
        return len(self.unique_prior_tuples)

    @property
    def total_free_parameters(self) -> int:
        """
        Returns the prior count, but with a name that is more easy to interpret for users.
        """
        return self.prior_count

    @property
    def priors(self):
        """A list of all Prior objects in this model (may contain duplicates for shared priors)."""
        return [prior_tuple.prior for prior_tuple in self.prior_tuples]

    @property
    def _prior_id_map(self):
        return {prior.id: prior for prior in self.priors}

    def prior_with_id(self, prior_id):
        """Retrieve a prior by its unique integer id.

        Parameters
        ----------
        prior_id : int
            The id of the prior to find.

        Returns
        -------
        Prior
            The prior with the matching id.

        Raises
        ------
        KeyError
            If no prior with the given id exists in this model.
        """
        return self._prior_id_map[prior_id]

    def name_for_prior(self, prior):
        """Get the underscore-separated name for a prior in this model.

        Searches child prior models recursively. Returns None if the prior
        is not found.

        Parameters
        ----------
        prior : Prior
            The prior to find.

        Returns
        -------
        str or None
            The name path joined by underscores, e.g. ``"galaxy_centre"``.
        """
        for prior_model_name, prior_model in self.direct_prior_model_tuples:
            prior_name = prior_model.name_for_prior(prior)
            if prior_name is not None:
                return "{}_{}".format(prior_model_name, prior_name)
        prior_tuples = self.prior_tuples
        for name, p in prior_tuples:
            if p == prior:
                return name

    def __hash__(self):
        return self.id

    def __add__(self, other):
        if isinstance(other, Prior):
            return other + self

        result = copy.deepcopy(self)

        for key, value in other.__dict__.items():
            if not hasattr(result, key) or isinstance(value, Prior):
                setattr(result, key, value)
                continue
            result_value = getattr(result, key)
            if isinstance(value, AbstractPriorModel):
                if isinstance(result_value, AbstractPriorModel):
                    setattr(result, key, result_value + value)
                else:
                    setattr(result, key, value)

        return result

    def copy_with_fixed_priors(self, instance, excluded_classes=tuple()):
        """
        Recursively overwrite priors in the mapper with instance values from the
        instance except where the containing class is the descendant of a listed class.
        Parameters
        ----------
        excluded_classes
            Classes that should be left model
        instance
            The best fit from the previous search
        """
        mapper = copy.deepcopy(self)
        transfer_classes(instance, mapper, excluded_classes)
        return mapper

    @property
    def path_priors_tuples(self) -> List[Tuple[Path, Prior]]:
        """All (path, prior) tuples in this model, sorted by prior id.

        Unlike ``unique_path_prior_tuples``, this includes duplicate entries
        when a prior appears at multiple paths (shared priors).
        """
        path_priors_tuples = self.path_instance_tuples_for_class(Prior)
        return sorted(path_priors_tuples, key=lambda item: item[1].id)

    @property
    def paths(self) -> List[Path]:
        """
        A list of paths to all the priors in the model, ordered by their
        ids
        """
        return [path for path, _ in self.path_priors_tuples]

    @property
    def paths_formatted(self) -> List[Path]:
        """
        A list of paths to all the priors in the model, ordered by their
        ids
        """
        return [path for path, _ in self.path_priors_tuples]

    @property
    def composition(self):
        """A list of dot-separated path strings for each prior, ordered by prior id.

        For example: ``["galaxy.centre", "galaxy.normalization", "galaxy.sigma"]``.
        """
        return [".".join(path) for path in self.paths]

    def sort_priors_alphabetically(self, priors: Iterable[Prior]) -> List[Prior]:
        """
        Sort priors by their paths according to this model.

        Parameters
        ----------
        priors
            A set of priors

        Returns
        -------
        Those priors sorted alphabetically by path.
        """
        return sorted(priors, key=lambda prior: self.path_for_prior(prior))

    def path_for_object(self, obj) -> Optional[Path]:
        """
        Find a path that points at the given object.

        Parameters
        ----------
        obj
            An object in the model.

        Returns
        -------
        A path, a series of attributes that point to one location of the object.
        """
        for path, instance in self.path_instance_tuples_for_class(
            object, ignore_children=False
        ):
            if instance is obj:
                return path
        return None

    def path_for_prior(self, prior: Prior) -> Optional[Path]:
        """
        Find a path that points at the given tuple.
        Returns the first path or None if no path is found.
        Parameters
        ----------
        prior
            A prior representing what's known about some dimension of the model.
        Returns
        -------
        A path, a series of attributes that point to one location of the prior.
        """
        for path in self.all_paths_for_prior(prior):
            return path
        return None

    def all_paths_for_prior(self, prior: Prior) -> Generator[Path, None, None]:
        """
        Find all paths that points at the given tuple.

        Parameters
        ----------
        prior
            A prior representing what's known about some dimension of the model.
        Yields
        -------
        Paths, each a series of attributes that point to one location of the prior.
        """
        for path, path_prior in reversed(self.path_priors_tuples):
            if path_prior == prior:
                yield path

    @property
    def path_float_tuples(self):
        """(path, float) tuples for all fixed float values, excluding Prior objects."""
        return self.path_instance_tuples_for_class(float, ignore_class=Prior)

    @property
    def unique_prior_paths(self):
        """Paths to each unique prior (deduplicated for shared priors), ordered by id."""
        return [item[0] for item in self.unique_path_prior_tuples]

    @property
    def unique_path_prior_tuples(self):
        """(path, prior) tuples deduplicated by prior identity, ordered by id.

        When a prior is shared across multiple paths, only one path is kept.
        """
        unique = {item[1]: item for item in self.path_priors_tuples}.values()
        return sorted(unique, key=lambda item: item[1].id)

    @property
    def prior_prior_model_dict(self):
        """
        Returns
        -------
        prior_prior_model_dict: {Prior: Model}
            A dictionary mapping priors to associated prior models. Each prior will only
            have one prior model; if a prior is shared by two prior models then one of
            those prior models will be in this dictionary.
        """
        return {
            prior: prior_model[1]
            for prior_model in self.prior_model_tuples + [("model", self)]
            for _, prior in prior_model[1].prior_tuples
        }

    def log_prior_list_from(self, parameter_lists: List[List]) -> List:
        """Compute the total log prior for each parameter vector in a list.

        Parameters
        ----------
        parameter_lists
            A list of physical parameter vectors.

        Returns
        -------
        list
            The summed log prior for each vector.
        """
        return [
            sum(self.log_prior_list_from_vector(vector=vector))
            for vector in parameter_lists
        ]

    @property
    def info(self) -> str:
        """
        Use the priors that make up the model_mapper to generate information on each
        parameter of the overall model.
        This information is extracted from each priors *model_info* property.
        """
        formatter = TextFormatter(line_length=info_whitespace())

        for t in find_groups(
            [
                t
                for t in self.path_instance_tuples_for_class(
                    (Prior, float, Constant, int, tuple, ConfigException),
                    ignore_children=True,
                )
                if t[0][-1] not in ("id", "item_number")
            ],
            limit=1,
        ):
            if isinstance(t[1], ConfigException):
                t = (t[0], "Prior Missing: Enter Manually or Add to Config")

            formatter.add(*t)

        return "\n\n".join(
            [
                f"Total Free Parameters = {self.prior_count}",
                f"{self.parameterization}",
                formatter.text,
            ]
        )

    @property
    def order_no(self) -> str:
        """
        A string that can be used to order models by their
        parametrisation.

        Priors and constants are ordered by their paths and then
        joined into a string which means that models with higher
        associated values are consistently ordered later in a
        collection.
        """
        values = [
            str(float(value))
            for _, value in sorted(
                self.path_instance_tuples_for_class((Prior, float)), key=lambda t: t[0]
            )
        ]
        return ":".join(values)

    @property
    def parameterization(self) -> str:
        """
        Describes the path to each of the PriorModels, its class
        and its number of free parameters
        """
        from .prior_model import Model

        formatter = TextFormatter(line_length=info_whitespace())

        paths = []

        for t in self.path_instance_tuples_for_class(
            (
                Prior,
                float,
                Constant,
                tuple,
            ),
            ignore_children=True,
        ):
            for i in range(len(t[0])):
                path = t[0][:i]
                obj = self.object_for_path(path)
                if isinstance(obj, TuplePrior):
                    continue
                if isinstance(obj, AbstractPriorModel):
                    n = obj.prior_count
                else:
                    n = 0
                if isinstance(obj, Model):
                    name = obj.cls.__name__
                else:
                    name = type(obj).__name__

                paths.append((("model",) + path, f"{name} (N={n})"))

        for group in find_groups(paths, limit=0):
            formatter.add(*group)

        return formatter.text

    @property
    def all_paths(self) -> List[Tuple[Path]]:
        """
        All possible paths to all priors grouped such that all
        paths to the same prior are collected together in a tuple.
        """
        if self.prior_count == 0:
            return []
        paths, _ = zip(*self.all_paths_prior_tuples)
        return paths

    @property
    def all_paths_prior_tuples(self) -> List[Tuple[Tuple[Path], Prior]]:
        """
        Maps a tuple containing all paths to a given prior to that prior.
        """
        prior_paths_dict = defaultdict(tuple)
        for path, prior in self.path_priors_tuples:
            prior_paths_dict[prior] += (path,)
        return sorted(
            [(paths, prior) for prior, paths in prior_paths_dict.items()],
            key=lambda item: item[1].id,
        )

    @property
    def all_names(self) -> List[Tuple[str]]:
        """
        All possible names for all priors grouped such that all
        names for a given prior are collected together in a tuple.
        """
        if self.prior_count == 0:
            return []
        names, _ = zip(*self.all_name_prior_tuples)
        return names

    @property
    def all_name_prior_tuples(self) -> List[Tuple[Tuple[str], Prior]]:
        """
        Maps a tuple containing all names for a given prior to that prior.
        """
        path_modifier = TuplePathModifier(self)
        return [
            (tuple(".".join(path_modifier(path)) for path in paths), prior)
            for paths, prior in self.all_paths_prior_tuples
        ]

    @property
    def model_component_and_parameter_names(self) -> List[str]:
        """
        Lists each parameter's name and path, and is used for labeling visualization with parameter labels.

        The parameter names are determined from the class instance names of the model_mapper. Latex tags are properties
        of each model class.
        """
        prior_paths = self.unique_prior_paths

        tuple_filter = TuplePathModifier(self)

        prior_paths = list(map(tuple_filter, prior_paths))

        return [".".join(path) for path in prior_paths]

    @property
    def joined_paths(self) -> List[str]:
        """Dot-joined path strings for each unique prior, ordered by id."""
        prior_paths = self.unique_prior_paths

        return [".".join(path) for path in prior_paths]

    @property
    def parameter_names(self) -> List[str]:
        """
        Returns a list of labels containing the name of every parameter in a model.

        This is used for displaying model results as text and for visualization.

        The parameter labels are defined for every parameter of every model component in the config files label.ini and
        label_format.ini.
        """
        return [parameter_name[-1] for parameter_name in self.unique_prior_paths]

    @property
    def parameter_labels(self) -> List[str]:
        """
        Returns a list of labels containing latex labels of every parameter in a model.

        This is used for displaying model results as text and for visualization.

        The parameter labels are defined for every parameter of every model component in the config files label.ini and
        label_format.ini.
        """

        parameter_labels = []

        for parameter_name in self.parameter_names:
            parameter_label = frm.convert_name_to_label(
                parameter_name=parameter_name, name_to_label=True
            )
            parameter_labels.append(parameter_label)

        return parameter_labels

    @property
    def superscripts(self) -> List[str]:
        """
        Returns a list of the model component superscripts for every parameter in a model.


        The class superscript labels are defined as the name of every model component in the `ModelMapper`. For
        the example of a 1D Gaussian, if the model component name is `gaussian` three superscripts
        with this string (corresponding to the parameters `centre`, `normalization` and `sigma`) will
        be returned.

        For a `Collection`, the name of the inner model components are used.

        These superscripts may be overwritten by those returned from the `superscripts_config_overwrite` property,
        which optionally loads the superscripts from a `.json` config file. This allows high levels of customization
        in what superscripts are used.

        This is used for displaying model results as text and for visualization.
        """

        prior_paths = self.unique_prior_paths

        tuple_filter = TuplePathModifier(self)

        prior_paths = map(tuple_filter, prior_paths)

        superscripts = [path[-2] if len(path) > 1 else None for path in prior_paths]

        return [
            superscript if not superscript_overwrite else superscript_overwrite
            for superscript, superscript_overwrite in zip(
                superscripts, self.superscripts_overwrite_via_config
            )
        ]

    @property
    def superscripts_overwrite_via_config(self) -> List[str]:
        """
        Returns a list of the model component superscripts for every parameter in a model, which can be used to
        overwrite the default superscripts used in the function above.

        The class superscript labels are defined for a model component in the config file `notation/label.ini`. By
        default, the model component names are used as superscripts (which are loaded via the method `superscripts`).
        These are overwritten by the superscripts loaded via a  config in this function. If no value is present in the
        config the model component names are used.

        For the example of a 1D Gaussian, when instatiated as a model component it is typically given the
        name `gaussian`. Thus, the string `gaussian` will be used as the supersript of every one its parameter labels
        (`centre`, `normalization` and `sigma`). However, if the config file `label.ini` reads `Gaussian=g`, every
        superscript for these parameters will instead be given the superscript `g`.

        This is used for displaying model results as text and for visualization with.
        """

        superscripts = []

        for prior_name, prior in self.prior_tuples_ordered_by_id:
            cls = self.prior_class_dict[prior]
            try:
                superscript = conf.instance["notation"]["label"]["superscript"][
                    cls.__name__
                ]

            except KeyError:
                superscript = ""

            superscripts.append(superscript)

        return superscripts

    @property
    def parameter_labels_with_superscripts(self) -> List[str]:
        """
        Returns a list of the latex parameter label and superscript of every parameter in a model.

        The parameter labels are defined for every parameter of every model component in the config file `label.ini`.
        This file can also be used to overwrite superscripts, that are assigned based on the model component name.

        This is used for displaying model results as text and for visualization, for example labelling parameters.
        """

        return [
            f"{label}^{{\\rm {superscript}}}" if superscript else f"{label}"
            for label, superscript in zip(self.parameter_labels, self.superscripts)
        ]

    @property
    def parameter_labels_with_superscripts_latex(self) -> List[str]:
        """
        Returns a list of the latex parameter label and superscript of every parameter in a model.

        The parameter labels are defined for every parameter of every model component in the config file `label.ini`.
        This file can also be used to overwrite superscripts, that are assigned based on the model component name.

        This is used for displaying model results as text and for visualization, for example labelling parameters.
        """

        return [f"${label}$" for label in self.parameter_labels_with_superscripts]


def transfer_classes(instance, mapper, model_classes=None):
    """
    Recursively overwrite priors in the mapper with instance values from the
    instance except where the containing class is the descendant of a listed class.
    Parameters
    ----------
    model_classes
        Classes whose descendants should not be overwritten
    instance
        The best fit from the previous search
    mapper
        The prior model from the previous search
    """
    from autofit.mapper.prior_model.annotation import AnnotationPriorModel

    model_classes = model_classes or []
    for key, instance_value in instance.__dict__.items():
        try:
            mapper_value = getattr(mapper, key)
            if isinstance(mapper_value, Prior) or isinstance(
                mapper_value, AnnotationPriorModel
            ):
                setattr(mapper, key, instance_value)
                continue
            if not any(isinstance(instance_value, cls) for cls in model_classes):
                try:
                    transfer_classes(instance_value, mapper_value, model_classes)
                except AttributeError:
                    setattr(mapper, key, instance_value)
        except AttributeError:
            pass
