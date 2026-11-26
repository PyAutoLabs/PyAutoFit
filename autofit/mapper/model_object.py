import copy
import itertools
from typing import Type, Union, Tuple, Optional, Dict
import logging

from autonerves.class_path import get_class
from autonerves.dictable import from_dict, to_dict
from .identifier import Identifier

logger = logging.getLogger(__name__)


def dereference(reference: Optional[dict], name: str):
    if reference is None:
        return None
    updated = {}
    for key, value in reference.items():
        array = key.split(".")
        if array[0] == name:
            updated[".".join(array[1:])] = value
    return updated


def _restore_prior_order(loaded_ids: dict):
    """
    Give the priors created while parsing a dictionary the relative order of
    the ids they were saved with.

    ``loaded_ids`` maps each saved prior id to the prior created for it. The
    new priors keep the set of ids they were given, so no id collides with a
    prior outside the loaded model, but those ids are handed out again in the
    order of the saved ids.

    Parameters
    ----------
    loaded_ids
        A dictionary mapping saved ids to the objects created for them.
    """
    from autofit.mapper.prior.abstract import Prior
    from autofit.mapper.prior.constant import Constant

    saved = {}
    for saved_id, obj in loaded_ids.items():
        if (
            isinstance(obj, Prior)
            and not isinstance(obj, Constant)
            and isinstance(saved_id, int)
        ):
            saved.setdefault(id(obj), (saved_id, obj))

    if len(saved) < 2:
        return

    by_saved_id = sorted(saved.values(), key=lambda item: item[0])
    new_ids = sorted(prior.id for _, prior in by_saved_id)
    for (_, prior), new_id in zip(by_saved_id, new_ids):
        prior.id = new_id
        prior.message.id_ = new_id


class ModelObject:
    _ids = itertools.count()

    @classmethod
    def next_id(cls):
        return next(cls._ids)

    def __init__(
        self,
        id_=None,
        label=None,
    ):
        """
        A generic object in AutoFit

        Parameters
        ----------
        id_
            A unique integer identifier. This is used to hash and order priors.
        label
            A label which can optionally be set for visualising this object in a
            graph.
        """
        self.id = int(self.next_id() if id_ is None else id_)
        self._label = label

    def replacing_for_path(self, path: Tuple[str, ...], value) -> "ModelObject":
        """
        Create a new model replacing the value for a given path with a new value

        Parameters
        ----------
        path
            A path indicating the sequence of names used to address an object
        value
            A value that should replace the object at the given path

        Returns
        -------
        A copy of this with an updated value
        """
        new = copy.deepcopy(self)
        obj = new
        for key in path[:-1]:
            if isinstance(key, int):
                obj = obj[key]
            else:
                obj = getattr(obj, key)

        key = path[-1]
        if isinstance(key, int):
            obj[key] = value
        else:
            setattr(obj, key, value)
        return new

    def has(self, cls: Union[Type, Tuple[Type, ...]]) -> bool:
        """
        Does this instance have an attribute which is of type cls?
        """
        for value in self.__dict__.values():
            if isinstance(value, cls):
                return True
        return False

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def component_number(self):
        return self.id

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        try:
            return self.id == other.id
        except AttributeError:
            return False

    @property
    def identifier(self):
        return str(Identifier(self))

    @classmethod
    def from_dict(
        cls,
        d,
        reference: Optional[Dict[str, str]] = None,
        loaded_ids: Optional[dict] = None,
    ):
        """
        Recursively parse a dictionary returning the model, collection or
        instance that is represents.

        A model's free parameters are ordered by prior id, which is creation
        order. The priors made while parsing are created in the order the
        dictionary is walked, which differs from the original order whenever a
        prior was replaced after the model was built. The outermost call
        therefore gives the new priors their original relative order once
        parsing is done (see ``_restore_prior_order``), so a parameter vector
        maps to the same parameters on the reloaded model as on the original.

        Parameters
        ----------
        d
            A dictionary representation of some object
        reference
            An optional dictionary mapping names to class paths. See
            ``_from_dict``.
        loaded_ids
            A dictionary mapping ids to instances. Callers leave it out: it is
            passed down the recursion so that all instances with the same id
            are the same object.

        Returns
        -------
        An instance
        """
        if loaded_ids is not None:
            return cls._from_dict(d, reference=reference, loaded_ids=loaded_ids)

        loaded_ids = {}
        instance = cls._from_dict(d, reference=reference, loaded_ids=loaded_ids)
        _restore_prior_order(loaded_ids)
        return instance

    @classmethod
    def _from_dict(
        cls,
        d,
        reference: Optional[Dict[str, str]] = None,
        loaded_ids: Optional[dict] = None,
    ):
        """
        Recursively parse a dictionary returning the model, collection or
        instance that is represents.

        Parameters
        ----------
        d
            A dictionary representation of some object
        reference
            An optional dictionary mapping names to class paths. This is used
            to specify the type of a model or instance.

            Maps paths to class paths. For example:
            "path.in.model": "path.to.Class"

            In this case, the class path "path.to.Class" will be used to
            instantiate the object at "path.in.model". If no class path is
            specified, or no type can be found for the class path in 'd', then
            a Collection will be used as a placeholder.

            This is used to specify the type of a model or instance.
        loaded_ids
            A dictionary mapping ids to instances. This is used to ensure that
            all instances with the same id are the same object.

        Returns
        -------
        An instance
        """
        from autofit.mapper.prior_model.collection import Collection
        from autofit.mapper.prior_model.prior_model import Model
        from autofit.mapper.prior.abstract import Prior
        from autofit.mapper.prior.tuple_prior import TuplePrior
        from autofit.mapper.prior.arithmetic.compound import Compound
        from autofit.mapper.prior.arithmetic.compound import ModifiedPrior
        from .prior.constant import Constant

        if isinstance(d, list):
            return [
                from_dict(
                    value,
                    reference=dereference(reference, str(index)),
                    loaded_ids=loaded_ids,
                )
                for index, value in enumerate(d)
            ]

        if not isinstance(d, dict):
            return d

        loaded_ids = {} if loaded_ids is None else loaded_ids

        type_ = d["type"]

        def get_class_path():
            try:
                return reference[""]
            except (KeyError, TypeError):
                return d.pop("class_path")

        if type_ == "model":
            class_path = get_class_path()
            try:
                instance = Model(get_class(class_path))
            except (ModuleNotFoundError, AttributeError):
                logger.warning(
                    f"Could not find type for class path {class_path}. Defaulting to Collection placeholder."
                )
                instance = Collection()
        elif type == "constant":
            return Constant(value=d["value"])
        elif type_ == "collection":
            instance = Collection()
        elif type_ == "tuple_prior":
            instance = TuplePrior()
        elif type_ == "compound":
            return Compound.from_dict(
                d,
                reference=dereference(reference, "assertion"),
                loaded_ids=loaded_ids,
            )
        elif type_ == "modified":
            return ModifiedPrior.from_dict(
                d,
                reference=dereference(reference, "prior"),
                loaded_ids=loaded_ids,
            )
        elif type_ == "dict":
            return {
                key: from_dict(
                    value,
                    reference=dereference(reference, key),
                    loaded_ids=loaded_ids,
                )
                for key, value in d["arguments"].items()
                if value is not None
            }
        elif type_ == "instance":
            class_path = get_class_path()
            try:
                cls_ = get_class(class_path)
                # noinspection PyArgumentList
                return cls_(
                    **{
                        key: from_dict(
                            value,
                            reference=dereference(reference, key),
                            loaded_ids=loaded_ids,
                        )
                        for key, value in d["arguments"].items()
                    }
                )
            except (ModuleNotFoundError, AttributeError):
                from autofit.mapper.model import ModelInstance

                logger.warning(
                    f"Could not find type for class path {class_path}. Defaulting to Instance placeholder."
                )
                instance = ModelInstance()
        elif type_ == "array":
            from autofit.mapper.prior_model.array import Array

            return Array.from_dict(d)
        else:
            try:
                return Prior.from_dict(d, loaded_ids=loaded_ids)
            except KeyError:
                cls_ = get_class(type_)
                instance = object.__new__(cls_)

        for key, value in d["arguments"].items():
            try:
                setattr(
                    instance,
                    key,
                    from_dict(
                        value,
                        reference=dereference(reference, key),
                        loaded_ids=loaded_ids,
                    ),
                )
            except KeyError:
                pass

        if "assertions" in d:
            instance.assertions = [
                from_dict(
                    value,
                    reference=dereference(reference, "assertions"),
                    loaded_ids=loaded_ids,
                )
                for value in d["assertions"]
            ]

        return instance

    def dict(self) -> dict:
        """
        A dictionary representation of this object
        """
        from autofit.mapper.prior_model.abstract import AbstractPriorModel
        from autofit.mapper.prior_model.collection import Collection
        from autofit.mapper.prior_model.prior_model import Model
        from autofit.mapper.prior.tuple_prior import TuplePrior
        from autofit.mapper.prior_model.array import Array
        from autofit.mapper.prior.constant import Constant

        if isinstance(self, Collection):
            type_ = "collection"
        elif (
            isinstance(self, Model)
            and self.__dict__.get("_serialize_as_instance", False)
            and self.prior_count == 0
        ):
            type_ = "instance"
        elif isinstance(self, Model):
            type_ = "model"
        elif isinstance(self, AbstractPriorModel) and self.prior_count == 0:
            type_ = "instance"
        elif isinstance(self, TuplePrior):
            type_ = "tuple_prior"
        elif isinstance(self, Array):
            type_ = "array"
        elif isinstance(self, Constant):
            type_ = "constant"
        else:
            raise AssertionError(
                f"{self.__class__.__name__} cannot be serialised to dict"
            )

        try:
            assertions = [assertion.dict() for assertion in self._assertions]
        except AttributeError:
            assertions = []

        dict_ = {
            "type": type_,
        }

        if assertions:
            dict_["assertions"] = assertions

        arguments = {}

        for key, value in self._dict.items():
            try:
                value = to_dict(value)
            except AttributeError:
                pass
            except TypeError:
                pass
            arguments[key] = value

        dict_["arguments"] = arguments
        return dict_

    @property
    def _dict(self):
        # Pick up any cached_property descriptors declared on the class so
        # their cached values don't propagate via `Collection.items()` (which
        # delegates here) or any other downstream consumer. The lookup is
        # gated on hasattr because ModelObject is the base for the whole
        # mapper module: a few non-AbstractModel descendants do not carry the
        # ``_cached_property_names`` classmethod.
        try:
            excluded = type(self)._cached_property_names()
        except AttributeError:
            excluded = frozenset()
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("component_number", "item_number", "id", "cls", "label")
            and not key.startswith("_")
            and key not in excluded
        }
