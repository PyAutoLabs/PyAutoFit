import inspect
from collections.abc import Iterable
from hashlib import md5
from typing import Optional

from autonerves.class_path import get_class_path

# floats are rounded to this increment so floating point errors
# have no impact on identifier value
RESOLUTION = 1e-8


class IdentifierField:
    """
    A field that the identifier depends on.

    If the value of the field is changed then the identifier
    must be recomputed prior to use.
    """

    def __set_name__(self, owner: object, name: str):
        """
        Called on instantiation

        Parameters
        ----------
        owner
            The object for which this field is an attribute
        name
            The name of the attribute
        """
        self.private = f"_{name}"
        setattr(owner, self.private, None)

    def __get__(self, obj: object, objtype=None) -> Optional:
        """
        Retrieve the value of this field.

        Parameters
        ----------
        obj
            The object for which this field is an attribute
        objtype

        Returns
        -------
        The value (or None if it has not been set)
        """
        return getattr(obj, self.private)

    def __set__(self, obj, value):
        """
        Set a value for this field

        Parameters
        ----------
        obj
            The object for which the field is an attribute
        value
            A new value for the attribute
        """
        obj._identifier = None
        setattr(obj, self.private, value)


class Identifier:
    def __init__(self, obj):
        """
        Wraps an object and recursively generates an identifier

        The version can be set in general.ini (output/identifier_version). It can
        be overridden by specifying it explicitly in the constructor.
        """
        self.hash_list = list()
        self._add_value_to_hash_list(obj)

    @property
    def description(self):
        return "\n".join(self.hash_list)

    def _add_value_to_hash_list(self, value):
        """
        Add some object and recursively add its children to the hash_list.

        The md5 hash of this object is taken to create an identifier.

        If an object specifies __identifier_fields__ then only attributes
        with a name in this list are included.

        Parameters
        ----------
        value
            An object
        """
        from .model_object import ModelObject
        from autofit.mapper.prior.constant import Constant

        if isinstance(value, Constant):
            self.add_value_to_hash_list(value.value)
            return

        if inspect.isclass(value):
            self.add_value_to_hash_list(get_class_path(value))
            return

        if isinstance(value, Exception):
            raise value
        if hasattr(value, "__dict__"):
            if hasattr(value, "__class__"):
                self.add_value_to_hash_list(value.__class__.__name__)
            d = value.__dict__

            if hasattr(value, "__identifier_fields__"):
                fields = value.__identifier_fields__

                try:
                    d = {k: getattr(value, k) for k in fields}
                except AttributeError as e:
                    raise AssertionError(
                        f"Missing identifier fields for {type(value)}"
                    ) from e
            elif (
                hasattr(value, "__class__")
                and not inspect.isclass(value)
                and not isinstance(value, ModelObject)
            ):
                args = inspect.getfullargspec(value.__class__).args
                d = {k: v for k, v in d.items() if k in args}
                if hasattr(value, "__exclude_identifier_fields__"):
                    excluded_fields = value.__exclude_identifier_fields__

                    d = {k: v for k, v in d.items() if k not in excluded_fields}
            self.add_value_to_hash_list(d)
        elif isinstance(value, dict):
            for key, value in value.items():
                key = str(key)
                if not (key.startswith("_") or key in ("id", "paths")):
                    self.hash_list.append(key)
                    self.add_value_to_hash_list(value)
        elif isinstance(value, float):
            try:
                value = RESOLUTION * round(value / RESOLUTION)
            except OverflowError:
                pass

            self.hash_list.append(str(value))
        elif isinstance(value, (str, int, bool)):
            self.hash_list.append(str(value))
        elif isinstance(value, Iterable):
            for value in value:
                self.add_value_to_hash_list(value)

    def add_value_to_hash_list(self, value):
        if isinstance(value, property):
            return
        self._add_value_to_hash_list(value)

    def __str__(self):
        return md5(".".join(self.hash_list).encode("utf-8")).hexdigest()

    def __repr__(self):
        return f"<{self.__class__.__name__} {self}>"

    def __eq__(self, other):
        return str(self) == str(other)
