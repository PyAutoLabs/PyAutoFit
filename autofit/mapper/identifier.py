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


def _assertion_operands(assertion) -> list:
    """
    The operands of an assertion, in order.

    A `CompoundAssertion` joins two assertions as `assertion_1` and `assertion_2`. Every
    other assertion is a `CompoundPrior`, which holds its two sides in `_left` and
    `_right`; its public aliases are named after the caller's local variables, so they
    are not a stable component of an identifier.

    Parameters
    ----------
    assertion
        An assertion attached to a model.

    Returns
    -------
    The operands of the assertion, in order.
    """
    if hasattr(assertion, "assertion_1"):
        return [assertion.assertion_1, assertion.assertion_2]
    if hasattr(assertion, "_left"):
        return [assertion._left, assertion._right]
    return []


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

                # The search-level seed (search-extensibility A4, D4) is hashed only
                # when not None, and only for a search whose fields do not already
                # name it (Nautilus and NSS hash it statically).
                identifier_seed = getattr(type(value), "_identifier_seed", None)
                if identifier_seed is not None and "seed" not in fields:
                    seed = identifier_seed(value)
                    if seed is not None:
                        d["seed"] = seed
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
            self._add_assertions_to_hash_list(value)
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
            if getattr(value, "ndim", None) == 0:
                # A 0-d NumPy array is Iterable by type but cannot be iterated:
                # hash the scalar it holds, so it matches the equivalent Python
                # value. NumPy scalars such as np.float64 are not Iterable and JAX
                # arrays take the __dict__ branch above, so neither reaches here.
                self.add_value_to_hash_list(value.item())
                return
            for value in value:
                self.add_value_to_hash_list(value)

    def _add_assertions_to_hash_list(self, model):
        """
        Add the assertions attached to a model to the hash_list.

        Assertions live in the underscore prefixed `_assertions` attribute, which the
        attribute walk above skips. Without this a model carrying an ordering assertion
        hashes identically to the same model without it, so an ordered fit is written to
        the unordered fit's output directory and loads its completed result instead of
        running.

        Nothing at all is added for a model with no assertions, so the identifier of
        every model which does not use them is unchanged.

        Parameters
        ----------
        model
            An object which may be an `AbstractPriorModel` carrying assertions.
        """
        assertions = getattr(model, "_assertions", None)
        if not assertions:
            return

        self.hash_list.append("assertions")

        prior_paths = {
            id(prior): ".".join(path) for path, prior in model.path_priors_tuples
        }

        for assertion in assertions:
            self._add_assertion_to_hash_list(assertion, prior_paths)

    def _add_assertion_to_hash_list(self, assertion, prior_paths):
        """
        Add a single assertion to the hash_list as its class name followed by its
        operands in order, so that `a < b` and `b < a` hash differently.

        An operand which is a prior of the model is identified by its path (e.g.
        "gaussian.sigma") rather than by its value, because two priors with identical
        limits hash identically and a value based hash could not tell the two directions
        apart. An operand which is itself compound (a chained assertion, or arithmetic on
        priors) is recursed into for the same reason: its own operands are underscore
        prefixed and would otherwise be skipped.

        Parameters
        ----------
        assertion
            An assertion attached to a model.
        prior_paths
            A map from the id of each prior in the model to its path in that model.
        """
        from autofit.mapper.prior.arithmetic.compound import Compound

        self.hash_list.append(type(assertion).__name__)

        for operand in _assertion_operands(assertion):
            if id(operand) in prior_paths:
                self.hash_list.append(prior_paths[id(operand)])
            elif isinstance(operand, Compound):
                self._add_assertion_to_hash_list(operand, prior_paths)
            else:
                self.add_value_to_hash_list(operand)

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
