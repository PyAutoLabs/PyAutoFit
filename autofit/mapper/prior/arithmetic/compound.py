import inspect
import logging
from abc import ABC
from copy import copy
from typing import Optional, Dict

import numpy as np

from autofit.mapper.model_object import dereference
from autofit.mapper.prior.arithmetic import ArithmeticMixin
from autofit.mapper.prior_model.abstract import AbstractPriorModel

logger = logging.getLogger(__name__)


def retrieve_name(var):
    first_name = None
    frame = inspect.currentframe()
    while frame is not None:
        for name, value in list(frame.f_locals.items()):
            if var is value:
                first_name = name
        frame = frame.f_back

    return first_name


class Compound:
    @classmethod
    def from_dict(
        cls,
        d,
        reference: Optional[Dict[str, str]] = None,
        loaded_ids: Optional[dict] = None,
    ):
        assertion_type = d.pop("compound_type")
        for subclass in cls.descendants():
            if subclass.__name__ == assertion_type:
                return subclass.from_dict(
                    d,
                    reference=reference,
                    loaded_ids=loaded_ids,
                )
        raise ValueError(f"Compound type {assertion_type} not recognised")

    @classmethod
    def descendants(cls):
        subclasses = cls.__subclasses__()

        for child in subclasses:
            yield child
            yield from child.descendants()


class CompoundPrior(AbstractPriorModel, ArithmeticMixin, Compound, ABC):
    cls = float

    def __init__(self, left, right):
        """
        Comprises objects that are to undergo some arithmetic
        operation after realisation.

        Parameters
        ----------
        left
            A prior, promise or float
        right
            A prior, promise or float
        """
        super().__init__()

        self._left_name = retrieve_name(left) or "left"
        self._right_name = retrieve_name(right) or "right"

        if self._left_name == "left":
            self._left_name = "left_"

        if self._right_name == "right":
            self._right_name = "right_"

        self._left = None
        self._right = None

        self.left = left
        self.right = right

    def __str__(self):
        return f"{type(self).__name__}(left={self.left!r}, right={self.right!r})"

    def __repr__(self):
        return str(self)

    def dict(self) -> dict:
        from autofit import ModelObject

        return {
            "type": "compound",
            "compound_type": self.__class__.__name__,
            "left": self._left.dict()
            if isinstance(self._left, ModelObject)
            else self._left,
            "right": self._right.dict()
            if isinstance(self._right, ModelObject)
            else self._right,
            "left_name": self._left_name,
            "right_name": self._right_name,
        }

    @classmethod
    def from_dict(
        cls,
        d,
        reference: Optional[Dict[str, str]] = None,
        loaded_ids: Optional[dict] = None,
    ):
        from autofit import ModelObject

        compound = cls(
            ModelObject.from_dict(d["left"], reference, loaded_ids),
            ModelObject.from_dict(d["right"], reference, loaded_ids),
        )
        compound._set_operand_names(d.get("left_name"), d.get("right_name"))
        return compound

    def _set_operand_names(
        self, left_name: Optional[str], right_name: Optional[str]
    ):
        """
        Restore the attribute names the operands were stored under when the
        compound was serialised.

        The names are found by inspecting the caller's frames when the
        compound is first built, so a compound rebuilt by ``from_dict`` would
        otherwise get different names. They are part of the model identifier,
        so a reloaded model must keep them. Dictionaries written before the
        names were serialised carry none and keep the rebuilt names.
        """
        left_name = left_name or self._left_name
        right_name = right_name or self._right_name

        for name in {self._left_name, self._right_name} - {left_name, right_name}:
            self.__dict__.pop(name, None)

        self._left_name = left_name
        self._right_name = right_name
        setattr(self, left_name, self._left)
        setattr(self, right_name, self._right)

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @left.setter
    def left(self, left):
        self._left = left
        setattr(self, self._left_name, left)

    @right.setter
    def right(self, right):
        self._right = right
        setattr(self, self._right_name, right)

    def gaussian_prior_model_for_arguments(self, arguments):
        new = copy(self)
        try:
            new.left = new.left.gaussian_prior_model_for_arguments(arguments)
        except AttributeError:
            pass
        try:
            new.right = new.right.gaussian_prior_model_for_arguments(arguments)
        except AttributeError:
            pass
        return new

    def left_for_arguments(
        self,
        arguments: dict,
        ignore_assertions=False,
        xp=np,
    ):
        """
        Instantiate the left object.

        Parameters
        ----------
        arguments
            A dictionary mapping priors to values
        ignore_assertions
            If True, ignore assertions

        Returns
        -------
        A value for the left object
        """
        try:
            return self._left.instance_for_arguments(
                arguments,
                ignore_assertions=ignore_assertions,
                xp=xp,
            )
        except AttributeError:
            return self._left

    def right_for_arguments(
        self,
        arguments: dict,
        ignore_assertions=False,
        xp=np,
    ):
        """
        Instantiate the right object.

        Parameters
        ----------
        arguments
            A dictionary mapping priors to values
        ignore_assertions
            If True, ignore assertions

        Returns
        -------
        A value for the right object
        """
        try:
            return self._right.instance_for_arguments(
                arguments,
                ignore_assertions=ignore_assertions,
                xp=xp,
            )
        except AttributeError:
            return self._right

    def __add__(self, other):
        return ArithmeticMixin.__add__(self, other)


class SumPrior(CompoundPrior):
    """
    The sum of two objects, computed after realisation.
    """

    def _instance_for_arguments(
        self,
        arguments,
        ignore_assertions=False,
        xp=np,
    ):
        return self.left_for_arguments(
            arguments,
            ignore_assertions=ignore_assertions,
            xp=xp,
        ) + self.right_for_arguments(
            arguments,
            ignore_assertions=ignore_assertions,
            xp=xp,
        )

    def __str__(self):
        return f"{self._left} + {self._right}"


class MultiplePrior(CompoundPrior):
    """
    The multiple of two objects, computed after realisation.
    """

    def __str__(self):
        return f"{self._left} * {self._right}"

    def _instance_for_arguments(
        self,
        arguments,
        ignore_assertions=False,
        xp=np,
    ):
        return self.left_for_arguments(
            arguments,
            ignore_assertions=ignore_assertions,
            xp=xp,
        ) * self.right_for_arguments(
            arguments,
            ignore_assertions=ignore_assertions,
            xp=xp,
        )


class DivisionPrior(CompoundPrior):
    """
    One object divided by another, computed after realisation
    """

    def _instance_for_arguments(
        self,
        arguments,
        ignore_assertions=False,
        xp=np,
    ):
        return self.left_for_arguments(
            arguments,
            ignore_assertions=ignore_assertions,
            xp=xp,
        ) / self.right_for_arguments(
            arguments,
            ignore_assertions=ignore_assertions,
            xp=xp,
        )


class FloorDivPrior(CompoundPrior):
    """
    One object divided by another and floored, computed after realisation.
    """

    def _instance_for_arguments(
        self,
        arguments,
        ignore_assertions=False,
        xp=np,
    ):
        return self.left_for_arguments(
            arguments,
            ignore_assertions=ignore_assertions,
            xp=xp,
        ) // self.right_for_arguments(
            arguments,
            ignore_assertions=ignore_assertions,
            xp=xp,
        )


class ModPrior(CompoundPrior):
    """
    The modulus of a pair of objects, computed after realisation.
    """

    def _instance_for_arguments(
        self,
        arguments,
        ignore_assertions=False,
        xp=np,
    ):
        return self.left_for_arguments(
            arguments,
            ignore_assertions=ignore_assertions,
            xp=xp,
        ) % self.right_for_arguments(
            arguments,
            ignore_assertions=ignore_assertions,
            xp=xp,
        )


class PowerPrior(CompoundPrior):
    """
    One object to the power of another, computed after realisation.
    """

    def _instance_for_arguments(
        self,
        arguments,
        ignore_assertions=False,
        xp=np,
    ):
        return self.left_for_arguments(
            arguments,
            ignore_assertions=ignore_assertions,
            xp=xp,
        ) ** self.right_for_arguments(
            arguments,
            ignore_assertions=ignore_assertions,
            xp=xp,
        )


class ModifiedPrior(AbstractPriorModel, ABC, ArithmeticMixin, Compound):
    def __init__(self, prior, name=None):
        super().__init__()
        self._prior_name = name or retrieve_name(prior)

        if self._prior_name == "prior":
            self._prior_name = "prior_"

        self.prior = prior

    def dict(self):
        from autofit import ModelObject

        return {
            "type": "modified",
            "modified_type": self.__class__.__name__,
            "name": self._prior_name,
            "prior": self.prior.dict()
            if isinstance(self.prior, ModelObject)
            else self.prior,
        }

    @classmethod
    def from_dict(
        cls,
        d,
        reference: Optional[Dict[str, str]] = None,
        loaded_ids: Optional[dict] = None,
    ):
        modified_type = d.pop("modified_type")
        for subclass in cls.descendants():
            if subclass.__name__ == modified_type:
                return subclass(
                    AbstractPriorModel.from_dict(
                        d["prior"],
                        reference=dereference(reference, "prior"),
                        loaded_ids=loaded_ids,
                    ),
                    name=d["name"],
                )
        raise ValueError(f"Modified type {modified_type} not recognised")

    def __add__(self, other):
        return ArithmeticMixin.__add__(self, other)

    @property
    def cls(self):
        if isinstance(self.prior, AbstractPriorModel):
            return self.prior.cls
        return float

    @property
    def prior(self):
        return getattr(self, self._prior_name)

    @prior.setter
    def prior(self, prior):
        setattr(self, self._prior_name, prior)

    def gaussian_prior_model_for_arguments(self, arguments):
        new = copy(self)
        try:
            new.prior = new.prior.gaussian_prior_model_for_arguments(arguments)
        except AttributeError:
            pass
        return new


class NegativePrior(ModifiedPrior):
    """
    The negation of an object, computed after realisation.
    """

    def _instance_for_arguments(
        self,
        arguments,
        ignore_assertions=False,
        xp=np,
    ):
        return -self.prior.instance_for_arguments(
            arguments,
            ignore_assertions=ignore_assertions,
            xp=xp,
        )


class AbsolutePrior(ModifiedPrior):
    """
    The absolute value of an object, computed after realisation.
    """

    def _instance_for_arguments(
        self,
        arguments,
        ignore_assertions=False,
        xp=np,
    ):
        return abs(
            self.prior.instance_for_arguments(
                arguments,
                ignore_assertions=ignore_assertions,
                xp=xp,
            )
        )


class Log(ModifiedPrior):
    """
    The natural logarithm of an object, computed after realisation.
    """

    def _instance_for_arguments(
        self,
        arguments,
        ignore_assertions=False,
        xp=np,
    ):
        return xp.log(
            self.prior.instance_for_arguments(
                arguments,
                ignore_assertions=ignore_assertions,
                xp=xp,
            )
        )


class Log10(ModifiedPrior):
    """
    The base10 logarithm of an object, computed after realisation.
    """

    def _instance_for_arguments(
        self,
        arguments,
        ignore_assertions=False,
        xp=np,
    ):
        return xp.log10(
            self.prior.instance_for_arguments(
                arguments,
                ignore_assertions=ignore_assertions,
                xp=xp,
            )
        )
