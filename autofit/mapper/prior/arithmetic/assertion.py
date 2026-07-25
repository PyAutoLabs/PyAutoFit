from abc import ABC
import numpy as np
from typing import Optional, Dict

from autofit.mapper.prior.arithmetic.compound import CompoundPrior, Compound
from autofit.mapper.prior_model.abstract import AbstractPriorModel


class ComparisonAssertion(CompoundPrior, Compound, ABC):
    def __init__(self, lower, greater, name=""):
        super().__init__(lower, greater)
        self._name = name

    def __gt__(self, other):
        return CompoundAssertion(self, self._left > other)

    def __lt__(self, other):
        return CompoundAssertion(self, self._right < other)

    def __ge__(self, other):
        return CompoundAssertion(self, self._left >= other)

    def __le__(self, other):
        return CompoundAssertion(self, self._right <= other)


class GreaterThanLessThanAssertion(ComparisonAssertion):
    def _instance_for_arguments(self, arguments, ignore_assertions=False, xp=np):
        """
        Assert that the value in the dictionary associated with the lower
        prior is lower than the value associated with the greater prior.

        Parameters
        ----------
        arguments
            A dictionary mapping priors to physical values.

        Raises
        ------
        FitException
            If the assertion is not met
        """
        lower = self.left_for_arguments(
            arguments,
            ignore_assertions=ignore_assertions,
        )
        greater = self.right_for_arguments(
            arguments,
            ignore_assertions=ignore_assertions,
        )
        return lower < greater


class GreaterThanLessThanEqualAssertion(ComparisonAssertion):
    def _instance_for_arguments(
        self,
        arguments,
        ignore_assertions=False,
        xp=np,
    ):
        """
        Assert that the value in the dictionary associated with the lower
        prior is lower than the value associated with the greater prior.

        Parameters
        ----------
        arguments
            A dictionary mapping priors to physical values.

        Raises
        ------
        FitException
            If the assertion is not met
        """
        return self.left_for_arguments(
            arguments,
            ignore_assertions=ignore_assertions,
        ) <= self.right_for_arguments(
            arguments,
            ignore_assertions=ignore_assertions,
        )


class CompoundAssertion(AbstractPriorModel, Compound):
    def __init__(self, assertion_1, assertion_2, name=""):
        super().__init__()
        self.assertion_1 = assertion_1
        self.assertion_2 = assertion_2
        self._name = name

    def _instance_for_arguments(
        self,
        arguments,
        ignore_assertions=False,
        xp=np,
    ):
        return self.assertion_1.instance_for_arguments(
            arguments,
            ignore_assertions,
        ) and self.assertion_2.instance_for_arguments(
            arguments,
            ignore_assertions,
        )

    def dict(self) -> dict:
        return {
            "type": "compound",
            "compound_type": self.__class__.__name__,
            "assertion_1": self.assertion_1.dict(),
            "assertion_2": self.assertion_2.dict(),
        }

    @classmethod
    def from_dict(
        cls,
        d,
        reference: Optional[Dict[str, str]] = None,
        loaded_ids: Optional[dict] = None,
    ):
        return cls(
            Compound.from_dict(d["assertion_1"], reference, loaded_ids),
            Compound.from_dict(d["assertion_2"], reference, loaded_ids),
        )


def unwrap(obj):
    try:
        return obj._value
    except AttributeError:
        return obj
