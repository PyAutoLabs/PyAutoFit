from abc import ABC, abstractmethod
from typing import List, Iterator

from .search_output import SearchOutput


class AttributePredicate:
    def __init__(self, *path):
        """
        Used to produce predicate objects for filtering in the aggregator.

        When an unrecognised attribute is called on an aggregator an instance
        of this object is created. This object implements comparison methods
        facilitating construction of predicates.

        Parameters
        ----------
        path
            A series of names of attributes that can be used to get a value.
            For example, (mask, pixel_size) would get the pixel size of a mask
            when evaluated for a given search.
        """
        self.path = path

    def value_for_search_output(
            self,
            search_output: SearchOutput
    ):
        """
        Recurse the search output by iterating the attributes in the path
        and getting a value for each attribute.
        """
        value = search_output
        for attribute in self.path:
            value = getattr(
                value,
                attribute
            )
        return value

    def __eq__(self, value):
        """
        Returns a predicate which asks whether the given value is equal to
        the attribute of a search.
        """
        return EqualityPredicate(
            self,
            value
        )

    def __le__(self, other):
        return OrPredicate(
            self == other,
            self < other
        )

    def __ge__(self, other):
        return OrPredicate(
            self == other,
            self > other
        )

    def __getattr__(self, item: str) -> "AttributePredicate":
        """
        Adds another item to the path
        """
        return AttributePredicate(
            *self.path, item
        )

    def __gt__(self, other):
        """
        Is the value of this attribute for a given search greater than some
        other value?
        """
        return GreaterThanPredicate(
            self, other
        )

    def __lt__(self, other):
        """
        Is the value of this attribute for a given search less than some
        other value?
        """
        return LessThanPredicate(
            self, other
        )

    def __ne__(self, other):
        """
        Returns a predicate which asks whether the given value is not equal to
        the attribute of a search.
        """
        return ~(self == other)

    def contains(self, value):
        """
        Returns a predicate which asks whether the given is contained within
        the attribute of a search.
        """
        return ContainsPredicate(
            self,
            value
        )


class AbstractPredicate(ABC):
    """
    Comparison between a value and some attribute of a search
    """

    def filter(
            self,
            search_outputs: List[SearchOutput]
    ) -> Iterator[SearchOutput]:
        """
        Only return searchs for which this predicate evaluates to True

        Parameters
        ----------
        search_outputs

        Returns
        -------

        """
        return filter(
            lambda search_output: self(search_output),
            search_outputs
        )

    def __invert__(self) -> "NotPredicate":
        """
        A predicate that evaluates to `True` when this predicate evaluates
        to False
        """
        return NotPredicate(
            self
        )

    def __or__(self, other: "AbstractPredicate") -> "OrPredicate":
        """
        Returns a predicate that is true if either predicate is true
        for a given search.
        """
        return OrPredicate(self, other)

    def __and__(self, other: "AbstractPredicate") -> "AndPredicate":
        """
        Returns a predicate that is true if both predicates are true
        for a given search.
        """
        return AndPredicate(self, other)

    @abstractmethod
    def __call__(self, search_output: SearchOutput) -> bool:
        """
        Does the attribute of the search match the requirement of this predicate?
        """


class CombinationPredicate(AbstractPredicate, ABC):
    def __init__(
            self,
            one: AbstractPredicate,
            two: AbstractPredicate
    ):
        """
        Abstract predicate combining two other predicates.

        Parameters
        ----------
        one
        two
            Child predicates
        """
        self.one = one
        self.two = two


class OrPredicate(CombinationPredicate):
    def __call__(self, search_output: SearchOutput):
        """
        The disjunction of two predicates.

        Parameters
        ----------
        search_output
            An object representing the output of a given search.

        Returns
        -------
        True if either predicate is `True` for the search
        """
        return self.one(search_output) or self.two(search_output)


class AndPredicate(CombinationPredicate):
    def __call__(self, search_output: SearchOutput):
        """
        The conjunction of two predicates.

        Parameters
        ----------
        search_output
            An object representing the output of a given search.

        Returns
        -------
        True if both predicates are `True` for the search
        """
        return self.one(search_output) and self.two(search_output)


class ComparisonPredicate(AbstractPredicate, ABC):
    def __init__(
            self,
            attribute_predicate: AttributePredicate,
            value
    ):
        """
        Compare an attribute of a search with a value.

        Parameters
        ----------
        attribute_predicate
            An attribute path of a search
        value
            A value to which the attribute is compared
        """
        self.attribute_predicate = attribute_predicate
        self._value = value

    def value(
            self,
            search_output
    ):
        if isinstance(self._value, AttributePredicate):
            return self._value.value_for_search_output(
                search_output
            )
        return self._value


class GreaterThanPredicate(ComparisonPredicate):
    def __call__(
            self,
            search_output: SearchOutput
    ) -> bool:
        """
        Parameters
        ----------
        search_output
            An object representing the output of a given search.

        Returns
        -------
        True iff the value of the attribute of the search is greater than
        the value associated with this predicate
        """

        return self.attribute_predicate.value_for_search_output(
            search_output
        ) > self.value(
            search_output
        )


class LessThanPredicate(ComparisonPredicate):
    def __call__(
            self,
            search_output: SearchOutput
    ) -> bool:
        """
        Parameters
        ----------
        search_output
            An object representing the output of a given search.

        Returns
        -------
        True iff the value of the attribute of the search is less than
        the value associated with this predicate
        """
        return self.attribute_predicate.value_for_search_output(
            search_output
        ) < self.value(
            search_output
        )


class ContainsPredicate(ComparisonPredicate):
    def __call__(
            self,
            search_output: SearchOutput
    ) -> bool:
        """
        Parameters
        ----------
        search_output
            An object representing the output of a given search.

        Returns
        -------
        True iff the value of the attribute of the search contains
        the value associated with this predicate
        """
        return self.value(
            search_output
        ) in self.attribute_predicate.value_for_search_output(
            search_output
        )


class EqualityPredicate(ComparisonPredicate):
    def __call__(self, search_output):
        """
        Parameters
        ----------
        search_output
            An object representing the output of a given search.

        Returns
        -------
        True iff the value of the attribute of the search is equal to
        the value associated with this predicate
        """
        try:
            value = self.value(
                search_output
            )
        except AttributeError:
            value = self.value
        return self.attribute_predicate.value_for_search_output(
            search_output
        ) == value


class NotPredicate(AbstractPredicate):
    def __init__(
            self,
            predicate: AbstractPredicate
    ):
        """
        Negates the output of a predicate.

        If the predicate would have returned `True` for a given search
        it now returns `False` and vice-versa.

        Parameters
        ----------
        predicate
            A predicate that is negated
        """
        self.predicate = predicate

    def __call__(self, search_output: SearchOutput) -> bool:
        """
        Evaluate the predicate for the search and return the negation
        of the result.

        Parameters
        ----------
        search_output
            The output of an AutoFit search

        Returns
        -------
        The negation of the underlying predicate
        """
        return not self.predicate(
            search_output
        )
