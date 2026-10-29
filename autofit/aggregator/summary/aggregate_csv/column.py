import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class AbstractColumn(ABC):
    def __init__(self, name: str):
        """
        A column in the summary table.

        Parameters
        ----------
        name
            An optional name for the column
        """
        self.name = name

    @abstractmethod
    def value(self, row: "Row"):
        pass


class ValueType(Enum):
    """
    Possible value types to create columns for.
    """

    Median = 0  # The median_pdf_sample value
    MaxLogLikelihood = 1  # The max likelihood value
    ValuesAt1Sigma = 2  # The values at 1 sigma. This includes a lower and upper value
    ValuesAt3Sigma = 3  # The values at 3 sigma. This includes a lower and upper value


class Column(AbstractColumn):
    def __init__(
        self,
        argument: str,
        name: Optional[str] = None,
        value_types: list[ValueType] = (ValueType.Median,),
        strict: bool = False,
    ):
        """
        A column in the summary table.

        Parameters
        ----------
        argument
            The argument as it appears in the median_pdf_sample arguments
        name
            An optional name for the column
        value_types
            Value types to include. See ValueType.
        strict
            If True an argument matching neither the samples_summary nor the
            latent_summary raises a KeyError. If False (the default) a warning
            is logged once for the column and the values are left empty.
        """
        super().__init__(
            name
            or argument.replace(
                ".",
                "_",
            )
        )
        self.argument = argument
        self.value_types = value_types
        self.strict = strict
        self._has_warned = False

    def _check_argument(self, row: "Row"):
        """
        Check that the argument matches the samples_summary or latent_summary
        of a row.

        If it matches neither then a KeyError is raised when strict, else a
        warning is logged once for the column.

        Parameters
        ----------
        row
            A row in the summary table
        """
        if self.path in row.known_paths:
            return

        available = sorted(".".join(path) for path in row.known_paths)
        message = (
            f"Argument '{self.argument}' matches no value in the samples "
            f"summary or latent summary. Available arguments include: "
            f"{', '.join(available[:10])}"
        )
        if self.strict:
            raise KeyError(message)
        if not self._has_warned:
            self._has_warned = True
            logger.warning(message)

    def value(self, row: "Row"):
        self._check_argument(row)

        result = {}

        if ValueType.Median in self.value_types:
            try:
                result[""] = row.median_pdf_sample_kwargs[self.path]
            except KeyError:
                result[""] = None

        if ValueType.MaxLogLikelihood in self.value_types:
            try:
                result["max_lh"] = row.max_likelihood_kwargs[self.path]
            except KeyError:
                result["max_lh"] = None

        if ValueType.ValuesAt1Sigma in self.value_types:
            try:
                lower, upper = row.values_at_sigma_1_kwargs[self.path]
                result["lower_1_sigma"] = lower
                result["upper_1_sigma"] = upper
            except KeyError:
                result["lower_1_sigma"] = None
                result["upper_1_sigma"] = None

        if ValueType.ValuesAt3Sigma in self.value_types:
            try:
                lower, upper = row.values_at_sigma_3_kwargs[self.path]
                result["lower_3_sigma"] = lower
                result["upper_3_sigma"] = upper
            except KeyError:
                result["lower_3_sigma"] = None
                result["upper_3_sigma"] = None

        return result

    @property
    def path(self) -> tuple:
        """
        The path of an argument in the median_pdf_sample arguments.
        """
        return tuple(self.argument.split("."))


class ComputedColumn(AbstractColumn):
    def __init__(self, name: str, compute: Callable):
        """
        A column in the summary table that is computed from other columns.

        Parameters
        ----------
        name
            The name of the column
        compute
            A function that takes the median_pdf_sample arguments and returns the computed value
        """
        super().__init__(name)
        self.compute = compute

    def value(self, row: "Row"):

        try:
            return self.compute(row.result)
        except AttributeError as e:
            raise AssertionError(
                "Cannot compute additional fields if no samples.json present"
            ) from e


class LabelColumn(AbstractColumn):
    def __init__(
        self,
        name: str,
        values,
        key: Optional[Callable] = None,
        strict: bool = False,
    ):
        """
        A column in the summary table that is a label.

        Parameters
        ----------
        name
            The name of the column
        values
            A dict joined to each row by key, or a list paired with rows by position.
        key
            Computes a row's join key from its search output (its ``id`` by default).
        strict
            If True a dict with no value for a row's key raises a KeyError; if False
            the value is left empty.
        """
        super().__init__(name)
        self.values = values
        self.key = key
        self.strict = strict

    def value(self, row: "Row"):
        if not isinstance(self.values, dict):
            return self.values[row.number]

        key = self.key(row.result) if self.key is not None else row.result.id
        try:
            return self.values[key]
        except KeyError:
            if self.strict:
                raise KeyError(
                    f"The label column '{self.name}' has no value for the search "
                    f"with key {key!r}."
                )
            return None
