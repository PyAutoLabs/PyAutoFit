import logging
import math
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

    def spec(self) -> list:
        """
        What the column computes, for the row cache key: a change to it invalidates
        cached rows.
        """
        return [type(self).__name__, self.name]

    def describe(self, field: str) -> dict:
        """
        The ``columns.json`` entry of the output field ``field`` this column writes.
        """
        return {"source": None, "kind": "derived"}


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
        unit: Optional[str] = None,
        prior_edge: bool = False,
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
        unit
            The unit of the values, recorded in ``columns.json``.
        prior_edge
            Add a ``<name>_at_prior_edge`` column: whether the median lies within 1% of
            the prior's width from one of its limits, measured in log space for a log
            prior (empty for a prior without two finite limits, e.g. a latent).
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
        self.unit = unit
        self.prior_edge = prior_edge
        self.kind = None
        self._has_warned = False

    def spec(self) -> list:
        return super().spec() + [
            self.argument,
            [value_type.name for value_type in self.value_types],
            self.strict,
            self.unit,
            self.prior_edge,
        ]

    _BOUNDS = {
        "": ("median", None),
        "max_lh": ("max_log_likelihood", None),
        "lower_1_sigma": ("lower", "quantile_1sigma"),
        "upper_1_sigma": ("upper", "quantile_1sigma"),
        "lower_3_sigma": ("lower", "quantile_3sigma"),
        "upper_3_sigma": ("upper", "quantile_3sigma"),
        "at_prior_edge": ("flag", None),
    }

    def fields(self) -> list:
        """
        The CSV fields this column writes, in order.
        """
        suffixes = []
        if ValueType.Median in self.value_types:
            suffixes.append("")
        if ValueType.MaxLogLikelihood in self.value_types:
            suffixes.append("max_lh")
        if ValueType.ValuesAt1Sigma in self.value_types:
            suffixes += ["lower_1_sigma", "upper_1_sigma"]
        if ValueType.ValuesAt3Sigma in self.value_types:
            suffixes += ["lower_3_sigma", "upper_3_sigma"]
        if self.prior_edge:
            suffixes.append("at_prior_edge")
        return [f"{self.name}_{suffix}" if suffix else self.name for suffix in suffixes]

    def describe(self, field: str) -> dict:
        suffix = "" if field == self.name else field[len(self.name) + 1 :]
        statistic, bound = self._BOUNDS.get(suffix, (None, None))
        return {
            "source": self.argument,
            "kind": "flag" if suffix == "at_prior_edge" else (self.kind or "parameter"),
            "statistic": statistic,
            "bound": bound,
            "unit": None if suffix == "at_prior_edge" else self.unit,
        }

    def _at_prior_edge(self, row: "Row", value) -> Optional[bool]:
        if value is None:
            return None
        try:
            prior = row.result.model.object_for_path(self.path)
            lower = float(prior.lower_limit)
            upper = float(prior.upper_limit)
        except Exception:
            return None
        if type(prior).__name__.startswith("Log"):
            # A log prior's width is in log space: 50 is mid-way in LogUniform(1e-4, 1e4).
            if lower <= 0 or value <= 0:
                return None
            lower, upper, value = math.log10(lower), math.log10(upper), math.log10(value)
        width = upper - lower
        if not (width > 0 and width != float("inf")):
            return None
        return bool(min(value - lower, upper - value) < 0.01 * width)

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
        if self.path in row.parameter_paths:
            self.kind = "parameter"
        elif self.path in row.known_paths:
            self.kind = "latent"
        else:
            row.reasons.append(
                "missing_latent_summary"
                if row.latent_summary is None
                else f"missing_value:{self.argument}"
            )

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

        if self.prior_edge:
            median = row.median_pdf_sample_kwargs.get(self.path)
            result["at_prior_edge"] = self._at_prior_edge(row, median)

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

    def spec(self) -> list:
        from autofit.aggregator.summary.aggregate_csv.cache import function_digest

        return super().spec() + [function_digest(self.compute)]

    def describe(self, field: str) -> dict:
        return {"source": getattr(self.compute, "__qualname__", None), "kind": "derived"}

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

    def describe(self, field: str) -> dict:
        return {"source": None, "kind": "label"}

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
