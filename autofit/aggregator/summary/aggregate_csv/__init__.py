import csv
import logging
from pathlib import Path
from typing import Optional, Callable, List, Union, Hashable

from autofit.aggregator.aggregator import Aggregator
from autofit.tools.util import open_atomic

from autofit.aggregator.summary.aggregate_csv.column import (
    LabelColumn,
    Column,
    ComputedColumn,
    ValueType,
)
from autofit.aggregator.summary.aggregate_csv.row import Row

logger = logging.getLogger(__name__)


class AggregateCSV:
    def __init__(self, aggregator: Aggregator, strict: bool = True):
        """
        Summarise results from the aggregator as a CSV.

        Parameters
        ----------
        aggregator
        strict
            If True (the default) an argument matching neither the samples_summary
            nor the latent_summary raises a KeyError, as does a keyed label with no
            value for a search and a label list whose length differs from the
            aggregator's. If False a warning is logged once for each such column and
            its values are left empty.
        """
        if len(aggregator) == 0:
            raise ValueError("The aggregator is empty.")

        self._aggregator = aggregator
        self._strict = strict
        self._columns = []

    def add_variable(
        self,
        argument: str,
        name: Optional[str] = None,
        value_types: list = (ValueType.Median,),
    ):
        """
        Add a variable to include columns for in the summary table.

        This will be taken from the samples_summary or latent_summary.

        Parameters
        ----------
        argument
            The argument as it appears in the median_pdf_sample arguments
            e.g. "galaxies.lens.bulge.centre.centre_0"
        name
            An optional name for the column. If not provided, the argument will be used.
        value_types
            Value types to include. See ValueType.
        """
        self._columns.append(
            Column(
                argument,
                name=name,
                value_types=value_types,
                strict=self._strict,
            )
        )

    def add_computed_column(
        self,
        name: str,
        compute: Callable,
    ):
        """
        Add a column to the summary table that is computed from other columns.

        Parameters
        ----------
        name
            The name of the column
        compute
            A function that takes the median_pdf_sample arguments and returns the computed value
        """
        self._columns.append(
            ComputedColumn(
                name,
                compute,
            )
        )

    def add_label_column(
        self,
        name: str,
        values: Union[list, dict],
        key: Optional[Callable[[object], Hashable]] = None,
    ):
        """
        Add a column to the summary table that is a label.

        Parameters
        ----------
        name
            The name of the column
        values
            Either a dict joined to each search by key (``key(search_output)``, the
            search's ``id`` by default), which pairs each label with its search
            whatever order the aggregator holds them in; or a list paired with the
            searches by position, which must then be as long as the aggregator.
        key
            Computes the join key of a search output for a dict of ``values``.
        """
        if (
            not isinstance(values, dict)
            and self._strict
            and len(values) != len(self._aggregator)
        ):
            raise ValueError(
                f"The label column '{name}' has {len(values)} values but the "
                f"aggregator holds {len(self._aggregator)} searches. Pass a dict "
                f"keyed by search id to join labels by key."
            )

        self._columns.append(
            LabelColumn(
                name,
                values,
                key=key,
                strict=self._strict,
            )
        )

    def _rows(self) -> List[dict]:
        return [
            Row(
                result,
                self._columns,
                i,
            ).dict()
            for i, result in enumerate(self._aggregator)
        ]

    @staticmethod
    def _union_fieldnames(rows: List[dict]) -> List[str]:
        fieldnames = {}
        for row in rows:
            fieldnames.update(dict.fromkeys(row))
        return list(fieldnames)

    @property
    def fieldnames(self) -> List[str]:
        """
        The fieldnames for the CSV file: the union of every row's columns, in the
        order they first appear, so a column only some searches have (e.g. a
        computed dict column) is not dropped because the first row lacks it.
        """
        return self._union_fieldnames(self._rows())

    def save(self, path: Union[str, Path]) -> List[str]:
        """
        Save the summary table to a CSV file.

        Every row is computed before the file is opened and the file is written
        atomically (``open_atomic``), so an error or an interrupted save leaves any
        previous CSV at ``path`` whole.

        Parameters
        ----------
        path
            The path to save the file to

        Returns
        -------
        The columns that are blank in every row, which are also logged as a warning.
        """
        path = Path(path)

        folder_path = path.parent if path.suffix else path
        folder_path.mkdir(parents=True, exist_ok=True)

        rows = self._rows()
        fieldnames = self._union_fieldnames(rows)

        with open_atomic(path, "w") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        blank = [
            name
            for name in fieldnames
            if all(row.get(name) in (None, "") for row in rows)
        ]
        if blank:
            logger.warning(
                f"The summary CSV {path} has columns blank in every row: "
                f"{', '.join(blank)}."
            )

        return blank
