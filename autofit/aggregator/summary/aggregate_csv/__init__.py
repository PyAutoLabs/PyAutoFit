import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional, Callable, Dict, List, Union, Hashable

from autofit.aggregator.aggregator import Aggregator
from autofit.tools.util import NumpyEncoder, open_atomic

from autofit.aggregator.summary.aggregate_csv.column import (
    LabelColumn,
    Column,
    ComputedColumn,
    ValueType,
)
from autofit.aggregator.summary.aggregate_csv.row import Row
from autofit.aggregator.summary.aggregate_csv.cache import (
    RowCache,
    csv_text,
    row_fingerprint,
    spec_digest,
)
from autofit.aggregator.summary.aggregate_csv.merge import (
    MergeException,
    merge_csv,
    sort_rows,
)

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
        self.stats: Dict[str, int] = {}
        """What the last ``save`` did: rows ``computed``, ``cached``, ``errors`` and
        ``expected`` (rows for expected results with no output)."""

    def add_variable(
        self,
        argument: str,
        name: Optional[str] = None,
        value_types: list = (ValueType.Median,),
        unit: Optional[str] = None,
        prior_edge: bool = False,
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
        unit
            The unit of the values, recorded in the ``columns.json`` sidecar.
        prior_edge
            Add a ``<name>_at_prior_edge`` flag column: whether the median lies within
            1% of the prior's width from one of its limits.
        """
        self._columns.append(
            Column(
                argument,
                name=name,
                value_types=value_types,
                strict=self._strict,
                unit=unit,
                prior_edge=prior_edge,
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

    @staticmethod
    def _fields(column, value) -> Dict[str, str]:
        """
        The CSV fields ``column`` writes for ``value``, as the text the CSV holds.
        """
        if isinstance(value, dict):
            fields = {}
            for key, item in value.items():
                fields[f"{column.name}_{key}" if key else column.name] = csv_text(item)
            return fields
        return {column.name: csv_text(value)}

    def _fieldnames(self, rows, computed_fields, expected, honest) -> List[str]:
        """
        The header, from the columns rather than from the rows: the same for every
        shard of an export and whatever row sorts first. A computed column returning a
        dict contributes the fields its rows produced (sorted).
        """
        fieldnames = ["id"] + (["dataset", "search"] if expected is not None else [])
        for column in self._columns:
            if isinstance(column, Column):
                fieldnames += column.fields()
            elif isinstance(column, ComputedColumn):
                fields = computed_fields.get(column.name) or {column.name}
                fieldnames += sorted(fields, key=lambda f: (f != column.name, f))
            else:
                fieldnames.append(column.name)
        if honest:
            fieldnames.append("status_reason")
        return fieldnames

    def save(
        self,
        path: Union[str, Path],
        cache: bool = False,
        shard=None,
        expected=None,
        sort_by: Optional[str] = None,
        on_error: str = "raise",
        columns_json: bool = False,
    ) -> List[str]:
        """
        Save the summary table to a CSV file.

        Every row is computed before the file is opened and the file is written
        atomically (``open_atomic``), so an error or an interrupted save leaves any
        previous CSV at ``path`` whole.

        Parameters
        ----------
        path
            The path to save the file to
        cache
            Keep the rows in ``<path>.rows.jsonl`` and reuse them for results whose
            files and column definitions are unchanged, so refreshing a large export
            after a few new results reads only those results.
        shard
            ``"i/N"`` or ``(i, N)``: export only shard ``i`` of ``N``
            (``autofit.results.shard``); merge the parts with ``merge_csv``. Rows are
            then sorted by ``sort_by`` (default ``id``).
        expected
            A ``CompletenessReport`` (``autofit.results.ResultsLibrary.check``): adds
            ``dataset`` and ``search`` columns, and a row with its ``status_reason``
            for every expected result that is not done, so nothing expected is missing
            from the table without a reason.
        sort_by
            Sort the rows by this column (ties broken by the whole row), as
            ``merge_csv`` does: a merge of shards is then byte-identical to a serial
            export with the same ``sort_by``.
        on_error
            ``"raise"`` (the default) stops at the first result that cannot be
            summarised; ``"row"`` gives it a row whose ``status_reason`` is the error
            and goes on. With ``"row"`` or ``expected`` every row has a
            ``status_reason`` (empty when nothing is missing).
        columns_json
            Write ``<name>.columns.json`` beside the CSV: for each column its source,
            kind (``parameter``, ``latent``, ``derived``, ``label``, ``flag``,
            ``meta``), statistic, bound (``quantile_1sigma`` / ``quantile_3sigma``),
            unit and how many rows leave it blank.

        Returns
        -------
        The columns that are blank in every row, which are also logged as a warning.
        """
        from autofit.results.shard import in_shard, parse_shard

        if on_error not in ("raise", "row"):
            raise ValueError(f"on_error is 'raise' or 'row', not {on_error!r}.")
        path = Path(path)
        folder_path = path.parent if path.suffix else path
        folder_path.mkdir(parents=True, exist_ok=True)

        shard = parse_shard(shard)
        if shard is not None and sort_by is None:
            sort_by = "id"
        honest = on_error == "row" or expected is not None

        data_columns = [c for c in self._columns if not isinstance(c, LabelColumn)]
        row_cache = None
        if cache:
            row_cache = RowCache(
                path, spec_digest([c.spec() for c in data_columns] + [honest])
            ).load()
            for column in data_columns:
                kind = row_cache.meta.get("kinds", {}).get(column.name)
                if getattr(column, "kind", "") is None and kind:
                    column.kind = kind

        checks = {}
        if expected is not None:
            for check in expected.results:
                if check.record is not None:
                    checks.setdefault(
                        str(Path(check.record.output_path).resolve()), check
                    )

        stats = Counter()
        owners = {}
        computed_fields = {}
        rows = []
        exported = set()
        out_of_shard = []
        for number, result in enumerate(self._aggregator):
            output_path = _output_path(result)
            if not in_shard(output_path, shard):
                out_of_shard.append(output_path)
                continue
            row = Row(result, self._columns, number)
            entry = None
            fingerprint = None
            if row_cache is not None:
                fingerprint = row_fingerprint(Path(output_path))
                entry = row_cache.get(output_path, fingerprint)
            if entry is not None:
                stats["cached"] += 1
            else:
                try:
                    values = {c.name: self._fields(c, c.value(row)) for c in data_columns}
                    entry = {
                        "id": result.id,
                        "values": values,
                        "reasons": list(dict.fromkeys(row.reasons)),
                    }
                    if row_cache is not None:
                        row_cache.put(output_path, fingerprint, entry)
                    stats["computed"] += 1
                except Exception as exception:
                    if on_error == "raise":
                        raise
                    logger.warning(f"Could not summarise {output_path}: {exception}")
                    stats["errors"] += 1
                    entry = {
                        "id": _id_or_none(result),
                        "values": {},
                        "reasons": [f"error:{type(exception).__name__}: {exception}"],
                    }

            data = {"id": entry["id"]}
            reasons = list(entry["reasons"])
            if expected is not None:
                check = checks.get(output_path)
                data["dataset"] = check.expected.dataset if check else None
                data["search"] = check.expected.search if check else None
                if check is not None:
                    exported.add(id(check))
                    if check.status != "done":
                        reasons.append(_check_reason(check))
            for column in self._columns:
                if isinstance(column, LabelColumn):
                    try:
                        fields = self._fields(column, column.value(row))
                    except Exception as exception:
                        if on_error == "raise":
                            raise
                        reasons.append(f"error:{type(exception).__name__}: {exception}")
                        fields = {}
                else:
                    fields = entry["values"].get(column.name, {})
                    if isinstance(column, ComputedColumn):
                        computed_fields.setdefault(column.name, set()).update(fields)
                for field in fields:
                    owners.setdefault(field, column)
                data.update(fields)
            if honest:
                data["status_reason"] = ";".join(dict.fromkeys(reasons))
            rows.append(data)

        if expected is not None:
            for check in expected.results:
                if id(check) in exported:
                    continue
                record = check.record
                key = (
                    str(Path(record.output_path).resolve())
                    if record
                    else f"{check.expected.dataset}/{check.expected.search}"
                )
                if not in_shard(key, shard):
                    continue
                reason = "not_exported" if check.status == "done" else _check_reason(check)
                rows.append(
                    {
                        "id": record.identifier if record else None,
                        "dataset": check.expected.dataset,
                        "search": check.expected.search,
                        "status_reason": reason,
                    }
                )
                stats["expected"] += 1

        if sort_by is not None:
            rows = sort_rows(rows, sort_by)

        fieldnames = self._fieldnames(rows, computed_fields, expected, honest)
        for column in self._columns:
            if isinstance(column, Column):
                for field in column.fields():
                    owners.setdefault(field, column)
            else:
                owners.setdefault(column.name, column)

        with open_atomic(path, "w") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        if row_cache is not None:
            row_cache.meta["kinds"] = {
                c.name: c.kind for c in data_columns if getattr(c, "kind", None)
            }
            row_cache.save(keep=out_of_shard)

        blank = [
            name
            for name in fieldnames
            if all(row.get(name) in (None, "") for row in rows)
        ]
        if columns_json:
            self._write_columns_json(path, fieldnames, rows, owners)
        if blank:
            logger.warning(
                f"The summary CSV {path} has columns blank in every row: "
                f"{', '.join(blank)}."
            )

        self.stats = {
            name: stats.get(name, 0)
            for name in ("computed", "cached", "errors", "expected")
        }
        return blank

    @staticmethod
    def _write_columns_json(path: Path, fieldnames, rows, owners):
        meta = {
            "id": "the search output's id",
            "dataset": "the expected dataset (ResultsLibrary)",
            "search": "the expected search (ResultsLibrary)",
            "status_reason": "why values of the row are missing; empty when none are",
        }
        columns = []
        for field in fieldnames:
            if field in meta and field not in owners:
                entry = {"source": meta[field], "kind": "meta"}
            else:
                entry = owners[field].describe(field)
            columns.append(
                {
                    "name": field,
                    **entry,
                    "n_blank": sum(1 for row in rows if row.get(field) in (None, "")),
                }
            )
        json_path = path.with_suffix(".columns.json") if path.suffix else Path(
            f"{path}.columns.json"
        )
        with open_atomic(json_path, "w") as f:
            json.dump({"csv": path.name, "n_rows": len(rows), "columns": columns}, f, indent=1)


def _output_path(search_output) -> str:
    """
    The resolved ``output_path`` of a search output, whether read from a folder or a
    zip: the key of its cached row, its shard and its expected result, so a tree read
    through a symlink is the same tree.
    """
    from autofit.aggregator.zip_member import ZipMember

    directory = search_output.directory
    if isinstance(directory, ZipMember):
        path = str(Path(directory.zip_path).resolve())
        return path[: -len(".zip")] if path.endswith(".zip") else path
    return str(Path(directory).resolve())


def _id_or_none(search_output):
    try:
        return search_output.id
    except Exception:
        return None


def _check_reason(check) -> str:
    if check.missing_products:
        return "missing_product:" + ",".join(check.missing_products)
    return check.reason or check.status
