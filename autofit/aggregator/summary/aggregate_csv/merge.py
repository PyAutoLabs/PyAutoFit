"""
Merge CSV parts exported in shards (``AggregateCSV.save(path, shard="i/N")``) into one
CSV, the way any scheduler can run them: Slurm array tasks, a laptop process pool, or
one after another.

The merge is strict where a silent result would be wrong:

- every part with rows must have the same header (the same columns in the same
  order); parts of one export do, so a disagreement means parts of different exports
  (or a computed column returning dicts with different keys in different shards). A
  part with no rows (an empty shard) is not compared;
- a key (``id`` by default) seen in several parts is kept once when the rows are
  identical, and refused when they differ; rows with an empty key (expected results
  with no output) are kept once per distinct row;
- rows are sorted (``sort_rows``), so a merge is byte-identical to a serial export saved
  with the same ``sort_by``, whatever the shard count.
"""
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union

from autofit.tools.util import open_atomic

PathLike = Union[str, os.PathLike]


class MergeException(Exception):
    pass


def sort_rows(rows: Iterable[Dict], by: str) -> List[Dict]:
    """
    ``rows`` sorted by the column ``by``, ties broken by the whole row, so the order
    never depends on how the rows were found.
    """
    return sorted(
        rows,
        key=lambda row: (
            str(row.get(by) if row.get(by) is not None else ""),
            tuple(str(value if value is not None else "") for value in row.values()),
        ),
    )


def write_rows(path: PathLike, fieldnames: Sequence[str], rows: Iterable[Dict]):
    """
    Write ``rows`` to the CSV at ``path`` atomically (``open_atomic``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_atomic(path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), lineterminator="\r\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def merge_csv(
    parts: Sequence[PathLike],
    path: PathLike,
    key: str = "id",
    sort_by: str = None,
) -> int:
    """
    Merge the CSV ``parts`` into ``path`` (see the module docstring).

    Parameters
    ----------
    parts
        The CSV files to merge, e.g. ``summary.part0.csv`` .. ``summary.part3.csv``.
    key
        The column that identifies a row; rows sharing a non-empty key are duplicates.
    sort_by
        The column the merged rows are sorted by (default: ``key``).

    Returns
    -------
    The number of rows written.

    Raises
    ------
    MergeException
        When the headers disagree, a duplicate key has different rows, or ``key`` is not
        a column.
    """
    if not parts:
        raise MergeException("No parts to merge.")
    sort_by = sort_by or key

    header = None
    first = None
    rows = []
    seen = {}
    seen_unkeyed = set()
    for part in parts:
        with open(part, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            part_rows = list(reader)
        if key not in fieldnames:
            raise MergeException(f"{part} has no '{key}' column.")
        if not part_rows:
            continue
        if header is None:
            header, first = fieldnames, part
        elif fieldnames != header:
            raise MergeException(
                f"The header of {part} differs from that of {first}: "
                f"{fieldnames} != {header}."
            )
        for row in part_rows:
            value = row[key]
            if value:
                if value in seen:
                    if seen[value] != row:
                        raise MergeException(
                            f"Key {key}={value} has different rows in two parts."
                        )
                    continue
                seen[value] = row
            else:
                identity = tuple(row.items())
                if identity in seen_unkeyed:
                    continue
                seen_unkeyed.add(identity)
            rows.append(row)

    if header is None:
        with open(parts[0], newline="") as f:
            header = list(csv.DictReader(f).fieldnames or [])
    write_rows(path, header, sort_rows(rows, sort_by))
    return len(rows)
