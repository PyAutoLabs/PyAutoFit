"""
Completeness of a ``ResultsLibrary``: every expected result is ``done``, ``missing``,
``incomplete``, ``failed`` or ``duplicate``, each with a reason, read from the results
index — a zip is opened only to list its members when a path product is required.

- ``done``: one completed result holding every required product;
- ``missing``: no result at all (``no_result``);
- ``incomplete``: a result that is not complete yet (``not_completed:<state>``:
  running, interrupted, never started), a completed one lacking a required product
  (``missing_product``, one gap per product), or one that sits where another object's
  result belongs (``location_mismatch``);
- ``failed``: the result failed or its zip / checkpoint is corrupt
  (``not_completed:<state>``), or where it should be is not known (a campaign manifest
  line missing a field, ``manifest_line:<index> ...``);
- ``duplicate``: several completed results match (``duplicate:<n>``) — the library
  cannot say which is the one.

Superseded results (an older sibling of a newer completed fit) never count, except at
an exact location: there the result asked about is the one at that path. An exact
(campaign) expectation also needs a zip (``no_zip``) holding its path products as
members, as ``autofit.campaign.collector.check_success`` does.

``CompletenessReport.write_missing_csv`` writes one row per gap to
``<name>.missing.csv``: an expected result that is missing, failed or incomplete as a
whole has one row with an empty ``product``; one lacking products has a row per product.
"""
import csv
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence

from autofit.non_linear.status.states import Product, ResultState
from autofit.results.library import ExpectedResult
from autofit.results.record import ResultRecord
from autofit.tools.util import open_atomic

logger = logging.getLogger(__name__)

NAMED_PRODUCTS = {
    "samples",
    "samples_summary",
    "latent",
    "image",
    "dataset_identity",
    "search_internal",
}

STATUSES = ("done", "missing", "incomplete", "failed", "duplicate")

MISSING_COLUMNS = ("dataset", "search", "product", "status", "reason", "output_path")

_FAILED_STATES = {
    str(ResultState.FAILED),
    str(ResultState.CORRUPT_ZIP),
    str(ResultState.CORRUPT_CKPT),
}

# Which unfinished result speaks for an expected result when there are several: the
# one with the most to say about why it is not done.
_PRECEDENCE = [
    str(state)
    for state in (
        ResultState.FAILED,
        ResultState.CORRUPT_ZIP,
        ResultState.CORRUPT_CKPT,
        ResultState.INTERRUPTED_VALID_CKPT,
        ResultState.RUNNING,
        ResultState.NEVER_STARTED,
    )
]


@dataclass
class ResultCheck:
    expected: ExpectedResult
    status: str
    reason: Optional[str] = None
    record: Optional[ResultRecord] = None
    missing_products: List[str] = field(default_factory=list)

    @property
    def output_path(self) -> Optional[str]:
        return self.record.output_path if self.record is not None else None

    def gaps(self) -> List[Dict[str, str]]:
        """
        The rows this result adds to ``<name>.missing.csv`` (none when ``done``).
        """
        if self.status == "done":
            return []
        row = {
            "dataset": self.expected.dataset,
            "search": self.expected.search,
            "status": self.status,
            "output_path": self.output_path or "",
        }
        if self.missing_products:
            return [
                {**row, "product": product, "reason": f"missing_product:{product}"}
                for product in self.missing_products
            ]
        return [{**row, "product": "", "reason": self.reason or ""}]


@dataclass
class CompletenessReport:
    name: str
    results: List[ResultCheck]
    unexpected: List[ResultRecord] = field(default_factory=list)
    """Completed results of a declared search that no expected result claims, e.g. a
    fit of an object outside the manifest or a tree indexed by mistake."""

    def counts(self) -> Dict[str, int]:
        """
        Expected results per status, plus ``expected`` and ``unexpected``.
        """
        statuses = Counter(result.status for result in self.results)
        return {
            "expected": len(self.results),
            **{status: statuses.get(status, 0) for status in STATUSES},
            "unexpected": len(self.unexpected),
        }

    def product_counts(self) -> Dict[str, Dict[str, int]]:
        """
        For each required product: how many expected results require it and how many
        of the completed ones lack it.
        """
        counts = defaultdict(lambda: {"required": 0, "missing": 0})
        for result in self.results:
            for product in result.expected.products:
                counts[product]["required"] += 1
            for product in result.missing_products:
                counts[product]["missing"] += 1
        return dict(counts)

    @property
    def is_complete(self) -> bool:
        return all(result.status == "done" for result in self.results)

    def gaps(self) -> List[Dict[str, str]]:
        return [row for result in self.results for row in result.gaps()]

    def write_missing_csv(self, directory: os.PathLike = ".") -> Path:
        """
        Write every gap to ``<directory>/<name>.missing.csv`` (header only when the
        library is complete), atomically: a reader never sees half a file.
        """
        path = Path(directory) / f"{self.name}.missing.csv"
        with open_atomic(path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=MISSING_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for row in self.gaps():
                writer.writerow(row)
        return path

    def summary(self) -> str:
        counts = self.counts()
        lines = [
            f"{self.name}: {counts['done']} of {counts['expected']} expected results done"
        ]
        for status in STATUSES[1:]:
            if counts[status]:
                lines.append(f"  {status:12s} {counts[status]}")
        for product, product_counts in sorted(self.product_counts().items()):
            if product_counts["missing"]:
                lines.append(
                    f"  missing {product}: {product_counts['missing']} of "
                    f"{product_counts['required']}"
                )
        if counts["unexpected"]:
            lines.append(
                f"  {'unexpected':12s} {counts['unexpected']} completed results no "
                f"expected result claims"
            )
        return "\n".join(lines)


def _resolved(path: str) -> str:
    return str(Path(path).resolve())


def _name_of(record: ResultRecord) -> Optional[str]:
    # A fit that failed before writing ``files/search.json`` leaves only its status
    # record: its name is then where it sits, ``<path_prefix>/<name>/<identifier>``.
    return record.name or Path(record.output_path).parent.name


def _dataset_keys(record: ResultRecord) -> List[str]:
    keys = [record.unique_tag, record.object_id, record.path_prefix]
    if record.path_prefix:
        keys.append(PurePosixPath(record.path_prefix).name)
    elif record.name is None:
        keys.append(Path(record.output_path).parent.parent.name)
    return [key for key in keys if key]


def has_product(record: ResultRecord, product: str, zip_only: bool = False) -> bool:
    """
    Whether the result ``record`` describes holds ``product`` (a named product or a
    path inside the result; a trailing ``/`` names a folder). With ``zip_only`` a path
    product must be a member of the zip (named products come from the index record).
    """
    if product in NAMED_PRODUCTS:
        return bool(record.products.get(product))

    from autofit.aggregator.zip_member import zip_names

    output_path = Path(record.output_path)
    folder = product.endswith("/")
    member = product.rstrip("/")
    if record.has_dir and not zip_only:
        path = output_path / member
        if path.is_dir() if folder else path.is_file():
            return True
    if record.has_zip:
        try:
            names = zip_names(f"{output_path}{Product.ZIP}")
        except OSError as exception:
            logger.warning(f"Could not list {output_path}{Product.ZIP}: {exception}")
            return False
        if folder:
            return any(name.startswith(f"{member}/") for name in names)
        return member in names
    return False


def _check_one(
    expected: ExpectedResult, candidates: Sequence[ResultRecord]
) -> ResultCheck:
    if expected.error is not None:
        return ResultCheck(expected, "failed", expected.error)
    if not expected.exact:
        # Only the latest of several fits of one search to one object counts; at an
        # exact location the result there is the one asked about, superseded or not.
        candidates = [
            record
            for record in candidates
            if record.state != str(ResultState.SUPERSEDED)
        ]
    if not candidates:
        return ResultCheck(expected, "missing", "no_result")

    finished = (str(ResultState.COMPLETED), str(ResultState.SUPERSEDED))
    completed = [record for record in candidates if record.state in finished]
    if len(completed) > 1:
        latest = max(completed, key=lambda r: (r.completed_at or "", r.output_path))
        return ResultCheck(expected, "duplicate", f"duplicate:{len(completed)}", latest)

    if not completed:
        record = min(
            candidates,
            key=lambda r: (
                _PRECEDENCE.index(r.state) if r.state in _PRECEDENCE else len(_PRECEDENCE),
                r.output_path,
            ),
        )
        reason = f"not_completed:{record.state}"
        if record.reason:
            reason = f"{reason} ({record.reason})"
        status = "failed" if record.state in _FAILED_STATES else "incomplete"
        return ResultCheck(expected, status, reason, record)

    record = completed[0]
    if expected.exact and not record.has_zip:
        return ResultCheck(expected, "incomplete", "no_zip", record)
    if record.location_mismatch and not expected.exact:
        return ResultCheck(expected, "incomplete", "location_mismatch", record)
    missing = [
        product
        for product in expected.products
        if not has_product(record, product, zip_only=expected.exact)
    ]
    if missing:
        return ResultCheck(
            expected,
            "incomplete",
            "missing_product",
            record,
            missing_products=missing,
        )
    return ResultCheck(expected, "done", None, record)


def check(library, records: Iterable[ResultRecord]) -> CompletenessReport:
    """
    The ``CompletenessReport`` of ``library`` against the index ``records``.
    """
    records = list(records)
    by_path = {}
    by_parent = defaultdict(list)
    by_name = defaultdict(list)
    for record in records:
        path = _resolved(record.output_path)
        by_path[path] = record
        by_parent[str(Path(path).parent)].append(record)
        by_name[_name_of(record)].append(record)

    claimed = set()
    results = []
    for expected in library.expected:
        if expected.error is not None:
            candidates = []
        elif expected.location is not None:
            location = _resolved(expected.location)
            if location in by_path:
                candidates = [by_path[location]]
            elif expected.exact:
                candidates = []
            else:
                candidates = by_parent.get(location, [])
        else:
            candidates = [
                record
                for record in by_name.get(expected.search, [])
                if expected.dataset in _dataset_keys(record)
            ]
        claimed.update(record.output_path for record in candidates)
        results.append(_check_one(expected, candidates))

    searches = {expected.search for expected in library.expected}
    unexpected = [
        record
        for record in records
        if record.state == str(ResultState.COMPLETED)
        and record.output_path not in claimed
        and record.name in searches
    ]
    return CompletenessReport(library.name, results, unexpected)
