"""
Parity: is one results table the same as another — a rebuild against its reference,
a rerun against the first run?

``compare(a, b, key=...)`` diffs two tables (CSV paths, or lists of row dicts) row by
row under three kinds of rule:

- **identity** — keys present on one side only, keys repeated on either side, and
  columns present on one side only. Always gated: a table that gained, lost or
  duplicated objects or columns is not the same table. A column named in ``exact``
  or ``gate`` that is not in both tables fails too, never silently passes;
- **exact** — the columns named in ``exact`` (positions read from a header, an
  object's name) must be equal — as integers, as numbers, else as text — and a value
  blank on one side only is a difference; gated;
- **tolerance by σ** — a value column ``x`` with ``x_lower_1_sigma`` /
  ``x_upper_1_sigma`` on both sides is compared as::

      sigma = max((upper - lower) / 2, floor, rel_floor * |x|)
      z     = |a - b| / sqrt(sigma_a ** 2 + sigma_b ** 2)

  and reported as the median and maximum ``z`` over the common rows. The floors keep a
  vanishing error bar from turning a negligible difference into an enormous ``z``.

Value columns are **informational unless named in** ``gate``: two runs of the same
fit are not bitwise reproducible, and two Nautilus runs of the same model on the same
data have been measured differing by a median ``z`` of 15 (results-library report
P19). So "within 3σ" is not a parity test until the run-to-run scatter has been
measured — ``scatter_between`` measures it — and a parity check of reruns reports
their scatter instead of failing on it.
"""
import csv
import math
import os
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

SIGMA_SUFFIXES = ("_lower_1_sigma", "_upper_1_sigma")
DERIVED_SUFFIXES = (
    "_lower_1_sigma",
    "_upper_1_sigma",
    "_lower_3_sigma",
    "_upper_3_sigma",
    "_max_lh",
    "_at_prior_edge",
)

Table = Union[str, os.PathLike, Sequence[Dict]]


def _rows(table: Table) -> List[Dict]:
    if isinstance(table, (str, os.PathLike)):
        with open(table, newline="") as f:
            return list(csv.DictReader(f))
    return [dict(row) for row in table]


def _float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass
class ColumnParity:
    name: str
    kind: str
    """``exact``, ``sigma`` (tolerance by σ), ``numeric`` or ``text``."""
    gated: bool
    n: int = 0
    """Rows compared (both sides non-blank)."""
    n_blank: int = 0
    n_mismatch: int = 0
    """Rows that differ (exact / text) or exceed the threshold (sigma)."""
    median_z: Optional[float] = None
    max_z: Optional[float] = None
    max_abs_diff: Optional[float] = None
    worst: List[Tuple] = field(default_factory=list)
    """Up to five ``(key, a, b, z or None)`` of the largest differences."""

    @property
    def passed(self) -> bool:
        return not self.gated or self.n_mismatch == 0


@dataclass
class ParityReport:
    key: Tuple[str, ...]
    threshold: float
    n_a: int
    n_b: int
    n_common: int
    only_a: List = field(default_factory=list)
    only_b: List = field(default_factory=list)
    duplicates_a: List = field(default_factory=list)
    duplicates_b: List = field(default_factory=list)
    columns_only_a: List[str] = field(default_factory=list)
    columns_only_b: List[str] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)
    """Columns named in ``exact`` or ``gate`` that are not in both tables."""
    columns: Dict[str, ColumnParity] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return (
            not self.only_a
            and not self.only_b
            and not self.duplicates_a
            and not self.duplicates_b
            and not self.columns_only_a
            and not self.columns_only_b
            and not self.unknown
            and all(column.passed for column in self.columns.values())
        )

    def to_markdown(self, title: str = "Parity") -> str:
        lines = [
            f"# {title}: {'PASS' if self.passed else 'FAIL'}",
            "",
            f"Key `{', '.join(self.key)}`: {self.n_a} rows in A, {self.n_b} in B, "
            f"{self.n_common} in both. Tolerance: z ≤ {self.threshold:g} on gated "
            f"σ columns.",
            "",
            "## Identity (gated)",
            "",
            f"- only in A: {len(self.only_a)} {self.only_a[:5]}",
            f"- only in B: {len(self.only_b)} {self.only_b[:5]}",
            f"- duplicate keys in A: {len(self.duplicates_a)} {self.duplicates_a[:5]}",
            f"- duplicate keys in B: {len(self.duplicates_b)} {self.duplicates_b[:5]}",
            f"- columns only in A: {self.columns_only_a}",
            f"- columns only in B: {self.columns_only_b}",
            f"- exact / gate columns not in both tables: {self.unknown}",
            "",
            "## Columns",
            "",
            "| column | kind | gated | n | blank | median z | max z | over / differing | result |",
            "|---|---|---|---|---|---|---|---|---|",
        ]

        def fmt(value):
            return "" if value is None else f"{value:.3g}"

        for column in self.columns.values():
            lines.append(
                f"| `{column.name}` | {column.kind} | {'yes' if column.gated else 'no'} | "
                f"{column.n} | {column.n_blank} | {fmt(column.median_z)} | "
                f"{fmt(column.max_z)} | {column.n_mismatch} | "
                f"{'pass' if column.passed else 'FAIL'} |"
            )
        failing = [c for c in self.columns.values() if c.worst and (c.gated or c.kind == "sigma")]
        if failing:
            lines += ["", "## Largest differences", ""]
            for column in failing:
                lines.append(f"- `{column.name}`: {column.worst}")
        return "\n".join(lines) + "\n"


def compare(
    a: Table,
    b: Table,
    key: Union[str, Sequence[str]] = "id",
    exact: Sequence[str] = (),
    gate: Sequence[str] = (),
    threshold: float = 3.0,
    floor: float = 0.0,
    rel_floor: float = 0.0,
    columns: Optional[Sequence[str]] = None,
) -> ParityReport:
    """
    Compare table ``a`` (under test) with table ``b`` (the reference); see the module
    docstring.

    Parameters
    ----------
    key
        The column (or columns) naming a row.
    exact
        Columns that must be equal (as numbers when both parse, else as text); gated.
    gate
        Value columns whose ``z`` (or, without σ columns, any difference) fails the
        comparison; every other value column is informational.
    threshold
        The largest ``z`` a gated σ column may reach.
    floor, rel_floor
        The smallest σ used: ``max(sigma, floor, rel_floor * |value|)``.
    columns
        Compare only these value columns (default: every column on both sides).
    """
    keys = (key,) if isinstance(key, str) else tuple(key)
    rows_a, rows_b = _rows(a), _rows(b)

    def index(rows):
        counts = Counter(tuple(row.get(k, "") for k in keys) for row in rows)
        by_key = {}
        for row in rows:
            by_key.setdefault(tuple(row.get(k, "") for k in keys), row)
        return by_key, sorted(k for k, n in counts.items() if n > 1)

    by_a, duplicates_a = index(rows_a)
    by_b, duplicates_b = index(rows_b)
    common = sorted(set(by_a) & set(by_b))

    def label(k):
        return k[0] if len(k) == 1 else k

    report = ParityReport(
        key=keys,
        threshold=threshold,
        n_a=len(rows_a),
        n_b=len(rows_b),
        n_common=len(common),
        only_a=[label(k) for k in sorted(set(by_a) - set(by_b))],
        only_b=[label(k) for k in sorted(set(by_b) - set(by_a))],
        duplicates_a=[label(k) for k in duplicates_a],
        duplicates_b=[label(k) for k in duplicates_b],
    )

    fields_a = _fieldnames(a, rows_a)
    fields_b = _fieldnames(b, rows_b)
    report.columns_only_a = [f for f in fields_a if f not in fields_b]
    report.columns_only_b = [f for f in fields_b if f not in fields_a]
    shared = [f for f in fields_a if f in fields_b and f not in keys]
    report.unknown = sorted(
        {name for name in list(exact) + list(gate) if name not in shared}
    )
    values = [f for f in shared if f in exact or not f.endswith(DERIVED_SUFFIXES)]
    if columns is not None:
        values = [f for f in values if f in columns or f in exact or f in gate]

    for name in values:
        has_sigma = all(f"{name}{suffix}" in shared for suffix in SIGMA_SUFFIXES)
        if name in exact:
            kind = "exact"
        elif has_sigma:
            kind = "sigma"
        else:
            numeric = all(
                _float(by_a[k].get(name)) is not None or by_a[k].get(name) in ("", None)
                for k in common
            )
            kind = "numeric" if numeric else "text"
        column = ColumnParity(name=name, kind=kind, gated=name in exact or name in gate)
        zs = []
        diffs = []
        for k in common:
            va, vb = by_a[k].get(name, ""), by_b[k].get(name, "")
            blank_a, blank_b = va in ("", None), vb in ("", None)
            if blank_a or blank_b:
                column.n_blank += 1
                if blank_a != blank_b and column.gated:
                    # A value lost (or gained) is a difference, not a blank.
                    column.n_mismatch += 1
                    column.worst.append((label(k), va, vb, None))
                continue
            column.n += 1
            fa, fb = _float(va), _float(vb)
            if kind == "sigma" and fa is not None and fb is not None:
                sa = _sigma(by_a[k], name, fa, floor, rel_floor)
                sb = _sigma(by_b[k], name, fb, floor, rel_floor)
                scale = math.sqrt(sa**2 + sb**2)
                z = abs(fa - fb) / scale if scale > 0 else (0.0 if fa == fb else math.inf)
                zs.append((z, label(k), fa, fb))
                if z > threshold:
                    column.n_mismatch += 1
            elif kind == "exact":
                if not _exact_equal(va, vb, fa, fb):
                    diffs.append((abs(fa - fb) if fa is not None and fb is not None else None, label(k), va, vb))
                    column.n_mismatch += 1
            elif fa is not None and fb is not None:
                diffs.append((abs(fa - fb), label(k), fa, fb))
                if fa != fb:
                    column.n_mismatch += 1
            elif va != vb:
                diffs.append((None, label(k), va, vb))
                column.n_mismatch += 1
        if zs:
            values_z = [z for z, *_ in zs]
            column.median_z = statistics.median(values_z)
            column.max_z = max(values_z)
            column.worst = [
                (k, fa, fb, round(z, 3))
                for z, k, fa, fb in sorted(zs, key=lambda item: -item[0])[:5]
                if z > 0
            ]
        if diffs:
            numeric = [d for d in diffs if d[0] is not None]
            if numeric:
                column.max_abs_diff = max(d[0] for d in numeric)
            column.worst += [
                (k, va, vb, None)
                for d, k, va, vb in sorted(diffs, key=lambda d: -(d[0] or 0))[:5]
                if d is None or d > 0
            ]
        column.worst = column.worst[:5]
        report.columns[name] = column
    return report


_INTEGER = re.compile(r"^[+-]?\d+$")


def _exact_equal(va: str, vb: str, fa, fb) -> bool:
    """
    Equal as integers when both are integers (a 19-digit object id is not a float), as
    numbers when both parse ("1.0" == "1"), else as text.
    """
    va, vb = str(va).strip(), str(vb).strip()
    if va == vb:
        return True
    if _INTEGER.match(va) and _INTEGER.match(vb):
        return int(va) == int(vb)
    if fa is not None and fb is not None:
        return fa == fb
    return False


def _fieldnames(table, rows) -> List[str]:
    if isinstance(table, (str, os.PathLike)):
        with open(table, newline="") as f:
            return list(csv.DictReader(f).fieldnames or [])
    names = {}
    for row in rows:
        names.update(dict.fromkeys(row))
    return list(names)


def _sigma(row, name, value, floor, rel_floor) -> float:
    lower = _float(row.get(f"{name}_lower_1_sigma"))
    upper = _float(row.get(f"{name}_upper_1_sigma"))
    sigma = (upper - lower) / 2 if lower is not None and upper is not None else 0.0
    return max(sigma, floor, rel_floor * abs(value))


@dataclass
class Scatter:
    path: str
    n: int
    median_z: Optional[float]
    p84_z: Optional[float]
    max_z: Optional[float]
    zs: List[float] = field(default_factory=list)


@dataclass
class ScatterReport:
    matched: int
    only_a: List
    only_b: List
    paths: Dict[str, Scatter]

    def __str__(self):
        lines = [
            f"scatter between runs: {self.matched} matched fits "
            f"({len(self.only_a)} only in A, {len(self.only_b)} only in B)"
        ]
        for scatter in self.paths.values():
            if scatter.n:
                lines.append(
                    f"  {scatter.path}: n={scatter.n} median z={scatter.median_z:.3g} "
                    f"p84 z={scatter.p84_z:.3g} max z={scatter.max_z:.3g}"
                )
            else:
                lines.append(f"  {scatter.path}: no matched values")
        return "\n".join(lines)


def _default_key(search_output):
    return tuple(
        str(getattr(search_output, attribute, None))
        for attribute in ("path_prefix", "unique_tag", "name")
    )


def scatter_between(
    a,
    b,
    paths: Iterable[str],
    key: Callable = _default_key,
) -> ScatterReport:
    """
    The run-to-run scatter of two sets of fits of the same objects: for each parameter
    (or latent) ``path``, the distribution of ``z = |a - b| / sqrt(sa**2 + sb**2)`` over
    the fits matched between ``a`` and ``b``, with ``sa`` / ``sb`` each run's 1σ
    half-width.

    It is the calibration a parity test needs: a median ``z`` well above 1 between two
    runs of the same code on the same data says the quoted errors understate the
    run-to-run variation, and no tolerance tighter than that scatter can pass a rerun.

    Parameters
    ----------
    a, b
        Aggregators (or lists of search outputs) of the two runs.
    paths
        Model paths, e.g. ``"galaxies.lens.mass.einstein_radius"`` or a latent
        ``"latent.value"``.
    key
        Matches a fit in ``a`` to its rerun in ``b`` (default: ``path_prefix``,
        ``unique_tag``, ``name``). It must name one fit per object on each side;
        a repeated key raises ``ValueError`` rather than silently dropping fits.
    """
    from autofit.aggregator.summary.aggregate_csv.row import Row

    def by_key(outputs, side):
        found = {}
        repeated = set()
        for output in outputs:
            k = key(output)
            if k in found:
                repeated.add(k)
            found[k] = output
        if repeated:
            raise ValueError(
                f"The key names several fits in {side}: {sorted(repeated)[:3]}. Pass a "
                f"key that identifies one fit of one object, e.g. "
                f"key=lambda output: output.unique_tag."
            )
        return found

    outputs_a = by_key(getattr(a, "search_outputs", a), "a")
    outputs_b = by_key(getattr(b, "search_outputs", b), "b")
    common = sorted(set(outputs_a) & set(outputs_b))

    rows = {k: (Row(outputs_a[k], [], 0), Row(outputs_b[k], [], 0)) for k in common}
    scatters = {}
    for path in paths:
        parts = tuple(path.split("."))
        zs = []
        for row_a, row_b in rows.values():
            try:
                va = row_a.median_pdf_sample_kwargs.get(parts)
                vb = row_b.median_pdf_sample_kwargs.get(parts)
                la, ua = row_a.values_at_sigma_1_kwargs.get(parts, (None, None))
                lb, ub = row_b.values_at_sigma_1_kwargs.get(parts, (None, None))
            except Exception:
                continue
            if None in (va, vb, la, ua, lb, ub):
                continue
            scale = math.sqrt(((ua - la) / 2) ** 2 + ((ub - lb) / 2) ** 2)
            if scale > 0:
                zs.append(abs(va - vb) / scale)
        zs.sort()
        scatters[path] = Scatter(
            path=path,
            n=len(zs),
            median_z=statistics.median(zs) if zs else None,
            p84_z=zs[min(len(zs) - 1, int(math.ceil(0.84 * len(zs))) - 1)] if zs else None,
            max_z=zs[-1] if zs else None,
            zs=zs,
        )
    return ScatterReport(
        matched=len(common),
        only_a=sorted(set(outputs_a) - set(outputs_b)),
        only_b=sorted(set(outputs_b) - set(outputs_a)),
        paths=scatters,
    )
