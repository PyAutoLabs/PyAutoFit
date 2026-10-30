"""
``ResultsIndex``: an incremental, layout-agnostic index of every result under one or
more root directories.

The index is a folder of per-result JSON records (``<index_dir>/records/<key>.json``,
``ResultRecord``) plus ``<index_dir>/index.json`` (roots, counts, when it was last
updated). The records are the truth: they are plain files, so they rsync with the
results and any tool can read them. ``ResultsCache`` builds a local SQLite cache from
them for queries; the cache is never the truth and never sits on a network filesystem.

The index is written only by ``ResultsIndex.update`` (``python -m autofit.results
index``); a fit writes only its status record and ``dataset_identity.json``.

``update`` walks the roots and fingerprints each result (``stat_fingerprint``); a
result whose fingerprint matches its record is not opened again, so re-indexing a
large tree after a few new results costs a directory walk. Records of results that no
longer exist are removed.

By default the index lives at ``<root>/.autofit_index``; pass ``index_dir`` to keep it
elsewhere, e.g. when the results tree is read-only.
"""
import json
import logging
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from autofit.non_linear.status.states import Product, ResultState
from autofit.results.classify import classify, stat_fingerprint
from autofit.results.record import SCHEMA, ResultRecord, record_key
from autofit.tools.util import open_atomic

logger = logging.getLogger(__name__)

INDEX_DIRNAME = ".autofit_index"

DEFAULT_LATEST_KEY = ("path_prefix", "unique_tag", "name")
"""
The key whose results replace one another: one fit of one search (``name``) of one
object (``path_prefix`` / ``unique_tag``). Different identifiers under one key are
reruns of that fit with another model or search setting (sibling hashes).
"""

_SKIP_DIRS = {"files", "image", INDEX_DIRNAME}

_STATUS_SUFFIX = str(Product.STATUS)
_ZIP_SUFFIX = str(Product.ZIP)


def _is_output_folder(path: Path, filenames: Sequence[str], dirs: Sequence[str]) -> bool:
    if "files" in dirs and (path / "files" / "search.json").exists():
        return True
    return ".completed" in filenames or "metadata" in filenames


def find_results(root: Union[str, os.PathLike]) -> Iterator[Path]:
    """
    Every result ``output_path`` under ``root``: a folder holding ``files/search.json``
    (or ``.completed`` / legacy ``metadata``), a ``<output_path>.zip`` or a
    ``<output_path>.status.json``, whatever the nesting.
    """
    seen = set()
    for current, dirs, filenames in os.walk(root, topdown=True):
        current_path = Path(current)
        for filename in filenames:
            if filename.endswith(_STATUS_SUFFIX):
                stem = filename[: -len(_STATUS_SUFFIX)]
            elif filename.endswith(_ZIP_SUFFIX):
                stem = filename[: -len(_ZIP_SUFFIX)]
            else:
                continue
            output_path = current_path / stem
            if output_path not in seen:
                seen.add(output_path)
                yield output_path
        if current_path != Path(root) and _is_output_folder(
            current_path, filenames, dirs
        ):
            if current_path not in seen:
                seen.add(current_path)
                yield current_path
        # Never descend into a result's own products or checkpoints; nested
        # searches (grid search children, ``analyses/``) still are.
        dirs[:] = sorted(
            d
            for d in dirs
            if d not in _SKIP_DIRS and not d.startswith("search_internal")
        )


def _now() -> str:
    # Microseconds: the cache is rebuilt when this stamp changes, and two updates can
    # fall within one second.
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class ResultsIndex:
    def __init__(
        self,
        roots: Union[str, os.PathLike, Iterable[Union[str, os.PathLike]]],
        index_dir: Optional[Union[str, os.PathLike]] = None,
        verify: bool = True,
        stale_after_s: float = 600.0,
        latest_key: Tuple[str, ...] = DEFAULT_LATEST_KEY,
    ):
        """
        The index of the results under ``roots``.

        Parameters
        ----------
        roots
            One or more directories searched recursively for results.
        index_dir
            Where the records are kept. Defaults to ``<root>/.autofit_index`` (only
            allowed with a single root).
        verify
            Check each new or changed zip's CRC when classifying it.
        stale_after_s
            How old a ``running`` status record may be before it is not trusted.
        latest_key
            The record fields whose results replace one another; within a key, every
            completed result but the latest is ``superseded``.
        """
        if isinstance(roots, (str, os.PathLike)):
            roots = [roots]
        self.roots = [Path(root).resolve() for root in roots]
        if index_dir is None:
            if len(self.roots) != 1:
                raise ValueError(
                    "Indexing several roots needs an explicit index_dir."
                )
            index_dir = self.roots[0] / INDEX_DIRNAME
        self.index_dir = Path(index_dir).resolve()
        self.verify = verify
        self.stale_after_s = stale_after_s
        self.latest_key = tuple(latest_key)

    @property
    def records_dir(self) -> Path:
        return self.index_dir / "records"

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "index.json"

    def _record_path(self, key: str) -> Path:
        return self.records_dir / f"{key}.json"

    def load(self) -> Dict[str, ResultRecord]:
        """
        Every record in the index, by output path. Unreadable records are skipped (and
        rebuilt by the next ``update``).
        """
        records = {}
        if not self.records_dir.is_dir():
            return records
        with os.scandir(self.records_dir) as entries:
            for entry in entries:
                if not entry.name.endswith(".json"):
                    continue
                try:
                    with open(entry.path) as f:
                        record = ResultRecord.from_dict(json.load(f))
                except (OSError, ValueError, TypeError):
                    continue
                if record.schema != SCHEMA:
                    continue
                records[record.output_path] = record
        return records

    def records(self) -> List[ResultRecord]:
        """
        Every record, sorted by output path.
        """
        return sorted(self.load().values(), key=lambda record: record.output_path)

    def _write(self, record: ResultRecord):
        with open_atomic(self._record_path(record.key)) as f:
            json.dump(record.to_dict(), f, indent=1)

    def update(self) -> Dict[str, int]:
        """
        Bring the index up to date with the roots: classify new and changed results,
        keep unchanged ones, drop vanished ones and re-mark ``superseded``.

        Returns
        -------
        Counts of ``scanned``, ``new``, ``changed``, ``unchanged`` and ``removed``
        results.
        """
        os.makedirs(self.records_dir, exist_ok=True)
        existing = self.load()
        found = {}
        counts = Counter()

        for root in self.roots:
            for output_path in find_results(root):
                if self.index_dir in output_path.parents:
                    continue
                counts["scanned"] += 1
                key = str(output_path)
                record = existing.get(key)
                if record is not None and record.stat_key == stat_fingerprint(
                    output_path
                ):
                    # A running record goes stale with time alone, not on disk.
                    if record.state != str(ResultState.RUNNING):
                        found[key] = record
                        counts["unchanged"] += 1
                        continue
                try:
                    record_new = classify(
                        output_path,
                        root=root,
                        verify=self.verify,
                        stale_after_s=self.stale_after_s,
                    )
                except Exception as exception:
                    logger.warning(f"Could not index {output_path}: {exception}")
                    continue
                counts["new" if record is None else "changed"] += 1
                found[key] = record_new
                self._write(record_new)

        for key, record in existing.items():
            if key not in found:
                try:
                    self._record_path(record.key).unlink()
                except FileNotFoundError:
                    pass
                counts["removed"] += 1

        for record in self._mark_superseded(found.values()):
            self._write(record)

        self._write_manifest(found.values())

        return {
            name: counts[name]
            for name in ("scanned", "new", "changed", "unchanged", "removed")
        }

    def _mark_superseded(self, records: Iterable[ResultRecord]) -> List[ResultRecord]:
        """
        Within each ``latest_key`` group, mark every completed result but the latest
        ``superseded`` (and restore a superseded result that became the latest).
        Returns the records that changed.
        """
        groups = defaultdict(list)
        for record in records:
            if record.state in (
                str(ResultState.COMPLETED),
                str(ResultState.SUPERSEDED),
            ):
                groups[tuple(getattr(record, k) for k in self.latest_key)].append(
                    record
                )

        changed = []
        for group in groups.values():
            latest = max(group, key=lambda r: (r.completed_at or "", r.output_path))
            for record in group:
                state = (
                    str(ResultState.COMPLETED)
                    if record is latest
                    else str(ResultState.SUPERSEDED)
                )
                if record.state != state:
                    record.state = state
                    record.reason = (
                        None
                        if record is latest
                        else f"a later result of the same key: {latest.output_path}"
                    )
                    changed.append(record)
        return changed

    def _write_manifest(self, records: Iterable[ResultRecord]):
        states = Counter(record.state for record in records)
        with open_atomic(self.manifest_path) as f:
            json.dump(
                {
                    "schema": SCHEMA,
                    "roots": [str(root) for root in self.roots],
                    "updated": _now(),
                    "latest_key": list(self.latest_key),
                    "counts": dict(sorted(states.items())),
                },
                f,
                indent=1,
            )

    def manifest(self) -> Optional[Dict]:
        try:
            with open(self.manifest_path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def counts(self, records: Optional[Iterable[ResultRecord]] = None) -> Dict[str, int]:
        """
        The number of results in each ``ResultState`` (every state listed, zeros
        included).
        """
        if records is None:
            records = self.load().values()
        states = Counter(record.state for record in records)
        return {str(state): states.get(str(state), 0) for state in ResultState}
