"""
``package``: ship a finished results library as a set of zips anyone can download,
check and unpack.

The source is a *library folder*: one sub-folder per object (a lens, a galaxy, a
dataset), kept whole, plus tables (CSVs, a README) at its root::

    library/
        summary.csv
        summary.columns.json
        obj0001/  ...
        obj0002/  ...

``package(source, dist)`` writes to ``dist``:

- ``tables.zip``: the root files;
- ``part_001.zip``, ``part_002.zip`` ...: the object folders, each wholly inside one
  part, the parts no larger than ``cap_gb`` (or exactly ``n_parts`` near-equal parts);
- ``parts_manifest.csv``: which part holds each object;
- ``package.json``: everything ``verify`` checks and a rebuild reuses — each object's
  part, each zip's members (size, modification time), its SHA256, the row count of
  every CSV, and the provenance;
- ``SHA256SUMS``: the checksum of every zip, the manifest and ``package.json``;
- ``README.md`` (regenerated) and ``KNOWN_ISSUES.md`` (written once, then yours).

Members are archived ``ZIP_STORED`` when already compressed or large binaries (PNG,
FITS, HDF5) and ``ZIP_DEFLATED`` otherwise, in sorted order with each file's own
modification time, so the same files always give the same zip. Every zip is written
to a temporary file and moved into place: a killed run leaves the previous zip whole.

**Rebuilds are deltas.** Packaging again into the same ``dist`` keeps every object in
the part it was in; new objects go into the last part while it has room, then into
new parts; a part whose files (names, sizes, modification times) are unchanged is not
rewritten, so it stays byte-identical and its checksum stays valid for anyone who
already downloaded it. ``replan=True`` plans from scratch.

``include_dataset(object_name)`` may return extra files to ship with an object (its
input data, a segmentation map), archived under ``<object>/dataset/``.
"""
import csv
import hashlib
import io
import json
import logging
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from autofit.tools.util import compress_type_for, open_atomic

logger = logging.getLogger(__name__)

SCHEMA = 1
TABLES = "tables.zip"
MANIFEST = "parts_manifest.csv"
PACKAGE_JSON = "package.json"
SUMS = "SHA256SUMS"

PathLike = Union[str, os.PathLike]

# (arcname, source path)
Member = Tuple[str, Path]


class PackageException(Exception):
    pass


@dataclass
class PackageReport:
    dist: Path
    parts: Dict[str, List[str]]
    """Each part zip's objects."""
    written: List[str] = field(default_factory=list)
    """The zips written by this run."""
    kept: List[str] = field(default_factory=list)
    """The zips left untouched because their files did not change."""

    def __str__(self):
        return (
            f"package {self.dist}: {sum(len(v) for v in self.parts.values())} objects in "
            f"{len(self.parts)} parts; wrote {len(self.written)} zips, kept "
            f"{len(self.kept)} unchanged"
        )


def part_name(index: int) -> str:
    return f"part_{index:03d}.zip"


def sha256(path: PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints"}


def _files(directory: Path) -> List[Path]:
    # Dot-files inside an object ship (a result's .completed and .identifier are part
    # of it); only version-control and cache folders are left out.
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and not any(part in _SKIP_DIRS for part in path.relative_to(directory).parts)
    )


def _stat_entries(members: Sequence[Member]) -> List[List]:
    entries = []
    for arcname, path in members:
        stat = path.stat()
        entries.append([arcname, stat.st_size, stat.st_mtime_ns])
    return entries


def _date_time(path: Path) -> Tuple[int, ...]:
    # UTC, so the same files give the same zip on any machine.
    stamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    if stamp.year < 1980:
        return (1980, 1, 1, 0, 0, 0)
    return stamp.timetuple()[:6]


csv.field_size_limit(2**31 - 1)


def csv_records(handle) -> List[List[str]]:
    """
    The non-blank records of a CSV (bytes ``handle``); undecodable bytes are replaced
    rather than fatal, so one odd table never stops a package or a verify.
    """
    reader = csv.reader(
        io.TextIOWrapper(handle, encoding="utf-8", errors="replace", newline="")
    )
    return [row for row in reader if row]


def _count_csv_rows(handle) -> Optional[int]:
    try:
        return max(len(csv_records(handle)) - 1, 0)
    except csv.Error as exception:
        logger.warning(f"package: a CSV cannot be counted: {exception}")
        return None


def write_zip(zip_path: Path, members: Sequence[Member]) -> Dict[str, int]:
    """
    Write ``members`` to ``zip_path`` atomically and deterministically (sorted, each
    member with its file's modification time). Returns the row count of every CSV
    member.
    """
    tmp = zip_path.with_name(f"{zip_path.name}.{os.getpid()}.tmp")
    csv_rows = {}
    try:
        with zipfile.ZipFile(tmp, "w") as archive:
            for arcname, path in sorted(members):
                info = zipfile.ZipInfo(arcname, date_time=_date_time(path))
                info.compress_type = compress_type_for(path)
                info.external_attr = 0o644 << 16
                with open(path, "rb") as source, archive.open(info, "w", force_zip64=True) as target:
                    shutil.copyfileobj(source, target, 1024 * 1024)
                if arcname.endswith(".csv"):
                    with open(path, "rb") as handle:
                        csv_rows[arcname] = _count_csv_rows(handle)
        os.replace(tmp, zip_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return csv_rows


def _plan(
    items: Sequence[str],
    sizes: Dict[str, int],
    cap_bytes: Optional[int],
    n_parts: Optional[int],
    previous: Dict[str, str],
) -> Dict[str, str]:
    """
    Object → part name: previous assignments kept; new objects appended to the last
    part while it fits ``cap_bytes``, then to new parts. Without a previous plan,
    ``n_parts`` splits the objects into that many contiguous near-equal parts.
    """
    if not previous and n_parts:
        n = min(n_parts, len(items))
        total = sum(sizes[item] for item in items) or 1
        assignment = {}
        index, filled, in_part = 1, 0, 0
        for position, item in enumerate(items):
            parts_left = n - index
            items_left = len(items) - position
            if in_part and parts_left and (
                filled >= total * index / n or items_left <= parts_left
            ):
                index += 1
                in_part = 0
            assignment[item] = part_name(index)
            filled += sizes[item]
            in_part += 1
        return assignment

    assignment = {item: previous[item] for item in items if item in previous}
    part_sizes = {}
    for item, part in assignment.items():
        part_sizes[part] = part_sizes.get(part, 0) + sizes[item]
    index = max((int(part[5:8]) for part in part_sizes), default=0)
    if index == 0:
        index = 1
        part_sizes[part_name(1)] = 0
    for item in items:
        if item in assignment:
            continue
        name = part_name(index)
        if (
            cap_bytes is not None
            and part_sizes.get(name, 0) > 0
            and part_sizes.get(name, 0) + sizes[item] > cap_bytes
        ):
            index += 1
            name = part_name(index)
        assignment[item] = name
        part_sizes[name] = part_sizes.get(name, 0) + sizes[item]
    return assignment


def _git_revision(repo: Path) -> Dict[str, Optional[str]]:
    import subprocess

    def run(*args):
        try:
            return subprocess.run(
                ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "repo": str(repo),
        "revision": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def index_hash(index) -> str:
    """
    A digest of a results index (``autofit.results.ResultsIndex``): its records'
    paths, states and completion times, so a package names exactly which results it
    was built from.
    """
    digest = hashlib.sha256()
    for record in index.records():
        digest.update(
            f"{record.output_path}|{record.state}|{record.completed_at}\n".encode()
        )
    return digest.hexdigest()


def package(
    source: PathLike,
    dist: PathLike,
    cap_gb: Optional[float] = None,
    n_parts: Optional[int] = None,
    include_dataset: Optional[Callable[[str], Iterable[PathLike]]] = None,
    title: Optional[str] = None,
    repos: Sequence[PathLike] = (),
    index=None,
    notes: str = "",
    replan: bool = False,
    force: bool = False,
) -> PackageReport:
    """
    Package the library folder ``source`` into ``dist`` (see the module docstring).

    Parameters
    ----------
    cap_gb
        The largest a part may grow when objects are added to it (an object larger
        than the cap gets a part of its own).
    n_parts
        Split into exactly this many near-equal parts (first build or ``replan``).
    include_dataset
        ``include_dataset(object_name)`` returns extra files shipped with that object
        under ``<object>/dataset/``.
    title
        The README heading (default: the previous build's, else the source folder's
        name).
    repos
        Git checkouts whose revisions the README and ``package.json`` record.
    index
        The ``ResultsIndex`` the library was built from; its hash is recorded.
    notes
        Free text for the README (default: the previous build's).
    replan
        Ignore the previous package's part assignments.
    force
        Rewrite every zip, even those whose files look unchanged (a file rewritten
        with its size and modification time preserved is otherwise not noticed).
    """
    source = Path(source)
    dist = Path(dist)
    if not source.is_dir():
        raise PackageException(f"{source} is not a folder.")
    if cap_gb is not None and n_parts is not None:
        raise PackageException("Pass cap_gb or n_parts, not both.")
    dist.mkdir(parents=True, exist_ok=True)
    root = source.name

    previous = {}
    previous_document = {}
    if not replan and (dist / PACKAGE_JSON).exists():
        previous_document = json.loads((dist / PACKAGE_JSON).read_text())
        if previous_document.get("schema") == SCHEMA and previous_document.get("root") == root:
            previous = dict(previous_document.get("objects") or {})
        else:
            previous_document = {}
    if previous:
        if cap_gb is None and n_parts is None:
            cap_gb = previous_document.get("cap_gb")
        n_previous = len(set(previous.values()))
        if n_parts is not None and n_parts != n_previous:
            raise PackageException(
                f"{dist} already holds {n_previous} parts; a rebuild keeps them. Pass "
                f"replan=True (--replan) to split into {n_parts} parts instead."
            )

    items = sorted(
        path.name
        for path in source.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    members_of = {}
    for item in items:
        members = [
            (f"{root}/{path.relative_to(source).as_posix()}", path)
            for path in _files(source / item)
        ]
        if include_dataset is not None:
            for extra in include_dataset(item) or ():
                extra = Path(extra)
                members.append((f"{root}/{item}/dataset/{extra.name}", extra))
        if not members:
            logger.warning(f"package: {item} holds no files; it is left out.")
            continue
        names = [arcname for arcname, _ in members]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise PackageException(
                f"{item}: several files would be archived as {duplicates[:3]} "
                f"(an include_dataset file colliding with another file?)."
            )
        members_of[item] = members
    items = [item for item in items if item in members_of]
    sizes = {
        item: sum(path.stat().st_size for _, path in members_of[item]) for item in items
    }

    cap_bytes = int(cap_gb * 1024**3) if cap_gb is not None else None
    assignment = _plan(items, sizes, cap_bytes, n_parts, previous)
    parts: Dict[str, List[str]] = {}
    for item in items:
        parts.setdefault(assignment[item], []).append(item)

    zips = {
        name: [member for item in parts[name] for member in members_of[item]]
        for name in sorted(parts)
    }
    tables = [
        (f"{root}/{path.name}", path)
        for path in sorted(source.iterdir())
        if path.is_file() and not path.name.startswith(".")
    ]
    zips[TABLES] = tables

    report = PackageReport(dist=dist, parts={name: parts[name] for name in sorted(parts)})
    old_zips = previous_document.get("zips") or {}
    zip_documents = {}
    for name, members in zips.items():
        entries = _stat_entries(members)
        old = old_zips.get(name)
        zip_path = dist / name
        if (
            not force
            and old is not None
            and old.get("members") == entries
            and zip_path.exists()
            and zip_path.stat().st_size == old.get("bytes")
        ):
            zip_documents[name] = old
            report.kept.append(name)
            continue
        csv_rows = write_zip(zip_path, members)
        zip_documents[name] = {
            "members": entries,
            "csv_rows": csv_rows,
            "bytes": zip_path.stat().st_size,
            "sha256": sha256(zip_path),
        }
        report.written.append(name)
        logger.info(f"package: wrote {name} ({len(members)} files)")

    # Every part zip not in this build goes — including one left by a replan, a renamed
    # source or an interrupted run, which no package.json names.
    for stale in sorted(set(old_zips) | {p.name for p in dist.glob("part_*.zip")}):
        if stale not in zips:
            (dist / stale).unlink(missing_ok=True)
            logger.info(f"package: removed stale {stale}")

    provenance = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(source.resolve()),
        "repos": [_git_revision(Path(repo)) for repo in repos],
        "index_hash": index_hash(index) if index is not None else None,
        "autofit_version": _autofit_version(),
    }
    title = title or previous_document.get("title") or root
    notes = notes or previous_document.get("notes") or ""
    document = {
        "schema": SCHEMA,
        "root": root,
        "title": title,
        "notes": notes,
        "cap_gb": cap_gb,
        "objects": {item: assignment[item] for item in items},
        "zips": zip_documents,
        "provenance": provenance,
    }

    with open_atomic(dist / MANIFEST, "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["object", "part_zip", "n_files", "bytes"])
        for item in items:
            writer.writerow([item, assignment[item], len(members_of[item]), sizes[item]])
    with open_atomic(dist / PACKAGE_JSON, "w") as f:
        json.dump(document, f, indent=1)

    from autofit.export.readme import write_readme

    write_readme(dist, document, title=title, notes=notes)

    with open_atomic(dist / SUMS, "w") as f:
        for name in sorted(zip_documents):
            f.write(f"{zip_documents[name]['sha256']}  {name}\n")
        for name in (MANIFEST, PACKAGE_JSON):
            f.write(f"{sha256(dist / name)}  {name}\n")

    logger.info(str(report))
    return report


def _autofit_version() -> Optional[str]:
    try:
        import autofit

        return getattr(autofit, "__version__", None)
    except Exception:
        return None
