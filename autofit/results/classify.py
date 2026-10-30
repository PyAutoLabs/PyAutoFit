"""
Classify one result on disk into a ``ResultState`` and read what the index records about
it, opening nothing but the result's own small files.

The order of evidence:

1. a readable zip with a ``.completed`` member (and, with ``verify``, a clean CRC check)
   or a folder with a ``.completed`` file is ``completed``; an unreadable zip is
   ``corrupt_zip``;
2. otherwise the status record: ``failed`` is ``failed``; ``running`` / ``pending`` /
   ``timeout_pending`` written within ``stale_after_s`` is ``running``;
3. otherwise the checkpoint under ``files/search_internal/``: readable is
   ``interrupted_valid_ckpt``, unreadable (or quarantined to
   ``search_internal.failed_<utc>/`` with nothing since) is ``corrupt_ckpt``;
4. otherwise ``never_started`` (a folder with no checkpoint: a rerun starts from
   scratch).

``superseded`` is assigned afterwards, across results (``ResultsIndex``).
"""
import json
import os
import zipfile
import zlib
from datetime import datetime, timezone
from hashlib import md5
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional

from autofit.non_linear.status.states import Product, ResultState
from autofit.results.record import ResultRecord

ZIP_CONTENT_ERRORS = (zipfile.BadZipFile, EOFError, zlib.error)
"""
What a damaged zip raises. An ``OSError`` is not among them: reading a healthy zip
over a network or WSL filesystem can fail transiently (``ENOMEM``, ``EIO``), and such
a result is not corrupt but unreadable this time (``UnreadableResult``).
"""


class UnreadableResult(OSError):
    """
    A result could not be read because of an I/O error that persisted after a retry;
    its state is unknown, not corrupt.
    """

RUNNING_STATES = ("running", "pending", "timeout_pending")

STATUS_SUMMARY_KEYS = ("state", "attempt", "failure_class", "updated", "started", "job")


def _iso_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds")


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def stat_fingerprint(output_path: Path) -> List:
    """
    The (size, mtime) of each file a result's record depends on: the zip, the status
    record, the folder's ``.completed`` and its ``files/`` and ``search_internal/``
    folders. A result whose fingerprint is unchanged is not classified again.
    """
    output_path = Path(output_path)
    fingerprint = []
    for path in (
        Path(f"{output_path}{Product.ZIP}"),
        Path(f"{output_path}{Product.STATUS}"),
        output_path / str(Product.COMPLETED),
        output_path / "files",
        output_path / "files" / "search_internal",
    ):
        try:
            stat = os.stat(path)
        except OSError:
            fingerprint.append(None)
        else:
            fingerprint.append([stat.st_size, stat.st_mtime_ns])
    return fingerprint


class _Contents:
    """
    The member names of a result (zip members, or files under the folder) and a way to
    read the small ones.
    """

    def __init__(self, names: Iterable[str], read):
        self.names = set(names)
        self._read = read

    def has(self, name: str) -> bool:
        return name in self.names

    def json(self, name: str) -> Optional[Dict]:
        if name not in self.names:
            return None
        try:
            return json.loads(self._read(name))
        except Exception:
            return None

    def text(self, name: str) -> Optional[str]:
        if name not in self.names:
            return None
        try:
            return self._read(name).decode("utf-8")
        except Exception:
            return None


def _folder_contents(directory: Path) -> _Contents:
    names = []
    for root, dirs, filenames in os.walk(directory):
        relative = Path(root).relative_to(directory)
        # Checkpoints are probed separately; their contents are not products.
        dirs[:] = [d for d in dirs if not d.startswith("search_internal")]
        for filename in filenames:
            names.append(str(PurePosixPath(*relative.parts, filename)))

    def read(name):
        return (directory / name).read_bytes()

    return _Contents(names, read)


def _identifier_of(description: Optional[str]) -> Optional[str]:
    """
    The identifier recorded by ``.identifier`` (its md5, as ``SearchOutput.id``).
    """
    if not description:
        return None
    return md5(".".join(description.split("\n")).encode("utf-8")).hexdigest()


def _path_of(value) -> Optional[str]:
    if isinstance(value, dict) and "path" in value:
        return str(value["path"])
    if value is None:
        return None
    return str(value)


def _read_metadata(record: ResultRecord, contents: _Contents, output_path: Path):
    search = contents.json("files/search.json") or {}
    arguments = search.get("arguments") or {}
    paths = (arguments.get("paths") or {}).get("arguments") or {}

    record.name = arguments.get("name") or paths.get("name")
    record.unique_tag = arguments.get("unique_tag") or paths.get("unique_tag")
    record.path_prefix = _path_of(
        arguments.get("path_prefix", paths.get("path_prefix"))
    )
    class_path = search.get("class_path")
    record.search = class_path.rsplit(".", 1)[-1] if class_path else None

    record.identifier = _identifier_of(contents.text(".identifier")) or (
        output_path.name if len(output_path.name) == 32 else None
    )

    # A result whose declared name, or the last segment of its declared path_prefix
    # (usually the object), is not where it sits was copied or moved there: the fit
    # it holds belongs elsewhere. Only the last segment is compared because whole
    # trees are routinely moved under another root.
    parts = output_path.parts
    prefix_parts = PurePosixPath(record.path_prefix or "").parts
    record.location_mismatch = bool(
        (record.name is not None and parts[-2:-1] != (record.name,))
        or (prefix_parts and prefix_parts[-1] not in parts[:-2])
    )

    names = contents.names
    record.files = sorted(
        name[len("files/") :]
        for name in names
        if name.startswith("files/") and not name.startswith("files/search_internal")
    )
    record.n_images = sum(1 for name in names if name.startswith("image/"))
    record.products = {
        "samples": "files/samples.csv" in names,
        "samples_summary": "files/samples_summary.json" in names,
        "latent": any(
            name.startswith("files/latent") or name == "latent.results"
            for name in names
        ),
        "image": record.n_images > 0,
        "dataset_identity": f"files/{Product.DATASET_IDENTITY}" in names,
    }

    identity = contents.json(f"files/{Product.DATASET_IDENTITY}")
    if identity:
        record.object_id = identity.get("object_id")
        record.inputs = {
            name: entry.get("sha256")
            for name, entry in (identity.get("inputs") or {}).items()
            if isinstance(entry, dict)
        }
        record.versions = dict(identity.get("versions") or {})


def _probe_search_internal(files_path: Path) -> Optional[ResultState]:
    """
    ``interrupted_valid_ckpt`` / ``corrupt_ckpt`` from the checkpoint under
    ``files_path``, or ``None`` when there is no checkpoint.
    """
    directory = files_path / "search_internal"
    entries = (
        [p for p in directory.rglob("*") if p.is_file()] if directory.is_dir() else []
    )
    if entries:
        for path in entries:
            if path.stat().st_size == 0:
                return ResultState.CORRUPT_CKPT
            if path.suffix in (".hdf5", ".h5"):
                from autofit.non_linear.checkpoint import (
                    HDF5_PROBE_ERRORS,
                    probe_hdf5,
                )

                try:
                    probe_hdf5(path)
                except HDF5_PROBE_ERRORS:
                    return ResultState.CORRUPT_CKPT
                except ImportError:
                    pass
        return ResultState.INTERRUPTED_VALID_CKPT
    if files_path.is_dir() and any(
        p.name.startswith("search_internal.failed_") for p in files_path.iterdir()
    ):
        return ResultState.CORRUPT_CKPT
    return None


def _read_zip(zip_path: Path, verify: bool):
    """
    The member names, the small JSON / marker members and the ``.completed`` time of
    the zip at ``zip_path``; raises one of ``ZIP_CONTENT_ERRORS`` if it is damaged.
    """
    with zipfile.ZipFile(zip_path) as f:
        names = f.namelist()
        small = {}
        for name in names:
            if name.startswith("files/search_internal"):
                continue
            if f.getinfo(name).file_size <= 1 << 20 and (
                name in (".identifier", ".completed") or name.endswith(".json")
            ):
                small[name] = f.read(name)
        completed_at = None
        if ".completed" in names:
            if verify and f.testzip() is not None:
                raise zipfile.BadZipFile("a member fails its CRC check")
            completed_at = datetime(*f.getinfo(".completed").date_time).isoformat()
    return names, small, completed_at


def _read_zip_with_retry(zip_path: Path, verify: bool, attempts: int = 2):
    for attempt in range(attempts):
        try:
            return _read_zip(zip_path, verify)
        except ZIP_CONTENT_ERRORS:
            raise
        except OSError as error:
            if attempt == attempts - 1:
                raise UnreadableResult(
                    f"{zip_path}: {type(error).__name__}: {error}"
                ) from error


def classify(
    output_path,
    root,
    verify: bool = True,
    stale_after_s: float = 600.0,
    now: Optional[datetime] = None,
) -> ResultRecord:
    """
    The ``ResultRecord`` of the result at ``output_path`` (the folder path, without
    ``.zip``), found under ``root``.

    Parameters
    ----------
    verify
        Check every zip member's CRC (``ZipFile.testzip``), which reads the whole zip.
        Without it a zip with a ``.completed`` member counts as completed unread.
    stale_after_s
        A status record saying ``running`` that was last written longer ago than this
        is not trusted: the process is assumed gone and the checkpoint decides.
    """
    output_path = Path(output_path)
    zip_path = Path(f"{output_path}{Product.ZIP}")
    status_path = Path(f"{output_path}{Product.STATUS}")
    now = now or datetime.now(timezone.utc)

    record = ResultRecord(
        output_path=str(output_path),
        root=str(root),
        state=str(ResultState.NEVER_STARTED),
        has_zip=zip_path.is_file(),
        has_dir=output_path.is_dir(),
        has_status=status_path.is_file(),
        stat_key=stat_fingerprint(output_path),
    )

    status = None
    if record.has_status:
        try:
            with open(status_path) as f:
                status = json.load(f)
        except (OSError, ValueError):
            status = None
        if status is not None:
            record.status = {k: status.get(k) for k in STATUS_SUMMARY_KEYS}

    state = None
    contents = None

    if record.has_zip:
        record.bytes = zip_path.stat().st_size
        try:
            names, small, completed_at = _read_zip_with_retry(zip_path, verify)
            if completed_at is not None:
                state = ResultState.COMPLETED
                record.completed_at = completed_at
                record.completed_at_source = "zip_member"
            contents = _Contents(names, small.__getitem__)
        except ZIP_CONTENT_ERRORS as error:
            state = ResultState.CORRUPT_ZIP
            record.reason = f"{type(error).__name__}: {error}"
        if state is None and contents is not None and not record.has_dir:
            state = ResultState.CORRUPT_ZIP
            record.reason = "zip has no .completed member"

    if state is None and record.has_dir:
        completed = output_path / str(Product.COMPLETED)
        if completed.exists():
            state = ResultState.COMPLETED
            record.completed_at = _iso_utc(completed.stat().st_mtime)
            record.completed_at_source = "completed_file"
        contents = _folder_contents(output_path)
    elif state == ResultState.CORRUPT_ZIP and record.has_dir:
        # The folder beside an unreadable zip may still be whole.
        if (output_path / str(Product.COMPLETED)).exists():
            state = ResultState.COMPLETED
            record.reason = "zip unreadable; the folder beside it is complete"
            record.completed_at = _iso_utc(
                (output_path / str(Product.COMPLETED)).stat().st_mtime
            )
            record.completed_at_source = "completed_file"
            contents = _folder_contents(output_path)

    if state == ResultState.COMPLETED and status and status.get("state") == "completed":
        record.completed_at = status.get("updated") or record.completed_at
        record.completed_at_source = "status"

    if state is None and status is not None:
        if status.get("state") == "failed":
            state = ResultState.FAILED
            record.reason = status.get("failure_class")
        elif status.get("state") in RUNNING_STATES:
            updated = _parse_time(status.get("updated"))
            if updated is not None and (now - updated).total_seconds() <= stale_after_s:
                state = ResultState.RUNNING
            else:
                record.reason = (
                    f"status says {status.get('state')} but was last written "
                    f"{status.get('updated')}"
                )

    if state is None and record.has_dir:
        probed = _probe_search_internal(output_path / "files")
        if probed is not None:
            state = probed
        else:
            state = ResultState.NEVER_STARTED
            record.reason = record.reason or "output folder with no checkpoint"

    if state is None:
        state = ResultState.NEVER_STARTED
        record.reason = record.reason or "status record only"

    record.state = str(state)
    if contents is not None:
        _read_metadata(record, contents, output_path)
    record.products["search_internal"] = (
        output_path / "files" / "search_internal"
    ).is_dir()
    return record
