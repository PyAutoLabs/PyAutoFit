"""
``RowCache``: the rows of a previous export, so a refresh only reads the results that
are new or changed.

The cache sits beside the CSV (``<csv>.rows.jsonl``): a header line with the column
specification it was built for, then one line per result — its ``output_path``, the
``row_fingerprint`` it had and its row, as the text the CSV holds. A row is reused
only while the result's fingerprint and the whole column specification are unchanged;
a different specification (a column added, renamed, or a computed column's code edited)
discards the cache. Label columns are never cached — they are cheap, and change often.

Rows of results that failed are not cached, so they are retried. The cache is written
atomically after the CSV: a run killed in between leaves an older cache, which only
costs re-reading results next time.
"""
import functools
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from autofit.tools.util import NumpyEncoder, open_atomic

logger = logging.getLogger(__name__)

SCHEMA = 1


def _stable_repr(value) -> str:
    """
    A representation of ``value`` that is the same in every process for the same
    value: sets are sorted (their order follows hash randomisation). A value whose
    ``repr`` holds a memory address stays unstable, so the cache never reuses rows
    computed with it — slower, never stale.
    """
    if isinstance(value, (set, frozenset)):
        return f"{type(value).__name__}({sorted(_stable_repr(v) for v in value)})"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}({[_stable_repr(v) for v in value]})"
    if isinstance(value, dict):
        return f"dict({sorted((_stable_repr(k), _stable_repr(v)) for k, v in value.items())})"
    return repr(value)


def _code_digest(code, digest) -> None:
    digest.update(code.co_code)
    for const in code.co_consts:
        if hasattr(const, "co_code"):
            _code_digest(const, digest)
        else:
            digest.update(_stable_repr(const).encode("utf-8"))
    digest.update(repr(code.co_names).encode("utf-8"))


def _function_digest(function, digest, seen) -> None:
    if id(function) in seen:
        return
    seen.add(id(function))

    if isinstance(function, functools.partial):
        digest.update(b"partial")
        _function_digest(function.func, digest, seen)
        digest.update(_stable_repr(function.args).encode("utf-8"))
        digest.update(_stable_repr(function.keywords).encode("utf-8"))
        return

    code = getattr(function, "__code__", None)
    if code is None:
        digest.update(repr(function).encode("utf-8"))
        return

    digest.update(
        f"{getattr(function, '__module__', '')}.{getattr(function, '__qualname__', '')}".encode()
    )
    _code_digest(code, digest)
    for value in (function.__defaults__ or ()) + tuple(
        (function.__kwdefaults__ or {}).items()
    ):
        digest.update(_stable_repr(value).encode("utf-8"))
    for cell in function.__closure__ or ():
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if callable(value) and hasattr(value, "__code__"):
            _function_digest(value, digest, seen)
        else:
            digest.update(_stable_repr(value).encode("utf-8"))
    # Functions the column calls by global name, from its own module (an edited
    # helper must invalidate the rows too).
    module = getattr(function, "__module__", None)
    for name in code.co_names:
        value = function.__globals__.get(name)
        if inspect.isfunction(value) and value.__module__ == module:
            _function_digest(value, digest, seen)


def function_digest(function) -> str:
    """
    A digest of what ``function`` computes, so editing a computed column's function
    (even a lambda), a value it captured, a default, or a helper function it calls
    from its module invalidates the rows it produced.
    """
    digest = hashlib.sha1()
    _function_digest(function, digest, set())
    return digest.hexdigest()


def row_fingerprint(output_path: Path) -> List:
    """
    What a cached row of the result at ``output_path`` is valid for: the zip's size and
    modification time, or for a folder every file a row can read — everything under
    ``files/`` but the search's internal checkpoint, plus ``.identifier`` and
    ``.completed`` — since rewriting a file in place does not touch its folder.
    """
    zip_path = Path(f"{output_path}.zip")
    entries = []
    if zip_path.exists():
        stat = zip_path.stat()
        entries.append(["zip", stat.st_size, stat.st_mtime_ns])
    if output_path.is_dir():
        for name in (".identifier", ".completed"):
            path = output_path / name
            if path.exists():
                stat = path.stat()
                entries.append([name, stat.st_size, stat.st_mtime_ns])
        files = output_path / "files"
        for root, dirs, names in os.walk(files):
            dirs[:] = sorted(d for d in dirs if not d.startswith("search_internal"))
            for name in sorted(names):
                path = Path(root) / name
                stat = path.stat()
                entries.append(
                    [str(path.relative_to(output_path)), stat.st_size, stat.st_mtime_ns]
                )
    return entries


def csv_text(value) -> str:
    """
    The text ``csv`` writes for ``value``; cached rows hold this, so a row read back
    from the cache is written exactly as a freshly computed one.
    """
    return "" if value is None else str(value)


def spec_digest(parts: Iterable) -> str:
    return hashlib.sha1(
        json.dumps(list(parts), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def cache_path(csv_path: os.PathLike) -> Path:
    return Path(f"{csv_path}.rows.jsonl")


class RowCache:
    def __init__(self, csv_path: os.PathLike, spec: str):
        """
        The row cache of the CSV at ``csv_path`` for the column specification ``spec``
        (``spec_digest``).
        """
        self.path = cache_path(csv_path)
        self.spec = spec
        self.meta: Dict = {}
        self._rows: Dict[str, Tuple[List, Dict]] = {}
        self._new: Dict[str, Tuple[List, Dict]] = {}

    def load(self) -> "RowCache":
        try:
            with open(self.path) as f:
                header = json.loads(f.readline() or "{}")
                if header.get("schema") != SCHEMA or header.get("spec") != self.spec:
                    return self
                self.meta = header.get("meta") or {}
                for line in f:
                    entry = json.loads(line)
                    self._rows[entry["output_path"]] = (
                        entry["fingerprint"],
                        entry["row"],
                    )
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError) as exception:
            logger.warning(f"Ignoring unreadable row cache {self.path}: {exception}")
            self._rows = {}
        return self

    def get(self, output_path: str, fingerprint: List) -> Optional[Dict]:
        cached = self._rows.get(output_path)
        if cached is not None and cached[0] == fingerprint:
            return dict(cached[1])
        return None

    def put(self, output_path: str, fingerprint: List, row: Dict):
        self._new[output_path] = (fingerprint, row)

    def save(self, keep: Iterable[str] = ()):
        """
        Write the rows put in this run, plus the cached rows of ``keep`` (results of
        other shards, which this run did not read).
        """
        rows = {path: self._rows[path] for path in keep if path in self._rows}
        rows.update(self._new)
        with open_atomic(self.path, "w") as f:
            f.write(
                json.dumps({"schema": SCHEMA, "spec": self.spec, "meta": self.meta})
                + "\n"
            )
            for output_path in sorted(rows):
                fingerprint, row = rows[output_path]
                f.write(
                    json.dumps(
                        {
                            "output_path": output_path,
                            "fingerprint": fingerprint,
                            "row": row,
                        },
                        cls=NumpyEncoder,
                    )
                    + "\n"
                )

    def cached_paths(self) -> List[str]:
        return list(self._rows)
