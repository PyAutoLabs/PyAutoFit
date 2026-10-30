"""
``ZipMember``: a path to a member (file or folder) inside a result zip, read in place
with nothing extracted to disk.

It supports the subset of ``pathlib.Path`` that ``SearchOutput`` and ``FileOutput``
use (``/``, ``exists``, ``is_file``, ``is_dir``, ``iterdir``, ``glob``, ``rglob``,
``open``, ``read_text``, ``read_bytes``, ``relative_to``, ``name``, ``suffix``,
``stem``), so a search output can be backed by a zip instead of a folder.

Unlike ``zipfile.Path``, a ``ZipMember`` holds no open file: every ``open`` opens the
zip, opens the member and closes the zip again (``zipfile`` keeps the underlying file
open until the returned stream is closed). Aggregating thousands of zipped results
therefore never runs into the open-file limit. The member names of each zip are read
once and cached while the zip's size and modification time are unchanged.
"""
import fnmatch
import io
import os
import zipfile
from pathlib import Path, PurePosixPath
from typing import Dict, Iterator, Tuple

_names_cache: Dict[str, Tuple[Tuple[int, int], Tuple[str, ...]]] = {}


def zip_names(zip_path) -> Tuple[str, ...]:
    """
    The member names of the zip at ``zip_path`` (files only, no folder entries),
    cached per path while its size and modification time are unchanged.
    """
    zip_path = str(zip_path)
    stat = os.stat(zip_path)
    key = (stat.st_size, stat.st_mtime_ns)
    cached = _names_cache.get(zip_path)
    if cached is not None and cached[0] == key:
        return cached[1]
    with zipfile.ZipFile(zip_path) as f:
        names = tuple(name for name in f.namelist() if not name.endswith("/"))
    _names_cache[zip_path] = (key, names)
    return names


class ZipMember:
    def __init__(self, zip_path, at: str = ""):
        """
        The member ``at`` (a POSIX path inside the zip, ``""`` for the zip's root) of
        the zip at ``zip_path``.
        """
        self.zip_path = Path(zip_path)
        self.at = str(PurePosixPath(at)) if at not in ("", ".") else ""

    @property
    def _names(self) -> Tuple[str, ...]:
        return zip_names(self.zip_path)

    def _child(self, at: str) -> "ZipMember":
        return ZipMember(self.zip_path, at)

    def __truediv__(self, other) -> "ZipMember":
        other = str(other)
        return self._child(f"{self.at}/{other}" if self.at else other)

    def __str__(self):
        return f"{self.zip_path}/{self.at}" if self.at else str(self.zip_path)

    def __repr__(self):
        return f"<ZipMember {self}>"

    def __eq__(self, other):
        return (
            isinstance(other, ZipMember)
            and self.zip_path == other.zip_path
            and self.at == other.at
        )

    def __hash__(self):
        return hash((self.zip_path, self.at))

    @property
    def name(self) -> str:
        return PurePosixPath(self.at).name

    @property
    def suffix(self) -> str:
        return PurePosixPath(self.at).suffix

    @property
    def stem(self) -> str:
        return PurePosixPath(self.at).stem

    @property
    def _prefix(self) -> str:
        return f"{self.at}/" if self.at else ""

    def is_file(self) -> bool:
        return bool(self.at) and self.at in self._names

    def is_dir(self) -> bool:
        prefix = self._prefix
        return any(name.startswith(prefix) for name in self._names)

    def exists(self) -> bool:
        return self.is_file() or self.is_dir()

    def _descendants(self) -> Iterator[str]:
        """
        Every file and folder below this member, as paths relative to it.
        """
        prefix = self._prefix
        seen = set()
        for name in self._names:
            if not name.startswith(prefix) or name == prefix:
                continue
            relative = PurePosixPath(name[len(prefix) :])
            parents = list(relative.parents)[:-1]
            for path in reversed(parents):
                if path not in seen:
                    seen.add(path)
                    yield str(path)
            yield str(relative)

    def iterdir(self) -> Iterator["ZipMember"]:
        for relative in self._descendants():
            if "/" not in relative:
                yield self / relative

    def glob(self, pattern: str) -> Iterator["ZipMember"]:
        depth = pattern.count("/")
        for relative in self._descendants():
            if relative.count("/") == depth and fnmatch.fnmatchcase(relative, pattern):
                yield self / relative

    def rglob(self, pattern: str) -> Iterator["ZipMember"]:
        for relative in self._descendants():
            if fnmatch.fnmatchcase(PurePosixPath(relative).name, pattern):
                yield self / relative

    def relative_to(self, other: "ZipMember") -> PurePosixPath:
        return PurePosixPath(self.at).relative_to(PurePosixPath(other.at or "."))

    def open(self, mode: str = "r", encoding: str = "utf-8"):
        """
        Open the member for reading (``"r"`` text, ``"rb"`` bytes). Raises
        ``FileNotFoundError`` if it is not a file in the zip.
        """
        if mode not in ("r", "rb"):
            raise ValueError(f"ZipMember is read-only; mode {mode!r} is not supported.")
        if not self.is_file():
            raise FileNotFoundError(str(self))
        with zipfile.ZipFile(self.zip_path) as f:
            stream = f.open(self.at)
        if mode == "rb":
            return stream
        return io.TextIOWrapper(stream, encoding=encoding)

    def read_bytes(self) -> bytes:
        with self.open("rb") as f:
            return f.read()

    def read_text(self, encoding: str = "utf-8") -> str:
        return self.read_bytes().decode(encoding)
