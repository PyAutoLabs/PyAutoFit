"""
Where a non-linear search keeps its internal state on disk, and for how long.

A search has two kinds of internal state (decision D5 of the search-extensibility
epic, ``docs/design/checkpointing.md``):

- ``result_internal``, the **archive**: the backend's final state, written when a
  fit completes and read back by ``Result.search_internal`` and by
  ``samples_via_internal_from`` on a completed output folder (the emcee HDF file,
  the NUTS and SMC dicts, the dill of every other search);
- ``resume_state``: the file an *interrupted* run resumes from (dynesty's
  ``savestate.save``, nautilus's ``checkpoint.hdf5``, NSS's ``nss_checkpoint.pkl``,
  MultiStart's dill). Only searches that genuinely resume have one.

Each search chooses its strategies with two class attributes,
``NonLinearSearch.checkpointer`` (the archive, ``DillCheckpointer()`` by default) and
``NonLinearSearch.resume_state`` (``None`` by default). Three strategies cover every
search:

- ``DillCheckpointer``: ``search_internal.dill`` (the default);
- ``PickleCheckpointer``: ``search_internal.pickle``, plain ``pickle`` of NumPy data
  (BlackJAX NUTS and SMC);
- ``NativeFileCheckpointer(filename, loader)``: a file the backend writes itself and
  only autofit reads (emcee's HDF backend, dynesty's savestate, nautilus's hdf5
  checkpoint, NSS's pickle).

Every strategy offers ``save``, ``load``, ``exists`` and ``finalize`` and declares
``retain_after_completion``: whether its file survives the end of a fit when the
``output.search_internal`` config is off. Strategies are stateless: they take the
search's ``paths`` on every call, so one class-level instance serves every copy of a
search (``copy_with_paths``) and is never serialised with it.

Only ``DirectoryPaths`` stores anything. ``NullPaths`` (no output) and
``DatabasePaths`` (results live in the database) make ``save`` a no-op and ``load``
return ``None``, as their ``save_search_internal`` / ``load_search_internal`` always
have.
"""
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DILL_FILENAME = "search_internal.dill"
PICKLE_FILENAME = "search_internal.pickle"
EMCEE_FILENAME = "search_internal.hdf"


def _directory(paths) -> Optional[Path]:
    """
    The ``search_internal`` folder of ``paths``, or ``None`` when ``paths`` stores no
    internal state (``NullPaths``, ``DatabasePaths``).
    """
    if paths is None or not getattr(paths, "stores_search_internal", True):
        return None
    try:
        directory = paths.search_internal_path
    except TypeError:
        return None
    if directory is None:
        return None
    return Path(directory)


def _atomic_dump(path: Path, obj, dump: Callable):
    from autofit.tools.util import open_atomic

    with open_atomic(path, "wb") as f:
        dump(obj, f)


def load_emcee_backend(path: Path):
    """
    Open an emcee ``HDFBackend`` file (the emcee archive and resume state).
    """
    import emcee

    return emcee.backends.HDFBackend(filename=str(path))


def _load_dill(path: Path):
    import dill

    with open(path, "rb") as f:
        return dill.load(f)


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_legacy_search_internal(directory) -> Any:
    """
    Load whatever internal state a ``search_internal`` folder holds, detecting its
    format from the files present.

    This is the format-aware detection ``DirectoryPaths.load_search_internal`` used
    to do inline (it probed for emcee's ``search_internal.hdf`` first), kept for
    callers that do not know which search wrote the folder and as the fallback for
    a folder written before its search chose a strategy. In order:

    - ``search_internal.hdf``, opened as an emcee ``HDFBackend`` (when emcee is
      installed);
    - ``search_internal.pickle`` (BlackJAX NUTS / SMC);
    - ``search_internal.dill`` (every other search).

    Raises
    ------
    FileNotFoundError
        If the folder holds none of them.
    """
    directory = Path(directory)

    hdf = directory / EMCEE_FILENAME
    if hdf.is_file():
        try:
            return load_emcee_backend(hdf)
        except ImportError:
            pass

    pickled = directory / PICKLE_FILENAME
    if pickled.is_file():
        return _load_pickle(pickled)

    return _load_dill(directory / DILL_FILENAME)


class Checkpointer:
    """
    A strategy for one internal-state file of a search (see the module docstring).

    Parameters
    ----------
    filename
        The file's name inside the search's ``search_internal`` folder.
    retain_after_completion
        Whether the file is kept when a fit completes even though the
        ``output.search_internal`` config is off. For an archive this replaces the
        old practice of a search forcing that config on in its ``__init__``; for a
        resume state it says the file is also the archive.
    """

    kind = "abstract"

    def __init__(self, filename: str, retain_after_completion: bool = False):
        self.filename = filename
        self.retain_after_completion = retain_after_completion

    def path(self, paths) -> Optional[Path]:
        """
        The file's full path, or ``None`` when ``paths`` stores no internal state.
        """
        directory = _directory(paths)
        if directory is None:
            return None
        return directory / self.filename

    def exists(self, paths) -> bool:
        path = self.path(paths)
        return path is not None and path.is_file()

    def save(self, paths, obj):
        """
        Write ``obj`` atomically (a sibling temporary file renamed into place), so an
        interrupted write leaves the previous file whole. A no-op when ``paths``
        stores nothing.
        """
        raise NotImplementedError

    def _load(self, path: Path):
        raise NotImplementedError

    def load(self, paths):
        """
        Read the file back.

        Returns ``None`` when ``paths`` stores nothing. When this strategy's own file
        is absent the folder is probed by ``load_legacy_search_internal``, so an
        output written before the search chose this strategy still loads.

        Raises
        ------
        FileNotFoundError
            If the folder holds no internal state at all.
        """
        path = self.path(paths)
        if path is None:
            return None
        if path.is_file():
            return self._load(path)
        return load_legacy_search_internal(path.parent)

    def finalize(self, paths, obj=None):
        """
        Write the final state when a fit completes. For an archive strategy that is
        ``save(obj)``; a strategy whose file the backend writes itself has nothing
        to add.
        """
        if obj is not None:
            self.save(paths, obj)

    def discard(self, paths):
        """
        Delete the file (a resume state, once the fit it would resume has completed).
        """
        path = self.path(paths)
        if path is None:
            return
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    def __repr__(self):
        return (
            f"{type(self).__name__}({self.filename!r}, "
            f"retain_after_completion={self.retain_after_completion})"
        )

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.filename == other.filename
            and self.retain_after_completion == other.retain_after_completion
        )

    def __hash__(self):
        return hash((type(self), self.filename, self.retain_after_completion))


class DillCheckpointer(Checkpointer):
    """
    ``search_internal.dill``: the backend object serialised with ``dill`` (the
    default for every search).

    Parameters
    ----------
    strip_attributes
        Attributes set to ``None`` for the duration of the dump and restored
        afterwards, for members that cannot be serialised (nautilus's
        multiprocessing pools ``pool_l`` / ``pool_s``).
    """

    kind = "dill"

    def __init__(
        self,
        filename: str = DILL_FILENAME,
        retain_after_completion: bool = False,
        strip_attributes=(),
    ):
        super().__init__(
            filename=filename, retain_after_completion=retain_after_completion
        )
        self.strip_attributes = tuple(strip_attributes)

    def save(self, paths, obj):
        path = self.path(paths)
        if path is None:
            return

        import dill

        stripped = {}
        for name in self.strip_attributes:
            if hasattr(obj, name):
                stripped[name] = getattr(obj, name)
                setattr(obj, name, None)
        try:
            _atomic_dump(path, obj, dill.dump)
        finally:
            for name, value in stripped.items():
                setattr(obj, name, value)

    def _load(self, path: Path):
        return _load_dill(path)


class PickleCheckpointer(Checkpointer):
    """
    ``search_internal.pickle``: plain ``pickle`` of already-NumPy data (BlackJAX NUTS
    and SMC, whose dicts ``dill`` does not need and which then load without it).
    """

    kind = "pickle"

    def __init__(
        self, filename: str = PICKLE_FILENAME, retain_after_completion: bool = True
    ):
        super().__init__(
            filename=filename, retain_after_completion=retain_after_completion
        )

    def save(self, paths, obj):
        path = self.path(paths)
        if path is None:
            return
        _atomic_dump(path, obj, pickle.dump)

    def _load(self, path: Path):
        return _load_pickle(path)


class NativeFileCheckpointer(Checkpointer):
    """
    A file in the backend's own format, which the backend writes itself (emcee's HDF
    backend, dynesty's ``savestate.save``, nautilus's ``checkpoint.hdf5``, NSS's
    ``nss_checkpoint.pkl``): autofit only locates, loads and (for a resume state)
    deletes it.

    Parameters
    ----------
    filename
        The file's name inside ``search_internal``.
    loader
        ``loader(path)`` opens the file (for example ``load_emcee_backend``).
    retain_after_completion
        See ``Checkpointer``.
    """

    kind = "native"

    def __init__(
        self,
        filename: str,
        loader: Callable[[Path], Any],
        retain_after_completion: bool = False,
    ):
        super().__init__(
            filename=filename, retain_after_completion=retain_after_completion
        )
        self.loader = loader

    def save(self, paths, obj):
        """
        A no-op: the backend writes this file itself, during the run.
        """

    def finalize(self, paths, obj=None):
        """
        A no-op: the backend has already written the final state.
        """

    def load(self, paths):
        """
        Open the file with ``loader``. A native file is the backend's own format, so
        there is no format probing; ``FileNotFoundError`` if it is absent, except
        that an absent archive falls back to the legacy probe like every strategy.
        """
        path = self.path(paths)
        if path is None:
            return None
        if path.is_file():
            return self.loader(path)
        if self.retain_after_completion:
            return load_legacy_search_internal(path.parent)
        raise FileNotFoundError(f"No {self.filename} exists at {path.parent}")

    def __eq__(self, other):
        return super().__eq__(other) and self.loader is other.loader

    def __hash__(self):
        return super().__hash__()


def checkpointer_kind(checkpointer: Optional[Checkpointer]) -> Optional[str]:
    """
    The manifest spelling of a strategy: ``"dill"``, ``"pickle"``,
    ``"native:<filename>"``, or ``None``.
    """
    if checkpointer is None:
        return None
    if checkpointer.kind == "native":
        return f"native:{checkpointer.filename}"
    return checkpointer.kind
