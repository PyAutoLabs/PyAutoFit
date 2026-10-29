"""
The files of a fit's status record, all siblings of its ``output_path`` so they survive
``output.remove_files`` (which deletes ``output_path`` and keeps only the ``.zip``):

- ``<output_path>.status.json``: the current status (``schema`` 1), at most
  ``MAX_BYTES``, replaced atomically on every write;
- ``<output_path>.status.jsonl``: the append-only history, one line per attempt start
  and per state change (never per cadence write), from which the attempt number is
  counted;
- ``<output_path>.traceback.txt``: the traceback of the last failed attempt.
"""
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

from autofit.non_linear.status.states import Product
from autofit.tools.util import open_atomic

SCHEMA = 1
MAX_BYTES = 2048

# Optional fields dropped, in this order, should a status document exceed `MAX_BYTES`
# (e.g. a very long `path_prefix`): the least useful to a collector goes first.
_TRIM_ORDER = (
    ("progress", ("n_eff", "f_live", "n_live", "log_z", "max_log_likelihood")),
    ("resources", ("gpu", "cpu_model")),
    ("outputs", ("dataset_identity", "traceback")),
    ("job", ("node", "time_limit_s")),
)


def _dumps(document: Dict) -> str:
    return json.dumps(document, separators=(",", ":"), default=str)


def compact(document: Dict, max_bytes: int = MAX_BYTES) -> str:
    """
    ``document`` as compact JSON of at most ``max_bytes``, dropping optional fields
    (``_TRIM_ORDER``) until it fits, then shortening ``path_prefix`` if it still
    does not.
    """
    text = _dumps(document)
    for section, keys in _TRIM_ORDER:
        if len(text.encode()) <= max_bytes:
            break
        value = document.get(section)
        if isinstance(value, dict):
            for key in keys:
                value.pop(key, None)
        text = _dumps(document)
    # Still too large: only a pathological `path_prefix` gets here. Keep its tail (the
    # most specific part), marked as cut.
    excess = len(text.encode()) - max_bytes
    if excess > 0 and isinstance(document.get("path_prefix"), str):
        document["path_prefix"] = "..." + document["path_prefix"][excess + 3 :]
        text = _dumps(document)
    return text


class StatusWriter:
    def __init__(self, output_path):
        """
        Reads and writes the status record of the fit whose output folder is
        ``output_path`` (the record's files sit beside that folder, not in it).

        Writes are serialised by a lock: the cadence thread and the main thread (an
        update, a signal, completion) may both write. A signal handler runs in the main
        thread, possibly while that thread holds the lock, so it passes a ``timeout``
        and a write that cannot take the lock in time is skipped (returns ``None``)
        rather than deadlocking.
        """
        output_path = str(output_path)
        self.status_path = Path(f"{output_path}{Product.STATUS}")
        self.history_path = Path(f"{output_path}{Product.HISTORY}")
        self.traceback_path = Path(f"{output_path}{Product.TRACEBACK}")
        self._lock = threading.Lock()

    @contextmanager
    def _locked(self, timeout: Optional[float]):
        acquired = self._lock.acquire(timeout=-1 if timeout is None else timeout)
        try:
            yield acquired
        finally:
            if acquired:
                self._lock.release()

    def write(self, document: Dict, timeout: Optional[float] = None) -> Optional[str]:
        """
        Atomically replace ``status.json`` with ``document`` (compacted to
        ``MAX_BYTES``), returning the text written (``None`` if the lock timed out).
        """
        text = compact(document)
        with self._locked(timeout) as acquired:
            if not acquired:
                return None
            with open_atomic(self.status_path, "w") as f:
                f.write(text)
        return text

    def read(self) -> Optional[Dict]:
        try:
            return json.loads(self.status_path.read_text())
        except (OSError, ValueError):
            return None

    def append(self, entry: Dict, timeout: Optional[float] = None) -> bool:
        """
        Append one line to the ``status.jsonl`` history (``False`` if the lock timed
        out).
        """
        with self._locked(timeout) as acquired:
            if not acquired:
                return False
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_path, "a") as f:
                f.write(_dumps(entry) + "\n")
        return True

    def history(self) -> List[Dict]:
        """
        The history's entries, skipping a line a killed process left truncated.
        """
        entries = []
        try:
            with open(self.history_path) as f:
                for line in f:
                    try:
                        entries.append(json.loads(line))
                    except ValueError:
                        pass
        except OSError:
            pass
        return entries

    def next_attempt(self) -> int:
        """
        The number of the attempt about to start: one more than the attempts the
        history records.
        """
        return 1 + sum(1 for entry in self.history() if entry.get("event") == "start")

    def write_traceback(self, text: str) -> Path:
        with self._locked(None):
            with open_atomic(self.traceback_path, "w") as f:
                f.write(text)
        return self.traceback_path
