"""
``ResultsCache``: a local SQLite cache of a ``ResultsIndex`` for queries.

The cache is rebuilt from the index's JSON records whenever the index is newer than
it, so it can be deleted at any time. It lives in the user's cache directory
(``$XDG_CACHE_HOME/autofit/results``, default ``~/.cache/autofit/results``), never
beside the results: SQLite is not safe on network filesystems and the records are the
truth.
"""
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from autofit.results.index import ResultsIndex
from autofit.results.record import ResultRecord

COLUMNS = (
    "output_path",
    "root",
    "state",
    "reason",
    "name",
    "unique_tag",
    "path_prefix",
    "identifier",
    "search",
    "object_id",
    "completed_at",
    "location_mismatch",
)


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "autofit" / "results"


class ResultsCache:
    def __init__(self, index: ResultsIndex, cache_dir: Optional[os.PathLike] = None):
        self.index = index
        cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        name = hashlib.sha1(str(index.index_dir).encode("utf-8")).hexdigest()[:16]
        self.path = cache_dir / f"{name}.sqlite"

    def _stamp(self) -> Optional[str]:
        manifest = self.index.manifest()
        return None if manifest is None else manifest.get("updated")

    def connect(self) -> sqlite3.Connection:
        """
        A connection to the cache, rebuilt first if the index changed since it was
        built. The ``results`` table has one row per record (``COLUMNS`` plus the full
        record as JSON in ``record``).
        """
        stamp = self._stamp()
        if self.path.exists():
            connection = sqlite3.connect(self.path)
            try:
                (built,) = connection.execute(
                    "SELECT value FROM meta WHERE key = 'index_updated'"
                ).fetchone()
                if built == stamp:
                    return connection
            except (sqlite3.Error, TypeError):
                pass
            connection.close()
        return self.rebuild(stamp)

    def rebuild(self, stamp: Optional[str] = None) -> sqlite3.Connection:
        os.makedirs(self.path.parent, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        if tmp.exists():
            tmp.unlink()
        connection = sqlite3.connect(tmp)
        connection.execute(
            f"CREATE TABLE results ({', '.join(COLUMNS)}, record TEXT, "
            f"PRIMARY KEY (output_path))"
        )
        connection.execute("CREATE INDEX results_state ON results (state)")
        connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        connection.executemany(
            f"INSERT INTO results VALUES ({', '.join('?' * (len(COLUMNS) + 1))})",
            (
                tuple(getattr(record, column) for column in COLUMNS)
                + (json.dumps(record.to_dict()),)
                for record in self.index.load().values()
            ),
        )
        connection.execute(
            "INSERT INTO meta VALUES ('index_updated', ?)",
            (stamp if stamp is not None else self._stamp(),),
        )
        connection.commit()
        connection.close()
        os.replace(tmp, self.path)
        return sqlite3.connect(self.path)

    def query(self, where: str = "1", parameters=()) -> List[ResultRecord]:
        """
        The records matching an SQL ``where`` clause over ``COLUMNS``, e.g.
        ``query("state = ?", ("completed",))``.
        """
        connection = self.connect()
        try:
            rows = connection.execute(
                f"SELECT record FROM results WHERE {where} ORDER BY output_path",
                parameters,
            ).fetchall()
        finally:
            connection.close()
        return [ResultRecord.from_dict(json.loads(row[0])) for row in rows]

    def counts(self) -> Dict[str, int]:
        connection = self.connect()
        try:
            rows = connection.execute(
                "SELECT state, COUNT(*) FROM results GROUP BY state"
            ).fetchall()
        finally:
            connection.close()
        return dict(rows)
