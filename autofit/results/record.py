"""
``ResultRecord``: one result as the results index sees it.

A result is one fit's ``output_path`` (``<root>/<path_prefix>/<name>/<identifier>``),
which may be on disk as a folder, a ``<output_path>.zip``, a
``<output_path>.status.json`` or any combination of them. The record is the index's
unit of truth: it is written as one JSON file per result
(``<index_dir>/records/<key>.json``) and holds everything ``status``, ``count`` and
``latest_per`` need, so none of them opens a result again.
"""
import hashlib
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

SCHEMA = 1


def record_key(output_path: str) -> str:
    """
    The file name (without ``.json``) of the record of the result at ``output_path``.
    """
    return hashlib.sha1(str(output_path).encode("utf-8")).hexdigest()


@dataclass
class ResultRecord:
    output_path: str
    """The result's ``output_path`` (absolute, without ``.zip``)."""
    root: str
    """The root directory the result was found under."""
    state: str
    """A ``ResultState`` value."""
    reason: Optional[str] = None
    """Why the result has this state, when it is not plainly ``completed``."""
    has_zip: bool = False
    has_dir: bool = False
    has_status: bool = False
    name: Optional[str] = None
    """The search's ``name``, from ``files/search.json``."""
    unique_tag: Optional[str] = None
    path_prefix: Optional[str] = None
    """The ``path_prefix`` the search declared, from ``files/search.json``."""
    location_mismatch: bool = False
    """Whether the declared ``path_prefix`` / ``name`` disagree with where the result
    sits on disk (e.g. a result copied or moved into another object's folder)."""
    identifier: Optional[str] = None
    search: Optional[str] = None
    """The class name of the search (e.g. ``Nautilus``)."""
    completed_at: Optional[str] = None
    """When the result completed (ISO 8601), never the zip's modification time."""
    completed_at_source: Optional[str] = None
    """``status`` (the status record), ``zip_member`` (the ``.completed`` member's
    timestamp, local time as zipped) or ``completed_file`` (its mtime)."""
    products: Dict[str, bool] = field(default_factory=dict)
    """Which standard products are present (``samples``, ``samples_summary``,
    ``latent``, ``image``, ``search_internal``, ``dataset_identity``)."""
    files: List[str] = field(default_factory=list)
    """The paths under ``files/``, relative to it."""
    n_images: int = 0
    object_id: Optional[str] = None
    """From ``dataset_identity.json``."""
    inputs: Dict[str, Optional[str]] = field(default_factory=dict)
    """Input name to SHA256 (``None`` when not hashed), from ``dataset_identity.json``."""
    versions: Dict[str, str] = field(default_factory=dict)
    status: Optional[Dict] = None
    """A summary of the status record: state, attempt, failure class, updated, job."""
    bytes: Optional[int] = None
    """The size of the zip, when there is one."""
    stat_key: List = field(default_factory=list)
    """The stat fingerprint the record was built from; an unchanged fingerprint means
    the result is not opened again."""
    schema: int = SCHEMA

    @property
    def key(self) -> str:
        return record_key(self.output_path)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, document: Dict) -> "ResultRecord":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in document.items() if k in known})
