"""
``dataset_identity.json``: what a fit was run on, written once at fit start into
``<output_path>/files/``.

A fit's output folder is named by a hash of its model and search, so two fits of the
same model to different data, or a refit after an input was replaced, are only told
apart by what they were given. This record names the object (``object_id`` plus any
external ids), the input files with their sizes and SHA256 and, where the dataset
exposes one, its centre, with the versions of the libraries the fit ran with. The
results index (``autofit.results``) joins on it, so duplicate fits of one object and
fits of the wrong input become detectable.

The inputs come from ``Analysis.dataset_identity()``, which returns ``None`` by default;
the record then carries only the ``object_id`` (the search's ``unique_tag``). Datasets
that know their input files (PyAutoArray / PyAutoLens) fill it in.

``general.output.dataset_identity: false`` turns the record off and
``general.output.dataset_identity_sha256: false`` skips the hashing (paths and sizes
are still recorded). Writing is best effort: a record that cannot be written logs a
warning and never fails the fit.
"""
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from autonerves import conf

from autofit.non_linear.status.states import Product
from autofit.tools.util import open_atomic

logger = logging.getLogger(__name__)

SCHEMA = 1

_CHUNK_BYTES = 1 << 20


def _output_setting(name: str, default: bool) -> bool:
    try:
        value = conf.instance["general"]["output"][name]
    except Exception:
        return default
    return default if value is None else bool(value)


def dataset_identity_enabled() -> bool:
    """
    Whether fits write ``dataset_identity.json`` (``general.output.dataset_identity``,
    default on).
    """
    return _output_setting("dataset_identity", True)


def dataset_identity_sha256() -> bool:
    """
    Whether ``dataset_identity.json`` records the SHA256 of each input
    (``general.output.dataset_identity_sha256``, default on).
    """
    return _output_setting("dataset_identity_sha256", True)


def sha256_of(path) -> str:
    """
    The SHA256 hex digest of the file at ``path``, read in 1 MB chunks.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


_VERSIONED_PACKAGES = (
    "autonerves",
    "autofit",
    "autoarray",
    "autogalaxy",
    "autolens",
    "numpy",
    "jax",
)


def library_versions() -> Dict[str, str]:
    """
    The ``__version__`` of each PyAuto library (and numpy / jax) already imported by
    this process; a library that is not imported is not imported here.
    """
    versions = {}
    for name in _VERSIONED_PACKAGES:
        module = sys.modules.get(name)
        version = getattr(module, "__version__", None)
        if version is not None:
            versions[name] = str(version)
    return versions


def _input_entry(path, sha256: bool) -> Dict:
    path = Path(path)
    entry = {"path": str(path.resolve()) if path.exists() else str(path)}
    if not path.is_file():
        entry["missing"] = True
        return entry
    entry["bytes"] = path.stat().st_size
    if sha256:
        entry["sha256"] = sha256_of(path)
    return entry


def dataset_identity_document(
    identity: Optional[Dict], unique_tag: Optional[str], sha256: bool = True
) -> Dict:
    """
    The ``dataset_identity.json`` document (``schema`` 1) for the identity an
    ``Analysis`` returned.

    Parameters
    ----------
    identity
        What ``Analysis.dataset_identity()`` returned: a dict with any of ``object_id``,
        ``external_ids`` (a dict), ``inputs`` (a dict of name to file path, or a list of
        paths) and ``centre`` (a sequence of numbers), or ``None``.
    unique_tag
        The search's ``unique_tag``, used as the ``object_id`` when the identity names
        none.
    sha256
        Whether to hash each input.
    """
    identity = dict(identity or {})

    inputs = identity.get("inputs") or {}
    if not isinstance(inputs, dict):
        inputs = {Path(path).name: path for path in inputs}

    centre = identity.get("centre")
    if centre is not None:
        centre = [float(value) for value in centre]

    return {
        "schema": SCHEMA,
        "object_id": identity.get("object_id") or unique_tag,
        "external_ids": dict(identity.get("external_ids") or {}),
        "inputs": {
            str(name): _input_entry(path, sha256=sha256)
            for name, path in inputs.items()
        },
        "centre": centre,
        "versions": library_versions(),
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _differences(old: Dict, new: Dict):
    for name, entry in new["inputs"].items():
        previous = old.get("inputs", {}).get(name)
        if previous is None:
            continue
        for key in ("sha256", "bytes"):
            if key in entry and key in previous and entry[key] != previous[key]:
                yield f"{name} ({key} {previous[key]} -> {entry[key]})"
                break
    if old.get("object_id") != new["object_id"]:
        yield f"object_id ({old.get('object_id')} -> {new['object_id']})"


def write_dataset_identity(search, analysis) -> Optional[Path]:
    """
    Write ``<output_path>/files/dataset_identity.json`` for the fit ``search`` is about
    to start, unless it is already there.

    A resumed fit keeps the record of its first start; if the inputs it is resumed with
    differ (another SHA256, size or ``object_id``) a warning names them, since the
    output then mixes two inputs. Returns the path written, or ``None`` when nothing was
    written (disabled, no output folder, already present, or an error).
    """
    from autofit.non_linear.paths.directory import DirectoryPaths

    paths = getattr(search, "paths", None)
    if not isinstance(paths, DirectoryPaths) or not dataset_identity_enabled():
        return None

    try:
        hook = getattr(type(analysis), "dataset_identity", None)
        identity = analysis.dataset_identity() if hook is not None else None
        document = dataset_identity_document(
            identity,
            unique_tag=getattr(paths, "unique_tag", None),
            sha256=dataset_identity_sha256(),
        )
        path = Path(paths._files_path) / str(Product.DATASET_IDENTITY)

        if path.exists():
            try:
                with open(path) as f:
                    previous = json.load(f)
            except (OSError, ValueError):
                previous = None
            if previous is not None:
                changed = list(_differences(previous, document))
                if changed:
                    logger.warning(
                        f"{path}: this fit is resumed with different inputs than it "
                        f"started with: {', '.join(changed)}. The first record is kept."
                    )
                return None

        os.makedirs(path.parent, exist_ok=True)
        with open_atomic(path) as f:
            json.dump(document, f, indent=2)
        return path
    except Exception as exception:
        logger.warning(f"dataset_identity.json not written ({exception}).")
        return None
