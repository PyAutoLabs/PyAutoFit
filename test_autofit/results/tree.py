"""
Synthetic results trees for the results index: each helper writes one result in the
on-disk form a real fit leaves (``<prefix>/<name>/<identifier>`` as a zip, a folder,
a status record or a mix), with only the small files the index reads.
"""
import json
import zipfile
from datetime import datetime, timezone
from hashlib import md5
from pathlib import Path


def _search_json(name, unique_tag, path_prefix):
    return json.dumps(
        {
            "type": "instance",
            "class_path": "autofit.non_linear.search.nest.nautilus.search.Nautilus",
            "arguments": {
                "name": name,
                "unique_tag": unique_tag,
                "path_prefix": {"type": "path", "path": str(path_prefix)},
            },
        }
    )


def _identifier(seed):
    return md5(f"Nautilus.{seed}".encode()).hexdigest()


def _members(name, unique_tag, path_prefix, seed, completed=True):
    members = {
        ".identifier": f"Nautilus\n{seed}",
        "files/search.json": _search_json(name, unique_tag, path_prefix),
        "files/samples_summary.json": "{}",
        "files/samples.csv": "a,b\n1,2\n",
        "image/fit.png": b"\x89PNG",
    }
    if completed:
        members[".completed"] = ""
    return members


def write_zip(
    root,
    prefix="dataset_0",
    name="search",
    unique_tag=None,
    seed=0,
    completed=True,
    date_time=(2026, 10, 1, 12, 0, 0),
):
    """
    A ``remove_files`` result: only ``<output_path>.zip`` is left.
    """
    output_path = Path(root) / prefix / name / _identifier(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(f"{output_path}.zip", "w") as f:
        for member, content in _members(
            name, unique_tag or prefix, prefix, seed, completed
        ).items():
            f.writestr(zipfile.ZipInfo(member, date_time=date_time), content)
    return output_path


def write_folder(
    root,
    prefix="dataset_0",
    name="search",
    seed=0,
    completed=True,
    checkpoint=None,
):
    """
    An unzipped result folder; ``checkpoint`` is ``None``, ``"valid"``,
    ``"empty"`` or ``"quarantined"``.
    """
    output_path = Path(root) / prefix / name / _identifier(seed)
    for member, content in _members(name, prefix, prefix, seed, completed).items():
        path = output_path / member
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)
    internal = output_path / "files" / "search_internal"
    if checkpoint == "valid":
        internal.mkdir(parents=True)
        (internal / "search_internal.dill").write_bytes(b"state")
    elif checkpoint == "empty":
        internal.mkdir(parents=True)
        (internal / "search_internal.dill").write_bytes(b"")
    elif checkpoint == "quarantined":
        failed = output_path / "files" / "search_internal.failed_20261001T000000Z"
        failed.mkdir(parents=True)
        (failed / "checkpoint.hdf5").write_bytes(b"x")
    return output_path


def write_status(output_path, state, updated=None, failure_class=None):
    updated = updated or datetime.now(timezone.utc).isoformat(timespec="seconds")
    Path(f"{output_path}.status.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "state": state,
                "attempt": 1,
                "failure_class": failure_class,
                "updated": updated,
                "started": updated,
                "job": None,
            }
        )
    )
