import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from autofit.aggregator.aggregator import Aggregator
from autofit.results.__main__ import main
from autofit.results.shard import in_shard, parse_shard, shard_of

from autofit.tools.util import zip_directory

from .tree import write_folder, write_zip


def _digests(zip_path):
    with zipfile.ZipFile(zip_path) as f:
        return {
            info.filename: (hashlib.sha256(f.read(info)).hexdigest(), info.compress_type)
            for info in f.infolist()
        }


@pytest.fixture(name="library")
def make_library(results_root):
    return [write_zip(results_root, prefix=f"d{i:03d}", seed=i) for i in range(100)]


def _extra(search_output):
    return {"identifier": search_output.directory.zip_path.stem}


def test_enrich_adds_member_and_keeps_the_rest(results_root, library):
    before = {path: _digests(f"{path}.zip") for path in library}

    report = Aggregator.from_index(results_root).enrich(_extra, name="extra")

    assert report.counts()["written"] == 100
    assert report.name == "files/extra.json"
    for path in library:
        after = _digests(f"{path}.zip")
        assert {k: v for k, v in after.items() if k != "files/extra.json"} == before[path]
        with zipfile.ZipFile(f"{path}.zip") as f:
            assert json.loads(f.read("files/extra.json")) == {"identifier": path.name}
    assert not list(results_root.rglob("*.tmp"))


def test_overwrite(results_root, library):
    aggregator = Aggregator.from_index(results_root)
    aggregator.enrich(lambda o: "one", name="note.txt")

    report = aggregator.enrich(lambda o: "two", name="note.txt")
    assert report.counts()["exists"] == 100

    report = aggregator.enrich(lambda o: "two", name="note.txt", overwrite=True)
    assert report.counts()["written"] == 100
    with zipfile.ZipFile(f"{library[0]}.zip") as f:
        assert f.read("files/note.txt") == b"two"
        assert f.namelist().count("files/note.txt") == 1


def test_shards_partition(results_root, library):
    aggregator = Aggregator.from_index(results_root)
    written = []
    for i in range(4):
        report = aggregator.enrich(_extra, name="extra", shard=f"{i}/4")
        assert report.skipped_shard == 100 - len(report.written)
        written.extend(report.written)
    assert sorted(written) == sorted(str(path) for path in library)


def test_shard_helpers():
    assert parse_shard("1/4") == (1, 4)
    assert parse_shard((0, 1)) == (0, 1)
    assert parse_shard(None) is None
    for bad in ("4/4", "-1/2", "x", "1/0"):
        with pytest.raises(ValueError):
            parse_shard(bad)
    assert shard_of("/x/d0/search/abc", 7) == shard_of("/x/d0/search/abc.zip", 7)
    # Objects whose identifiers coincide still spread over the shards.
    shards = {shard_of(f"/x/d{i}/slam/abc", 4) for i in range(40)}
    assert shards == {0, 1, 2, 3}
    assert in_shard("/x", None)


def test_failure_keeps_zip_and_goes_on(results_root, library):
    original = Path(f"{library[0]}.zip").read_bytes()

    def fn(search_output):
        if search_output.directory.zip_path.stem == library[0].name:
            raise RuntimeError("boom")
        return {"ok": True}

    report = Aggregator.from_index(results_root).enrich(fn, name="extra")

    assert report.counts()["written"] == 99
    assert report.errors == [(str(library[0]), "RuntimeError: boom")]
    assert Path(f"{library[0]}.zip").read_bytes() == original


def test_killed_write_leaves_zip_intact(results_root, monkeypatch):
    path = write_zip(results_root, prefix="d0")
    original = Path(f"{path}.zip").read_bytes()
    aggregator = Aggregator.from_index(results_root)

    import autofit.tools.util as util

    def killed(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(util.shutil, "copyfileobj", killed)
    with pytest.raises(KeyboardInterrupt):
        aggregator.enrich(_extra, name="extra")

    assert Path(f"{path}.zip").read_bytes() == original
    assert not list(results_root.rglob("*.tmp"))


def test_declined_and_bad_values(results_root, library):
    aggregator = Aggregator.from_index(results_root)
    assert aggregator.enrich(lambda o: None, name="x").counts()["declined"] == 100
    report = aggregator.enrich(lambda o: 1.5, name="x")
    assert report.counts()["errors"] == 100
    assert "TypeError" in report.errors[0][1]
    with pytest.raises(ValueError):
        aggregator.enrich(_extra, name="../escape")


def test_file_path_value(results_root, library, tmp_path):
    image = tmp_path / "model.fits"
    image.write_bytes(b"SIMPLE")
    Aggregator.from_index(results_root).enrich(lambda o: image, name="model.fits")
    with zipfile.ZipFile(f"{library[0]}.zip") as f:
        info = f.getinfo("files/model.fits")
        assert f.read(info) == b"SIMPLE"
        assert info.compress_type == zipfile.ZIP_STORED


def test_folder_result(results_root):
    path = write_folder(results_root, prefix="d0")
    report = Aggregator.from_directory(results_root).enrich(lambda o: [1, 2], name="extra")
    assert report.counts()["written"] == 1
    assert json.loads((path / "files" / "extra.json").read_text()) == [1, 2]


def test_temporary_extraction_refused(results_root, library):
    aggregator = Aggregator.from_directory(results_root, unzip_temporary=True)
    with pytest.raises(ValueError, match="temporary"):
        aggregator.enrich(_extra, name="extra")


def test_cli_enrich(results_root, library, tmp_path, monkeypatch):
    (tmp_path / "products.py").write_text(
        "def note(search_output):\n    return 'hello'\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    assert main(["enrich", str(results_root), "--fn", "products:note", "--name", "note.txt", "--shard", "0/2"]) == 0
    assert main(["enrich", str(results_root), "--fn", "products:note", "--name", "note.txt", "--shard", "1/2"]) == 0
    for path in library:
        with zipfile.ZipFile(f"{path}.zip") as f:
            assert f.read("files/note.txt") == b"hello"
    assert main(["enrich", str(results_root), "--fn", "products:nope", "--name", "x"]) == 2


def _folder_and_zip(root, prefix="d0"):
    path = write_folder(root, prefix=prefix)
    zip_directory(path, f"{path}.zip")
    return path


def test_folder_and_zip_both_written(results_root):
    path = _folder_and_zip(results_root)

    report = Aggregator.from_index(results_root).enrich(_extra, name="extra")

    assert report.counts()["written"] == 1
    assert (path / "files" / "extra.json").is_file()
    # A rerun of the search re-zips the folder: the product survives it.
    zip_directory(path, f"{path}.zip")
    with zipfile.ZipFile(f"{path}.zip") as f:
        assert "files/extra.json" in f.namelist()


def test_stopped_between_copies_is_finished_by_a_rerun(results_root, monkeypatch):
    path = _folder_and_zip(results_root)
    aggregator = Aggregator.from_index(results_root)

    import importlib

    enrich_module = importlib.import_module("autofit.results.enrich")

    real_open_atomic = enrich_module.open_atomic

    def full_disk(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(enrich_module, "open_atomic", full_disk)
    assert aggregator.enrich(_extra, name="extra").counts()["errors"] == 1
    monkeypatch.setattr(enrich_module, "open_atomic", real_open_atomic)

    report = aggregator.enrich(_extra, name="extra")
    assert report.counts() == {
        "written": 1,
        "exists": 0,
        "declined": 0,
        "errors": 0,
        "other_shards": 0,
    }
    assert (path / "files" / "extra.json").is_file()
    assert aggregator.enrich(_extra, name="extra").counts()["exists"] == 1


def test_cli_enrich_errors_exit_2(results_root, library):
    assert main(["enrich", str(results_root), "--fn", "json:dumps", "--name", "x", "--shard", "9/2"]) == 2
