import json
import zipfile
from pathlib import Path

import pytest

from autofit.export import PackageException, package, verify
from autofit.export.__main__ import main


def _library(root: Path, n: int = 9) -> Path:
    library = root / "library"
    for i in range(n):
        item = library / f"obj{i:03d}"
        (item / "image").mkdir(parents=True)
        (item / "result.json").write_text(json.dumps({"object": i}))
        (item / "image" / "fit.png").write_bytes(b"\x89PNG" + bytes(1000 + 100 * i))
        (item / "table.csv").write_text("a,b\n1,2\n3,4\n")
    (library / "summary.csv").write_text(
        "id,value\n" + "".join(f"obj{i:03d},{i}\n" for i in range(n))
    )
    return library


def _sha(dist):
    return {path.name: path.read_bytes() for path in dist.glob("*.zip")}


@pytest.fixture(name="library")
def make_library(tmp_path):
    return _library(tmp_path)


def test_package_into_three_parts(library, tmp_path):
    dist = tmp_path / "dist"
    report = package(library, dist, n_parts=3)

    assert sorted(report.parts) == ["part_001.zip", "part_002.zip", "part_003.zip"]
    assert sorted(item for items in report.parts.values() for item in items) == [
        f"obj{i:03d}" for i in range(9)
    ]
    assert {p.name for p in dist.iterdir()} >= {
        "tables.zip",
        "parts_manifest.csv",
        "package.json",
        "SHA256SUMS",
        "README.md",
        "KNOWN_ISSUES.md",
    }
    with zipfile.ZipFile(dist / "part_001.zip") as archive:
        info = archive.getinfo("library/obj000/image/fit.png")
        assert info.compress_type == zipfile.ZIP_STORED
        assert archive.getinfo("library/obj000/result.json").compress_type == zipfile.ZIP_DEFLATED
    document = json.loads((dist / "package.json").read_text())
    assert document["zips"]["tables.zip"]["csv_rows"] == {"library/summary.csv": 9}
    assert "9 objects in 3 part zips" in (dist / "README.md").read_text()

    report = verify(dist)
    assert report.ok, report.problems
    assert report.csv_rows == 9 + 9 * 2


def test_rebuild_rewrites_only_the_changed_part(library, tmp_path):
    dist = tmp_path / "dist"
    package(library, dist, n_parts=3)
    before = _sha(dist)

    (library / "obj004" / "result.json").write_text(json.dumps({"object": 4, "v": 2}))
    report = package(library, dist, n_parts=3)

    assert report.written == ["part_002.zip"]
    after = _sha(dist)
    assert [name for name in after if after[name] != before[name]] == ["part_002.zip"]
    assert verify(dist).ok


def test_same_files_same_zip(library, tmp_path):
    package(library, tmp_path / "a", n_parts=3)
    package(library, tmp_path / "b", n_parts=3)
    assert _sha(tmp_path / "a") == _sha(tmp_path / "b")


def test_new_objects_fill_the_last_part_then_a_new_one(library, tmp_path):
    dist = tmp_path / "dist"
    package(library, dist, cap_gb=8000 / 1024**3)
    first = json.loads((dist / "package.json").read_text())["objects"]

    for i in range(9, 14):
        item = library / f"obj{i:03d}"
        item.mkdir()
        (item / "result.json").write_text("x" * 3000)
    report = package(library, dist, cap_gb=8000 / 1024**3)
    objects = json.loads((dist / "package.json").read_text())["objects"]

    assert {k: objects[k] for k in first} == first
    assert len(report.parts) > len(set(first.values()))
    assert set(report.kept) >= set(first.values()) - {max(first.values())}
    assert verify(dist).ok


def test_removed_object_and_stale_part(library, tmp_path):
    import shutil

    dist = tmp_path / "dist"
    package(library, dist, n_parts=3)
    for i in (6, 7, 8):
        shutil.rmtree(library / f"obj{i:03d}")
    report = package(library, dist)
    assert "part_003.zip" not in report.parts
    assert not (dist / "part_003.zip").exists()
    assert verify(dist).ok


def test_include_dataset(library, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "obj000.fits").write_bytes(b"SIMPLE")
    dist = tmp_path / "dist"
    package(
        library,
        dist,
        n_parts=1,
        include_dataset=lambda item: [data / f"{item}.fits"] if item == "obj000" else [],
    )
    with zipfile.ZipFile(dist / "part_001.zip") as archive:
        assert archive.read("library/obj000/dataset/obj000.fits") == b"SIMPLE"


def test_provenance(library, tmp_path):
    from autofit.results import ResultsIndex

    repo = Path(__file__).parents[2]
    dist = tmp_path / "dist"
    package(library, dist, n_parts=1, repos=[repo], index=ResultsIndex(tmp_path, index_dir=tmp_path / "i"))
    provenance = json.loads((dist / "package.json").read_text())["provenance"]
    assert len(provenance["repos"][0]["revision"]) == 40
    assert len(provenance["index_hash"]) == 64
    assert provenance["repos"][0]["revision"] in (dist / "README.md").read_text()

    (dist / "KNOWN_ISSUES.md").write_text("mine")
    package(library, dist, n_parts=1, title="My release", notes="Read me.")
    package(library, dist, n_parts=1)
    assert (dist / "KNOWN_ISSUES.md").read_text() == "mine"
    readme = (dist / "README.md").read_text()
    assert readme.startswith("# My release") and "Read me." in readme


def test_refusals(library, tmp_path):
    with pytest.raises(PackageException):
        package(library, tmp_path / "d", cap_gb=1, n_parts=2)
    with pytest.raises(PackageException):
        package(tmp_path / "absent", tmp_path / "d")


def _rewrite_member(zip_path, name, data):
    entries = {}
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            entries[info.filename] = archive.read(info)
    entries[name] = data
    with zipfile.ZipFile(zip_path, "w") as archive:
        for member, content in entries.items():
            archive.writestr(member, content)


@pytest.fixture(name="dist")
def make_dist(library, tmp_path):
    dist = tmp_path / "dist"
    package(library, dist, n_parts=3)
    return dist


def test_verify_catches_an_empty_member(dist):
    _rewrite_member(dist / "part_001.zip", "library/obj000/result.json", b"")
    problems = verify(dist).problems
    assert any("library/obj000/result.json is empty" in p for p in problems)
    assert any("part_001.zip: SHA256 does not match" in p for p in problems)


def test_verify_catches_a_corrupt_byte(dist):
    path = dist / "part_002.zip"
    data = bytearray(path.read_bytes())
    data[60] ^= 0xFF
    path.write_bytes(bytes(data))
    problems = verify(dist).problems
    assert any("part_002.zip" in p for p in problems)


def test_verify_catches_row_count_and_ragged_rows(dist):
    _rewrite_member(dist / "tables.zip", "library/summary.csv", b"id,value\nobj000,0\nobj001\n")
    problems = verify(dist).problems
    assert any("has 2 rows, package.json records 9" in p for p in problems)
    assert any("do not have 2 fields" in p for p in problems)


def test_verify_catches_manifest_drift(dist):
    text = (dist / "parts_manifest.csv").read_text().replace("obj000,part_001.zip", "obj000,part_002.zip")
    (dist / "parts_manifest.csv").write_text(text)
    problems = verify(dist).problems
    assert any("parts_manifest.csv disagrees" in p for p in problems)


def test_verify_catches_a_missing_part(dist):
    (dist / "part_003.zip").unlink()
    problems = verify(dist).problems
    assert any("part_003.zip: listed in SHA256SUMS but missing" in p for p in problems)
    assert any("obj008" in p for p in problems)


def test_cli(library, tmp_path, capsys):
    dist = tmp_path / "dist"
    assert main(["package", str(library), str(dist), "--n-parts", "2"]) == 0
    assert main(["verify", str(dist)]) == 0
    assert "OK" in capsys.readouterr().out
    _rewrite_member(dist / "part_001.zip", "library/obj000/result.json", b"")
    assert main(["verify", str(dist)]) == 1
    assert main(["package", str(tmp_path / "absent"), str(dist)]) == 2


# Regressions from the independent review.


def test_replan_removes_stale_parts_and_verify_flags_foreign_zips(library, tmp_path):
    dist = tmp_path / "dist"
    package(library, dist, n_parts=3)
    package(library, dist, n_parts=2, replan=True)
    assert sorted(p.name for p in dist.glob("part_*.zip")) == ["part_001.zip", "part_002.zip"]
    assert verify(dist).ok
    (dist / "part_009.zip").write_bytes((dist / "part_001.zip").read_bytes())
    assert any("part_009.zip: a zip package.json does not list" in p for p in verify(dist).problems)


def test_rebuild_refuses_a_different_part_count(library, tmp_path):
    dist = tmp_path / "dist"
    package(library, dist, n_parts=2)
    with pytest.raises(PackageException, match="replan"):
        package(library, dist, n_parts=9)


def test_cap_is_remembered(library, tmp_path):
    dist = tmp_path / "dist"
    package(library, dist, cap_gb=8000 / 1024**3)
    n = len(set(json.loads((dist / "package.json").read_text())["objects"].values()))
    for i in range(9, 20):
        item = library / f"obj{i:03d}"
        item.mkdir()
        (item / "result.json").write_text("x" * 3000)
    report = package(library, dist)  # no cap passed
    assert len(report.parts) > n + 2


def test_exactly_n_parts():
    from autofit.export.package import _plan

    items = ["a", "b", "c", "d"]
    sizes = {"a": 1, "b": 1, "c": 1, "d": 100}
    assert len(set(_plan(items, sizes, None, 2, {}).values())) == 2
    assert len(set(_plan(items, sizes, None, 9, {}).values())) == 4


def test_duplicate_member_refused(library, tmp_path):
    (library / "obj000" / "dataset").mkdir()
    (library / "obj000" / "dataset" / "img.fits").write_bytes(b"A")
    extra = tmp_path / "img.fits"
    extra.write_bytes(b"B")
    with pytest.raises(PackageException, match="several files"):
        package(library, tmp_path / "dist", include_dataset=lambda item: [extra])


def test_dot_files_ship_and_vcs_folders_do_not(library, tmp_path):
    (library / "obj000" / ".completed").write_text("")
    (library / "obj000" / ".git").mkdir()
    (library / "obj000" / ".git" / "HEAD").write_text("x")
    dist = tmp_path / "dist"
    package(library, dist, n_parts=1)
    with zipfile.ZipFile(dist / "part_001.zip") as archive:
        names = archive.namelist()
    assert "library/obj000/.completed" in names
    assert not any(".git" in name for name in names)
    assert verify(dist).ok  # .completed is allowed to be empty


def test_odd_csvs_do_not_stop_a_package(library, tmp_path):
    (library / "obj000" / "latin.csv").write_bytes("name\nM\xfcller\n".encode("latin-1"))
    (library / "obj001" / "wide.csv").write_text("a\n" + "x" * 200_000 + "\n")
    (library / "obj002" / "blank_end.csv").write_text("a,b\n1,2\n\n")
    dist = tmp_path / "dist"
    package(library, dist, n_parts=1)
    report = verify(dist)
    assert report.ok, report.problems


def test_same_zip_in_any_timezone(library, tmp_path, monkeypatch):
    import time

    hashes = []
    for tz in ("UTC", "America/New_York"):
        monkeypatch.setenv("TZ", tz)
        time.tzset()
        package(library, tmp_path / tz.replace("/", "_"), n_parts=1)
        hashes.append((tmp_path / tz.replace("/", "_") / "part_001.zip").read_bytes())
    monkeypatch.delenv("TZ")
    time.tzset()
    assert hashes[0] == hashes[1]


def test_force_rewrites(library, tmp_path):
    dist = tmp_path / "dist"
    package(library, dist, n_parts=2)
    assert package(library, dist).written == []
    assert sorted(package(library, dist, force=True).written) == [
        "part_001.zip",
        "part_002.zip",
        "tables.zip",
    ]


def test_verify_reports_corruption_instead_of_crashing(dist):
    path = dist / "tables.zip"
    with zipfile.ZipFile(path) as archive:
        info = archive.infolist()[0]
        offset = info.header_offset + 30 + len(info.filename) + len(info.extra) + 2
    data = bytearray(path.read_bytes())
    for i in range(8):
        data[offset + i] ^= 0xFF
    path.write_bytes(bytes(data))
    (dist / "SHA256SUMS").write_text((dist / "SHA256SUMS").read_text() + "garbage\n")
    report = verify(dist)
    assert not report.ok
    assert any("SHA256SUMS line" in p for p in report.problems)
    assert main(["verify", str(dist)]) == 1
