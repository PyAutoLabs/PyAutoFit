"""
The results index over synthetic trees: every result state, incremental re-indexing,
superseded siblings, the SQLite cache and the command line.
"""
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from autofit.non_linear.status.states import ResultState
from autofit.results import ResultsCache, ResultsIndex, classify, find_results
from autofit.results.__main__ import main

from .tree import write_folder, write_status, write_zip


def _states(index):
    return {Path(r.output_path).parts[-3]: r.state for r in index.records()}


def test_every_state_is_classified(results_root):
    write_zip(results_root, prefix="zipped")
    write_folder(results_root, prefix="folder")
    corrupt = write_zip(results_root, prefix="corrupt_zip")
    Path(f"{corrupt}.zip").write_bytes(b"not a zip")
    write_folder(results_root, prefix="resumable", completed=False, checkpoint="valid")
    write_folder(results_root, prefix="bad_ckpt", completed=False, checkpoint="empty")
    write_folder(
        results_root, prefix="quarantined", completed=False, checkpoint="quarantined"
    )
    write_folder(results_root, prefix="fresh", completed=False)
    failed = write_folder(results_root, prefix="failed", completed=False)
    write_status(failed, "failed", failure_class="bad_input")
    running = write_folder(results_root, prefix="running", completed=False)
    write_status(running, "running")
    stale = write_folder(
        results_root, prefix="stale", completed=False, checkpoint="valid"
    )
    write_status(
        stale,
        "running",
        updated=(datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
    )

    index = ResultsIndex(results_root)
    index.update()

    assert _states(index) == {
        "zipped": "completed",
        "folder": "completed",
        "corrupt_zip": "corrupt_zip",
        "resumable": "interrupted_valid_ckpt",
        "bad_ckpt": "corrupt_ckpt",
        "quarantined": "corrupt_ckpt",
        "fresh": "never_started",
        "failed": "failed",
        "running": "running",
        "stale": "interrupted_valid_ckpt",
    }
    records = {Path(r.output_path).parts[-3]: r for r in index.records()}
    assert records["failed"].reason == "bad_input"
    assert "last written" in records["stale"].reason
    assert set(index.counts()) == {str(state) for state in ResultState}


def test_record_holds_metadata_products_and_completion_time(results_root):
    output_path = write_zip(
        results_root, prefix="lens_1", name="vis_lp", date_time=(2026, 9, 28, 0, 47, 52)
    )

    record = classify(output_path, root=results_root)

    assert record.state == "completed"
    assert record.name == "vis_lp"
    assert record.unique_tag == "lens_1"
    assert record.path_prefix == "lens_1"
    assert record.search == "Nautilus"
    assert record.identifier == output_path.name
    assert record.completed_at == "2026-09-28T00:47:52"
    assert record.completed_at_source == "zip_member"
    assert record.products["samples"] and record.products["image"]
    assert not record.products["latent"]
    assert "samples_summary.json" in record.files
    assert not record.location_mismatch


def test_status_record_gives_the_completion_time(results_root):
    output_path = write_zip(results_root)
    write_status(output_path, "completed", updated="2026-10-02T09:00:00+00:00")

    record = classify(output_path, root=results_root)

    assert record.completed_at == "2026-10-02T09:00:00+00:00"
    assert record.completed_at_source == "status"


def test_crc_failure_is_a_corrupt_zip(results_root):
    output_path = write_zip(results_root)
    zip_path = Path(f"{output_path}.zip")
    data = bytearray(zip_path.read_bytes())
    offset = data.index(b"a,b")
    data[offset] = ord("X")
    zip_path.write_bytes(bytes(data))

    assert classify(output_path, root=results_root).state == "corrupt_zip"
    assert (
        classify(output_path, root=results_root, verify=False).state == "completed"
    )


def test_result_moved_into_another_objects_folder_is_flagged(results_root):
    output_path = write_zip(results_root, prefix="lens_1")
    moved = results_root / "lens_2" / "search" / output_path.name
    moved.parent.mkdir(parents=True)
    Path(f"{output_path}.zip").rename(f"{moved}.zip")

    assert classify(moved, root=results_root).location_mismatch


def test_nested_and_custom_layouts_are_found(results_root):
    write_zip(results_root / "a" / "b" / "c", prefix="deep")
    write_folder(results_root, prefix="flat")
    nested = write_folder(results_root, prefix="outer")
    write_folder(nested / "analyses", prefix="analysis_0")

    found = sorted(str(p.relative_to(results_root)) for p in find_results(results_root))

    assert len(found) == 4
    assert all("files" not in Path(p).parts for p in found)


def test_reindex_only_opens_new_and_changed_results(results_root, monkeypatch):
    for seed in range(30):
        write_zip(results_root, prefix=f"dataset_{seed}", seed=seed)
    index = ResultsIndex(results_root)
    assert index.update()["new"] == 30

    opened = []
    import autofit.results.index as index_module

    original = index_module.classify

    def counting(output_path, **kwargs):
        opened.append(output_path)
        return original(output_path, **kwargs)

    monkeypatch.setattr(index_module, "classify", counting)

    for seed in range(30, 40):
        write_zip(results_root, prefix=f"dataset_{seed}", seed=seed)
    counts = index.update()

    assert counts == {
        "scanned": 40,
        "new": 10,
        "changed": 0,
        "unchanged": 30,
        "removed": 0,
    }
    assert len(opened) == 10

    removed = results_root / "dataset_0"
    for path in removed.rglob("*.zip"):
        path.unlink()
    assert index.update()["removed"] == 1
    assert len(index.records()) == 39


def test_sibling_hashes_older_completed_is_superseded(results_root):
    old = write_zip(results_root, seed=1, date_time=(2026, 9, 1, 0, 0, 0))
    new = write_zip(results_root, seed=2, date_time=(2026, 10, 1, 0, 0, 0))
    other = write_zip(results_root, prefix="dataset_1", seed=3)

    index = ResultsIndex(results_root)
    index.update()
    states = {r.output_path: r.state for r in index.records()}

    assert states[str(old)] == "superseded"
    assert states[str(new)] == "completed"
    assert states[str(other)] == "completed"

    # The latest disappearing restores the older one.
    Path(f"{new}.zip").unlink()
    index.update()
    assert {r.output_path: r.state for r in index.records()}[str(old)] == "completed"


def test_index_dir_keeps_a_read_only_tree_untouched(results_root, tmp_path):
    write_zip(results_root)
    before = sorted(p for p in results_root.rglob("*"))

    index = ResultsIndex([results_root], index_dir=tmp_path / "index")
    index.update()

    assert sorted(p for p in results_root.rglob("*")) == before
    assert (tmp_path / "index" / "index.json").exists()
    assert len(index.records()) == 1


def test_several_roots_need_an_index_dir(tmp_path):
    with pytest.raises(ValueError):
        ResultsIndex([tmp_path / "a", tmp_path / "b"])


def test_sqlite_cache_is_rebuilt_from_the_records(results_root, tmp_path):
    write_zip(results_root)
    corrupt = write_zip(results_root, prefix="dataset_1")
    Path(f"{corrupt}.zip").write_bytes(b"junk")
    index = ResultsIndex(results_root)
    index.update()

    cache = ResultsCache(index, cache_dir=tmp_path / "cache")

    assert cache.counts() == {"completed": 1, "corrupt_zip": 1}
    assert [r.output_path for r in cache.query("state = ?", ("corrupt_zip",))] == [
        str(corrupt)
    ]

    write_zip(results_root, prefix="dataset_2")
    index.update()
    assert cache.counts()["completed"] == 2


def test_cli_index_status_count(results_root, tmp_path, capsys):
    write_zip(results_root)
    corrupt = write_zip(results_root, prefix="dataset_1")
    Path(f"{corrupt}.zip").write_bytes(b"junk")
    index_dir = str(tmp_path / "index")

    assert main(["status", str(results_root), "--index-dir", index_dir]) == 1
    capsys.readouterr()

    assert main(["index", str(results_root), "--index-dir", index_dir]) == 0
    assert "indexed 2 results" in capsys.readouterr().out

    main(["status", str(results_root), "--index-dir", index_dir])
    out = capsys.readouterr().out
    assert "2 results" in out
    assert "completed                1" in out
    assert "corrupt_zip              1" in out

    main(["status", str(results_root), "--index-dir", index_dir, "--state", "corrupt_zip"])
    assert str(corrupt) in capsys.readouterr().out

    main(["count", str(results_root), "--index-dir", index_dir, "--state", "completed"])
    assert capsys.readouterr().out.strip() == "1"

    main(["status", str(results_root), "--index-dir", index_dir, "--json"])
    assert json.loads(capsys.readouterr().out)["completed"] == 1
