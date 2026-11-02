import csv
import json
import shutil
from pathlib import Path

import pytest

import autofit as af
from autofit.aggregator import Aggregator
from autofit.aggregator.search_output import plain
from autofit.aggregator.summary.aggregate_csv import (
    AggregateCSV,
    MergeException,
    ValueType,
    merge_csv,
)
from autofit.aggregator.summary.aggregate_csv.cache import cache_path
from autofit.results.__main__ import main

FIXTURE = Path(__file__).parent / "aggregate_summary"
VARIABLE = "galaxies.lens.bulge.centre.centre_0"


def _copy(root: Path, n: int, start: int = 0):
    for i in range(start, start + n):
        target = root / f"obj{i:04d}" / "search" / f"{i:032x}"
        shutil.copytree(FIXTURE / ("fit_1" if i % 2 else "fit_2"), target)
        (target / ".identifier").write_text(f"copy\n{i}")
        (target / ".completed").write_text("")
        search = json.loads((target / "files" / "search.json").read_text())
        search["arguments"] = {"name": "search", "unique_tag": f"obj{i:04d}"}
        (target / "files" / "search.json").write_text(json.dumps(search))
    return root


def _summary(root, **kwargs):
    summary = AggregateCSV(Aggregator.from_directory(root), **kwargs)
    summary.add_variable(VARIABLE, value_types=[ValueType.Median, ValueType.ValuesAt1Sigma])
    summary.add_computed_column("doubled", lambda result: 2)
    return summary


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(name="root")
def make_root(tmp_path):
    return _copy(tmp_path / "output", 120)


def test_refresh_reads_only_new_results(root, tmp_path):
    path = tmp_path / "summary.csv"
    summary = _summary(root)
    summary.save(path, cache=True, sort_by="id")
    assert summary.stats["computed"] == 120

    _copy(root, 30, start=120)
    summary = _summary(root)
    summary.save(path, cache=True, sort_by="id")
    assert summary.stats == {"computed": 30, "cached": 120, "errors": 0, "expected": 0}

    fresh = tmp_path / "fresh.csv"
    _summary(root).save(fresh, sort_by="id")
    assert path.read_bytes() == fresh.read_bytes()


def test_changed_result_or_spec_is_read_again(root, tmp_path):
    path = tmp_path / "summary.csv"
    _summary(root).save(path, cache=True)

    changed = next(root.glob("obj0003/search/*"))
    (changed / "files" / "extra.json").write_text("{}")
    summary = _summary(root)
    summary.save(path, cache=True)
    assert (summary.stats["computed"], summary.stats["cached"]) == (1, 119)

    summary = AggregateCSV(Aggregator.from_directory(root))
    summary.add_variable(VARIABLE)
    summary.add_computed_column("doubled", lambda result: 3)  # edited code
    summary.save(path, cache=True)
    assert summary.stats["computed"] == 120


def test_killed_export_leaves_previous_files(root, tmp_path, monkeypatch):
    path = tmp_path / "summary.csv"
    _summary(root).save(path, cache=True)
    before = (path.read_bytes(), cache_path(path).read_bytes())

    import csv as csv_module

    class Killed(csv_module.DictWriter):
        def writerow(self, row):
            raise KeyboardInterrupt

    monkeypatch.setattr(
        "autofit.aggregator.summary.aggregate_csv.csv.DictWriter", Killed
    )
    _copy(root, 5, start=120)
    with pytest.raises(KeyboardInterrupt):
        _summary(root).save(path, cache=True)

    assert (path.read_bytes(), cache_path(path).read_bytes()) == before
    assert not list(tmp_path.glob("*.tmp"))


def test_shards_merge_to_the_serial_export(root, tmp_path):
    serial = tmp_path / "serial.csv"
    _summary(root).save(serial, sort_by="id")

    parts = []
    for i in range(4):
        part = tmp_path / f"part{i}.csv"
        _summary(root).save(part, shard=f"{i}/4", cache=True)
        parts.append(part)
    assert sum(len(_read(part)) for part in parts) == 120

    merged = tmp_path / "merged.csv"
    assert merge_csv(parts, merged) == 120
    assert merged.read_bytes() == serial.read_bytes()

    # Merging a part twice is harmless: identical duplicates are kept once.
    merge_csv(parts + parts[:1], merged)
    assert merged.read_bytes() == serial.read_bytes()


def test_merge_refusals(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("id,x\r\n1,2\r\n")
    b.write_text("id,y\r\n3,4\r\n")
    with pytest.raises(MergeException, match="header"):
        merge_csv([a, b], tmp_path / "m.csv")

    b.write_text("id,x\r\n1,5\r\n")
    with pytest.raises(MergeException, match="different rows"):
        merge_csv([a, b], tmp_path / "m.csv")

    with pytest.raises(MergeException, match="no 'name' column"):
        merge_csv([a], tmp_path / "m.csv", key="name")

    assert main(["merge", str(tmp_path / "m.csv"), str(a), str(b)]) == 2
    b.write_text("id,x\r\n3,5\r\n")
    assert main(["merge", str(tmp_path / "m.csv"), str(a), str(b)]) == 0
    assert [row["id"] for row in _read(tmp_path / "m.csv")] == ["1", "3"]


def test_errors_become_rows(root, tmp_path):
    summary = _summary(root)

    def explode(result):
        if result.id == summary_ids[0]:
            raise RuntimeError("boom")
        return 1

    summary_ids = sorted(output.id for output in summary._aggregator)
    summary.add_computed_column("explode", explode)

    with pytest.raises(RuntimeError):
        summary.save(tmp_path / "raise.csv")

    path = tmp_path / "rows.csv"
    summary.save(path, on_error="row", sort_by="id")
    rows = _read(path)
    assert len(rows) == 120
    assert summary.stats["errors"] == 1
    assert rows[0]["status_reason"] == "error:RuntimeError: boom"
    assert rows[0][VARIABLE.replace(".", "_")] == ""
    assert all(row["status_reason"] == "" for row in rows[1:])


def test_missing_latent_summary_reason(root, tmp_path):
    for latent in root.glob("obj000[0-4]/search/*/files/latent"):
        shutil.rmtree(latent)
    summary = AggregateCSV(Aggregator.from_directory(root), strict=False)
    summary.add_variable("latent.value")
    path = tmp_path / "s.csv"
    summary.save(path, on_error="row")
    reasons = [row["status_reason"] for row in _read(path)]
    assert reasons.count("missing_latent_summary") == 5


def test_expected_results_get_rows(root, tmp_path):
    library = af.ResultsLibrary.grid(
        [f"obj{i:04d}" for i in range(122)],
        {"search": {"products": ["files/model.json", "files/coolest.json"],
                    "location": "{dataset}/search"}},
        roots=root,
        index_dir=tmp_path / "index",
    )
    report = library.check()

    summary = _summary(root)
    path = tmp_path / "s.csv"
    summary.save(path, expected=report, sort_by="dataset")
    rows = {row["dataset"]: row for row in _read(path)}

    assert len(rows) == 122
    assert rows["obj0121"]["status_reason"] == "no_result"
    assert rows["obj0121"]["id"] == ""
    assert rows["obj0000"]["search"] == "search"
    assert rows["obj0000"]["status_reason"] == "missing_product:files/coolest.json"
    assert summary.stats["expected"] == 2


def test_columns_json(root, tmp_path):
    summary = AggregateCSV(Aggregator.from_directory(root))
    summary.add_variable(
        VARIABLE,
        value_types=[ValueType.Median, ValueType.ValuesAt1Sigma],
        unit="arcsec",
        prior_edge=True,
    )
    summary.add_variable("latent.value")
    summary.add_label_column("label", {})
    path = tmp_path / "s.csv"
    summary.save(path, columns_json=True, on_error="row")

    document = json.loads((tmp_path / "s.columns.json").read_text())
    columns = {column["name"]: column for column in document["columns"]}
    name = VARIABLE.replace(".", "_")

    assert document["n_rows"] == 120
    assert columns[name] == {
        "name": name,
        "source": VARIABLE,
        "kind": "parameter",
        "statistic": "median",
        "bound": None,
        "unit": "arcsec",
        "n_blank": 0,
    }
    assert columns[f"{name}_lower_1_sigma"]["bound"] == "quantile_1sigma"
    assert columns[f"{name}_at_prior_edge"]["kind"] == "flag"
    assert columns["latent_value"]["kind"] == "latent"
    assert columns["label"]["kind"] == "label"
    assert columns["label"]["n_blank"] == 120
    assert columns["status_reason"]["kind"] == "meta"
    edges = {row[f"{name}_at_prior_edge"] for row in _read(path)}
    assert edges <= {"True", "False", ""}


def test_plain():
    typed = {
        "type": "instance",
        "class_path": "astropy.wcs.WCS",
        "arguments": {
            "crpix": {"type": "ndarray", "array": [1.0, 2.0], "dtype": "float64"},
            "names": {"type": "tuple", "values": ["a", {"type": "path", "path": "/x"}]},
            "scale": {"type": "np.number", "dtype": "float32", "value": 0.1},
            "meta": {"type": "dict", "arguments": {"k": {"type": "list", "values": [1]}}},
        },
    }
    assert plain(typed) == {
        "class_path": "astropy.wcs.WCS",
        "crpix": [1.0, 2.0],
        "names": ["a", "/x"],
        "scale": 0.1,
        "meta": {"k": [1]},
    }
    assert plain([1, {"a": 2}]) == [1, {"a": 2}]


def test_search_output_plain(root):
    output = Aggregator.from_directory(root).search_outputs[0]
    model = output.plain("model")
    assert "type" not in model
    assert set(model["galaxies"]["lens"]["bulge"]) >= {"centre"}
    assert output.plain("absent") is None


# Regressions from the independent review.


def test_header_is_the_same_for_every_shard(tmp_path):
    root = _copy(tmp_path / "output", 4)
    serial = tmp_path / "serial.csv"
    summary = _summary(root)
    summary.add_label_column("lab", {})
    summary.save(serial, sort_by="id", on_error="row")

    parts = []
    for i in range(8):  # most shards are empty
        part = tmp_path / f"part{i}.csv"
        summary.save(part, shard=f"{i}/8", on_error="row")
        parts.append(part)
    header = serial.read_text().splitlines()[0]
    assert {part.read_text().splitlines()[0] for part in parts} == {header}
    merged = tmp_path / "merged.csv"
    merge_csv(parts, merged)
    assert merged.read_bytes() == serial.read_bytes()


def test_error_row_first_does_not_reorder_header(root, tmp_path):
    summary = _summary(root)
    summary.add_label_column("lab", {})
    first = sorted(output.id for output in summary._aggregator)[0]

    def explode(result):
        if result.id == first:
            raise RuntimeError("boom")
        return 1

    summary.add_computed_column("explode", explode)
    path = tmp_path / "s.csv"
    summary.save(path, on_error="row", sort_by="id")
    assert path.read_text().splitlines()[0] == (
        f"id,{VARIABLE.replace('.', '_')},"
        f"{VARIABLE.replace('.', '_')}_lower_1_sigma,"
        f"{VARIABLE.replace('.', '_')}_upper_1_sigma,doubled,lab,explode,status_reason"
    )


def test_symlinked_root_shards_with_expected(tmp_path):
    real = _copy(tmp_path / "real", 12)
    link = tmp_path / "link"
    link.symlink_to(real)
    library = af.ResultsLibrary.grid(
        [f"obj{i:04d}" for i in range(14)],
        {"search": {"products": ["files/model.json"], "location": "{dataset}/search"}},
        roots=real,
        index_dir=tmp_path / "index",
    )
    report = library.check()

    def summary():
        return _summary(link)

    serial = tmp_path / "serial.csv"
    s = summary()
    s.save(serial, expected=report, sort_by="id")
    assert s.stats["expected"] == 2

    parts = []
    for i in range(4):
        part = tmp_path / f"part{i}.csv"
        summary().save(part, expected=report, shard=f"{i}/4")
        parts.append(part)
    merged = tmp_path / "merged.csv"
    assert merge_csv(parts, merged) == 14
    assert merged.read_bytes() == serial.read_bytes()


def test_in_place_latent_change_is_read_again(tmp_path):
    root = _copy(tmp_path / "output", 2)
    summary = AggregateCSV(Aggregator.from_directory(root))
    summary.add_variable("latent.value")
    path = tmp_path / "s.csv"
    summary.save(path, cache=True)

    for latent in root.glob("*/search/*/files/latent/latent_summary.json"):
        latent.write_text(latent.read_text().replace('"latent.value": 1.0', '"latent.value": 99.0'))
    summary = AggregateCSV(Aggregator.from_directory(root))
    summary.add_variable("latent.value")
    summary.save(path, cache=True)

    assert summary.stats["computed"] == 2
    assert "99.0" in path.read_text()


def test_captured_value_change_is_read_again(tmp_path):
    root = _copy(tmp_path / "output", 2)
    path = tmp_path / "s.csv"
    for k, computed in ((1.0, 2), (2.0, 2), (2.0, 0)):
        summary = AggregateCSV(Aggregator.from_directory(root))
        summary.add_computed_column("k", (lambda k: lambda r: k)(k))
        summary.save(path, cache=True)
        assert summary.stats["computed"] == computed
        assert f",{k}" in path.read_text()


def test_cached_csv_equals_fresh_csv(tmp_path):
    import numpy as np

    root = _copy(tmp_path / "output", 3)

    def summary():
        s = AggregateCSV(Aggregator.from_directory(root))
        s.add_computed_column("f32", lambda r: np.float32(0.1))
        s.add_computed_column("tup", lambda r: (1, 2))
        s.add_computed_column("d", lambda r: {0: 5, "a": 6})
        s.add_computed_column("arr", lambda r: np.array([1.0, 2.0]))
        return s

    fresh = tmp_path / "fresh.csv"
    summary().save(fresh, sort_by="id")
    cached = tmp_path / "cached.csv"
    summary().save(cached, cache=True, sort_by="id")
    s = summary()
    s.save(cached, cache=True, sort_by="id")
    assert s.stats["cached"] == 3
    assert cached.read_bytes() == fresh.read_bytes()
    assert "0.1," in fresh.read_text() and "(1, 2)" in fresh.read_text()


def test_spec_digest_stable_across_processes():
    import subprocess
    import sys

    code = (
        "from autofit.aggregator.summary.aggregate_csv.cache import function_digest\n"
        "f = lambda r: r in {'alpha', 'beta', 'gamma', 'delta'}\n"
        "print(function_digest(f))\n"
    )
    digests = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={**__import__("os").environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("1", "2", "3")
    }
    assert len(digests) == 1


def test_plain_compound_dict():
    from autonerves.dictable import to_dict

    assert sorted(plain(to_dict({("a", "b"): 1, ("c", "d"): 2}))) == [
        [["a", "b"], 1],
        [["c", "d"], 2],
    ]


def test_merge_twice_with_unkeyed_rows(tmp_path):
    part = tmp_path / "p.csv"
    part.write_text("id,dataset\r\n,obj1\r\n1,obj0\r\n")
    merged = tmp_path / "m.csv"
    assert merge_csv([part, part], merged) == 2


def test_prior_edge_in_log_space():
    from autofit.aggregator.summary.aggregate_csv.column import Column

    class Model:
        def __init__(self, prior):
            self.prior = prior

        def object_for_path(self, path):
            return self.prior

    class Result:
        pass

    class FakeRow:
        result = Result()

    column = Column("x", prior_edge=True)
    FakeRow.result.model = Model(af.LogUniformPrior(lower_limit=1e-4, upper_limit=1e4))
    assert column._at_prior_edge(FakeRow, 50.0) is False
    assert column._at_prior_edge(FakeRow, 1.1e-4) is True
    FakeRow.result.model = Model(af.UniformPrior(lower_limit=0.0, upper_limit=100.0))
    assert column._at_prior_edge(FakeRow, 0.5) is True
    assert column._at_prior_edge(FakeRow, 50.0) is False
