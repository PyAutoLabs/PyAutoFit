import csv
import random
import shutil
from pathlib import Path

import pytest

from autofit.aggregator import Aggregator
from autofit.export import compare, scatter_between
from autofit.export.__main__ import main

FIXTURE = Path(__file__).parents[1] / "aggregator" / "summary_files" / "aggregate_summary"


def _table(n=50, shift=0.0, seed=0, sigma=0.1):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        value = i + shift + rng.gauss(0, sigma)
        rows.append(
            {
                "id": f"obj{i:03d}",
                "ra": f"{10 + i * 0.01:.6f}",
                "einstein_radius": value,
                "einstein_radius_lower_1_sigma": value - sigma,
                "einstein_radius_upper_1_sigma": value + sigma,
                "flag": "ok",
            }
        )
    return rows


def test_reruns_report_scatter_not_fail():
    report = compare(_table(seed=1), _table(seed=2))
    column = report.columns["einstein_radius"]
    assert report.passed
    assert column.kind == "sigma" and not column.gated
    assert 0.3 < column.median_z < 2.0
    assert column.max_z > column.median_z
    assert "PASS" in report.to_markdown()
    assert compare(_table(), _table()).columns["einstein_radius"].worst == []


def test_gated_column_fails_beyond_threshold():
    report = compare(_table(), _table(shift=1.0), gate=["einstein_radius"])
    assert not report.passed
    assert report.columns["einstein_radius"].n_mismatch == 50
    assert "FAIL" in report.to_markdown()


def test_identity_and_duplicates_fail():
    a = _table()
    b = _table()[:-2] + [_table()[0]]
    report = compare(a, b)
    assert report.only_a == ["obj048", "obj049"]
    assert report.duplicates_b == ["obj000"]
    assert not report.passed


def test_exact_columns():
    b = _table()
    b[3]["ra"] = "99.0"
    report = compare(_table(), b, exact=["ra"])
    assert report.columns["ra"].kind == "exact"
    assert report.columns["ra"].n_mismatch == 1
    assert not report.passed
    assert compare(_table(), _table(), exact=["ra", "flag"]).passed


def test_error_floor():
    a = _table(sigma=0.0)
    b = _table(sigma=0.0, shift=1e-6)
    assert compare(a, b).columns["einstein_radius"].max_z == float("inf")
    assert compare(a, b, floor=1e-3).columns["einstein_radius"].max_z < 0.01
    assert compare(a, b, rel_floor=0.01).columns["einstein_radius"].median_z < 1


def test_blank_and_composite_key():
    a = _table()
    b = _table()
    a[0]["einstein_radius"] = ""
    for row in a + b:
        row["band"] = "vis"
    report = compare(a, b, key=["id", "band"])
    assert report.key == ("id", "band")
    assert report.columns["einstein_radius"].n_blank == 1


def test_cli_compare(tmp_path):
    def write(path, rows):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    a = write(tmp_path / "a.csv", _table(seed=1))
    b = write(tmp_path / "b.csv", _table(seed=2))
    assert main(["compare", str(a), str(b), "--report", str(tmp_path / "r.md")]) == 0
    assert (tmp_path / "r.md").read_text().startswith("# Parity: PASS")
    assert main(["compare", str(a), str(b), "--gate", "einstein_radius", "--threshold", "0.01"]) == 1
    assert main(["compare", str(a), str(tmp_path / "absent.csv")]) == 2


def _copies(root, n):
    for i in range(n):
        shutil.copytree(FIXTURE / ("fit_1" if i % 2 else "fit_2"), root / f"obj{i}" / "fit")
    return Aggregator.from_directory(root)


def test_scatter_between(tmp_path):
    a = _copies(tmp_path / "a", 2)
    b = _copies(tmp_path / "b", 2)

    def key(output):
        return Path(output.directory).parent.name

    report = scatter_between(
        a, b, ["galaxies.lens.bulge.centre.centre_0", "latent.value", "nope"], key=key
    )
    assert report.matched == 2
    centre = report.paths["galaxies.lens.bulge.centre.centre_0"]
    assert centre.n == 2 and centre.median_z == 0.0
    assert report.paths["nope"].n == 0
    assert "matched fits" in str(report)


# Regressions from the independent review.


def test_missing_and_unknown_columns_fail():
    a = _table()
    b = [{k: v for k, v in row.items() if k != "flag"} for row in _table()]
    report = compare(a, b)
    assert report.columns_only_a == ["flag"] and not report.passed
    report = compare(_table(), _table(), exact=["typo"], gate=["nope"])
    assert report.unknown == ["nope", "typo"] and not report.passed


def test_value_lost_on_one_side_fails_a_gate():
    a = [{"id": "o1", "n": "x"}]
    b = [{"id": "o1", "n": ""}]
    assert not compare(a, b, exact=["n"]).passed
    assert compare(a, b).passed  # informational column: reported as blank


def test_columns_keeps_sigma_companions():
    report = compare(_table(seed=1), _table(seed=2), gate=["einstein_radius"], columns=["einstein_radius"])
    assert report.columns["einstein_radius"].kind == "sigma"
    assert list(report.columns) == ["einstein_radius"]


def test_exact_large_integer_ids():
    a = [{"id": "1", "oid": "9007199254740993"}]
    b = [{"id": "1", "oid": "9007199254740992"}]
    assert not compare(a, b, exact=["oid"]).passed
    assert compare([{"id": "1", "x": "1.0"}], [{"id": "1", "x": "1"}], exact=["x"]).passed


def test_scatter_key_must_be_unique(tmp_path):
    a = _copies(tmp_path / "a", 4)
    b = _copies(tmp_path / "b", 4)
    with pytest.raises(ValueError, match="several fits"):
        scatter_between(a, b, ["galaxies.lens.bulge.centre.centre_0"], key=lambda o: "same")
