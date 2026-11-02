import csv
import textwrap
from pathlib import Path

import pytest

from autofit.results import (
    LibraryException,
    ResultsLibrary,
)
from autofit.results.__main__ import main
from autofit.tools.util import write_zip_member

from .tree import write_folder, write_status, write_zip

PRODUCTS = ["image", "files/model.fits"]


def _with_model(output_path):
    write_zip_member(f"{output_path}.zip", "files/model.fits", data=b"SIMPLE")
    return output_path


@pytest.fixture(name="holes")
def make_holes(results_root):
    """
    One result per way a library can have a hole.
    """
    _with_model(write_zip(results_root, prefix="d0"))  # done
    # d1: no result at all
    write_zip(results_root, prefix="d2", completed=False)  # zip without .completed
    write_zip(results_root, prefix="d3")  # completed, no files/model.fits
    write_folder(results_root, prefix="d4", completed=False, checkpoint="valid")
    _with_model(write_zip(results_root, prefix="d5", seed=1, date_time=(2026, 9, 1, 0, 0, 0)))
    _with_model(write_zip(results_root, prefix="d5", seed=2, date_time=(2026, 10, 1, 0, 0, 0)))
    failed = results_root / "d6" / "search" / ("f" * 32)
    failed.parent.mkdir(parents=True)
    write_status(failed, "failed", failure_class="oom")  # failed before search.json
    _with_model(write_zip(results_root, prefix="a/d7", unique_tag="d7", seed=3))
    _with_model(write_zip(results_root, prefix="b/d7", unique_tag="d7", seed=4))
    _with_model(write_zip(results_root, prefix="zz"))  # not declared
    return results_root


def _by_dataset(report):
    return {result.expected.dataset: result for result in report.results}


def test_every_hole_has_its_reason(holes):
    library = ResultsLibrary.grid(
        [f"d{i}" for i in range(8)], {"search": PRODUCTS}, roots=holes
    )
    report = library.check()
    results = _by_dataset(report)

    assert results["d0"].status == "done"
    assert (results["d1"].status, results["d1"].reason) == ("missing", "no_result")
    assert results["d2"].status == "failed"
    assert results["d2"].reason.startswith("not_completed:corrupt_zip")
    assert results["d3"].status == "incomplete"
    assert results["d3"].missing_products == ["files/model.fits"]
    assert results["d4"].status == "incomplete"
    assert results["d4"].reason.startswith("not_completed:interrupted_valid_ckpt")
    assert results["d5"].status == "done"  # the superseded sibling does not count
    assert results["d6"].status == "failed"
    assert results["d6"].reason.startswith("not_completed:failed")
    assert (results["d7"].status, results["d7"].reason) == ("duplicate", "duplicate:2")

    assert report.counts() == {
        "expected": 8,
        "done": 2,
        "missing": 1,
        "incomplete": 2,
        "failed": 2,
        "duplicate": 1,
        "unexpected": 1,
    }
    assert report.product_counts()["files/model.fits"] == {"required": 8, "missing": 1}
    assert [record.path_prefix for record in report.unexpected] == ["zz"]
    assert not report.is_complete


def test_missing_csv(holes, tmp_path):
    library = ResultsLibrary.grid(
        [f"d{i}" for i in range(8)], {"search": PRODUCTS}, roots=holes, name="lib"
    )
    path = library.check().write_missing_csv(tmp_path)

    assert path == tmp_path / "lib.missing.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    gaps = {(row["dataset"], row["product"]): row for row in rows}
    assert set(gaps) == {
        ("d1", ""),
        ("d2", ""),
        ("d3", "files/model.fits"),
        ("d4", ""),
        ("d6", ""),
        ("d7", ""),
    }
    assert gaps[("d3", "files/model.fits")]["reason"] == "missing_product:files/model.fits"
    assert gaps[("d3", "files/model.fits")]["output_path"].endswith(".zip") is False
    assert gaps[("d1", "")]["output_path"] == ""
    assert not list(tmp_path.glob("*.tmp"))


def test_named_and_folder_products(results_root):
    write_zip(results_root, prefix="d0")
    write_folder(results_root, prefix="d1")
    library = ResultsLibrary.grid(
        ["d0", "d1"], {"search": ["image", "image/", "samples", "latent"]}, roots=results_root
    )
    results = _by_dataset(library.check())

    assert results["d0"].missing_products == ["latent"]
    assert results["d1"].missing_products == ["latent"]


def test_location(results_root):
    write_zip(results_root, prefix="d0")
    write_zip(results_root, prefix="moved", unique_tag="d1")
    library = ResultsLibrary.grid(
        ["d0", "d1"],
        {"search": {"products": ["image"], "location": "{dataset}/search"}},
        roots=results_root,
    )
    results = _by_dataset(library.check())

    assert results["d0"].status == "done"
    # Found by key it would be done; at its declared location there is nothing.
    assert results["d1"].reason == "no_result"


def test_from_yaml(holes, tmp_path):
    (tmp_path / "manifest.txt").write_text("# objects\nd0 extra\n\nd3\n")
    path = tmp_path / "lib.yaml"
    path.write_text(
        textwrap.dedent(
            f"""
            roots: [{holes}]
            datasets_file: manifest.txt
            searches:
              search: [image, files/model.fits]
            """
        )
    )
    library = ResultsLibrary.from_yaml(path)
    report = library.check()

    assert library.name == "lib"
    assert [e.dataset for e in library.expected] == ["d0", "d3"]
    assert report.counts()["done"] == 1


@pytest.mark.parametrize(
    "text",
    [
        "roots: [o]\nsearches: {s: []}\n",
        "roots: [o]\ndatasets: [a]\ndatasets_file: m.txt\nsearches: {s: []}\n",
        "roots: [o]\ndatasets: [a]\n",
        "datasets: [a]\nsearches: {s: []}\n",
        "roots: [o]\ndatasets: [a]\nsearches: {s: []}\nstages: [x]\n",
        "roots: [o]\ndatasets: [a]\nsearches: {s: {product: [x]}}\n",
        "roots: [o]\ndatasets: [a]\nsearches: {s: []}\ntypo: 1\n",
        "from_campaign: c.yaml\ndatasets: [a]\n",
    ],
)
def test_from_yaml_refuses(tmp_path, text):
    path = tmp_path / "lib.yaml"
    path.write_text(text)
    with pytest.raises(LibraryException):
        ResultsLibrary.from_yaml(path)


def _campaign_zip(output, tile, completed=True, members=True):
    """
    A result zip exactly where the test campaign's ``success.zip_complete`` names it:
    ``output/dr1/<tile>/vis_lp.zip``.
    """
    path = write_zip(output, prefix="dr1", name=tile, completed=completed)
    target = output / "dr1" / tile / "vis_lp"
    Path(f"{path}.zip").rename(f"{target}.zip")
    return target


def test_from_campaign(tmp_path):
    from autofit.campaign.collector import check_success
    from autofit.campaign.spec import Campaign
    from test_autofit.campaign.conftest import write_campaign

    project = tmp_path / "project"
    campaign_file = write_campaign(project, n=4)
    output = project / "output"
    _campaign_zip(output, "Tile000")
    _campaign_zip(output, "Tile001", completed=False)
    # Under <zip_complete>/<identifier>: not where the campaign's success check looks.
    write_zip(output, prefix="dr1/Tile003", name="vis_lp")

    library = ResultsLibrary.from_campaign(campaign_file, products={"vis_lp": "image"})
    report = library.check()
    results = _by_dataset(report)

    assert library.name == "demo"
    assert [e.search for e in library.expected] == ["vis_lp"] * 4
    assert library.expected[0].products == ("files/samples_summary.json", "image")
    assert results["dr1 Tile000"].status == "done"
    assert results["dr1 Tile001"].status == "failed"
    assert results["dr1 Tile002"].reason == "no_result"
    assert results["dr1 Tile003"].reason == "no_result"

    # The library and the campaign collector agree on every task.
    campaign = Campaign.load(campaign_file)
    stage = campaign.stages["vis_lp"]
    for index in campaign.indices():
        done = results[campaign.fields(index)["line"]].status == "done"
        assert done == bool(check_success(campaign, stage, index))

    yaml_path = project / "lib.yaml"
    yaml_path.write_text("from_campaign: campaign.yaml\nproducts: {vis_lp: [image]}\n")
    assert ResultsLibrary.from_yaml(yaml_path).check().counts() == report.counts()

    with pytest.raises(LibraryException):
        ResultsLibrary.from_campaign(campaign_file, stages=["vis_pix"])


def test_from_campaign_short_manifest_line(tmp_path):
    from test_autofit.campaign.conftest import write_campaign

    project = tmp_path / "project"
    campaign_file = write_campaign(project, n=2)
    (project / "manifest.txt").write_text("dr1 Tile000\ndr1\n")

    results = _by_dataset(ResultsLibrary.from_campaign(campaign_file).check())

    assert results["dr1"].status == "failed"
    assert results["dr1"].reason.startswith("manifest_line:1 has no field")
    assert results["dr1 Tile000"].reason == "no_result"


def test_exact_location_counts_a_superseded_result(results_root):
    from autofit.results import ExpectedResult

    original = write_zip(results_root, prefix="d0")
    write_zip(results_root / "zbackup", prefix="d0")  # same key, wins the tie
    library = ResultsLibrary(
        [ExpectedResult("d0", "search", (), str(original), exact=True)],
        roots=results_root,
    )
    assert library.check().results[0].status == "done"


def test_string_product_is_one_product(results_root):
    write_zip(results_root, prefix="d0")
    library = ResultsLibrary.grid(
        ["d0"], {"search": {"products": "image"}}, roots=results_root
    )
    assert library.expected[0].products == ("image",)
    assert library.check().results[0].status == "done"


def test_several_roots_need_an_index_dir(tmp_path):
    with pytest.raises(LibraryException):
        ResultsLibrary([], roots=[tmp_path / "a", tmp_path / "b"])


def test_cli_check(holes, tmp_path, capsys):
    path = tmp_path / "lib.yaml"
    path.write_text(f"roots: [{holes}]\ndatasets: [d0, d3]\nsearches: {{search: [files/model.fits]}}\n")

    assert main(["check", str(path)]) == 1
    out = capsys.readouterr().out
    assert "1 of 2 expected results done" in out
    assert (tmp_path / "lib.missing.csv").exists()

    path.write_text(f"roots: [{holes}]\ndatasets: [d0]\nsearches: {{search: [image]}}\n")
    assert main(["check", str(path), "--no-update", "--output-dir", str(tmp_path / "o")]) == 0
    assert (tmp_path / "o" / "lib.missing.csv").read_text().count("\n") == 1

    # An error is 2, never 1 ("incomplete").
    path.write_text(f"roots: [{holes}, {tmp_path}]\ndatasets: [d0]\nsearches: {{search: []}}\n")
    assert main(["check", str(path)]) == 2
