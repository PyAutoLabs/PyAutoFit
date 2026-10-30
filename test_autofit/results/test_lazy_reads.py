"""
Reading zipped results in place: ``ZipMember``, ``SearchOutput.from_zip``,
``open_member``, ``value(kind=...)`` and ``Aggregator.from_index`` with nothing
extracted to disk.
"""
import logging
import zipfile
from pathlib import Path

import numpy as np
import pytest

import autofit as af
from autofit.aggregator import search_output as search_output_module
from autofit.aggregator.aggregator import Aggregator
from autofit.aggregator.search_output import SearchOutput
from autofit.aggregator.zip_member import ZipMember
from autonerves import conf

CONFIG = Path(__file__).parents[1] / "config"

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

FIXTURE = Path(__file__).parents[1] / "aggregator" / "search_output"


@pytest.fixture(name="output", autouse=True)
def make_output(tmp_path):
    conf.instance.push(new_path=str(CONFIG), output_path=str(tmp_path / "output"))
    return tmp_path / "output"


@pytest.fixture(name="zipped_fixture")
def make_zipped_fixture(tmp_path):
    """
    The aggregator's directory fixture (json, pickle, csv and fits files, analyses)
    archived as a zip.
    """
    zip_path = tmp_path / "search_output.zip"
    with zipfile.ZipFile(zip_path, "w") as f:
        for path in FIXTURE.rglob("*"):
            if path.is_file():
                f.write(path, path.relative_to(FIXTURE).as_posix())
    return zip_path


def _fit(name, unique_tag, total_draws=5):
    search = af.Drawer(name=name, unique_tag=unique_tag, total_draws=total_draws)
    search.fit(
        model=af.Model(af.ex.Gaussian),
        analysis=af.ex.Analysis(
            data=np.ones(20), noise_map=np.ones(20) * 0.1, use_jax=False
        ),
    )
    return search


def test_zip_member_paths(zipped_fixture):
    root = ZipMember(zipped_fixture)
    files = root / "files"

    assert files.is_dir() and not files.is_file()
    assert (files / "model.json").is_file()
    assert not (files / "absent.json").exists()
    assert (files / "model.json").suffix == ".json"
    assert (files / "model.json").stem == "model"
    assert (files / "model.json").relative_to(files).as_posix() == "model.json"
    assert {p.name for p in root.iterdir()} >= {"files", "analyses", ".completed"}
    assert sorted(p.name for p in root.glob("analyses/*")) == [
        "analysis_0",
        "analysis_1",
    ]
    assert any(p.name == "psf.fits" for p in root.rglob("*.fits"))
    with pytest.raises(FileNotFoundError):
        (files / "absent.json").open()
    with pytest.raises(ValueError):
        (files / "model.json").open("w")


def test_zipped_output_reads_like_the_folder(zipped_fixture):
    folder = SearchOutput(FIXTURE)
    zipped = SearchOutput.from_zip(zipped_fixture)

    assert zipped.is_complete
    assert zipped.id == folder.id
    assert zipped.model.prior_count == folder.model.prior_count
    assert zipped.search.__class__ is folder.search.__class__
    assert len(zipped.children) == len(folder.children) == 2
    assert sorted(o.name for o in zipped.jsons) == sorted(o.name for o in folder.jsons)
    assert zipped.value("psf", kind="fits")[0].data.shape == (
        folder.value("psf", kind="fits")[0].data.shape
    )
    assert zipped.value("info") == folder.value("info")


def test_open_member(zipped_fixture):
    zipped = SearchOutput.from_zip(zipped_fixture)
    with zipped.open_member("files/search.json", "r") as f:
        assert "class_path" in f.read()
    with SearchOutput(FIXTURE).open_member("files/search.json") as f:
        assert b"class_path" in f.read()


def test_value_kind_resolves_a_name_shared_by_json_and_fits(tmp_path, caplog):
    directory = tmp_path / "out"
    (directory / "files").mkdir(parents=True)
    (directory / "files" / "tracer.json").write_text('{"a": 1}')
    from astropy.io import fits

    fits.HDUList([fits.PrimaryHDU(np.ones((2, 2)))]).writeto(
        directory / "files" / "tracer.fits"
    )
    output = SearchOutput(directory)
    search_output_module._warned_ambiguous.discard("tracer")

    with caplog.at_level(logging.WARNING):
        assert output.value("tracer") == {"a": 1}
    assert "Pass kind= to choose" in caplog.text

    assert output.value("tracer", kind="fits")[0].data.shape == (2, 2)
    assert output.value("tracer", kind="json") == {"a": 1}
    assert output.value("tracer", kind="pickle") is None
    with pytest.raises(ValueError):
        output.value("tracer", kind="png")


def test_from_index_extracts_nothing(output, tmp_path, monkeypatch):
    _fit("search", "dataset_0", total_draws=5)
    _fit("search", "dataset_0", total_draws=6)
    _fit("search", "dataset_1")

    def refuse(*args, **kwargs):
        raise AssertionError("from_index must not extract")

    monkeypatch.setattr(zipfile.ZipFile, "extractall", refuse)
    monkeypatch.setattr(zipfile.ZipFile, "extract", refuse)
    before = sorted(output.rglob("*"))

    aggregator = Aggregator.from_index(output, index_dir=tmp_path / "index")

    # The older dataset_0 fit is superseded and left out by default.
    assert len(aggregator) == 2
    assert all(isinstance(o.directory, ZipMember) for o in aggregator)
    assert sorted(o.unique_tag for o in aggregator) == ["dataset_0", "dataset_1"]
    assert all(o.samples_summary is not None for o in aggregator)
    assert all(o.completed_at is not None for o in aggregator)
    assert len(list(aggregator.values("samples_summary"))) == 2
    assert sorted(output.rglob("*")) == before

    everything = Aggregator.from_index(
        output,
        index_dir=tmp_path / "index",
        states=("completed", "superseded"),
        update=False,
    )
    assert len(everything) == 3
    assert len(everything.latest_per()) == 2
