import shutil
import zipfile
from pathlib import Path

import pytest

from autofit.aggregator import Aggregator
from autofit.aggregator.search_output import SearchOutput


@pytest.fixture(name="scan_directory")
def make_scan_directory(tmp_path):
    source = Path(__file__).parent / "search_output"
    destination = tmp_path / "search_output"
    shutil.copytree(source, destination)
    return tmp_path


@pytest.fixture(name="zipped_directory")
def make_zipped_directory(tmp_path):
    source = Path(__file__).parent / "search_output"
    with zipfile.ZipFile(tmp_path / "search_output.zip", "w") as f:
        for path in source.rglob("*"):
            if path.is_file():
                f.write(path, path.relative_to(source))
    return tmp_path


def test_from_directory(scan_directory):
    aggregator = Aggregator.from_directory(scan_directory)
    assert len(aggregator) == 1


def test_zip_extracted_and_loaded(zipped_directory):
    aggregator = Aggregator.from_directory(zipped_directory)
    assert len(aggregator) == 1
    assert (zipped_directory / "search_output" / "metadata").exists()


def test_zip_not_re_extracted(zipped_directory):
    Aggregator.from_directory(zipped_directory)

    (zipped_directory / "search_output" / ".completed").unlink()
    aggregator = Aggregator.from_directory(zipped_directory)

    assert len(aggregator) == 1
    assert not (zipped_directory / "search_output" / ".completed").exists()


def test_outputs_by_suffix(scan_directory):
    search_output = SearchOutput(scan_directory / "search_output")

    assert {output.name for output in search_output.jsons} == {
        "directory.example",
        "model",
        "samples_info",
        "search",
    }
    assert {output.name for output in search_output.pickles} == {"info"}
    assert {output.name for output in search_output.fits} == {"psf"}
    assert search_output.arrays == []


def test_samples_summary_cached():
    directory = (
        Path(__file__).parent / "summary_files" / "aggregate_summary" / "fit_1"
    )
    search_output = SearchOutput(directory)

    assert search_output.samples_summary is search_output.samples_summary


@pytest.fixture(name="test_mode_directory")
def make_test_mode_directory(tmp_path):
    """
    Mirrors the on-disk layout a search produces under test mode: the results
    live beneath an inserted ``test_mode`` segment (``output/test_mode/prefix``)
    while the caller points ``from_directory`` at the real-run location
    (``output/prefix``), which holds no metadata.
    """
    source = Path(__file__).parent / "search_output"
    real_directory = tmp_path / "output" / "prefix"
    real_directory.mkdir(parents=True)
    shutil.copytree(source, real_directory.parent / "test_mode" / "prefix" / "search_output")
    return real_directory


def test_from_directory_test_mode_fallback(test_mode_directory, monkeypatch):
    monkeypatch.setattr(
        "autofit.aggregator.aggregator.is_test_mode", lambda: True
    )
    aggregator = Aggregator.from_directory(test_mode_directory)
    assert len(aggregator) == 1


def test_from_directory_no_fallback_when_not_test_mode(test_mode_directory, monkeypatch):
    monkeypatch.setattr(
        "autofit.aggregator.aggregator.is_test_mode", lambda: False
    )
    aggregator = Aggregator.from_directory(test_mode_directory)
    assert len(aggregator) == 0
