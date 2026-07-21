"""
Tests for the results-inspector MCP tool functions (`autofit/mcp/tools.py`, the
``autofit[mcp]`` extra).

The fixture runs a real (tiny) `af.LBFGS` fit of the `af.ex.Gaussian` toy so the
output directory always matches the current on-disk format — a frozen fixture
directory would silently drift. LBFGS is an MLE search, so `log_evidence` is
legitimately absent; tests assert that shape rather than skipping it.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import autofit as af
from autonerves import conf

from autofit.mcp import tools

# The shipped config (autofit/config), not test_autofit/config: it keeps search
# output files on disk (remove_files: false), whereas the test config zips them
# away — and the read-only tools operate on real, unzipped output directories,
# exactly as an end user's `pip install autofit` config leaves them.
CONFIG = Path(__file__).resolve().parents[2] / "autofit" / "config"


@pytest.fixture(scope="module")
def output_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("output")
    conf.instance.push(new_path=str(CONFIG), output_path=str(root))

    xvalues = np.arange(100.0)
    gaussian = af.ex.Gaussian(centre=50.0, normalization=25.0, sigma=10.0)
    data = gaussian.model_data_from(xvalues=xvalues)
    noise_map = np.full(fill_value=1.0, shape=data.shape)
    analysis = af.ex.Analysis(data=data, noise_map=noise_map)

    for name in ("fit_0", "fit_1"):
        search = af.LBFGS(name=name, path_prefix="mcp_fixture")
        search.fit(model=af.Model(af.ex.Gaussian), analysis=analysis)

    return root


@pytest.fixture(scope="module")
def fit_directory(output_root):
    directory = sorted(
        marker.parent for marker in output_root.rglob(".completed")
    )[0]
    (directory / "image").mkdir(exist_ok=True)
    Image.new("RGB", (32, 16), color=(200, 40, 40)).save(
        directory / "image" / "subplot_fit.png"
    )
    return directory


def test_list_searches(output_root, fit_directory):
    rows = tools.list_searches(str(output_root), sort_by="max_log_likelihood")

    assert len(rows) == 2
    assert {row["name"] for row in rows} == {"fit_0", "fit_1"}
    for row in rows:
        assert row["is_complete"] is True
        assert isinstance(row["max_log_likelihood"], float)
        assert row["log_evidence"] is None
        assert row["model_free_parameters"] == 3


def test_list_searches_limit(output_root):
    assert len(tools.list_searches(str(output_root), limit=1)) == 1


def test_list_searches_sorts_none_last(output_root):
    rows = tools.list_searches(str(output_root), sort_by="log_evidence")
    assert len(rows) == 2


def test_get_model(fit_directory):
    result = tools.get_model(str(fit_directory))

    assert "Gaussian" in result["info"]
    assert result["model"]["class_path"].endswith("Gaussian")


def test_get_result_summary(fit_directory):
    text = tools.get_result_summary(str(fit_directory))

    assert "Maximum Log Likelihood" in text


def test_get_samples_summary(fit_directory):
    summary = tools.get_samples_summary(str(fit_directory))

    assert summary["log_evidence"] is None
    assert isinstance(summary["max_log_likelihood"], float)
    assert summary["parameter_paths"] == [
        "centre",
        "normalization",
        "sigma",
    ]
    assert len(summary["max_log_likelihood_parameters"]) == 3
    assert summary["median_pdf_parameters"] is None
    assert summary["max_log_likelihood_parameters"][0] == pytest.approx(
        50.0, abs=1.0
    )


def test_get_search_info(fit_directory):
    info = tools.get_search_info(str(fit_directory))

    assert info["name"] == "fit_0"
    assert info["is_complete"] is True
    assert info["search"] is not None


def test_list_images(fit_directory):
    assert "subplot_fit.png" in tools.list_images(str(fit_directory))


def test_fetch_image(fit_directory):
    image = tools.fetch_image(str(fit_directory), name="subplot_fit")

    assert image.size == (32, 16)


def test_fetch_image_missing(fit_directory):
    with pytest.raises(Exception):
        tools.fetch_image(str(fit_directory), name="no_such_image")
