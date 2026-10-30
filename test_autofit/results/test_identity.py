"""
``dataset_identity.json`` written by real (cheap) fits, and ``Aggregator.latest_per``
over their outputs.
"""
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

import autofit as af
from autofit.non_linear.status.identity import (
    dataset_identity_document,
    sha256_of,
)
from autofit.aggregator.aggregator import Aggregator
from autofit.results import ResultsIndex
from autonerves import conf

CONFIG = Path(__file__).parents[1] / "config"

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(name="output", autouse=True)
def make_output(tmp_path):
    conf.instance.push(new_path=str(CONFIG), output_path=str(tmp_path / "output"))
    return tmp_path / "output"


class IdentifiedAnalysis(af.ex.Analysis):
    def __init__(self, data_path, **kwargs):
        super().__init__(**kwargs)
        self.data_path = data_path

    def dataset_identity(self):
        return {
            "object_id": "lens_42",
            "external_ids": {"tile": "102022527"},
            "inputs": {"data": self.data_path},
            "centre": (0.1, -0.2),
        }


def _analysis(cls=af.ex.Analysis, **kwargs):
    return cls(data=np.ones(20), noise_map=np.ones(20) * 0.1, use_jax=False, **kwargs)


def _fit(analysis, name="identity_fit", unique_tag=None, total_draws=5):
    search = af.Drawer(name=name, unique_tag=unique_tag, total_draws=total_draws)
    search.fit(model=af.Model(af.ex.Gaussian), analysis=analysis)
    return search


def _identity_in_zip(search):
    with zipfile.ZipFile(f"{search.paths.output_path}.zip") as f:
        return json.loads(f.read("files/dataset_identity.json"))


def test_fit_records_its_inputs_with_sha256(tmp_path):
    data_path = tmp_path / "data.fits"
    data_path.write_bytes(b"pixels" * 100)

    search = _fit(_analysis(IdentifiedAnalysis, data_path=data_path))
    identity = _identity_in_zip(search)

    assert identity["schema"] == 1
    assert identity["object_id"] == "lens_42"
    assert identity["external_ids"] == {"tile": "102022527"}
    assert identity["centre"] == [0.1, -0.2]
    assert identity["inputs"]["data"]["sha256"] == sha256_of(data_path)
    assert identity["inputs"]["data"]["bytes"] == 600
    assert "autofit" in identity["versions"]


def test_default_identity_is_the_unique_tag():
    search = _fit(_analysis(), unique_tag="dataset_7")

    identity = _identity_in_zip(search)

    assert identity["object_id"] == "dataset_7"
    assert identity["inputs"] == {}


def test_sha256_switch_and_off_switch(tmp_path, monkeypatch):
    data_path = tmp_path / "data.fits"
    data_path.write_bytes(b"pixels")

    document = dataset_identity_document(
        {"inputs": [data_path]}, unique_tag=None, sha256=False
    )
    assert document["inputs"]["data.fits"] == {
        "path": str(data_path.resolve()),
        "bytes": 6,
    }

    output = conf.instance["general"]["output"]
    monkeypatch.setitem(output, "dataset_identity", False)
    search = _fit(_analysis(), name="no_identity")
    with zipfile.ZipFile(f"{search.paths.output_path}.zip") as f:
        assert "files/dataset_identity.json" not in f.namelist()


def test_missing_input_is_recorded_not_raised(tmp_path):
    document = dataset_identity_document(
        {"inputs": {"psf": tmp_path / "absent.fits"}}, unique_tag="x"
    )
    assert document["inputs"]["psf"]["missing"] is True


def test_index_reads_identity_from_a_real_fit(output, tmp_path):
    data_path = tmp_path / "data.fits"
    data_path.write_bytes(b"pixels")
    _fit(_analysis(IdentifiedAnalysis, data_path=data_path))

    index = ResultsIndex(output)
    index.update()
    (record,) = index.records()

    assert record.state == "completed"
    assert record.object_id == "lens_42"
    assert record.inputs == {"data": sha256_of(data_path)}
    assert record.products["dataset_identity"]
    assert record.completed_at_source == "status"
    assert record.status["state"] == "completed"


def test_latest_per_keeps_the_latest_sibling(output):
    _fit(_analysis(), name="search", unique_tag="dataset_0", total_draws=5)
    second = _fit(_analysis(), name="search", unique_tag="dataset_0", total_draws=6)
    _fit(_analysis(), name="search", unique_tag="dataset_1", total_draws=5)
    second_completed = json.loads(
        Path(f"{second.paths.output_path}.status.json").read_text()
    )["updated"]

    aggregator = Aggregator.from_directory(output)
    assert len(aggregator) == 3

    latest = aggregator.latest_per()

    assert len(latest) == 2
    kept = {o.unique_tag: o for o in latest}
    # From the status record beside each output, not the extraction's file times.
    assert kept["dataset_0"].completed_at == second_completed
    assert Path(kept["dataset_0"].directory) == Path(second.paths.output_path)
