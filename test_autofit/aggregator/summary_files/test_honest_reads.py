"""
Honest reads (results-library phase 0b): the aggregator and AggregateCSV report
problems (a corrupt zip, an unknown column, a column only some rows have, an
interrupted save) instead of hiding them.
"""
import csv
import logging
import shutil
import zipfile
from pathlib import Path

import pytest

import autofit as af
from autofit.aggregator import Aggregator
from autofit.aggregator.summary.aggregate_csv import AggregateCSV, ValueType
from autofit.aggregator.summary.aggregate_csv.column import Column
from autofit.aggregator.summary.aggregate_csv.row import Row
from autofit.example.model import Gaussian
from autofit.non_linear.samples.sample import Sample
from autofit.non_linear.samples.summary import SamplesSummary

AGGREGATE_SUMMARY = Path(__file__).parent / "aggregate_summary"


@pytest.fixture(name="tree")
def make_tree(tmp_path):
    """
    A results tree of the two summary fixtures, one zipped, plus one corrupt zip.
    """
    tree = tmp_path / "tree"
    shutil.copytree(AGGREGATE_SUMMARY, tree)

    shutil.make_archive(str(tree / "fit_2"), "zip", tree / "fit_2")
    shutil.rmtree(tree / "fit_2")

    (tree / "fit_3.zip").write_bytes(b"PK\x03\x04 truncated")

    return tree


def test__from_directory__corrupt_zip_is_collected(tree):
    aggregator = Aggregator.from_directory(tree)

    assert len(aggregator) == 2
    assert len(aggregator.errors) == 1

    path, error = aggregator.errors[0]
    assert Path(path).name == "fit_3.zip"
    assert isinstance(error, zipfile.BadZipFile)


def test__from_directory__corrupt_zip_raises_on_request(tree):
    with pytest.raises(zipfile.BadZipFile, match="fit_3.zip"):
        Aggregator.from_directory(tree, on_error="raise")


def test__from_directory__unzip_temporary_collects_corrupt_zip(tree):
    with Aggregator.from_directory(tree, unzip_temporary=True) as aggregator:
        assert len(aggregator) == 2
        assert len(aggregator.errors) == 1


def test__from_directory__damaged_member_is_collected(tree):
    """
    A zip whose central directory reads but whose member data is damaged fails only
    on extraction (``zlib.error`` / CRC ``BadZipFile``), and is collected the same way.
    """
    (tree / "fit_3.zip").unlink()

    damaged = tree / "fit_4.zip"
    with zipfile.ZipFile(damaged, "w", zipfile.ZIP_DEFLATED) as f:
        f.writestr("data.txt", "".join(str(i) for i in range(5000)))

    data = bytearray(damaged.read_bytes())
    start = zipfile.sizeFileHeader + len("data.txt")
    for i in range(start, start + 64):
        data[i] ^= 0xFF
    damaged.write_bytes(bytes(data))

    aggregator = Aggregator.from_directory(tree)

    assert len(aggregator) == 2
    assert [Path(path).name for path, _ in aggregator.errors] == ["fit_4.zip"]

    with pytest.raises(zipfile.BadZipFile, match="fit_4.zip"):
        Aggregator.from_directory(tree, on_error="raise")


def test__from_directory__rejects_unknown_on_error(tree):
    with pytest.raises(ValueError, match="on_error"):
        Aggregator.from_directory(tree, on_error="ignore")


def test__from_directory__empty_warns_naming_directory_and_filters(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="autofit.aggregator.aggregator"):
        aggregator = Aggregator.from_directory(tmp_path, completed_only=True)

    assert len(aggregator) == 0
    messages = [record.getMessage() for record in caplog.records]
    assert any(str(tmp_path) in m and "completed_only=True" in m for m in messages)


def test__from_directory__reports_through_logging_not_print(tree, capsys):
    Aggregator.from_directory(tree)

    assert capsys.readouterr().out == ""


# --- AggregateCSV ----------------------------------------------------------------------


@pytest.fixture(name="aggregator")
def make_aggregator():
    return Aggregator.from_directory(AGGREGATE_SUMMARY)


def _load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test__aggregate_csv__unknown_column_raises_by_default(aggregator, tmp_path):
    summary = AggregateCSV(aggregator)
    summary.add_variable("does.not.exist")

    with pytest.raises(KeyError, match="does.not.exist"):
        summary.save(tmp_path / "summary.csv")

    assert not (tmp_path / "summary.csv").exists()


def _name(output):
    # The two summary fixtures are copies with the same search id, so the tests
    # join on the directory name instead.
    return Path(output.directory).name


def test__aggregate_csv__header_is_union_of_rows(aggregator, tmp_path):
    def compute(result):
        if _name(result) == "fit_1":
            return {"only_first": 1}
        return {"only_second": 2}

    summary = AggregateCSV(aggregator)
    summary.add_computed_column("computed", compute)
    summary.save(tmp_path / "summary.csv")

    with open(tmp_path / "summary.csv") as f:
        header = next(csv.reader(f))

    assert header == ["id", "computed_only_first", "computed_only_second"]

    rows = _load(tmp_path / "summary.csv")
    assert rows[0]["computed_only_first"] == "1"
    assert rows[0]["computed_only_second"] == ""
    assert rows[1]["computed_only_second"] == "2"


def test__aggregate_csv__interrupted_save_leaves_previous_csv(
    aggregator, tmp_path, monkeypatch
):
    path = tmp_path / "summary.csv"

    summary = AggregateCSV(aggregator)
    summary.add_variable("galaxies.lens.bulge.centre.centre_0")
    summary.save(path)
    before = path.read_bytes()

    original = csv.DictWriter.writerow
    calls = []

    def writerow(self, row):
        calls.append(row)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return original(self, row)

    monkeypatch.setattr(csv.DictWriter, "writerow", writerow)

    summary.add_label_column("label", ["a", "b"])
    with pytest.raises(KeyboardInterrupt):
        summary.save(path)

    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test__aggregate_csv__save_reports_blank_columns(aggregator, tmp_path, caplog):
    summary = AggregateCSV(aggregator)
    summary.add_computed_column("nothing", lambda result: None)

    with caplog.at_level(
        logging.WARNING, logger="autofit.aggregator.summary.aggregate_csv"
    ):
        blank = summary.save(tmp_path / "summary.csv")

    assert blank == ["nothing"]
    assert any("nothing" in record.getMessage() for record in caplog.records)


def test__aggregate_csv__label_column_joined_by_key(aggregator, tmp_path):
    summary = AggregateCSV(aggregator)
    summary.add_label_column("name", {"fit_2": "fit_2", "fit_1": "fit_1"}, key=_name)
    summary.add_computed_column("directory", _name)
    summary.save(tmp_path / "summary.csv")

    for row in _load(tmp_path / "summary.csv"):
        assert row["name"] == row["directory"]


def test__aggregate_csv__label_column_joined_by_search_id(aggregator, tmp_path):
    summary = AggregateCSV(aggregator)
    summary.add_label_column("label", {aggregator[0].id: "labelled"})
    summary.save(tmp_path / "summary.csv")

    assert [row["label"] for row in _load(tmp_path / "summary.csv")] == [
        "labelled",
        "labelled",
    ]


def test__aggregate_csv__label_column_missing_key_raises(aggregator, tmp_path):
    summary = AggregateCSV(aggregator)
    summary.add_label_column("label", {"fit_1": "first"}, key=_name)

    with pytest.raises(KeyError, match="label"):
        summary.save(tmp_path / "summary.csv")


def test__aggregate_csv__label_list_of_wrong_length_raises(aggregator):
    summary = AggregateCSV(aggregator)

    with pytest.raises(ValueError, match="label"):
        summary.add_label_column("label", ["a", "b", "c"])


# --- Row: asymmetric latent PDF --------------------------------------------------------


class _AsymmetricLatentResult:
    """
    A search output whose latent PDF is asymmetric about its median, so a lower and
    upper bound, a 1-sigma and a 3-sigma interval and the median and max-likelihood
    values are all distinct numbers: any swap between them shows up as a wrong value.
    """

    id = "asymmetric"

    def __init__(self):
        self.model = af.Model(Gaussian)
        sample = Sample(
            log_likelihood=1.0,
            log_prior=0.0,
            weight=1.0,
            kwargs={"centre": 0.5, "normalization": 2.0, "sigma": 3.0},
        )
        self.samples_summary = SamplesSummary(
            model=self.model,
            max_log_likelihood_sample=sample,
            median_pdf_sample=sample,
        )
        self.latent_summary = SamplesSummary(
            model=None,
            median_pdf_sample=Sample(
                log_likelihood=1.0,
                log_prior=0.0,
                weight=1.0,
                kwargs={"latent.flux": 3.0},
            ),
            max_log_likelihood_sample=Sample(
                log_likelihood=2.0,
                log_prior=0.0,
                weight=1.0,
                kwargs={"latent.flux": 3.6},
            ),
            values_at_sigma_1={"latent.flux": (2.5, 5.0)},
            values_at_sigma_3={"latent.flux": (0.4, 11.0)},
        )

    def value(self, name):
        if name == "latent.latent_summary":
            return self.latent_summary
        return None


def test__row__asymmetric_latent_values():
    column = Column(
        "latent.flux",
        value_types=[
            ValueType.Median,
            ValueType.MaxLogLikelihood,
            ValueType.ValuesAt1Sigma,
            ValueType.ValuesAt3Sigma,
        ],
        strict=True,
    )

    row = Row(_AsymmetricLatentResult(), [column], 0)

    assert row.dict() == {
        "id": "asymmetric",
        "latent_flux": 3.0,
        "latent_flux_max_lh": 3.6,
        "latent_flux_lower_1_sigma": 2.5,
        "latent_flux_upper_1_sigma": 5.0,
        "latent_flux_lower_3_sigma": 0.4,
        "latent_flux_upper_3_sigma": 11.0,
    }
