import csv
import logging

from pathlib import Path

import autofit as af
from autofit.example.model import Gaussian
from autofit.aggregator.summary.aggregate_csv import AggregateCSV, ValueType
from autofit.aggregator.summary.aggregate_csv.column import Column
from autofit.aggregator.summary.aggregate_csv.row import Row
from autofit.non_linear.samples.sample import Sample
from autofit.non_linear.samples.summary import SamplesSummary

import pytest


@pytest.fixture
def output_path():
    path = Path("/tmp/summary.csv")
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def summary(aggregator):
    return AggregateCSV(aggregator)


@pytest.fixture
def load_output(output_path):
    def _load_output():
        with open(output_path) as f:
            return list(csv.DictReader(f))

    return _load_output


def test_writes(output_path, summary):
    summary.save(output_path)

    with open(output_path) as f:
        dicts = list(csv.DictReader(f))

    assert dicts[0]["id"] is not None
    assert dicts[1]["id"] is not None


def test_add_label_colum(
    output_path,
    summary,
    load_output,
):
    summary.add_label_column("label", ["a", "b"])
    summary.save(output_path)

    dicts = load_output()

    assert dicts[0]["label"] == "a"
    assert dicts[1]["label"] == "b"


def test_add_column(
    output_path,
    summary,
    load_output,
):
    summary.add_variable("galaxies.lens.bulge.centre.centre_0")
    summary.save(output_path)

    dicts = load_output()

    assert dicts[0]["galaxies_lens_bulge_centre_centre_0"] == "-1.0"
    assert dicts[1]["galaxies_lens_bulge_centre_centre_0"] == "-5.0"


def test_use_max_log_likelihood(
    output_path,
    summary,
    load_output,
):
    summary.add_variable(
        "galaxies.lens.bulge.centre.centre_0",
        value_types=[ValueType.MaxLogLikelihood],
    )
    summary.save(output_path)

    dicts = load_output()

    assert dicts[0]["galaxies_lens_bulge_centre_centre_0_max_lh"] == "-1.5"
    assert dicts[1]["galaxies_lens_bulge_centre_centre_0_max_lh"] == "-5.5"


def test_add_named_column(
    output_path,
    summary,
    load_output,
):
    summary.add_variable(
        "galaxies.lens.bulge.centre.centre_0",
        name="centre_0",
    )
    summary.save(output_path)

    dicts = load_output()

    assert dicts[0]["centre_0"] == "-1.0"
    assert dicts[1]["centre_0"] == "-5.0"


def test_add_latent_column(
    output_path,
    summary,
    load_output,
):
    summary.add_variable(
        "latent.value",
        value_types=[ValueType.Median, ValueType.MaxLogLikelihood],
    )
    summary.save(output_path)

    dicts = load_output()

    assert dicts[0]["latent_value"] == "1.0"
    assert dicts[1]["latent_value"] == "2.0"
    assert dicts[0]["latent_value_max_lh"] == "2.0"
    assert dicts[1]["latent_value_max_lh"] == "3.0"


def test_computed_column(
    output_path,
    summary,
    load_output,
):
    def compute(samples):
        return 1

    summary.add_computed_column(
        "computed",
        compute,
    )
    summary.save(output_path)

    dicts = load_output()

    assert dicts[0]["computed"] == "1"


def test_dict_computed_column(
    output_path,
    summary,
    load_output,
):
    def compute(samples):
        return {"a": 1, "b": 2}

    summary.add_computed_column(
        "computed",
        compute,
    )
    summary.save(output_path)

    dicts = load_output()

    first = dicts[0]
    assert first["computed_a"] == "1"
    assert first["computed_b"] == "2"


def test_values_at_1_sigma(
    output_path,
    summary,
    load_output,
):
    summary.add_variable(
        "galaxies.lens.bulge.centre.centre_0",
        value_types=[ValueType.ValuesAt1Sigma],
    )
    summary.save(output_path)

    dicts = load_output()

    first = dicts[0]
    assert (
        first["galaxies_lens_bulge_centre_centre_0_lower_1_sigma"]
        == "3.4319440071038327"
    )
    assert (
        first["galaxies_lens_bulge_centre_centre_0_upper_1_sigma"]
        == "5.134685987907622"
    )


def test_values_at_3_sigma(
    output_path,
    summary,
    load_output,
):
    summary.add_variable(
        "galaxies.lens.bulge.centre.centre_0",
        value_types=[ValueType.ValuesAt3Sigma],
    )
    summary.save(output_path)

    dicts = load_output()

    first = dicts[0]
    assert (
        first["galaxies_lens_bulge_centre_centre_0_lower_3_sigma"]
        == "1.6742483855526449"
    )
    assert (
        first["galaxies_lens_bulge_centre_centre_0_upper_3_sigma"] == "5.93749816282317"
    )


def test_latent_values_at_1_sigma(
    output_path,
    summary,
    load_output,
):
    summary.add_variable(
        "galaxies.lens.bulge.centre.latent",
        value_types=[ValueType.ValuesAt1Sigma],
    )
    summary.save(output_path)

    dicts = load_output()

    first = dicts[0]
    assert (
        first["galaxies_lens_bulge_centre_latent_lower_1_sigma"] == "3.4319440071038327"
    )


def test_latent_values_at_3_sigma(
    output_path,
    summary,
    load_output,
):
    summary.add_variable(
        "galaxies.lens.bulge.centre.latent",
        value_types=[ValueType.ValuesAt1Sigma, ValueType.ValuesAt3Sigma],
    )
    summary.save(output_path)

    dicts = load_output()

    first = dicts[0]
    assert (
        first["galaxies_lens_bulge_centre_latent_lower_3_sigma"] == "1.6742483855526449"
    )
    assert (
        first["galaxies_lens_bulge_centre_latent_upper_3_sigma"] == "5.93749816282317"
    )
    assert (
        first["galaxies_lens_bulge_centre_latent_lower_3_sigma"]
        != first["galaxies_lens_bulge_centre_latent_lower_1_sigma"]
    )


def test_max_log_likelihood_differs_from_median(
    output_path,
    summary,
    load_output,
):
    summary.add_variable(
        "galaxies.lens.bulge.centre.centre_0",
        value_types=[ValueType.Median, ValueType.MaxLogLikelihood],
    )
    summary.save(output_path)

    dicts = load_output()

    assert dicts[0]["galaxies_lens_bulge_centre_centre_0"] == "-1.0"
    assert dicts[1]["galaxies_lens_bulge_centre_centre_0"] == "-5.0"
    assert dicts[0]["galaxies_lens_bulge_centre_centre_0_max_lh"] == "-1.5"
    assert dicts[1]["galaxies_lens_bulge_centre_centre_0_max_lh"] == "-5.5"

    for row in dicts:
        assert (
            row["galaxies_lens_bulge_centre_centre_0"]
            != row["galaxies_lens_bulge_centre_centre_0_max_lh"]
        )


def test_unresolvable_argument_warns(
    output_path,
    aggregator,
    load_output,
    caplog,
):
    summary = AggregateCSV(aggregator, strict=False)
    summary.add_variable("does.not.exist")

    with caplog.at_level(
        logging.WARNING,
        logger="autofit.aggregator.summary.aggregate_csv.column",
    ):
        summary.save(output_path)

    records = [
        record for record in caplog.records if "does.not.exist" in record.getMessage()
    ]
    assert len(records) == 1

    dicts = load_output()

    assert dicts[0]["does_not_exist"] == ""
    assert dicts[1]["does_not_exist"] == ""


def test_resolvable_argument_does_not_warn(
    output_path,
    summary,
    load_output,
    caplog,
):
    summary.add_variable("galaxies.lens.bulge.centre.centre_0")

    with caplog.at_level(
        logging.WARNING,
        logger="autofit.aggregator.summary.aggregate_csv.column",
    ):
        summary.save(output_path)

    assert caplog.records == []


def test_unresolvable_argument_strict_raises(
    output_path,
    aggregator,
):
    summary = AggregateCSV(aggregator, strict=True)
    summary.add_variable("does.not.exist")

    with pytest.raises(KeyError, match="does.not.exist"):
        summary.save(output_path)


class _NoPDFResult:
    """
    A stand-in search output whose summary holds only a max_log_likelihood_sample,
    as produced by a search without a PDF (no median sample, no sigma bounds).
    """

    id = "no_pdf"

    def __init__(self):
        self.model = af.Model(Gaussian)
        self.samples_summary = SamplesSummary(
            model=self.model,
            max_log_likelihood_sample=Sample(
                log_likelihood=1.0,
                log_prior=0.0,
                weight=1.0,
                kwargs={"centre": 0.5, "normalization": 2.0, "sigma": 3.0},
            ),
        )

    def value(self, name):
        return None


def test_summary_without_pdf_does_not_raise(caplog):
    row = Row(_NoPDFResult(), [], 0)

    assert row.known_paths == {("centre",), ("normalization",), ("sigma",)}

    column = Column(
        "centre",
        value_types=[
            ValueType.Median,
            ValueType.MaxLogLikelihood,
            ValueType.ValuesAt1Sigma,
            ValueType.ValuesAt3Sigma,
        ],
    )

    with caplog.at_level(
        logging.WARNING,
        logger="autofit.aggregator.summary.aggregate_csv.column",
    ):
        value = column.value(row)

    assert value == {
        "": None,
        "max_lh": 0.5,
        "lower_1_sigma": None,
        "upper_1_sigma": None,
        "lower_3_sigma": None,
        "upper_3_sigma": None,
    }
    assert caplog.records == []
