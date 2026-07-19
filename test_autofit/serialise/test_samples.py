import autofit as af
import pytest

from autofit.non_linear.samples.summary import SamplesSummary
from autonerves.dictable import from_dict, to_dict


@pytest.fixture(name="model")
def make_model():
    return af.Model(af.ex.Gaussian)


@pytest.fixture(name="sample")
def make_sample(model):
    return af.Sample(
        log_likelihood=4.0,
        log_prior=5.0,
        weight=6.0,
        kwargs={"centre": 2.0, "normalization": 4.0, "sigma": 6.0},
    )


@pytest.fixture(name="samples_pdf")
def make_samples_pdf(model, sample):
    return af.SamplesPDF(
        sample_list=[
            af.Sample(
                log_likelihood=1.0,
                log_prior=2.0,
                weight=3.0,
                kwargs={"centre": 0.0, "normalization": 1.0, "sigma": 2.0},
            ),
            sample,
        ],
        model=model,
    )


@pytest.fixture(name="summary")
def make_summary(samples_pdf):
    return samples_pdf.summary()


def test_summary(summary, model, sample):
    assert summary.model is model
    assert summary.max_log_likelihood_sample == sample


@pytest.fixture(name="summary_dict")
def make_summary_dict():
    return {
        "type": "instance",
        "class_path": "autofit.non_linear.samples.summary.SamplesSummary",
        "arguments": {
            "values_at_sigma_1": {
                "type": "dict",
                "arguments": {
                    "centre": {"type": "tuple", "values": [0.0, 2.0]},
                    "normalization": {"type": "tuple", "values": [1.0, 4.0]},
                    "sigma": {"type": "tuple", "values": [2.0, 6.0]},
                },
            },
            "errors_at_sigma_3": {
                "type": "dict",
                "arguments": {
                    "centre": {"type": "tuple", "values": [2.0, 0.0]},
                    "normalization": {"type": "tuple", "values": [3.0, 0.0]},
                    "sigma": {"type": "tuple", "values": [4.0, 0.0]},
                },
            },
            "median_pdf_sample": {
                "type": "instance",
                "class_path": "autofit.non_linear.samples.sample.Sample",
                "arguments": {
                    "log_likelihood": 4.0,
                    "log_prior": 5.0,
                    "weight": 6.0,
                    "kwargs": {
                        "type": "dict",
                        "arguments": {
                            "centre": 2.0,
                            "normalization": 4.0,
                            "sigma": 6.0,
                        },
                    },
                },
            },
            "max_log_likelihood_sample": {
                "type": "instance",
                "class_path": "autofit.non_linear.samples.sample.Sample",
                "arguments": {
                    "log_likelihood": 4.0,
                    "log_prior": 5.0,
                    "weight": 6.0,
                    "kwargs": {
                        "type": "dict",
                        "arguments": {
                            "centre": 2.0,
                            "normalization": 4.0,
                            "sigma": 6.0,
                        },
                    },
                },
            },
            "errors_at_sigma_1": {
                "type": "dict",
                "arguments": {
                    "centre": {"type": "tuple", "values": [2.0, 0.0]},
                    "normalization": {"type": "tuple", "values": [3.0, 0.0]},
                    "sigma": {"type": "tuple", "values": [4.0, 0.0]},
                },
            },
            "model": {
                "class_path": "autofit.example.model.Gaussian",
                "type": "model",
                "arguments": {
                    "centre": {
                        "lower_limit": 0.0,
                        "upper_limit": 1.0,
                        "type": "Uniform",
                    },
                    "normalization": {
                        "lower_limit": 0.0,
                        "upper_limit": 1.0,
                        "type": "Uniform",
                    },
                    "sigma": {
                        "lower_limit": 0.0,
                        "upper_limit": 1.0,
                        "type": "Uniform",
                    },
                },
            },
            "log_evidence": None,
            "values_at_sigma_3": {
                "type": "dict",
                "arguments": {
                    "centre": {"type": "tuple", "values": [0.0, 2.0]},
                    "normalization": {"type": "tuple", "values": [1.0, 4.0]},
                    "sigma": {"type": "tuple", "values": [2.0, 6.0]},
                },
            },
        },
    }


def test_dict(summary, summary_dict, remove_ids):
    assert remove_ids(to_dict(summary)) == summary_dict


def test_from_dict(summary_dict):
    summary = from_dict(summary_dict)
    assert isinstance(summary, SamplesSummary)


def test_generic_from_dict(summary_dict):
    summary = from_dict(summary_dict)
    assert isinstance(summary, SamplesSummary)
    assert isinstance(summary.max_log_likelihood_sample, af.Sample)


def test_sample_zero_valued_kwargs_round_trip():
    """
    Sample kwargs whose value is exactly 0.0 (e.g. a prior-median centre) must
    survive to_dict -> from_dict — the dict branch previously filtered falsy
    values, silently dropping them.
    """
    sample = af.Sample(
        log_likelihood=1.0,
        log_prior=0.0,
        weight=1.0,
        kwargs={"centre.centre_0": 0.0, "sigma": 6.0},
    )
    loaded = from_dict(to_dict(sample))
    assert loaded.kwargs == {("centre", "centre_0"): 0.0, ("sigma",): 6.0}
