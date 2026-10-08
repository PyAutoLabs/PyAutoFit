"""
Golden ``samples.csv`` fixtures for the search-extensibility phase A3 samples adapter.

Each case under ``golden/`` holds a backend internal stored from a small real fit
(Emcee, DynestyStatic, Nautilus, BlackJAXNUTS, SMC), the fitted model, and the
``samples.csv`` / ``samples_info.json`` that ``samples_via_internal_from`` produced
from that internal on ``origin/main`` before the adapter landed (``golden/generate.py``).
Converting the same internal today must reproduce ``samples.csv`` byte for byte:
parameters, log likelihoods, log priors, log posteriors and weights, in order, with no
renormalisation (decision D13).
"""
import importlib.util
import json

import pytest

from test_autofit.non_linear.samples.golden import cases

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def _skip_missing(name):
    missing = [
        module
        for module in cases.CASES[name][3]
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        pytest.skip(f"golden case {name} needs {missing}")


@pytest.fixture(autouse=True)
def reference_mode(monkeypatch):
    # The fixtures were converted outside test mode, where the MCMC burn-in and
    # thinning come from the auto-correlation times rather than fixed values.
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)


@pytest.mark.parametrize("name", list(cases.CASES))
def test_samples_csv_is_byte_identical(name, tmp_path):
    _skip_missing(name)

    samples = cases.samples_from(name)
    samples.write_table(filename=tmp_path / "samples.csv")

    assert (tmp_path / "samples.csv").read_bytes() == (
        cases.HERE / name / "samples.csv"
    ).read_bytes()


@pytest.mark.parametrize("name", list(cases.CASES))
def test_samples_info_is_unchanged(name):
    _skip_missing(name)

    samples = cases.samples_from(name)

    with open(cases.HERE / name / "samples_info.json") as f:
        expected = json.load(f)

    assert cases.samples_info_for_comparison(samples) == expected
