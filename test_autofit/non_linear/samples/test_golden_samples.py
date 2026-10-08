"""
Golden ``samples.csv`` fixtures for the search-extensibility phase A3 samples adapter.

Each case under ``golden/`` holds a backend internal stored from a small real fit
(Emcee, DynestyStatic, Nautilus, BlackJAXNUTS, SMC), the fitted model, and the
``samples.csv`` / ``samples_info.json`` that ``samples_via_internal_from`` produced
from that internal on ``origin/main`` before the adapter landed (``golden/generate.py``).
Converting the same internal today must reproduce ``samples.csv``: the same header,
the same rows in the same order, and every parameter, log likelihood, log prior, log
posterior and weight equal to within floating-point roundoff, with no renormalisation
(decision D13).

The comparison is to ``CSV_RTOL`` rather than byte for byte because NumPy's
vectorised ``exp`` / ``log`` loops are dispatched per CPU (AVX2 vs AVX-512), so the
same NumPy version on a different runner can land one unit in the last place away:
on CI the dynesty and nautilus weights (and two nautilus log priors) differed from the
AVX2-generated fixtures by at most 1 ULP (relative 2.2e-16). Any renormalisation,
reordering or dropped row changes values by many orders of magnitude more.

``samples_info`` is compared exactly except for the convergence diagnostics BlackJAX
itself computes (ESS, bulk/tail ESS, rank-normalised R-hat), whose estimators change
between BlackJAX releases; for those only the shape is pinned.
"""
import csv
import importlib.util
import json

import numpy as np
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


#: Relative tolerance on every numeric ``samples.csv`` entry: ~4500x the largest drift
#: measured across CI runners (2.2e-16), far below any change the adapter could make.
CSV_RTOL = 1e-12

#: ``samples_info`` keys computed by ``blackjax.diagnostics`` rather than by autofit.
BACKEND_DIAGNOSTIC_KEYS = (
    "ess_min",
    "ess_per_param",
    "ess_bulk_min",
    "ess_bulk_per_param",
    "ess_tail_min",
    "ess_tail_per_param",
    "rhat_max",
    "rhat_per_param",
)


def _read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header = [column.strip() for column in rows[0]]
    values = np.array([[float(value) for value in row] for row in rows[1:]])
    return header, values


@pytest.mark.parametrize("name", list(cases.CASES))
def test_samples_csv_is_unchanged(name, tmp_path):
    _skip_missing(name)

    samples = cases.samples_from(name)
    samples.write_table(filename=tmp_path / "samples.csv")

    header, values = _read_csv(tmp_path / "samples.csv")
    expected_header, expected_values = _read_csv(cases.HERE / name / "samples.csv")

    assert header == expected_header
    assert values.shape == expected_values.shape
    np.testing.assert_allclose(values, expected_values, rtol=CSV_RTOL, atol=0.0)


@pytest.mark.parametrize("name", list(cases.CASES))
def test_samples_info_is_unchanged(name):
    _skip_missing(name)

    samples = cases.samples_from(name)

    with open(cases.HERE / name / "samples_info.json") as f:
        expected = json.load(f)

    info = cases.samples_info_for_comparison(samples)

    for key in BACKEND_DIAGNOSTIC_KEYS:
        if key in expected:
            assert np.shape(info.pop(key)) == np.shape(expected.pop(key)), key

    assert info == expected
