"""
The fail-fast gate for ``jax_use='required'`` searches (search-extensibility D2).

``NonLinearSearch.start_resume_fit`` raises one shared ``SearchException``
(``capabilities.JAX_REQUIRED_MESSAGE``) when a JAX-required search is given a numpy
analysis, before the backend runs. The gate sits after the test-mode bypass return, so
``PYAUTO_DISABLE_JAX=1`` + ``PYAUTO_TEST_MODE=2`` smoke runs still complete.

No backend executes here: the gate raises before ``_fit``, and test mode 2 bypasses it,
so these cases run on every CI leg (``NSS`` alone needs blackjax to be constructed).
"""

import pytest

import autofit as af
from autofit import exc
from autofit.non_linear.search import capabilities as cap

from test_autofit.non_linear.search.conformance_roster import searches_under_test

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

REQUIRED = [entry for entry in searches_under_test() if entry.jax_use == "required"]


class GaussianAnalysis(af.Analysis):
    def log_likelihood_function(self, instance):
        return -0.5 * float(
            (instance.centre - 50.0) ** 2
            + (instance.normalization - 25.0) ** 2
            + (instance.sigma - 10.0) ** 2
        )


def _search(entry):
    missing = entry.missing_requirements()
    if missing:
        pytest.skip(f"{entry.name} cannot be constructed without {missing}")
    return entry.resolve()()


def test_seven_searches_require_jax():
    assert sorted(entry.name for entry in REQUIRED) == sorted(
        [
            "BlackJAXNUTS",
            "SMC",
            "NSS",
            "MultiStartAdam",
            "MultiStartADABelief",
            "MultiStartLion",
            "MultiStartProdigy",
        ]
    )


@pytest.mark.parametrize("entry", REQUIRED, ids=[entry.name for entry in REQUIRED])
def test_required_search_with_numpy_analysis_raises_the_shared_message(
    entry, monkeypatch
):
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)
    search = _search(entry)

    def backend_must_not_run(*args, **kwargs):
        raise AssertionError("the backend ran before the gate")

    monkeypatch.setattr(search, "_fit", backend_must_not_run)

    with pytest.raises(exc.SearchException) as error:
        search.fit(model=af.Model(af.ex.Gaussian), analysis=GaussianAnalysis())

    assert str(error.value) == cap.JAX_REQUIRED_MESSAGE.format(search=entry.name)


@pytest.mark.parametrize("entry", REQUIRED, ids=[entry.name for entry in REQUIRED])
def test_required_search_completes_with_disable_jax_and_test_mode_2(
    entry, monkeypatch
):
    monkeypatch.setenv("PYAUTO_DISABLE_JAX", "1")
    monkeypatch.setenv("PYAUTO_TEST_MODE", "2")
    search = _search(entry)
    analysis = GaussianAnalysis(use_jax=True)

    assert analysis.is_jax is False

    result = search.fit(model=af.Model(af.ex.Gaussian), analysis=analysis)

    assert isinstance(result, af.Result)


@pytest.mark.parametrize("entry", REQUIRED, ids=[entry.name for entry in REQUIRED])
def test_direct_fit_call_raises_the_shared_message(entry, monkeypatch):
    """
    The JAX-native searches repeat the gate at the top of ``_fit``, so a caller that
    bypasses ``fit`` gets the same error rather than a tracer error deep in the backend.
    """
    for module in entry.jax_modules:
        pytest.importorskip(module)
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)
    search = _search(entry)

    with pytest.raises(exc.SearchException, match=entry.name):
        search._fit(model=af.Model(af.ex.Gaussian), analysis=GaussianAnalysis())


def test_optional_search_with_numpy_analysis_passes_the_gate():
    cap.check_jax_required(af.Drawer(), GaussianAnalysis())
    cap.check_jax_required(af.DynestyStatic(), GaussianAnalysis())
