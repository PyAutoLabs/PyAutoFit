"""
The static capability attributes every public search declares
(``autofit.non_linear.search.capabilities``), parametrised per class over the
conformance roster.

Classes are resolved inside each test, so the module collects on every CI leg. The
values themselves are mirrored by ``search/registry.py`` and compared there
(``test_registry.py``); these tests check that each declaration is well formed, that
none of them is an identifier field and that ``test_mode_budget`` is what
``PYAUTO_TEST_MODE=1`` actually applies.
"""

import json
import math
import re
from pathlib import Path

import pytest

from autofit.non_linear.search import capabilities as cap
from autofit.non_linear.search.abstract_search import NonLinearSearch

from test_autofit.non_linear.search.conformance_roster import searches_under_test

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

ROSTER = searches_under_test()

CITATIONS_BIB = Path(__file__).parents[3] / "files" / "citations.bib"


def _params():
    return [pytest.param(entry, id=entry.name) for entry in ROSTER]


def _bib_keys():
    return set(re.findall(r"^@\w+\{([^,]+),", CITATIONS_BIB.read_text(), re.M))


def _search_class(entry):
    missing = entry.missing_requirements()
    if missing:
        pytest.skip(f"{entry.name} cannot be constructed without {missing}")
    return entry.resolve()


def test_defaults_on_non_linear_search():
    assert NonLinearSearch.jax_use == cap.JaxUse.NONE
    assert NonLinearSearch.gradient == cap.Gradient.NONE
    assert NonLinearSearch.batched is False
    assert NonLinearSearch.posterior_kind is None
    assert NonLinearSearch.status == cap.Status.EXPERIMENTAL
    assert NonLinearSearch.objective_target is None
    assert NonLinearSearch.invalid_value == -math.inf


@pytest.mark.parametrize("entry", _params())
def test_capabilities_are_declared_and_well_formed(entry):
    cls = entry.resolve()

    assert isinstance(cls.jax_use, cap.JaxUse)
    assert isinstance(cls.gradient, cap.Gradient)
    assert isinstance(cls.batched, bool)
    assert isinstance(cls.honours_gradient_mode, bool)
    assert isinstance(cls.posterior_kind, cap.PosteriorKind)
    assert isinstance(cls.produces_evidence, bool)
    assert isinstance(cls.resumable, bool)
    assert isinstance(cls.warm_start, cap.WarmStart)
    assert cls.install_extra in ("", "optional")
    assert cls.upstream_url.startswith("https://")
    assert isinstance(cls.citation_keys, tuple)
    assert isinstance(cls.status, cap.Status)
    assert isinstance(cls.test_mode_budget, dict)
    assert isinstance(cls.objective_target, cap.ObjectiveTarget)
    assert isinstance(cls.objective_target.quantity, cap.ObjectiveQuantity)
    assert isinstance(cls.objective_target.space, cap.CoordinateSpace)
    assert isinstance(cls.invalid_value, float)


@pytest.mark.parametrize("entry", _params())
def test_capability_rules(entry):
    """
    Rules that hold between attributes: a gradient backend is JAX-native, only a
    gradient backend can honour ``gradient_mode``, evidence comes with weighted
    samples and point estimators never claim evidence.
    """
    cls = entry.resolve()

    if cls.gradient == cap.Gradient.USES:
        assert cls.jax_use == cap.JaxUse.REQUIRED
    if cls.honours_gradient_mode:
        assert cls.gradient == cap.Gradient.USES
    if cls.produces_evidence:
        assert cls.posterior_kind == cap.PosteriorKind.WEIGHTED
    if cls.posterior_kind == cap.PosteriorKind.POINT:
        assert not cls.produces_evidence


@pytest.mark.parametrize("entry", _params())
def test_roster_jax_use_matches_the_class(entry):
    assert entry.jax_use == str(entry.resolve().jax_use)


@pytest.mark.parametrize("entry", _params())
def test_capabilities_are_never_identifier_fields(entry):
    cls = entry.resolve()

    assert not set(cap.CAPABILITY_ATTRIBUTES) & set(cls.__identifier_fields__)


@pytest.mark.parametrize("entry", _params())
def test_citation_keys_are_in_citations_bib(entry):
    assert set(entry.resolve().citation_keys) <= _bib_keys()


@pytest.mark.parametrize("entry", _params())
def test_capabilities_serialise_to_json(entry):
    capabilities = cap.capabilities_from(entry.resolve())

    assert list(capabilities) == list(cap.CAPABILITY_ATTRIBUTES)
    assert json.loads(json.dumps(capabilities)) == capabilities


def _attribute(search, dotted):
    value = search
    for part in dotted.split("."):
        value = getattr(value, part)
    return value


@pytest.mark.parametrize("entry", _params())
def test_test_mode_budget_is_what_test_mode_applies(entry, monkeypatch):
    """
    Each budget entry is the value ``PYAUTO_TEST_MODE=1`` sets, or the default when the
    default is already below it (``num_chains`` and ``num_particles`` are capped with
    ``min``).
    """
    cls = _search_class(entry)

    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)
    default = cls()

    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")
    reduced = cls()

    assert reduced is not default
    for key, budget in cls.test_mode_budget.items():
        value = _attribute(reduced, key)
        default_value = _attribute(default, key)
        assert value == budget or value == default_value <= budget, key


def test_jax_required_message_names_the_search_and_the_fix():
    message = cap.JAX_REQUIRED_MESSAGE.format(search="BlackJAXNUTS")

    assert "BlackJAXNUTS" in message
    assert "use_jax=True" in message
    assert "PYAUTO_DISABLE_JAX" in message


def test_invalid_value_spelling():
    assert cap.invalid_value_to_str(-math.inf) == "-inf"
    assert cap.invalid_value_to_str(-1.0e99) == "-1e+99"
    assert cap.invalid_value_to_str(-1.0e30) == "-1e+30"
