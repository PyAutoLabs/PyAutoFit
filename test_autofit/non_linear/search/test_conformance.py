"""
Search conformance suite, layer (i): metadata and serialization of the 15 public
searches.

Every case is parametrised over ``searches_under_test()`` (ids are the short names)
and checked against the frozen ``GOLDEN`` table in ``conformance_roster.py``, whose
docstring states the roster and the coverage contract. Classes are resolved with
``importlib`` inside each test, so the module collects on every CI leg, including
``unittest-nojax``. A construction-dependent case skips only when a module in the
entry's ``requires`` list cannot be found.

Strict xfails mark behaviour that is known to be wrong today and is repaired by a later
phase of the search-extensibility epic; a repair flips the xfail to XPASS, which fails
the suite until the marker is removed.
"""

import copy
import inspect
import json

import pytest

import autofit as af
from autofit.mapper.identifier import Identifier
from autonerves import conf

from test_autofit.non_linear.search.conformance_roster import (
    FAMILY_BASES,
    GOLDEN,
    resolve_class_path,
    searches_under_test,
)

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

ROSTER = searches_under_test()

ROUND_TRIP_XFAIL = {
    "BlackJAXNUTS": (
        "The serialised inverse_mass_matrix is a kind string ('none') the "
        "constructor rejects; repaired in search-extensibility phase A0b."
    ),
    "SMC": (
        "The serialised inverse_mass_matrix is a kind string SMC's constructor "
        "rejects; repaired in search-extensibility phase A0b."
    ),
}

CONFIG_MUTATION_XFAIL = {
    name: (
        "__init__ sets conf.instance['output']['search_internal'] = True; "
        "removed in search-extensibility phase A3."
    )
    for name in ("Emcee", "BlackJAXNUTS", "SMC")
}


def _params(xfails=None, raises=None):
    """
    The roster as pytest params, with strict xfail marks for the named entries.

    ``raises`` restricts the expected failure to one exception type, so an xfail
    cannot hide an unrelated error.
    """
    xfails = xfails or {}
    params = []
    for entry in ROSTER:
        marks = []
        if entry.name in xfails:
            marks.append(
                pytest.mark.xfail(strict=True, reason=xfails[entry.name], raises=raises)
            )
        params.append(pytest.param(entry, id=entry.name, marks=marks))
    return params


def _search_class(entry):
    """
    Resolve the class of a roster entry, skipping when a required module is missing.
    """
    missing = entry.missing_requirements()
    if missing:
        pytest.skip(f"{entry.name} cannot be constructed without {missing}")
    return entry.resolve()


def _plain(value):
    """
    Deep-copy a config mapping into plain nested dicts, for before/after comparison.
    """
    if hasattr(value, "items"):
        return {key: _plain(item) for key, item in value.items()}
    return copy.deepcopy(value)


def test_golden_table_matches_roster():
    names = [entry.name for entry in ROSTER]

    assert len(ROSTER) == 15
    assert len(set(names)) == 15
    assert set(GOLDEN) == set(names)
    assert {entry.family for entry in ROSTER} == set(FAMILY_BASES)


@pytest.mark.parametrize("entry", _params())
def test_default_construction(entry):
    cls = _search_class(entry)

    assert isinstance(cls(), cls)


@pytest.mark.parametrize("entry", _params())
def test_family_base(entry):
    cls = entry.resolve()

    assert issubclass(cls, resolve_class_path(FAMILY_BASES[entry.family]))


@pytest.mark.parametrize("entry", _params())
def test_identifier_fields_golden(entry):
    cls = entry.resolve()

    assert tuple(cls.__identifier_fields__) == GOLDEN[entry.name].identifier_fields


@pytest.mark.parametrize("entry", _params())
def test_identifier_golden(entry, monkeypatch):
    """
    Test mode alters the Emcee and Zeus identifiers through ``apply_test_mode``, so the
    golden identifiers are computed and checked with ``PYAUTO_TEST_MODE`` unset.
    """
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)
    cls = _search_class(entry)

    assert str(Identifier(cls())) == GOLDEN[entry.name].identifier


@pytest.mark.parametrize("entry", _params())
def test_constructor_argument_set(entry):
    cls = entry.resolve()

    arguments = set(inspect.signature(cls.__init__).parameters) - {"self"}

    assert arguments == GOLDEN[entry.name].constructor_arguments


@pytest.mark.parametrize("entry", _params(ROUND_TRIP_XFAIL, raises=ValueError))
def test_search_json_round_trip(entry):
    """
    A default-constructed search serialises with ``NullPaths``, so the dictionary
    carries no computed identifier or path and is compared without normalisation.
    """
    cls = _search_class(entry)

    search_dict = af.to_dict(cls())
    loaded = af.from_dict(json.loads(json.dumps(search_dict)))

    assert af.to_dict(loaded) == search_dict


@pytest.fixture(name="search_internal_off")
def make_search_internal_off(tmp_path):
    """
    Push a config override with ``output.search_internal: false``.

    The test config sets it to true, which would mask a constructor writing ``True``
    into the config. The original config stack is restored afterwards, which also
    discards any mutation made during the test.
    """
    original_configs = list(conf.instance.configs)
    (tmp_path / "output.yaml").write_text("search_internal: false\n")
    conf.instance.push(new_path=str(tmp_path))
    yield
    conf.instance.configs = original_configs


@pytest.mark.parametrize("entry", _params(CONFIG_MUTATION_XFAIL, raises=AssertionError))
def test_config_unchanged_by_construction(entry, search_internal_off):
    cls = _search_class(entry)
    assert conf.instance["output"]["search_internal"] is False

    before = _plain(conf.instance.dict)
    cls()

    assert _plain(conf.instance.dict) == before
