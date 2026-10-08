"""
The declarative search registry (``autofit/non_linear/search/registry.py``) and the
versioned ``search-manifest@1``.

- completeness: every search ``autofit`` exports is registered, and every entry is
  exported, so deleting an entry (or adding an export without one) fails;
- entry == class attributes, skipped per entry only when a module in its ``requires``
  is not installed;
- the registry is data: building the manifest imports no search module and no optional
  backend;
- ``python -m autofit search-manifest --json`` emits the manifest.

None of these tests needs a sibling repository; the ``example`` / ``integration_test``
anchors are only checked for their format here.
"""

import inspect
import json
import re
import subprocess
import sys

import pytest

import autofit as af
from autofit.non_linear.search import registry
from autofit.non_linear.search.abstract_search import NonLinearSearch
from autofit.non_linear.search.capabilities import CAPABILITY_ATTRIBUTES, capabilities_from

from test_autofit.non_linear.search.conformance_roster import searches_under_test

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

_SEARCH_PACKAGE = "autofit.non_linear.search."

OPTIONAL_BACKENDS = (
    "jax",
    "jaxlib",
    "blackjax",
    "optax",
    "prodigyopt",
    "nautilus",
    "dynesty",
    "emcee",
    "zeus",
)


def exported_search_names():
    """
    The names ``autofit`` exports for concrete non-linear searches. Lazy exports are
    recognised by their module path, so this never imports a lazy search module.
    """
    names = {
        name
        for name, (module, _) in af._LAZY_ATTRS.items()
        if module.startswith(_SEARCH_PACKAGE)
    }
    for name, value in vars(af).items():
        if (
            inspect.isclass(value)
            and issubclass(value, NonLinearSearch)
            and not inspect.isabstract(value)
            and value.__module__.startswith(_SEARCH_PACKAGE)
        ):
            names.add(name)
    return names


def _params():
    return [pytest.param(entry, id=entry.name) for entry in registry.entries()]


def test_every_exported_search_is_registered():
    registered = {entry.name for entry in registry.entries()}

    assert exported_search_names() == registered
    assert len(registered) == 15


def test_deleting_an_entry_fails_completeness(monkeypatch):
    monkeypatch.setattr(registry, "SEARCHES", registry.SEARCHES[1:])

    assert exported_search_names() != {entry.name for entry in registry.entries()}


def test_registry_and_conformance_roster_agree():
    roster = {entry.name: entry for entry in searches_under_test()}

    assert set(roster) == {entry.name for entry in registry.entries()}
    for entry in registry.entries():
        assert roster[entry.name].class_path == entry.class_path
        assert roster[entry.name].family == entry.family
        assert roster[entry.name].jax_use == entry.capabilities["jax_use"]


@pytest.mark.parametrize("entry", _params())
def test_entry_names_its_export(entry):
    if entry.lazy:
        module, attribute = af._LAZY_ATTRS[entry.name]
        assert f"{module}.{attribute}" == entry.class_path
    else:
        assert entry.name not in af._LAZY_ATTRS
        cls = vars(af)[entry.name]
        assert f"{cls.__module__}.{cls.__qualname__}" == entry.class_path


@pytest.mark.parametrize("entry", _params())
def test_entry_equals_class_attributes(entry):
    missing = entry.missing_requirements()
    if missing:
        pytest.skip(f"{entry.name} needs {missing}, which is not installed")

    cls = entry.resolve()

    assert cls.__name__ == entry.name
    assert capabilities_from(cls) == dict(entry.to_dict()["capabilities"])


@pytest.mark.parametrize("entry", _params())
def test_entry_is_well_formed(entry):
    assert entry.family in registry.FAMILIES
    assert list(entry.capabilities) == list(CAPABILITY_ATTRIBUTES)
    if entry.example is not None:
        assert re.fullmatch(r"\w+:scripts/[\w/]+\.py#Search: \w+", entry.example)
    if entry.integration_test is not None:
        assert re.fullmatch(r"\w+:scripts/[\w/]+\.py", entry.integration_test)


def test_manifest_schema_and_round_trip():
    manifest = registry.manifest()

    assert manifest["schema"] == registry.MANIFEST_SCHEMA == "search-manifest@1"
    assert manifest["autofit_version"] == af.__version__
    assert manifest["capability_attributes"] == list(CAPABILITY_ATTRIBUTES)
    assert list(manifest["families"]) == ["mcmc", "nest", "mle"]
    assert [search["name"] for search in manifest["searches"]] == [
        entry.name for entry in registry.entries()
    ]
    assert json.loads(json.dumps(manifest)) == manifest

    for search in manifest["searches"]:
        assert set(search) == {
            "name",
            "class_path",
            "family",
            "lazy",
            "requires",
            "example",
            "integration_test",
            "capabilities",
        }


def _run_python(code):
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_manifest_imports_no_search_module_and_no_backend():
    """
    ``import autofit`` plus building the manifest imports neither the lazy search
    modules nor any optional sampler backend: the registry is data.
    """
    output = _run_python(
        "import sys, json\n"
        "import autofit\n"
        "from autofit.non_linear.search import registry\n"
        "registry.manifest()\n"
        "print(json.dumps(sorted(sys.modules)))\n"
    )
    modules = set(json.loads(output.strip().splitlines()[-1]))

    for entry in registry.entries():
        if entry.lazy:
            assert entry.class_path.rpartition(".")[0] not in modules
    for backend in OPTIONAL_BACKENDS:
        assert backend not in modules, backend


def test_python_m_autofit_search_manifest_json():
    output = subprocess.run(
        [sys.executable, "-m", "autofit", "search-manifest", "--json"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    manifest = json.loads(output)

    assert manifest["schema"] == "search-manifest@1"
    assert len(manifest["searches"]) == 15
    assert manifest == json.loads(json.dumps(registry.manifest()))


def test_python_m_autofit_search_manifest_table():
    output = subprocess.run(
        [sys.executable, "-m", "autofit", "search-manifest"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    assert output.startswith("search-manifest@1")
    assert "Nautilus" in output
