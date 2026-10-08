"""
The declarative registry of PyAutoFit's public non-linear searches.

One entry per search, written as **data**: the class is named by a ``class_path``
string and is never imported by this module, so ``import autofit`` never imports a
search backend through it (``autofit.__init__._LAZY_ATTRS`` and the no-JAX CI leg keep
holding). Each entry mirrors the search's static capability class attributes
(``autofit.non_linear.search.capabilities``) in their JSON form; a unit test resolves
every class whose dependencies are installed and asserts entry == class attributes, and
a completeness test asserts every search ``autofit`` exports has an entry.

The registry is published as a versioned JSON manifest,

    python -m autofit search-manifest --json

whose schema is ``search-manifest@1`` (``MANIFEST_SCHEMA``). The manifest is the only
cross-repo format: the generated docs (``docs/_generate_searches.py``), workspaces and
assistants read it, never this module's Python objects.

The capabilities are **static** (a property of the class). What a particular run can
deliver may be narrower: for example ``SMC`` estimates the evidence only when it starts
from a prior-sampling initializer. Effective run capabilities are documented, not
recorded here.

Adding a search costs one entry here plus its ``autofit`` export; the completeness
test fails until both exist.
"""

import importlib
import importlib.util
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from autofit.non_linear.search.capabilities import CAPABILITY_ATTRIBUTES

MANIFEST_SCHEMA = "search-manifest@1"
"""
The manifest schema identifier. Bump the suffix on any incompatible change to the
manifest layout (a removed or renamed key, a changed value type); adding a key is
compatible.
"""

FAMILIES = {
    "mcmc": "Markov chain Monte Carlo",
    "nest": "Nested sampling",
    "mle": "Maximum likelihood / maximum a posteriori",
}
"""
The search families, in documentation order, with their display titles.
"""

_WORKSPACE = "autofit_workspace:scripts/searches"
_WORKSPACE_TEST = "autofit_workspace_test:scripts/searches"


@dataclass(frozen=True)
class RegistryEntry:
    """
    One public search.

    Parameters
    ----------
    name
        The public short name, as exported by ``autofit``.
    class_path
        The dotted ``module.Class`` path of the search class.
    family
        ``mcmc``, ``nest`` or ``mle`` (a key of ``FAMILIES``).
    lazy
        Whether ``autofit`` exports the class lazily (``_LAZY_ATTRS``), because its
        module imports a heavy optional backend.
    requires
        The top-level modules the search's backend needs to run.
    capabilities
        The search's static capability attributes in JSON form, keyed and ordered by
        ``CAPABILITY_ATTRIBUTES``.
    example
        A runnable example anchor, ``repo:path#section``, or ``None``.
    integration_test
        The integration-test script anchor, ``repo:path``, or ``None``.
    """

    name: str
    class_path: str
    family: str
    lazy: bool
    requires: Tuple[str, ...]
    capabilities: Mapping[str, Any]
    example: Optional[str] = None
    integration_test: Optional[str] = None

    def missing_requirements(self) -> List[str]:
        """
        The modules in ``requires`` that are not installed.
        """
        return [
            module
            for module in self.requires
            if importlib.util.find_spec(module) is None
        ]

    def resolve(self):
        """
        Import and return the search class (imports its module).
        """
        module_name, _, attribute = self.class_path.rpartition(".")
        return getattr(importlib.import_module(module_name), attribute)

    def to_dict(self) -> Dict[str, Any]:
        """
        The entry as it appears in the manifest.
        """
        return {
            "name": self.name,
            "class_path": self.class_path,
            "family": self.family,
            "lazy": self.lazy,
            "requires": list(self.requires),
            "example": self.example,
            "integration_test": self.integration_test,
            "capabilities": {
                name: _plain(self.capabilities[name]) for name in CAPABILITY_ATTRIBUTES
            },
        }


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _capabilities(
    *,
    jax_use: str,
    gradient: str,
    batched: bool,
    honours_gradient_mode: bool,
    posterior_kind: str,
    produces_evidence: bool,
    resumable: bool,
    warm_start: str,
    install_extra: str,
    upstream_url: str,
    citation_keys: Tuple[str, ...],
    status: str,
    test_mode_budget: Dict[str, Any],
    objective_target: Tuple[str, str],
    invalid_value: str,
) -> Dict[str, Any]:
    quantity, space = objective_target
    values = dict(
        jax_use=jax_use,
        gradient=gradient,
        batched=batched,
        honours_gradient_mode=honours_gradient_mode,
        posterior_kind=posterior_kind,
        produces_evidence=produces_evidence,
        resumable=resumable,
        warm_start=warm_start,
        install_extra=install_extra,
        upstream_url=upstream_url,
        citation_keys=list(citation_keys),
        status=status,
        test_mode_budget=dict(test_mode_budget),
        objective_target={"quantity": quantity, "space": space},
        invalid_value=invalid_value,
    )
    return {name: values[name] for name in CAPABILITY_ATTRIBUTES}


_SEARCH = "autofit.non_linear.search"
_MULTI_START = f"{_SEARCH}.mle.multi_start_gradient.search"

_BLACKJAX = "https://github.com/blackjax-devs/blackjax"
_DYNESTY = "https://github.com/joshspeagle/dynesty"
_SCIPY = "https://github.com/scipy/scipy"
_OPTAX = "https://github.com/google-deepmind/optax"

_EMCEE_ZEUS_BUDGET = {"nwalkers": 20, "nsteps": 10}
_MULTI_START_BUDGET = {"convergence.window": 1, "convergence.min_steps": 1}


def _dynesty(name: str, module: str) -> RegistryEntry:
    return RegistryEntry(
        name=name,
        class_path=f"{_SEARCH}.nest.dynesty.search.{module}.{name}",
        family="nest",
        lazy=False,
        requires=("dynesty",),
        capabilities=_capabilities(
            jax_use="optional",
            gradient="none",
            batched=False,
            honours_gradient_mode=False,
            posterior_kind="weighted",
            produces_evidence=True,
            resumable=True,
            warm_start="provider",
            install_extra="",
            upstream_url=_DYNESTY,
            citation_keys=("dynesty",),
            status="stable",
            test_mode_budget={"maxcall": 1},
            objective_target=("log_likelihood", "unit_cube"),
            invalid_value="-1e+99",
        ),
        example=f"{_WORKSPACE}/nest.py#Search: {name}",
        integration_test=f"{_WORKSPACE_TEST}/{name}.py",
    )


def _bfgs(name: str) -> RegistryEntry:
    return RegistryEntry(
        name=name,
        class_path=f"{_SEARCH}.mle.bfgs.search.{name}",
        family="mle",
        lazy=False,
        requires=("scipy",),
        capabilities=_capabilities(
            jax_use="optional",
            gradient="none",
            batched=False,
            honours_gradient_mode=False,
            posterior_kind="point",
            produces_evidence=False,
            resumable=False,
            warm_start="provider",
            install_extra="",
            upstream_url=_SCIPY,
            citation_keys=("scipy",),
            status="stable",
            test_mode_budget={},
            objective_target=("neg2_log_posterior", "physical"),
            invalid_value="-inf",
        ),
        example=f"{_WORKSPACE}/mle.py#Search: LBFGS" if name == "LBFGS" else None,
        integration_test=f"{_WORKSPACE_TEST}/{name}.py",
    )


def _multi_start(
    name: str,
    status: str = "stable",
    upstream_url: str = _OPTAX,
    citation_keys: Tuple[str, ...] = ("optax",),
    example: Optional[str] = None,
) -> RegistryEntry:
    return RegistryEntry(
        name=name,
        class_path=f"{_MULTI_START}.{name}",
        family="mle",
        lazy=False,
        requires=("jax", "optax"),
        capabilities=_capabilities(
            jax_use="required",
            gradient="uses",
            batched=True,
            honours_gradient_mode=True,
            posterior_kind="point",
            produces_evidence=False,
            resumable=True,
            warm_start="provider",
            install_extra="",
            upstream_url=upstream_url,
            citation_keys=citation_keys,
            status=status,
            test_mode_budget=_MULTI_START_BUDGET,
            objective_target=("neg2_log_posterior", "physical"),
            invalid_value="-inf",
        ),
        example=example,
        integration_test=f"{_WORKSPACE_TEST}/{name}.py",
    )


SEARCHES: Tuple[RegistryEntry, ...] = (
    RegistryEntry(
        name="Emcee",
        class_path=f"{_SEARCH}.mcmc.emcee.search.Emcee",
        family="mcmc",
        lazy=False,
        requires=("emcee",),
        capabilities=_capabilities(
            jax_use="none",
            gradient="none",
            batched=False,
            honours_gradient_mode=False,
            posterior_kind="chain",
            produces_evidence=False,
            resumable=False,
            warm_start="consumer",
            install_extra="",
            upstream_url="https://github.com/dfm/emcee",
            citation_keys=("emcee",),
            status="stable",
            test_mode_budget=_EMCEE_ZEUS_BUDGET,
            objective_target=("log_posterior", "physical"),
            invalid_value="-inf",
        ),
        example=f"{_WORKSPACE}/mcmc.py#Search: Emcee",
        integration_test=f"{_WORKSPACE_TEST}/Emcee.py",
    ),
    RegistryEntry(
        name="Zeus",
        class_path=f"{_SEARCH}.mcmc.zeus.search.Zeus",
        family="mcmc",
        lazy=False,
        requires=("zeus",),
        capabilities=_capabilities(
            jax_use="none",
            gradient="none",
            batched=False,
            honours_gradient_mode=False,
            posterior_kind="chain",
            produces_evidence=False,
            resumable=False,
            warm_start="consumer",
            install_extra="optional",
            upstream_url="https://github.com/minaskar/zeus",
            citation_keys=("zeus1", "zeus2"),
            status="stable",
            test_mode_budget=_EMCEE_ZEUS_BUDGET,
            objective_target=("log_posterior", "physical"),
            invalid_value="-inf",
        ),
        example=f"{_WORKSPACE}/mcmc.py#Search: Zeus",
        integration_test=f"{_WORKSPACE_TEST}/Zeus.py",
    ),
    RegistryEntry(
        name="BlackJAXNUTS",
        class_path=f"{_SEARCH}.mcmc.blackjax.nuts.search.BlackJAXNUTS",
        family="mcmc",
        lazy=False,
        requires=("jax", "blackjax"),
        capabilities=_capabilities(
            jax_use="required",
            gradient="uses",
            batched=True,
            honours_gradient_mode=False,
            posterior_kind="chain",
            produces_evidence=False,
            resumable=False,
            warm_start="consumer",
            install_extra="optional",
            upstream_url=_BLACKJAX,
            citation_keys=("blackjax",),
            status="stable",
            test_mode_budget={"num_warmup": 20, "num_samples": 20, "num_chains": 2},
            objective_target=("log_posterior", "physical"),
            invalid_value="-inf",
        ),
        example=f"{_WORKSPACE}/mcmc.py#Search: BlackJAXNUTS",
        integration_test=f"{_WORKSPACE_TEST}/BlackJAXNUTS.py",
    ),
    RegistryEntry(
        name="SMC",
        class_path=f"{_SEARCH}.mcmc.blackjax.smc.search.SMC",
        family="mcmc",
        lazy=True,
        requires=("jax", "blackjax"),
        capabilities=_capabilities(
            jax_use="required",
            gradient="uses",
            batched=True,
            honours_gradient_mode=False,
            posterior_kind="weighted",
            produces_evidence=True,
            resumable=False,
            warm_start="provider",
            install_extra="optional",
            upstream_url=_BLACKJAX,
            citation_keys=("blackjax",),
            status="experimental",
            test_mode_budget={
                "num_particles": 16,
                "num_mcmc_steps": 2,
                "max_smc_steps": 5,
            },
            objective_target=("log_likelihood", "physical"),
            invalid_value="-1e+99",
        ),
        example=None,
        integration_test=f"{_WORKSPACE_TEST}/SMC.py",
    ),
    _dynesty("DynestyStatic", "static"),
    _dynesty("DynestyDynamic", "dynamic"),
    RegistryEntry(
        name="Nautilus",
        class_path=f"{_SEARCH}.nest.nautilus.search.Nautilus",
        family="nest",
        lazy=False,
        requires=("nautilus",),
        capabilities=_capabilities(
            jax_use="optional",
            gradient="none",
            batched=True,
            honours_gradient_mode=False,
            posterior_kind="weighted",
            produces_evidence=True,
            resumable=True,
            warm_start="provider",
            install_extra="optional",
            upstream_url="https://github.com/johannesulf/nautilus",
            citation_keys=("nautilus",),
            status="stable",
            test_mode_budget={"n_like_max": 1},
            objective_target=("log_likelihood", "unit_cube"),
            invalid_value="-1e+99",
        ),
        example=f"{_WORKSPACE}/nest.py#Search: Nautilus",
        integration_test=f"{_WORKSPACE_TEST}/Nautilus.py",
    ),
    RegistryEntry(
        name="NSS",
        class_path=f"{_SEARCH}.nest.nss.search.NSS",
        family="nest",
        lazy=True,
        requires=("jax", "blackjax"),
        capabilities=_capabilities(
            jax_use="required",
            gradient="none",
            batched=True,
            honours_gradient_mode=False,
            posterior_kind="weighted",
            produces_evidence=True,
            resumable=True,
            warm_start="provider",
            install_extra="optional",
            upstream_url=_BLACKJAX,
            citation_keys=("blackjax",),
            status="experimental",
            test_mode_budget={"termination": -1.0},
            objective_target=("log_likelihood", "physical"),
            invalid_value="-1e+30",
        ),
        example=None,
        integration_test=f"{_WORKSPACE_TEST}/NSS.py",
    ),
    RegistryEntry(
        name="Drawer",
        class_path=f"{_SEARCH}.mle.drawer.search.Drawer",
        family="mle",
        lazy=False,
        requires=(),
        capabilities=_capabilities(
            jax_use="none",
            gradient="none",
            batched=False,
            honours_gradient_mode=False,
            posterior_kind="point",
            produces_evidence=False,
            resumable=False,
            warm_start="neutral",
            install_extra="",
            upstream_url="https://github.com/PyAutoLabs/PyAutoFit",
            citation_keys=(),
            status="stable",
            test_mode_budget={},
            objective_target=("log_posterior", "physical"),
            invalid_value="-inf",
        ),
        example=f"{_WORKSPACE}/mle.py#Search: Drawer",
        integration_test=f"{_WORKSPACE_TEST}/Drawer.py",
    ),
    _bfgs("BFGS"),
    _bfgs("LBFGS"),
    _multi_start("MultiStartAdam", example=f"{_WORKSPACE}/mle.py#Search: MultiStartAdam"),
    _multi_start("MultiStartADABelief", status="experimental"),
    _multi_start("MultiStartLion", status="experimental"),
    _multi_start(
        "MultiStartProdigy",
        upstream_url="https://github.com/konstmish/prodigy",
        citation_keys=("optax", "prodigy"),
    ),
)
"""
Every public search, in documentation order (by family, then by name order above).
"""


def entries() -> Tuple[RegistryEntry, ...]:
    """
    Every registered search.
    """
    return SEARCHES


def entry(name: str) -> RegistryEntry:
    """
    The registry entry of the search exported as ``autofit.<name>``.
    """
    for search_entry in SEARCHES:
        if search_entry.name == name:
            return search_entry
    raise KeyError(f"No search named {name!r} is registered")


def manifest() -> Dict[str, Any]:
    """
    The versioned search manifest (``search-manifest@1``), the only cross-repo format.

    Keys: ``schema``, ``autofit_version``, ``families`` (family key -> title),
    ``capability_attributes`` (the capability keys, in order) and ``searches`` (one
    object per search: ``name``, ``class_path``, ``family``, ``lazy``, ``requires``,
    ``example``, ``integration_test`` and ``capabilities``).
    """
    import autofit

    return {
        "schema": MANIFEST_SCHEMA,
        "autofit_version": autofit.__version__,
        "families": dict(FAMILIES),
        "capability_attributes": list(CAPABILITY_ATTRIBUTES),
        "searches": [search_entry.to_dict() for search_entry in SEARCHES],
    }
