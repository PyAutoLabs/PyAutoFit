"""
Roster and golden table for the search conformance suite (``test_conformance.py``).

The roster is the 15 public non-linear searches of PyAutoFit:

- MCMC: ``Emcee``, ``Zeus``, ``BlackJAXNUTS``, ``SMC``.
- Nested sampling: ``DynestyStatic``, ``DynestyDynamic``, ``Nautilus``, ``NSS``.
- Maximum likelihood: ``Drawer``, ``BFGS``, ``LBFGS``, ``MultiStartAdam``,
  ``MultiStartADABelief``, ``MultiStartLion``, ``MultiStartProdigy``.

Each entry names its class by a class-path string. Tests resolve the class with
``importlib`` inside the test body, never at module import or collection time, so
this module and the suite collect on every CI leg.

Coverage contract
-----------------
Layer (i), this suite, checks metadata and serialization only: default construction,
family base class, ``__identifier_fields__``, the default-construction identifier,
the constructor argument names, the ``search.json`` round trip and that construction
leaves the config untouched. No sampler backend is executed. It runs on every CI leg,
including PyAutoHeart's ``unittest-nojax`` leg, where jax, jaxlib, optax and blackjax
are uninstalled but nautilus, zeus, emcee and dynesty remain installed. Every search
except ``NSS`` constructs with defaults when its backend is missing; ``NSS`` raises at
construction without blackjax (>= 1.6), so its construction-dependent cases skip when
a module in ``requires`` cannot be found.

Layer (ii), backend execution and ``samples_info`` key sets (only obtainable from live
sampler objects), comes later in the search-extensibility epic.

Golden table
------------
``GOLDEN`` freezes, per search, the ``__identifier_fields__`` tuple, the identifier of
a default-constructed search (``str(Identifier(cls()))``, computed with
``PYAUTO_TEST_MODE`` unset and the ``test_autofit/config`` configuration pushed), and
the ``__init__`` parameter names. Identifiers must not change: a failure here means a
refactor changed a search identifier, which must be flagged to a human before it lands.

The table was generated from PyAutoFit main 908ca61a5. To regenerate it (only after a
human has approved an identifier change), run from the PyAutoFit repo root::

    PYAUTO_TEST_MODE= python -c "
    import importlib, inspect
    from autonerves import conf
    conf.instance.push(new_path='test_autofit/config', output_path='test_autofit/output')
    from autofit.mapper.identifier import Identifier
    from test_autofit.non_linear.search.conformance_roster import searches_under_test
    for e in searches_under_test():
        cls = e.resolve()
        params = sorted(p for p in inspect.signature(cls.__init__).parameters if p != 'self')
        print(e.name, str(Identifier(cls())), cls.__identifier_fields__, params)
    "

with ``PYAUTO_TEST_MODE`` removed from the environment (``env -u PYAUTO_TEST_MODE``).
"""

import importlib
import importlib.util
from dataclasses import dataclass, field
from typing import List, Tuple

FAMILY_BASES = {
    "mcmc": "autofit.non_linear.search.mcmc.abstract_mcmc.AbstractMCMC",
    "nest": "autofit.non_linear.search.nest.abstract_nest.AbstractNest",
    "mle": "autofit.non_linear.search.mle.abstract_mle.AbstractMLE",
}


def resolve_class_path(class_path: str):
    """
    Import and return the object named by a dotted ``module.attribute`` path.
    """
    module_name, _, attribute = class_path.rpartition(".")
    return getattr(importlib.import_module(module_name), attribute)


@dataclass(frozen=True)
class SearchEntry:
    """
    One search under test.

    Parameters
    ----------
    name
        The public short name, as exported by ``autofit`` and used as the test id.
    class_path
        The dotted path of the search class.
    family
        One of ``mcmc``, ``nest`` or ``mle``.
    requires
        Top-level module names whose absence makes default construction impossible.
    """

    name: str
    class_path: str
    family: str
    requires: List[str] = field(default_factory=list)

    def missing_requirements(self) -> List[str]:
        """
        The modules in ``requires`` that cannot be found in this environment.
        """
        return [
            module
            for module in self.requires
            if importlib.util.find_spec(module) is None
        ]

    def resolve(self):
        """
        Import and return the search class.
        """
        return resolve_class_path(self.class_path)


_SEARCH = "autofit.non_linear.search"
_MULTI_START = f"{_SEARCH}.mle.multi_start_gradient.search"


def searches_under_test() -> List[SearchEntry]:
    """
    The 15 public searches of PyAutoFit, in a fixed order.
    """
    return [
        SearchEntry("Emcee", f"{_SEARCH}.mcmc.emcee.search.Emcee", "mcmc"),
        SearchEntry("Zeus", f"{_SEARCH}.mcmc.zeus.search.Zeus", "mcmc"),
        SearchEntry(
            "BlackJAXNUTS",
            f"{_SEARCH}.mcmc.blackjax.nuts.search.BlackJAXNUTS",
            "mcmc",
        ),
        SearchEntry("SMC", f"{_SEARCH}.mcmc.blackjax.smc.search.SMC", "mcmc"),
        SearchEntry(
            "DynestyStatic",
            f"{_SEARCH}.nest.dynesty.search.static.DynestyStatic",
            "nest",
        ),
        SearchEntry(
            "DynestyDynamic",
            f"{_SEARCH}.nest.dynesty.search.dynamic.DynestyDynamic",
            "nest",
        ),
        SearchEntry("Nautilus", f"{_SEARCH}.nest.nautilus.search.Nautilus", "nest"),
        SearchEntry(
            "NSS", f"{_SEARCH}.nest.nss.search.NSS", "nest", requires=["blackjax"]
        ),
        SearchEntry("Drawer", f"{_SEARCH}.mle.drawer.search.Drawer", "mle"),
        SearchEntry("BFGS", f"{_SEARCH}.mle.bfgs.search.BFGS", "mle"),
        SearchEntry("LBFGS", f"{_SEARCH}.mle.bfgs.search.LBFGS", "mle"),
        SearchEntry("MultiStartAdam", f"{_MULTI_START}.MultiStartAdam", "mle"),
        SearchEntry(
            "MultiStartADABelief", f"{_MULTI_START}.MultiStartADABelief", "mle"
        ),
        SearchEntry("MultiStartLion", f"{_MULTI_START}.MultiStartLion", "mle"),
        SearchEntry("MultiStartProdigy", f"{_MULTI_START}.MultiStartProdigy", "mle"),
    ]


@dataclass(frozen=True)
class Golden:
    """
    The frozen metadata of one search.

    Parameters
    ----------
    identifier
        ``str(Identifier(cls()))`` for a default-constructed search.
    identifier_fields
        ``cls.__identifier_fields__``.
    constructor_arguments
        The ``cls.__init__`` parameter names, excluding ``self``.
    """

    identifier: str
    identifier_fields: Tuple[str, ...]
    constructor_arguments: frozenset


_COMMON = frozenset(
    {
        "name",
        "path_prefix",
        "unique_tag",
        "iterations_per_full_update",
        "iterations_per_quick_update",
        "silence",
        "kwargs",
    }
)

_EMCEE_ZEUS_COMMON = _COMMON | {
    "session",
    "initializer",
    "auto_correlation_settings",
    "number_of_cores",
    "nsteps",
    "nwalkers",
}

_BFGS_ARGUMENTS = _COMMON | {
    "session",
    "initializer",
    "clipper",
    "disp",
    "eps",
    "ftol",
    "gtol",
    "iprint",
    "maxcor",
    "maxfun",
    "maxiter",
    "maxls",
    "tol",
}

_MULTI_START_ARGUMENTS = _COMMON | {
    "session",
    "initializer",
    "batch_size",
    "bijector",
    "clipper",
    "convergence",
    "gradient_mode",
    "iterations_per_log",
    "learning_rate",
    "max_consecutive_nan",
    "n_starts",
    "n_steps",
    "record_lane_nan_history",
    "reset_momentum_on_clip",
    "resurrect",
    "scaler",
    "seed",
    "start_lower_limit",
    "start_upper_limit",
    "trace_param_indices",
}

GOLDEN = {
    "Emcee": Golden(
        identifier="f804af6499430da65cf8fba466dcfdd3",
        identifier_fields=("nwalkers",),
        constructor_arguments=_EMCEE_ZEUS_COMMON,
    ),
    "Zeus": Golden(
        identifier="6d171637e7fbdd903c674391faa0b197",
        identifier_fields=(
            "nwalkers",
            "tune",
            "tolerance",
            "patience",
            "mu",
            "light_mode",
        ),
        constructor_arguments=_EMCEE_ZEUS_COMMON
        | {
            "check_walkers",
            "light_mode",
            "maxcall",
            "maxiter",
            "maxsteps",
            "mu",
            "patience",
            "shuffle_ensemble",
            "tolerance",
            "tune",
            "vectorize",
        },
    ),
    "BlackJAXNUTS": Golden(
        identifier="2a6f5217539f82fab8ea42eda9bf3441",
        identifier_fields=(
            "num_warmup",
            "num_samples",
            "num_chains",
            "inverse_mass_matrix",
        ),
        constructor_arguments=_COMMON
        | {
            "session",
            "initializer",
            "auto_correlation_settings",
            "number_of_cores",
            "inverse_mass_matrix",
            "mass_matrix_shrinkage",
            "max_num_doublings",
            "num_chains",
            "num_samples",
            "num_warmup",
            "seed",
            "share_adaptation",
            "target_accept",
        },
    ),
    "SMC": Golden(
        identifier="3baa494f2ca7b37a8c4b85884cadd1f7",
        identifier_fields=(
            "num_particles",
            "kernel",
            "num_mcmc_steps",
            "num_integration_steps",
            "target_ess",
            "inverse_mass_matrix",
        ),
        constructor_arguments=_COMMON
        | {
            "session",
            "initializer",
            "auto_correlation_settings",
            "number_of_cores",
            "batch_size",
            "inverse_mass_matrix",
            "kernel",
            "max_smc_steps",
            "num_integration_steps",
            "num_mcmc_steps",
            "num_particles",
            "seed",
            "step_size",
            "target_ess",
            "whiten_inflate",
        },
    ),
    "DynestyStatic": Golden(
        identifier="d75351d847c453ff56e2a64b2d071d6f",
        identifier_fields=(
            "nlive",
            "bound",
            "sample",
            "bootstrap",
            "enlarge",
            "walks",
            "facc",
            "slices",
            "fmove",
            "max_move",
        ),
        constructor_arguments=_COMMON
        | {
            "session",
            "number_of_cores",
            "dlogz",
            "logl_max",
            "maxiter",
            "nlive",
        },
    ),
    "DynestyDynamic": Golden(
        identifier="d39dcd32c65eb17c624b91343a46ee17",
        identifier_fields=(
            "bound",
            "sample",
            "enlarge",
            "bootstrap",
            "walks",
            "facc",
            "slices",
            "fmove",
            "max_move",
        ),
        constructor_arguments=_COMMON
        | {
            "number_of_cores",
            "dlogz_init",
            "logl_max_init",
            "maxcall_init",
            "maxiter",
            "maxiter_init",
            "nlive_init",
        },
    ),
    "Nautilus": Golden(
        identifier="e521e1535fb162729c321a4faf34d235",
        identifier_fields=(
            "n_live",
            "n_update",
            "enlarge_per_dim",
            "n_points_min",
            "split_threshold",
            "n_networks",
            "n_like_new_bound",
            "seed",
            "n_shell",
            "n_eff",
        ),
        constructor_arguments=_COMMON
        | {
            "session",
            "number_of_cores",
            "discard_exploration",
            "enlarge_per_dim",
            "f_live",
            "force_x1_cpu",
            "n_batch",
            "n_eff",
            "n_like_max",
            "n_like_new_bound",
            "n_live",
            "n_networks",
            "n_points_min",
            "n_shell",
            "n_update",
            "seed",
            "split_threshold",
            "use_jax_vmap",
            "vectorized",
            "verbose",
        },
    ),
    "NSS": Golden(
        identifier="a52ce705ede0c3f57e2e3fc3dbb0a3bd",
        identifier_fields=(
            "n_live",
            "num_mcmc_steps",
            "num_delete",
            "termination",
            "seed",
        ),
        constructor_arguments=_COMMON
        | {
            "session",
            "number_of_cores",
            "checkpoint_interval",
            "chunk_size",
            "n_live",
            "num_delete",
            "num_mcmc_steps",
            "seed",
            "termination",
        },
    ),
    "Drawer": Golden(
        identifier="0f80f695c0906253e9526e4c2f242dc7",
        identifier_fields=("total_draws",),
        constructor_arguments=_COMMON | {"session", "initializer", "total_draws"},
    ),
    "BFGS": Golden(
        identifier="979826a042aacbe511b97be84d8c2255",
        identifier_fields=("clipper",),
        constructor_arguments=_BFGS_ARGUMENTS,
    ),
    "LBFGS": Golden(
        identifier="eb33da63306e10e4fd4276f1e83f372b",
        identifier_fields=("clipper",),
        constructor_arguments=_BFGS_ARGUMENTS,
    ),
    "MultiStartAdam": Golden(
        identifier="bd3370e335bd0c947364b769700d1513",
        identifier_fields=("clipper",),
        constructor_arguments=_MULTI_START_ARGUMENTS,
    ),
    "MultiStartADABelief": Golden(
        identifier="305f4ce87416b2ca58a74b9c597ac0f5",
        identifier_fields=("clipper",),
        constructor_arguments=_MULTI_START_ARGUMENTS,
    ),
    "MultiStartLion": Golden(
        identifier="3d7d54515bf9e6224ba6f6b6ee6b848d",
        identifier_fields=("clipper",),
        constructor_arguments=_MULTI_START_ARGUMENTS,
    ),
    "MultiStartProdigy": Golden(
        identifier="676f33a7e8e071125e92c4ffcd754c27",
        identifier_fields=("clipper",),
        constructor_arguments=_MULTI_START_ARGUMENTS,
    ),
}
