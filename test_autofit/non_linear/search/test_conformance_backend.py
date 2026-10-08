"""
Search conformance suite, layer (ii): backend execution of the 15 public searches.

Every case is parametrised over ``searches_under_test()`` (ids are the short names) and
runs the search's real sampler at ``PYAUTO_TEST_MODE=1``, the reduced-budget mode in
which each search's ``apply_test_mode`` shrinks its own budget. Budgets are reduced
only through that mode; no search is edited and no budget argument is passed. The one
constructor argument passed (``Emcee``'s ``auto_correlation_settings``) stands in for a
default built at import time; see ``DEFAULT_AUTO_CORRELATION_XFAIL``.

Per search the suite asserts:

- (a) a fit with real ``DirectoryPaths`` writes the expected output file set, checked
  in the ``.zip`` archive the test configuration (``remove_files: true``) leaves behind:
  the generic files, ``samples.csv`` / ``samples_info.json`` (``covariance.csv`` for the
  PDF families) and the backend's own ``search_internal`` file;
- (b) a second ``fit`` on the completed path restores the result without calling the
  backend (``_fit`` is wrapped by a call counter, which counts one call on the first fit);
- (c) a ``NullPaths`` fit (no ``name``) returns a ``Result`` with samples;
- (d) ``samples_from(model, None)`` follows the narrowed fallback of search-extensibility
  phase A0b: under ``NullPaths`` a missing internal state raises its real error rather
  than falling back to ``paths.samples``; a ``FileNotFoundError`` or
  ``NotImplementedError`` falls back with a WARNING; and on a restored output folder the
  samples load from the backend's own internal state;
- (e) the ``samples_info`` key set, frozen per search and sharing a per-family core.

Coverage contract
-----------------
Every backend executes on PyAutoHeart's full-extras CI legs: ``lib-tests.yml`` installs
the ``[optional]`` extras on its ``unittest`` legs, so jax, blackjax, optax, nautilus,
zeus, emcee and dynesty are all present there. ``importorskip`` (or any other
import-availability skip) may never hide a required backend on those legs: a missing
backend for a ``jax_use`` ``"none"`` or ``"optional"`` search fails the suite.

On the ``unittest-nojax`` leg the jax family (jax, jaxlib, optax, blackjax, ...) is
uninstalled. There a search is skipped only by its DECLARED capability in
``conformance_roster.py``: ``jax_use == "required"`` together with a missing module in
its ``jax_modules``. That skips exactly ``NSS``, ``BlackJAXNUTS``, ``SMC`` and the four
``MultiStart`` searches; every other search runs with a numpy likelihood.

Likelihoods are capability-matched: ``jax_use`` ``"none"`` / ``"optional"`` searches fit
a numpy analysis (``use_jax=False``) and ``"required"`` searches fit a JAX analysis
(``jax.numpy`` log likelihood, ``use_jax=True``).

Each search's fits run once per module (``backend_run``) and the cases read the cached
outcome. Strict xfails mark behaviour that is known to be wrong today and is repaired by
a later phase of the search-extensibility epic; a repair flips the xfail to XPASS, which
fails the suite until the marker is removed.
"""

import importlib.util
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

import autofit as af
from autofit.non_linear.paths.directory import DirectoryPaths
from autofit.non_linear.paths.null import NullPaths
from autofit.non_linear.test_mode import test_mode_level
from autonerves import conf

from test_autofit.non_linear.search.conformance_roster import searches_under_test

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

ROSTER = searches_under_test()

CONFIG_PATH = Path(__file__).parents[2] / "config"

_TARGET = (50.0, 25.0, 10.0)
_SCALE = (10.0, 5.0, 2.0)

_FALLBACK_MESSAGE = "could not be loaded from its internal results"

# --- (a) expected output files ---------------------------------------------------------

_GENERIC_FILES = frozenset(
    {
        ".completed",
        ".identifier",
        "model.info",
        "model.results",
        "search.log",
        "search.summary",
        "files/model.json",
        "files/search.json",
        "files/samples.csv",
        "files/samples_info.json",
        "files/samples_summary.json",
        "files/search_internal/.start_time",
        "files/search_internal/.time",
    }
)

# ``covariance.csv`` is written only for ``SamplesPDF`` (the MCMC and nested families).
_PDF_FILES = frozenset({"files/covariance.csv"})

_DILL = frozenset({"files/search_internal/search_internal.dill"})

# The backend's own checkpoint / internal-state files (survey 01 §5.3 item 2, §8).
SEARCH_INTERNAL_FILES = {
    "Emcee": frozenset({"files/search_internal/search_internal.hdf"}),
    "Zeus": _DILL,
    "BlackJAXNUTS": frozenset({"files/search_internal/search_internal.pickle"}),
    "SMC": frozenset({"files/search_internal/search_internal.pickle"}),
    "DynestyStatic": _DILL | {"files/search_internal/uses_pool.save"},
    "DynestyDynamic": _DILL | {"files/search_internal/uses_pool.save"},
    "Nautilus": _DILL,
    "NSS": _DILL,
    "Drawer": _DILL,
    "BFGS": _DILL,
    "LBFGS": _DILL,
    "MultiStartAdam": _DILL,
    "MultiStartADABelief": _DILL,
    "MultiStartLion": _DILL,
    "MultiStartProdigy": _DILL,
}

# --- (e) samples_info key sets ---------------------------------------------------------

_CORE_KEYS = frozenset({"class_path", "time"})

FAMILY_SAMPLES_INFO_CORE = {
    "mcmc": _CORE_KEYS,
    "nest": _CORE_KEYS
    | {"log_evidence", "total_samples", "total_accepted_samples", "number_live_points"},
    "mle": _CORE_KEYS,
}

_EMCEE_ZEUS_KEYS = _CORE_KEYS | {
    "check_size",
    "required_length",
    "change_threshold",
    "total_walkers",
    "total_steps",
}

_DYNESTY_NAUTILUS_KEYS = FAMILY_SAMPLES_INFO_CORE["nest"]

_BFGS_KEYS = _CORE_KEYS | {"total_iterations", "clipper"}

_MULTI_START_KEYS = _CORE_KEYS | {
    "n_starts",
    "n_steps",
    "batch_size",
    "gradient_mode",
    "total_steps",
    "optax_method",
    "learning_rate",
    "max_consecutive_nan",
    "resurrect",
    "n_resurrections",
    "n_value_nan_lane_steps",
    "n_constrained_lane_steps",
    "n_grad_nan_lane_steps",
    "clipper",
    "scaler",
    "bijector",
    "bijector_kinds",
    "reset_momentum_on_clip",
    "n_clipped_lane_steps",
    "seed",
    "stop_reason",
    "converged",
    "convergence",
    "fom_history",
}

SAMPLES_INFO_KEYS = {
    "Emcee": _EMCEE_ZEUS_KEYS,
    "Zeus": _EMCEE_ZEUS_KEYS,
    "BlackJAXNUTS": _EMCEE_ZEUS_KEYS
    | {
        "num_warmup",
        "num_samples",
        "num_chains",
        "ess_min",
        "ess_per_param",
        "ess_bulk_per_param",
        "ess_tail_per_param",
        "ess_bulk_min",
        "ess_tail_min",
        "rhat_per_param",
        "rhat_max",
        "mean_acceptance",
        "n_divergent",
        "divergent_indices",
        "tree_depth_histogram",
        "n_logl_evals",
        "warm_start_source",
        "inverse_mass_matrix_kind",
    },
    "SMC": _CORE_KEYS
    | {
        "num_particles",
        "kernel",
        "num_mcmc_steps",
        "num_integration_steps",
        "target_ess",
        "step_size",
        "log_evidence",
        "log_jacobian",
        "lambda_list",
        "log_likelihood_increment_list",
        "acceptance_rate_list",
        "ess_list",
        "max_log_likelihood_list",
        "n_smc_steps",
        "converged",
        "is_warm_start",
        "whitening_kind",
        "warm_start_source",
        "inverse_mass_matrix_kind",
        "total_samples",
    },
    "DynestyStatic": _DYNESTY_NAUTILUS_KEYS,
    "DynestyDynamic": _DYNESTY_NAUTILUS_KEYS,
    "Nautilus": _DYNESTY_NAUTILUS_KEYS,
    "NSS": _DYNESTY_NAUTILUS_KEYS
    | {
        "log_evidence_error",
        "sampling_time",
        "num_mcmc_steps",
        "num_delete",
        "termination",
        "ess",
    },
    "Drawer": _CORE_KEYS | {"parameter_lists", "log_posterior_list"},
    "BFGS": _BFGS_KEYS,
    "LBFGS": _BFGS_KEYS,
    "MultiStartAdam": _MULTI_START_KEYS,
    "MultiStartADABelief": _MULTI_START_KEYS,
    "MultiStartLion": _MULTI_START_KEYS,
    "MultiStartProdigy": _MULTI_START_KEYS,
}

# --- (d) the Samples class each search returns -----------------------------------------

_SAMPLES_CLASS = {
    "mcmc": "SamplesMCMC",
    "nest": "SamplesNest",
    "mle": "Samples",
}

SAMPLES_CLASS = {
    **{entry.name: _SAMPLES_CLASS[entry.family] for entry in ROSTER},
    "SMC": "SamplesSMC",
    "NSS": "NSSamples",
}

# ``Emcee``, ``Zeus``, ``BlackJAXNUTS`` and ``SMC`` take ``auto_correlation_settings=
# AutoCorrelationsSettings()`` as a default argument, built once at import time, so the
# test-mode ``check_size = 1`` reduction applies only when ``PYAUTO_TEST_MODE`` is set
# before ``autofit`` is imported. Set at run time (as here), the default keeps
# ``check_size = 100`` and a 10-step test-mode Emcee chain fails its auto-correlation
# check with ``IndexError``. Per-posterior-kind default objects and a generic
# ``apply_test_mode`` are phase A4.
DEFAULT_AUTO_CORRELATION_XFAIL = {
    name: (
        "auto_correlation_settings default is built at import time, so a run-time "
        "PYAUTO_TEST_MODE leaves check_size=100; per-kind defaults and a generic "
        "apply_test_mode in search-extensibility phase A4."
    )
    for name in ("Emcee", "Zeus", "BlackJAXNUTS", "SMC")
}


def _construction_overrides(entry) -> dict:
    """
    Constructor arguments that stand in for a default argument built at import time.

    Only ``Emcee`` needs one: it receives the ``AutoCorrelationsSettings()`` its default
    would be had test mode been on at import (see ``DEFAULT_AUTO_CORRELATION_XFAIL``),
    so its backend still executes here. No budget is changed.
    """
    if entry.name == "Emcee":
        return {"auto_correlation_settings": af.AutoCorrelationsSettings()}
    return {}


def _params(xfails=None, raises=None):
    """
    The roster as pytest params, with strict xfail marks for the named entries.
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


# --- capability-matched analyses -------------------------------------------------------


class NumpyMockAnalysis(af.Analysis):
    """
    A cheap, smooth numpy log likelihood over the three ``Gaussian`` parameters, for
    searches whose declared ``jax_use`` is ``"none"`` or ``"optional"``.
    """

    def __init__(self):
        super().__init__(use_jax=False)

    def log_likelihood_function(self, instance):
        vector = np.array([instance.centre, instance.normalization, instance.sigma])
        return float(
            -0.5 * np.sum(((vector - np.array(_TARGET)) / np.array(_SCALE)) ** 2)
        )


class JaxMockAnalysis(af.Analysis):
    """
    The same log likelihood written in ``jax.numpy`` (``use_jax=True``), for searches
    whose declared ``jax_use`` is ``"required"``.
    """

    def __init__(self):
        super().__init__(use_jax=True)

    def log_likelihood_function(self, instance):
        import jax.numpy as jnp

        vector = jnp.array([instance.centre, instance.normalization, instance.sigma])
        return -0.5 * jnp.sum(((vector - jnp.array(_TARGET)) / jnp.array(_SCALE)) ** 2)


def _analysis_for(entry):
    if entry.jax_use == "required":
        return JaxMockAnalysis()
    return NumpyMockAnalysis()


def _model():
    return af.Model(af.ex.Gaussian)


def skip_by_declared_capability(entry):
    """
    Skip a search only when it declares ``jax_use == "required"`` and a module in its
    ``jax_modules`` is missing (the ``unittest-nojax`` leg). Nothing else ever skips.
    """
    if entry.jax_use != "required":
        return
    missing = [
        module
        for module in entry.jax_modules
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        pytest.skip(
            f"{entry.name} declares jax_use='required' and {missing} is not installed "
            f"(the unittest-nojax leg)"
        )


# --- the cached backend run ------------------------------------------------------------


class _RecordHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages: List[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@dataclass
class SamplesFromOutcome:
    """
    What ``samples_from(model, None)`` did: returned ``samples`` or raised ``error``,
    and whether it logged the A0b fallback warning.
    """

    samples: Any = None
    error: Optional[BaseException] = None
    fell_back: bool = False


@dataclass
class BackendRun:
    """
    The outcome of one search's three fits (directory, resumed directory, null paths).
    """

    search: Any
    result: Any
    archived_files: frozenset
    samples_info: Dict[str, Any]
    first_fit_calls: int
    resumed_result: Any
    resume_fit_calls: int
    null_search: Any
    null_result: Any
    disk_samples_from: SamplesFromOutcome
    null_samples_from: SamplesFromOutcome
    test_mode_level: int


def _samples_from_outcome(search, model) -> SamplesFromOutcome:
    handler = _RecordHandler()
    logger = logging.getLogger("autofit.non_linear.search.abstract_search")
    logger.addHandler(handler)
    outcome = SamplesFromOutcome()
    try:
        outcome.samples = search.samples_from(model, None)
    except Exception as error:
        outcome.error = error
    finally:
        logger.removeHandler(handler)
    outcome.fell_back = any(_FALLBACK_MESSAGE in m for m in handler.messages)
    return outcome


def _execute(entry, output_path: Path) -> BackendRun:
    import json

    cls = entry.resolve()

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("PYAUTO_TEST_MODE", "1")
        level = test_mode_level()

        conf.instance.push(new_path=str(CONFIG_PATH), output_path=str(output_path))
        # The test configuration disables samples.csv output; the package default
        # (and every user run) writes it, and layer (ii) asserts on it.
        conf.instance["general"]["output"]["samples_to_csv"] = True

        calls = []
        original_fit = cls._fit

        def counted_fit(self, *args, **kwargs):
            calls.append(type(self).__name__)
            return original_fit(self, *args, **kwargs)

        patch.setattr(cls, "_fit", counted_fit)

        search = cls(name=entry.name, **_construction_overrides(entry))
        assert isinstance(search.paths, DirectoryPaths)
        result = search.fit(model=_model(), analysis=_analysis_for(entry))
        first_fit_calls = len(calls)

        with zipfile.ZipFile(search.paths._zip_path) as archive:
            archived_files = frozenset(
                name
                for name in archive.namelist()
                if not name.endswith("/") and not name.startswith("image/")
            )
            samples_info = json.loads(archive.read("files/samples_info.json"))

        resumed_search = cls(name=entry.name, **_construction_overrides(entry))
        resumed_result = resumed_search.fit(
            model=_model(), analysis=_analysis_for(entry)
        )
        resume_fit_calls = len(calls) - first_fit_calls

        resumed_search.paths.restore()
        disk_samples_from = _samples_from_outcome(
            resumed_search, resumed_search.paths.model
        )

        null_search = cls(**_construction_overrides(entry))
        assert isinstance(null_search.paths, NullPaths)
        null_model = _model()
        null_result = null_search.fit(model=null_model, analysis=_analysis_for(entry))
        null_samples_from = _samples_from_outcome(null_search, null_model)

    return BackendRun(
        search=search,
        result=result,
        archived_files=archived_files,
        samples_info=samples_info,
        first_fit_calls=first_fit_calls,
        resumed_result=resumed_result,
        resume_fit_calls=resume_fit_calls,
        null_search=null_search,
        null_result=null_result,
        disk_samples_from=disk_samples_from,
        null_samples_from=null_samples_from,
        test_mode_level=level,
    )


@pytest.fixture(name="backend_run", scope="module")
def make_backend_run(tmp_path_factory):
    """
    Run each search's fits once per module and hand every case the cached outcome.

    A failure is cached too, so every case of a broken search reports the same error
    without re-running the backend.
    """
    cache = {}

    def backend_run(entry) -> BackendRun:
        skip_by_declared_capability(entry)
        if entry.name not in cache:
            try:
                cache[entry.name] = _execute(
                    entry, tmp_path_factory.mktemp(f"backend_{entry.name}")
                )
            except Exception as error:
                cache[entry.name] = error
        outcome = cache[entry.name]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return backend_run


# --- the cases -------------------------------------------------------------------------


@pytest.mark.parametrize("entry", _params())
def test_backend_runs_in_reduced_test_mode(entry, backend_run):
    run = backend_run(entry)

    assert run.test_mode_level == 1
    assert run.first_fit_calls == 1


@pytest.mark.parametrize("entry", _params())
def test_directory_paths_output_file_set(entry, backend_run):
    run = backend_run(entry)

    expected = _GENERIC_FILES | SEARCH_INTERNAL_FILES[entry.name]
    if entry.family in ("mcmc", "nest"):
        expected = expected | _PDF_FILES

    assert run.archived_files == expected
    assert isinstance(run.result, af.Result)


@pytest.mark.parametrize("entry", _params())
def test_completed_path_resumes_without_rerunning_the_backend(entry, backend_run):
    run = backend_run(entry)

    assert run.resume_fit_calls == 0
    assert isinstance(run.resumed_result, af.Result)
    assert type(run.resumed_result.samples).__name__ == SAMPLES_CLASS[entry.name]
    assert run.resumed_result.samples_summary.max_log_likelihood_sample is not None


@pytest.mark.parametrize("entry", _params())
def test_null_paths_fit_returns_a_result(entry, backend_run):
    run = backend_run(entry)

    assert isinstance(run.null_search.paths, NullPaths)
    assert isinstance(run.null_result, af.Result)
    assert type(run.null_result.samples).__name__ == SAMPLES_CLASS[entry.name]
    assert len(run.null_result.samples.sample_list) > 0


@pytest.mark.parametrize("entry", _params())
def test_null_paths_samples_from_none_raises_rather_than_falling_back(
    entry, backend_run
):
    """
    Under ``NullPaths`` there is no internal state to load; A0b narrowed the fallback so
    the real error surfaces instead of ``paths.samples`` (``None``) being returned.
    """
    outcome = backend_run(entry).null_samples_from

    assert outcome.samples is None
    assert outcome.error is not None
    assert not isinstance(outcome.error, (FileNotFoundError, NotImplementedError))
    assert not outcome.fell_back


@pytest.mark.parametrize("entry", _params())
@pytest.mark.parametrize("error_type", [FileNotFoundError, NotImplementedError])
def test_samples_from_falls_back_to_paths_samples(
    entry, error_type, backend_run, monkeypatch, caplog
):
    search = backend_run(entry).null_search

    def raise_error(*args, **kwargs):
        raise error_type("no internal results")

    monkeypatch.setattr(search, "samples_via_internal_from", raise_error)

    with caplog.at_level(logging.WARNING):
        samples = search.samples_from(_model(), None)

    assert samples is None
    assert any(_FALLBACK_MESSAGE in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("entry", _params())
def test_restored_output_samples_from_none_loads_the_internal_state(entry, backend_run):
    """
    A completed, restored output folder keeps the archive ``samples_via_internal_from``
    reads (for Dynesty ``search_internal.dill``; its ``savestate.save`` resume state is
    deleted on completion), so the samples load from it without the ``samples.csv``
    fallback.
    """
    outcome = backend_run(entry).disk_samples_from

    assert outcome.error is None
    assert type(outcome.samples).__name__ == SAMPLES_CLASS[entry.name]
    assert not outcome.fell_back


@pytest.mark.parametrize("entry", _params())
def test_samples_info_key_set(entry, backend_run):
    keys = set(backend_run(entry).samples_info)

    assert FAMILY_SAMPLES_INFO_CORE[entry.family] <= keys
    assert keys == SAMPLES_INFO_KEYS[entry.name]


@pytest.mark.parametrize(
    "entry",
    [
        param
        for param in _params(DEFAULT_AUTO_CORRELATION_XFAIL, raises=AssertionError)
        if param.values[0].family == "mcmc"
    ],
)
def test_default_auto_correlation_settings_follow_run_time_test_mode(
    entry, monkeypatch
):
    """
    A search constructed after ``PYAUTO_TEST_MODE`` is switched on gets the test-mode
    auto-correlation settings, whatever the mode was when its module was imported.
    """
    skip_by_declared_capability(entry)
    cls = entry.resolve()

    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)
    cls()

    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")
    search = cls()

    assert search.auto_correlation_settings.check_size == 1
