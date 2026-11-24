"""
`number_of_cores="auto"` (likelihood-lifecycle phase 5D): validation, resolution after
`modify_before_fit` (`autofit.non_linear.parallel.pool.resolve_number_of_cores`), the
available-memory probe, the one `Parallel:` banner line, `search.summary` and
`files/parallel.json`.

The core and memory probes are always monkeypatched: no test depends on the machine's
core count or forks a worker.
"""
import json
import logging
import warnings
import zipfile

import numpy as np
import pytest

import autofit as af
from autofit.non_linear.parallel import pool as pool_module
from autofit.non_linear.parallel import resources
from autofit.non_linear.parallel import (
    PoolFactory,
    check_factor_search_cores,
    resolve_number_of_cores,
    validate_number_of_cores,
)

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(autouse=True)
def fake_machine(monkeypatch):
    """
    A fake 64-core machine with 120 GB available, and no forking allowed.
    """
    monkeypatch.setattr(pool_module, "detected_cores", lambda: 64)
    monkeypatch.setattr(resources, "detected_cores", lambda: 64)
    monkeypatch.setattr(pool_module, "available_memory_gb", lambda: 120.0)

    def no_fork(*args, **kwargs):
        raise AssertionError("must not fork")

    monkeypatch.setattr(pool_module, "fork_context", no_fork)


class _Setup:
    def __init__(self, setup_id):
        self.setup_id = setup_id


class _Analysis:
    def __init__(self, is_jax=False, setup_id=None, resources_per_eval=None):
        self.is_jax = is_jax
        if setup_id is not None:
            self.setup = _Setup(setup_id)
        if resources_per_eval is not None:
            self.resources_per_eval = lambda: resources_per_eval


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "value, expected",
    [(1, 1), (8, 8), (np.int64(3), 3), (None, 1), ("auto", "auto"), ("AUTO", "auto"),
     (" Auto ", "auto")],
)
def test_validate_accepts(value, expected):
    assert validate_number_of_cores(value) == expected


@pytest.mark.parametrize("value", [0, -2, 2.0, 1.5, True, "all", "", [2]])
def test_validate_rejects_with_a_clear_error(value):
    with pytest.raises(ValueError, match="number_of_cores="):
        validate_number_of_cores(value, where="Emcee")


def test_searches_and_settings_accept_auto_and_reject_garbage():
    assert af.Emcee(number_of_cores="auto").number_of_cores == "auto"
    assert af.DynestyStatic(number_of_cores="Auto").number_of_cores == "auto"
    assert af.DynestyDynamic(number_of_cores="auto").number_of_cores == "auto"
    assert af.Nautilus(number_of_cores="auto").number_of_cores == "auto"
    assert af.Zeus(number_of_cores="auto").number_of_cores == "auto"
    assert af.Emcee(number_of_cores=None).number_of_cores == 1
    assert af.SettingsSearch(path_prefix="x", number_of_cores="AUTO").number_of_cores == "auto"

    with pytest.raises(ValueError, match="Emcee"):
        af.Emcee(number_of_cores=0)
    with pytest.raises(ValueError, match="SettingsSearch"):
        af.SettingsSearch(path_prefix="x", number_of_cores="many")


def test_settings_search_passes_auto_through_to_a_search():
    settings = af.SettingsSearch(path_prefix="x", number_of_cores="auto")
    search = af.Emcee(**settings.search_dict_for(af.Emcee))
    assert search.number_of_cores == "auto"


def test_auto_defers_the_thread_warning(monkeypatch):
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(key, "4")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        af.Emcee(number_of_cores="auto")

    assert not [w for w in caught if issubclass(w.category, af.exc.SearchWarning)]


def test_grid_search_and_sensitivity_resolve_auto_to_detected_cores():
    grid = af.SearchGridSearch(search=af.m.MockSearch(), number_of_cores="auto")
    assert grid.number_of_cores == 64
    assert resources.resolve_job_cores(None, where="x") == 1
    with pytest.raises(ValueError):
        resources.resolve_job_cores(0, where="x")


# ---------------------------------------------------------------- resolution


def test_int_request_is_honoured():
    resolution = resolve_number_of_cores(4, _Analysis())
    assert (resolution.number_of_cores, resolution.reason) == (4, "requested")
    assert resolution.banner == "Parallel: 4 workers (requested 4)"


def test_auto_resolves_to_detected_cores():
    resolution = resolve_number_of_cores("auto", _Analysis(setup_id="numba_cpu_sparse_fp64"))
    assert resolution.number_of_cores == 64
    assert resolution.reason == "detected_cores"
    assert resolution.detected_cores == 64
    assert resolution.available_memory_gb is None
    assert resolution.banner == (
        "Parallel: 64 workers (requested auto; detected cores) · setup numba_cpu_sparse_fp64"
    )


def test_auto_under_jax_is_one():
    resolution = resolve_number_of_cores("auto", _Analysis(is_jax=True))
    assert resolution.number_of_cores == 1
    assert resolution.banner == (
        "Parallel: 1 worker (requested auto; JAX analyses run single-process)"
    )


def test_int_request_under_jax_is_one():
    resolution = resolve_number_of_cores(4, _Analysis(is_jax=True))
    assert (resolution.number_of_cores, resolution.reason) == (1, "jax")


def test_auto_ep_factor_search_is_one():
    resolution = resolve_number_of_cores("auto", _Analysis(), ep_factor_search=True)
    assert (resolution.number_of_cores, resolution.reason) == (1, "expectation_propagation")


def test_ep_check_accepts_auto_and_refuses_an_explicit_count():
    check_factor_search_cores(af.Emcee(number_of_cores="auto"))
    check_factor_search_cores(af.Emcee(number_of_cores=1))
    with pytest.raises(af.exc.SearchException):
        check_factor_search_cores(af.Emcee(number_of_cores=2))


def test_no_resources_hook_means_no_cap():
    resolution = resolve_number_of_cores("auto", _Analysis())
    assert resolution.number_of_cores == 64


def test_memory_cap(monkeypatch):
    analysis = _Analysis(resources_per_eval={"cpus_per_eval": 1, "peak_rss_gb": 2.5})
    resolution = resolve_number_of_cores("auto", analysis)
    assert resolution.number_of_cores == 48
    assert resolution.reason == "memory"
    assert resolution.available_memory_gb == 120.0
    assert resolution.banner == (
        "Parallel: 48 workers (requested auto; capped by memory: 120.0 GB available "
        "/ 2.5 GB per evaluation)"
    )


def test_cores_per_eval_cap():
    analysis = _Analysis(resources_per_eval={"cpus_per_eval": 4, "peak_rss_gb": 0.5})
    resolution = resolve_number_of_cores("auto", analysis)
    assert (resolution.number_of_cores, resolution.reason) == (16, "cores_per_eval")


def test_memory_cap_never_below_one(monkeypatch):
    monkeypatch.setattr(pool_module, "available_memory_gb", lambda: 1.0)
    analysis = _Analysis(resources_per_eval={"peak_rss_gb": 50.0})
    assert resolve_number_of_cores("auto", analysis).number_of_cores == 1


def test_unknown_memory_means_no_memory_cap(monkeypatch):
    monkeypatch.setattr(pool_module, "available_memory_gb", lambda: None)
    analysis = _Analysis(resources_per_eval={"peak_rss_gb": 50.0})
    assert resolve_number_of_cores("auto", analysis).number_of_cores == 64


def test_resources_hook_returning_none_means_no_cap():
    analysis = _Analysis()
    analysis.resources_per_eval = lambda: None
    assert resolve_number_of_cores("auto", analysis).number_of_cores == 64


def test_int_request_is_not_capped_by_memory():
    analysis = _Analysis(resources_per_eval={"peak_rss_gb": 50.0})
    assert resolve_number_of_cores(8, analysis).number_of_cores == 8


def test_pool_factory_uses_the_resolved_count():
    search = af.Emcee(number_of_cores="auto")
    factory = PoolFactory(search, _Analysis())
    assert factory.requested == "auto"
    assert factory.number_of_cores == 64
    assert "Number of cores = 64 (requested auto): detected cores" in factory.summary


def test_pool_factory_under_jax_auto_builds_no_pool():
    factory = PoolFactory(af.Emcee(number_of_cores="auto"), _Analysis(is_jax=True))
    assert factory.number_of_cores == 1
    assert factory() is None
    assert factory.sneaky(fitness=None) is None


def test_pool_factory_ep_flag_resolves_to_one():
    search = af.Emcee(number_of_cores="auto")
    search._ep_factor_search = True
    assert PoolFactory(search, _Analysis()).number_of_cores == 1


def test_summary_records_the_setup():
    resolution = resolve_number_of_cores(2, _Analysis(setup_id="numpy_cpu_fp64"))
    assert resolution.summary == (
        "Number of cores = 2 (requested 2)\nSetup = numpy_cpu_fp64\n"
    )
    assert resolution.to_dict() == {
        "requested": 2,
        "number_of_cores": 2,
        "reason": "requested",
        "detected_cores": None,
        "available_memory_gb": None,
        "setup_id": "numpy_cpu_fp64",
    }


# ---------------------------------------------------------------- memory probe


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_available_memory_reads_memavailable(tmp_path):
    meminfo = tmp_path / "meminfo"
    _write(meminfo, "MemTotal:       16777216 kB\nMemAvailable:    8388608 kB\n")
    assert resources.available_memory_gb(
        meminfo_path=str(meminfo),
        cgroup_root=str(tmp_path / "nocgroup"),
        proc_cgroup_path=str(tmp_path / "nocgroupfile"),
    ) == pytest.approx(8.0)


def test_available_memory_falls_back_to_total(tmp_path):
    meminfo = tmp_path / "meminfo"
    _write(meminfo, "MemTotal:       16777216 kB\n")
    assert resources.available_memory_gb(
        meminfo_path=str(meminfo),
        cgroup_root=str(tmp_path / "nocgroup"),
        proc_cgroup_path=str(tmp_path / "nocgroupfile"),
    ) == pytest.approx(16.0)


def test_available_memory_respects_a_lower_cgroup_limit(tmp_path):
    meminfo = tmp_path / "meminfo"
    _write(meminfo, "MemTotal:       16777216 kB\nMemAvailable:    8388608 kB\n")
    root = tmp_path / "cgroup"
    _write(root / "job" / "memory.max", str(4 * 1024 ** 3))
    _write(root / "job" / "memory.current", str(1 * 1024 ** 3))
    proc = tmp_path / "proc_cgroup"
    _write(proc, "0::/job\n")

    assert resources.available_memory_gb(
        meminfo_path=str(meminfo), cgroup_root=str(root), proc_cgroup_path=str(proc)
    ) == pytest.approx(3.0)


def test_available_memory_ignores_an_unlimited_cgroup(tmp_path):
    meminfo = tmp_path / "meminfo"
    _write(meminfo, "MemAvailable:    8388608 kB\n")
    root = tmp_path / "cgroup"
    _write(root / "memory.max", "max\n")

    assert resources.available_memory_gb(
        meminfo_path=str(meminfo),
        cgroup_root=str(root),
        proc_cgroup_path=str(tmp_path / "missing"),
    ) == pytest.approx(8.0)


def test_available_memory_with_nothing_readable_uses_nerves(tmp_path, monkeypatch):
    import autonerves.environment

    monkeypatch.setattr(autonerves.environment, "detect_ram_gb", lambda: 32.0)
    assert resources.available_memory_gb(
        meminfo_path=str(tmp_path / "missing"),
        cgroup_root=str(tmp_path / "missing_root"),
        proc_cgroup_path=str(tmp_path / "missing_proc"),
    ) == 32.0


# ---------------------------------------------------------------- a fit


class _SetupAnalysis(af.ex.Analysis):
    def modify_before_fit(self, paths, model):
        self.setup = _Setup("numba_cpu_sparse_fp64")
        return self


def _banner_records(caplog):
    return [r for r in caplog.records if r.getMessage().startswith("Parallel:")]


def test_fit_logs_one_banner_after_modify_before_fit_and_writes_parallel_json(
    monkeypatch, caplog, tmp_path
):
    monkeypatch.setenv("PYAUTO_TEST_MODE", "2")

    analysis = _SetupAnalysis(data=np.ones(10), noise_map=np.ones(10))
    search = af.Emcee(
        name="auto_cores", path_prefix=str(tmp_path), number_of_cores="auto"
    )

    with caplog.at_level(logging.INFO):
        search.fit(model=af.Model(af.ex.Gaussian), analysis=analysis)

    banners = _banner_records(caplog)
    assert [r.getMessage() for r in banners] == [
        "Parallel: 64 workers (requested auto; detected cores) · setup numba_cpu_sparse_fp64"
    ]
    assert not [r for r in caplog.records if "cores." in r.getMessage()
                and r.getMessage().startswith("Starting non-linear search")]

    assert search.number_of_cores == "auto"

    with zipfile.ZipFile(search.paths._zip_path) as archive:
        names = archive.namelist()
        record = json.loads(archive.read("files/parallel.json"))
        search_json = json.loads(archive.read("files/search.json"))

    # Test mode 2 writes no `search.summary`; its run-time lines come from here.
    from autofit.non_linear.search.capabilities import run_summary_from

    summary = run_summary_from(search)
    assert "Number of cores = 64 (requested auto): detected cores" in summary
    assert "Setup = numba_cpu_sparse_fp64" in summary

    assert record == {
        "requested": "auto",
        "number_of_cores": 64,
        "reason": "detected_cores",
        "detected_cores": 64,
        "available_memory_gb": None,
        "setup_id": "numba_cpu_sparse_fp64",
        "estimate_line": None,
    }

    assert search_json["arguments"]["number_of_cores"] == "auto"


def test_fit_banner_for_jax_auto(monkeypatch, caplog, tmp_path):
    pytest.importorskip("jax")
    monkeypatch.setenv("PYAUTO_TEST_MODE", "2")

    analysis = af.ex.Analysis(data=np.ones(10), noise_map=np.ones(10), use_jax=True)
    search = af.Emcee(name="auto_jax", path_prefix=str(tmp_path), number_of_cores="auto")

    with caplog.at_level(logging.INFO):
        search.fit(model=af.Model(af.ex.Gaussian), analysis=analysis)

    assert [r.getMessage() for r in _banner_records(caplog)] == [
        "Parallel: 1 worker (requested auto; JAX analyses run single-process)"
    ]


# ------------------------------------------------- runtime_estimate hook (5E-2)


class _EstimateAnalysis(_SetupAnalysis):
    """An analysis with a duck-typed ``runtime_estimate`` hook that records its call."""

    def __init__(self, *args, line="Estimate: ~2 ms/eval", raises=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.line = line
        self.raises = raises
        self.calls = []

    def runtime_estimate(self, search, model, number_of_cores):
        self.calls.append((search, model, number_of_cores))
        if self.raises:
            raise RuntimeError("no estimate data")
        return self.line


def _fit_with(analysis, tmp_path, monkeypatch, caplog, name):
    monkeypatch.setenv("PYAUTO_TEST_MODE", "2")
    search = af.Emcee(name=name, path_prefix=str(tmp_path), number_of_cores="auto")
    model = af.Model(af.ex.Gaussian)
    with caplog.at_level(logging.INFO):
        search.fit(model=model, analysis=analysis)
    with zipfile.ZipFile(search.paths._zip_path) as archive:
        record = json.loads(archive.read("files/parallel.json"))
    messages = [r.getMessage() for r in caplog.records]
    return search, model, record, messages


def test_runtime_estimate_hook_is_logged_after_the_banner_and_recorded(
    monkeypatch, caplog, tmp_path
):
    analysis = _EstimateAnalysis(data=np.ones(10), noise_map=np.ones(10))
    search, model, record, messages = _fit_with(
        analysis, tmp_path, monkeypatch, caplog, "estimate_hook"
    )

    banner = next(i for i, m in enumerate(messages) if m.startswith("Parallel:"))
    assert messages[banner + 1] == "Estimate: ~2 ms/eval"
    assert record["estimate_line"] == "Estimate: ~2 ms/eval"

    assert len(analysis.calls) == 1
    called_search, called_model, called_cores = analysis.calls[0]
    assert called_search is search
    assert called_model.prior_count == model.prior_count
    assert called_cores == 64


def test_runtime_estimate_hook_that_raises_is_ignored(monkeypatch, caplog, tmp_path):
    analysis = _EstimateAnalysis(
        data=np.ones(10), noise_map=np.ones(10), raises=True
    )
    _, _, record, messages = _fit_with(
        analysis, tmp_path, monkeypatch, caplog, "estimate_raises"
    )

    assert record["estimate_line"] is None
    assert not [m for m in messages if m.startswith("Estimate:")]
    assert len([m for m in messages if m.startswith("Parallel:")]) == 1


@pytest.mark.parametrize("line", [None, "", "   ", 42])
def test_runtime_estimate_hook_non_string_or_empty_is_ignored(line):
    from autofit.non_linear.search.runtime import runtime_estimate_line

    class _Analysis:
        def runtime_estimate(self, search, model, number_of_cores):
            return line

    assert (
        runtime_estimate_line(
            search=None, analysis=_Analysis(), model=None, number_of_cores=1
        )
        is None
    )


def test_runtime_estimate_line_without_hook_is_none():
    from autofit.non_linear.search.runtime import runtime_estimate_line

    assert (
        runtime_estimate_line(
            search=None, analysis=object(), model=None, number_of_cores=4
        )
        is None
    )

