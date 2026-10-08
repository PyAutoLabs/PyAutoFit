"""
Loading a search's internal state per output kind (search-extensibility phase A3, D5):
directory (through the search's injected ``checkpointer`` or, without a search, the
format-aware legacy probe), zipped, database, null and summary-only outputs.
"""
import logging
from pathlib import Path

import dill
import numpy as np
import pytest

import autofit as af
from autofit.non_linear.checkpoint import load_legacy_search_internal
from autofit.non_linear.paths.directory import DirectoryPaths
from autofit.non_linear.paths.null import NullPaths
from autofit.non_linear.result import Result
from autonerves import conf

CONFIG = Path(__file__).parents[2] / "config"


@pytest.fixture(name="output", autouse=True)
def make_output(tmp_path):
    conf.instance.push(new_path=str(CONFIG), output_path=str(tmp_path))
    return tmp_path


def _write_dill(paths, obj):
    with open(paths.search_internal_path / "search_internal.dill", "wb") as f:
        dill.dump(obj, f)


def test_directory_without_a_search_uses_the_legacy_probe():
    paths = DirectoryPaths(name="loading_legacy")
    _write_dill(paths, {"state": 1})

    assert paths.search is None
    assert paths.load_search_internal() == {"state": 1}


def test_directory_with_a_search_loads_through_its_checkpointer():
    search = af.Drawer(name="loading_injected")
    _write_dill(search.paths, {"state": 2})

    assert search.paths.search is search
    assert search.paths.load_search_internal() == {"state": 2}
    assert search.load_search_internal() == {"state": 2}


def _emcee_hdf(paths):
    emcee = pytest.importorskip("emcee")
    pytest.importorskip("h5py")

    backend = emcee.backends.HDFBackend(
        str(paths.search_internal_path / "search_internal.hdf")
    )
    sampler = emcee.EnsembleSampler(
        nwalkers=4,
        ndim=1,
        log_prob_fn=lambda x: -0.5 * float(np.sum(x**2)),
        backend=backend,
    )
    sampler.run_mcmc(np.random.default_rng(1).normal(size=(4, 1)), 3)
    return emcee


def test_emcee_hdf_loads_with_and_without_the_search():
    """
    The emcee HDF detection that used to live inline in
    ``DirectoryPaths.load_search_internal`` is the Emcee checkpointer's loader, and is
    kept as the first format of the legacy probe for a folder opened on its own.
    """
    search = af.Emcee(name="loading_emcee")
    emcee = _emcee_hdf(search.paths)

    via_search = search.paths.load_search_internal()
    assert isinstance(via_search, emcee.backends.HDFBackend)
    assert via_search.iteration == 3

    legacy = load_legacy_search_internal(search.paths.search_internal_path)
    assert isinstance(legacy, emcee.backends.HDFBackend)


def test_zipped_output_loads_after_restore():
    search = af.Drawer(name="loading_zipped")
    paths = search.paths
    _write_dill(paths, {"state": 3})

    paths.zip_remove()
    assert Path(paths._zip_path).exists()

    paths.restore()

    assert paths.load_search_internal() == {"state": 3}


def test_database_paths_load_nothing(session):
    paths = af.DatabasePaths(session=session, name="loading_database")

    assert paths.load_search_internal() is None
    assert Result(samples_summary=None, paths=paths).search_internal is None


def test_null_paths_load_nothing():
    paths = NullPaths()

    assert paths.load_search_internal() is None
    assert Result(samples_summary=None, paths=paths).search_internal is None


@pytest.fixture(name="search_internal_off")
def make_search_internal_off(tmp_path):
    original_configs = list(conf.instance.configs)
    config = tmp_path / "config_off"
    config.mkdir()
    (config / "output.yaml").write_text("search_internal: false\n")
    conf.instance.push(new_path=str(config))
    conf.instance["general"]["output"]["samples_to_csv"] = True
    yield
    conf.instance.configs = original_configs


def test_summary_only_output_falls_back_to_samples_csv(
    search_internal_off, monkeypatch, caplog
):
    """
    A completed output without ``search_internal/`` (``output.search_internal:
    false``) has no archive: loading raises ``FileNotFoundError``,
    ``Result.search_internal`` is ``None`` and the samples come from ``samples.csv``.
    """
    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")

    analysis = af.ex.Analysis(data=np.full(10, 5.0), noise_map=np.ones(10))
    search = af.Drawer(name="loading_summary_only", total_draws=5)
    search.fit(model=af.Model(af.ex.Gaussian), analysis=analysis)

    resumed = af.Drawer(name="loading_summary_only", total_draws=5)
    result = resumed.fit(model=af.Model(af.ex.Gaussian), analysis=analysis)
    resumed.paths.restore()

    with pytest.raises(FileNotFoundError):
        resumed.paths.load_search_internal()

    assert Result(samples_summary=None, paths=resumed.paths).search_internal is None

    with caplog.at_level(logging.WARNING):
        samples = resumed.samples_from(resumed.paths.model)

    assert len(samples.sample_list) == len(result.samples.sample_list)
    assert any(
        "could not be loaded from its internal results" in record.getMessage()
        for record in caplog.records
    )
