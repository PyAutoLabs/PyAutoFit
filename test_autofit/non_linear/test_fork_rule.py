"""
The one JAX fork rule and `PoolFactory` (`autofit.non_linear.parallel.pool`,
search-extensibility phase A2, decision D11).

A JAX analysis is never evaluated through a forked pool: a search given one with
`number_of_cores > 1` runs on one core, says so in exactly one INFO line, and records the
requested and effective counts in `search.summary`. Forking is unchanged everywhere else.
"""
import logging
import zipfile

import numpy as np
import pytest

import autofit as af
from autofit import exc
from autofit.non_linear.parallel import pool as pool_module
from autofit.non_linear.parallel import PoolFactory, effective_number_of_cores

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


class _Analysis:
    def __init__(self, is_jax):
        self.is_jax = is_jax


def _rule_records(caplog):
    return [
        record
        for record in caplog.records
        if record.name == pool_module.logger.name and "running with 1 core" in record.message
    ]


def test_rule_downgrades_jax_only(caplog):
    with caplog.at_level(logging.INFO):
        assert effective_number_of_cores(4, is_jax=False, where="X") == 4
        assert effective_number_of_cores(1, is_jax=True, where="X") == 1
        assert _rule_records(caplog) == []

        assert effective_number_of_cores(4, is_jax=True, where="X") == 1

    assert len(_rule_records(caplog)) == 1


def test_pool_factory_under_jax_builds_no_pool(monkeypatch):
    def no_fork():
        raise AssertionError("must not fork")

    monkeypatch.setattr(pool_module, "fork_context", no_fork)

    search = af.Emcee(number_of_cores=2)
    factory = PoolFactory(search, _Analysis(is_jax=True))

    assert factory.requested == 2
    assert factory.number_of_cores == 1
    assert factory() is None
    assert factory(n_cores=8) is None
    assert factory.sneaky(fitness=None) is None
    assert "Number of cores = 1 (requested 2)" in factory.summary
    assert "JAX analyses never fork" in factory.summary


def test_pool_factory_on_numpy_keeps_the_request():
    search = af.Emcee(number_of_cores=3)
    factory = PoolFactory(search, _Analysis(is_jax=False))

    assert factory.number_of_cores == 3
    assert factory.summary == "Number of cores = 3 (requested 3)\n"


def test_emcee_jax_two_cores_runs_single_core_and_records_it(monkeypatch, caplog, tmp_path):
    pytest.importorskip("jax")
    from autonerves import conf

    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")

    def no_fork(*args, **kwargs):
        raise AssertionError("must not fork")

    monkeypatch.setattr(pool_module, "fork_context", no_fork)
    monkeypatch.setattr("autofit.non_linear.parallel.sneaky.SneakyPool", no_fork)

    analysis = af.ex.Analysis(data=np.ones(20), noise_map=np.ones(20), use_jax=True)

    search = af.Emcee(
        name="fork_rule_emcee",
        path_prefix=str(tmp_path),
        number_of_cores=2,
        nwalkers=8,
        nsteps=4,
        auto_correlation_settings=af.AutoCorrelationsSettings(check_for_convergence=False),
    )

    with caplog.at_level(logging.INFO):
        search.fit(model=af.Model(af.ex.Gaussian), analysis=analysis)

    assert len(_rule_records(caplog)) == 1
    assert search.number_of_cores == 2  # the user's request is never rewritten

    with zipfile.ZipFile(search.paths._zip_path) as archive:
        summary = archive.read("search.summary").decode()

    assert "Number of cores = 1 (requested 2)" in summary


def test_analysis_pool_refuses_a_jax_analysis():
    from autofit.non_linear.analysis.multiprocessing import AnalysisPool

    with pytest.raises(exc.SearchException, match="JAX analyses never fork"):
        AnalysisPool([_Analysis(is_jax=True)], n_cores=2)
