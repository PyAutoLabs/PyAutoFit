import importlib.util

import numpy as np
import pytest

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

# The regression test below runs a real `search.fit`, which imports the
# `nautilus` sampler; it ships via the `[optional]` extras. Envs installed
# without those extras must skip rather than fail with
# `No module named 'nautilus'` — the other tests here only read config.
requires_nautilus = pytest.mark.skipif(
    importlib.util.find_spec("nautilus") is None,
    reason="requires nautilus-sampler (installed via the [optional] extras)",
)


def test__explicit_params():
    search = af.Nautilus(
        n_live=500,
        n_batch=200,
        f_live=0.05,
        n_eff=1000,
        iterations_per_full_update=501,
        number_of_cores=4,
    )

    assert search.iterations_per_full_update == 501

    assert search.n_live == 500
    assert search.n_batch == 200
    assert search.f_live == 0.05
    assert search.n_eff == 1000
    assert search.number_of_cores == 4

    search = af.Nautilus()

    assert search.n_live == 3000
    assert search.n_batch == 100
    assert search.enlarge_per_dim == 1.1
    assert search.split_threshold == 100
    assert search.n_networks == 4
    assert search.vectorized is False
    assert search.f_live == 0.01
    assert search.n_shell == 1
    assert search.n_eff == 500
    assert search.discard_exploration is False
    assert search.number_of_cores == 1


def test__identifier_fields():
    search = af.Nautilus()

    assert "n_live" in search.__identifier_fields__
    assert "n_eff" in search.__identifier_fields__
    assert "n_shell" in search.__identifier_fields__


def test__test_mode():
    # #1541: `n_like_max = 1` stopped nautilus after its first batch of prior
    # draws, returning a posterior whose effective sample size was 1. Test mode
    # must instead deliver a small but real posterior.
    search = af.Nautilus()
    search.apply_test_mode()

    assert search.n_like_max > 1
    assert search.n_like_max <= 2000
    assert search.n_live < af.Nautilus().n_live
    assert search.n_networks == 0


@requires_nautilus
def test__single_core_builds_no_pool(monkeypatch):
    """
    number_of_cores=1 must not construct a multiprocessing pool: nautilus
    treats pool=None as fully serial, whereas a Pool(1) object forces every
    likelihood call into a forked worker — which deadlocks in XLA compilation
    when the likelihood touches JAX (#1442).
    """
    from autofit.non_linear.search.nest.nautilus import search as nautilus_search

    def no_fork_context():
        raise AssertionError(
            "fork_context must not be used when number_of_cores == 1"
        )

    monkeypatch.setattr(nautilus_search, "fork_context", no_fork_context)
    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")

    model = af.Model(af.ex.Gaussian)
    analysis = af.ex.Analysis(
        data=np.full(100, 5.0),
        noise_map=np.full(100, 1.0),
    )

    search = af.Nautilus(
        name="nautilus_single_core",
        unique_tag="single_core_no_pool_test",
        n_live=10,
        number_of_cores=1,
    )

    search.fit(model=model, analysis=analysis)
