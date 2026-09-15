import numpy as np
import pytest

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


class MockDynestyResults:
    def __init__(self, samples, logl, logwt, ncall, logz, nlive):
        self.samples = samples
        self.logl = logl
        self.logwt = logwt
        self.ncall = ncall
        self.logz = logz
        self.nlive = nlive


class MockDynestySampler:
    def __init__(self, results):
        self.results = results


def test__explicit_params():

    search = af.DynestyStatic(
        nlive=151,
        dlogz=0.1,
        iterations_per_full_update=501,
        number_of_cores=2,
    )

    assert search.iterations_per_full_update == 501

    assert search.nlive == 151
    assert search.dlogz == 0.1
    assert search.number_of_cores == 2

    search = af.DynestyStatic()

    assert search.nlive == 50
    assert search.dlogz is None
    assert search.number_of_cores == 1

    search = af.DynestyDynamic(
        facc=0.4,
        iterations_per_full_update=501,
        dlogz_init=0.2,
        number_of_cores=3,
    )

    assert search.iterations_per_full_update == 501

    assert search.facc == 0.4
    assert search.dlogz_init == 0.2
    assert search.number_of_cores == 3

    search = af.DynestyDynamic()

    assert search.facc == 0.2
    assert search.dlogz_init == 0.01
    assert search.number_of_cores == 1


@pytest.mark.parametrize("search_cls", [af.DynestyStatic, af.DynestyDynamic])
def test__single_core_builds_no_pool(search_cls, monkeypatch):
    """
    number_of_cores=1 must not construct a multiprocessing pool: dynesty treats
    pool=None as fully serial, whereas a Pool(1) object forces every likelihood
    call into a forked worker — which deadlocks in XLA compilation when the
    likelihood touches JAX, and hangs forever if that single worker dies,
    because the pool never re-issues a dead worker's in-flight task. Nautilus
    received the same fix in #1442/#1443; this is #1630 for dynesty.
    """
    import importlib

    def no_fork_context(*args, **kwargs):
        raise AssertionError("fork_context must not be used when number_of_cores == 1")

    parallel = importlib.import_module("autofit.non_linear.parallel")

    # `_fork_pool_cls` imports `fork_context` from the package at call time, so
    # patching the package attribute is what intercepts the pool build.
    monkeypatch.setattr(parallel, "fork_context", no_fork_context)

    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")

    model = af.Model(af.ex.Gaussian)
    analysis = af.ex.Analysis(
        data=np.full(100, 5.0),
        noise_map=np.full(100, 1.0),
    )

    live_point_kwargs = (
        {"nlive": 10} if search_cls is af.DynestyStatic else {"nlive_init": 10}
    )

    search = search_cls(
        name="dynesty_single_core",
        unique_tag="single_core_no_pool_test",
        number_of_cores=1,
        **live_point_kwargs,
    )

    search.fit(model=model, analysis=analysis)
