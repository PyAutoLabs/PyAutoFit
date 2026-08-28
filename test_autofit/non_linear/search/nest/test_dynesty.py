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
