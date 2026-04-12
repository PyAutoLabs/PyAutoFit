import pytest

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def test__explicit_params():
    search = af.Drawer(
        total_draws=5,
        initializer=af.InitializerBall(lower_limit=0.2, upper_limit=0.8),
    )

    assert search.total_draws == 5
    assert isinstance(search.initializer, af.InitializerBall)
    assert search.initializer.lower_limit == 0.2
    assert search.initializer.upper_limit == 0.8
    assert search.number_of_cores == 1

    search = af.Drawer()

    assert search.total_draws == 50
    assert isinstance(search.initializer, af.InitializerBall)
