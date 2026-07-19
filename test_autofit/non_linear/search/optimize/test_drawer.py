import pytest

import autofit as af
from autonerves.dictable import from_dict, to_dict

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


def test__dict_round_trip_with_number_of_cores():
    # A round-tripped Drawer.search.json carries `number_of_cores` at the top
    # level (the resolved value, always 1). Before the fix, this collided with
    # the hardcoded super().__init__(number_of_cores=1, **kwargs) call and
    # raised `TypeError: got multiple values for keyword argument`.
    dictionary = to_dict(af.Drawer(total_draws=3))
    assert "number_of_cores" in dictionary["arguments"]

    restored = from_dict(dictionary)
    assert isinstance(restored, af.Drawer)
    assert restored.total_draws == 3
    assert restored.number_of_cores == 1
