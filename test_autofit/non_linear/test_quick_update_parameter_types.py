import numpy as np
import pytest

import autofit as af
from autofit.non_linear.fitness import Fitness

# `manage_quick_update` receives whatever type the calling search holds its
# parameters in. Nautilus passes an ndarray, so `self.quick_update_max_lh_parameters
# .tolist()` worked for as long as Nautilus was the only search wired up to quick
# updates. Dynesty's initializer passes a plain Python list, and the moment the
# other searches were wired up (PyAutoFit#1434) that call raised
# `AttributeError: 'list' object has no attribute 'tolist'` mid-fit.


class RecordingPaths:
    """Captures the rendered result info instead of writing it to disk."""

    def __init__(self):
        self.results = []

    def output_model_results(self, result_info):
        self.results.append(result_info)


def _fitness(iterations_per_quick_update):
    model = af.Model(af.ex.Gaussian)
    data = np.array(
        af.ex.Gaussian(centre=50.0, normalization=25.0, sigma=10.0)(
            xvalues=np.arange(30)
        )
    )
    analysis = af.ex.Analysis(data=data, noise_map=np.ones(30))
    fitness = Fitness(
        model=model,
        analysis=analysis,
        iterations_per_quick_update=iterations_per_quick_update,
    )
    fitness.paths = RecordingPaths()
    return fitness


@pytest.mark.parametrize(
    "parameters",
    [
        [50.0, 25.0, 10.0],  # Dynesty's initializer -- the case that crashed
        np.array([50.0, 25.0, 10.0]),  # Nautilus
        (50.0, 25.0, 10.0),
    ],
    ids=["list", "ndarray", "tuple"],
)
def test__a_quick_update_survives_any_parameter_container(parameters):
    fitness = _fitness(iterations_per_quick_update=1)

    # A cadence of 1 means this single call crosses the threshold and runs the
    # whole update body, which is where the `.tolist()` call lives.
    fitness.manage_quick_update(parameters=parameters, log_likelihood=-10.0)

    assert fitness.quick_update_count == 0  # reset after the update fired

    # The rendered result info is what `.tolist()` feeds, so a non-empty record
    # proves the update body ran to completion rather than dying mid-way.
    assert len(fitness.paths.results) == 1
    assert "centre" in fitness.paths.results[0]


def test__no_update_body_runs_when_the_cadence_is_not_reached():
    # Guards the other side: the update must not fire early, or the parameter
    # handling above would never be the thing under test.
    fitness = _fitness(iterations_per_quick_update=1000)

    fitness.manage_quick_update(parameters=[50.0, 25.0, 10.0], log_likelihood=-10.0)

    assert fitness.quick_update_count == 1
