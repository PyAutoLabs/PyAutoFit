import numpy as np

import autofit as af
from autofit.non_linear.fitness import Fitness


def test_fitness_returns_resample_fom_on_assertion_failure():
    """
    If an added assertion rejects a sampled parameter vector, ``Fitness.call``
    must return ``resample_figure_of_merit`` rather than letting the
    ``FitException`` escape into the non-linear search. Regression for
    the DynestyStatic-blocking bug where ``instance_from_vector`` was
    called outside the ``try/except FitException`` block.
    """
    gaussian_0 = af.Model(af.ex.Gaussian)
    gaussian_1 = af.Model(af.ex.Gaussian)
    gaussian_0.add_assertion(gaussian_0.centre > gaussian_1.centre)
    model = af.Collection(gaussian_0=gaussian_0, gaussian_1=gaussian_1)

    data = np.ones(20)
    noise_map = np.ones(20) * 0.1
    analysis = af.ex.Analysis(data=data, noise_map=noise_map)

    fitness = Fitness(model=model, analysis=analysis)

    # centre_0 (index 0) = 10 < centre_1 (index 3) = 20  → assertion violated
    violating = [10.0, 1.0, 1.0, 20.0, 1.0, 1.0]
    assert fitness.call(violating) == fitness.resample_figure_of_merit

    # centre_0 = 20 > centre_1 = 10  → assertion satisfied, real FOM returned
    satisfying = [20.0, 1.0, 1.0, 10.0, 1.0, 1.0]
    assert fitness.call(satisfying) != fitness.resample_figure_of_merit
