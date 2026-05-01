import pickle

import numpy as np

import autofit as af
from autofit.non_linear.fitness import Fitness


def _make_fitness(**kwargs):
    model = af.Model(af.ex.Gaussian)
    data = np.ones(20)
    noise_map = np.ones(20) * 0.1
    analysis = af.ex.Analysis(data=data, noise_map=noise_map)
    return Fitness(model=model, analysis=analysis, **kwargs)


def test_default_dispatch_is_call():
    fitness = _make_fitness()
    # `self.call` produces a fresh bound method each access — compare the
    # underlying function instead of the bound-method instance.
    assert fitness._call.__func__ is Fitness.call
    assert fitness.use_jax_jit is False
    assert fitness.use_jax_vmap is False


def test_pickle_default_path_unchanged():
    fitness = _make_fitness()

    blob = pickle.dumps(fitness)
    restored = pickle.loads(blob)

    assert restored.use_jax_jit is False
    assert restored.use_jax_vmap is False
    assert restored._call.__func__ is Fitness.call
