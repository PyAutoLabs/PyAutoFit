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


def test_jit_dispatch_sets_call_to_jit():
    fitness = _make_fitness(use_jax_jit=True)
    assert fitness.use_jax_jit is True
    assert fitness._call is fitness._jit


def test_vmap_takes_precedence_over_jit():
    fitness = _make_fitness(use_jax_jit=True, use_jax_vmap=True)
    assert fitness._call is fitness._vmap


def test_pickle_strips_jax_cached_attrs():
    """
    Dynesty's checkpoint writes pickle the loglikelihood. JAX-compiled
    callables (jax.jit / jax.vmap / jax.grad) carry C++ XLA state that
    cannot roundtrip through pickle. ``Fitness.__getstate__`` must drop
    them; ``Fitness.__setstate__`` re-derives the dispatch on resume.
    """
    fitness = _make_fitness(use_jax_jit=True)

    state = fitness.__getstate__()
    assert "_call" not in state
    assert "_jit" not in state
    assert "_vmap" not in state
    assert "_grad" not in state

    blob = pickle.dumps(fitness)
    restored = pickle.loads(blob)

    assert restored.use_jax_jit is True
    assert restored._call is restored._jit


def test_pickle_default_path_unchanged():
    fitness = _make_fitness()

    blob = pickle.dumps(fitness)
    restored = pickle.loads(blob)

    assert restored.use_jax_jit is False
    assert restored.use_jax_vmap is False
    assert restored._call.__func__ is Fitness.call
