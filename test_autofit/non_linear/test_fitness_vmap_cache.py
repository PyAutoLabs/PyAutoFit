"""
What the batched JAX likelihood path compiles, and how often (PyAutoFit#1636).

`Fitness._vmap` composes the batched likelihood as `jax.jit(jax.vmap(call))` rather than
`jax.vmap(jax.jit(call))`, so the whole batch is a single XLA program cached by the outer jit and
keyed on the batch shape. These tests witness that ordering from the outside: the likelihood is
traced exactly once per batch shape, a repeat call at the same shape is a cache hit, and the values
are bitwise identical to the old composition's.

The compile-count probe is `_vmap.__wrapped__._cache_size()` — `log_on_first_compile` keeps the
jitted callable reachable on `__wrapped__` for exactly this reason, so on the old composition (where
the wrapped callable was the vmap object, which has no cache) these tests cannot pass.
"""

import pickle

import numpy as np
import pytest

import autofit as af
from autofit.non_linear.fitness import Fitness

jax = pytest.importorskip("jax")


class CountingAnalysis(af.ex.Analysis):
    """
    The example 1D Gaussian analysis, counting how many times its likelihood function runs.

    Under `jax.jit` the Python body runs only while tracing, so the counter is a count of traces —
    not of evaluations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_count = 0

    def log_likelihood_function(self, instance, **kwargs):
        self.trace_count += 1
        return super().log_likelihood_function(instance=instance, **kwargs)


def _make_fitness(analysis_cls=af.ex.Analysis, **kwargs):
    model = af.Model(af.ex.Gaussian)
    analysis = analysis_cls(
        data=np.ones(20),
        noise_map=np.ones(20) * 0.1,
        use_jax=True,
    )
    return Fitness(model=model, analysis=analysis, **kwargs)


def _batch(model, rows):
    """A `(rows, d)` batch of the model's prior-median parameter vector."""
    return np.tile(np.array(model.physical_values_from_prior_medians), (rows, 1))


def test_the_batched_likelihood_is_traced_once_and_cached_per_shape():
    fitness = _make_fitness(analysis_cls=CountingAnalysis, use_jax_vmap=True)
    parameters = _batch(fitness.model, 4)

    first = fitness._vmap(parameters)
    second = fitness._vmap(parameters)

    # One trace for the whole batch, and the second call never re-enters Python: it is served
    # from the outer jit's C++ fast path on the executable cached for this batch shape.
    assert fitness.analysis.trace_count == 1
    assert fitness._vmap.__wrapped__._cache_size() == 1

    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert np.asarray(first).shape == (4,)

    # A different batch length is a different shape, so it is a different executable. That the
    # cache grows per batch length is the separate, known cost written up in the PyAutoMind draft
    # `vmap_jit_recompiles_per_nautilus_batch_length.md`; this test pins it, it does not bless it.
    fitness._vmap(_batch(fitness.model, 3))
    assert fitness._vmap.__wrapped__._cache_size() == 2


def test_jit_of_vmap_matches_vmap_of_jit_bitwise():
    fitness = _make_fitness(use_jax_vmap=True)

    medians = np.array(fitness.model.physical_values_from_prior_medians)
    parameters = medians[None, :] * np.linspace(0.8, 1.2, 5)[:, None]

    jit_of_vmap = np.asarray(jax.jit(jax.vmap(fitness.call))(parameters))
    vmap_of_jit = np.asarray(jax.vmap(jax.jit(fitness.call))(parameters))

    assert np.array_equal(jit_of_vmap, vmap_of_jit)


def test_call_wrap_promotes_a_vector_to_a_batch_on_the_vmap_path():
    fitness = _make_fitness(use_jax_vmap=True)
    vector = np.array(fitness.model.physical_values_from_prior_medians)

    batched = np.asarray(fitness.call_wrap(vector))

    assert batched.shape == (1,)
    assert np.array_equal(batched, np.asarray(fitness.call(vector))[None])


def test_vmap_survives_a_pickle_round_trip():
    fitness = _make_fitness(use_jax_vmap=True)
    parameters = _batch(fitness.model, 4)

    expected = np.asarray(fitness._vmap(parameters))

    restored = pickle.loads(pickle.dumps(fitness))
    restored_result = np.asarray(restored._vmap(parameters))

    assert np.array_equal(restored_result, expected)
    # `__getstate__` strips `_vmap`, so the restored object rebuilt it: a cache of exactly one
    # entry after one call is what a freshly compiled function looks like.
    assert restored._vmap.__wrapped__._cache_size() == 1
